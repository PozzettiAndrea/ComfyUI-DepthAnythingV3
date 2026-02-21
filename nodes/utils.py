"""Shared utilities for DepthAnythingV3 nodes."""
import json
import torch
import torch.nn.functional as F
import logging

import comfy.model_management as mm

logger = logging.getLogger("DepthAnythingV3")

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_PATCH_SIZE = 14


def imagenet_normalize(images_pt):
    """Apply ImageNet normalization without torchvision dependency.

    Args:
        images_pt: Tensor with shape [B, C, H, W] in [0, 1] range

    Returns:
        Normalized tensor
    """
    mean = torch.tensor(IMAGENET_MEAN, device=images_pt.device, dtype=images_pt.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images_pt.device, dtype=images_pt.dtype).view(1, 3, 1, 1)
    return (images_pt - mean) / std


def format_camera_params(param_list, param_name):
    """Format camera parameters as JSON string.

    Args:
        param_list: List of camera parameter tensors (or None values)
        param_name: Name of the parameter type (e.g., 'intrinsics', 'extrinsics')

    Returns:
        JSON string with formatted parameters
    """
    if all(p is None for p in param_list):
        return json.dumps({param_name: "Not available (mono/metric model)"})

    formatted = []
    for i, param in enumerate(param_list):
        if param is not None:
            # Convert tensor to list for JSON serialization
            formatted.append({
                f"image_{i}": param.squeeze().tolist()
            })
        else:
            formatted.append({
                f"image_{i}": None
            })

    return json.dumps({param_name: formatted}, indent=2)


def check_model_capabilities(model):
    """Check what capabilities a model has.

    Args:
        model: The DA3 model

    Returns:
        Dictionary of capabilities
    """
    has_camera = (
        hasattr(model, 'cam_enc') and model.cam_enc is not None and
        hasattr(model, 'cam_dec') and model.cam_dec is not None
    )

    # Main series models have camera support, Mono/Metric don't
    # Sky is available on Mono/Metric (DPT head), not on Main series (DualDPT head)
    has_sky = not has_camera  # Inverse relationship

    # Nested model has both (camera from main branch, sky from metric branch)
    is_nested = hasattr(model, 'da3') and hasattr(model, 'da3_metric')
    if is_nested:
        has_sky = True
        has_camera = True

    has_gs = (
        hasattr(model, 'gs_head') and model.gs_head is not None and
        hasattr(model, 'gs_adapter') and model.gs_adapter is not None
    )

    return {
        "has_camera_conditioning": has_camera,
        "has_sky_segmentation": has_sky,
        "has_multiview_attention": has_camera,
        "has_3d_gaussians": has_gs,
        "is_nested": is_nested,
    }


def process_tensor_to_image(tensor_list, orig_H, orig_W, normalize_output=False, skip_resize=False):
    """Convert list of depth/conf tensors to ComfyUI IMAGE format.

    Args:
        tensor_list: List of tensors with shape [1, H, W] or [H, W]
        orig_H: Original image height
        orig_W: Original image width
        normalize_output: If True, clamp output to 0-1 range
        skip_resize: If True, keep model's native output size instead of resizing back

    Returns:
        Tensor with shape [B, H, W, 3] in ComfyUI IMAGE format
    """
    # Concatenate all tensors
    out = torch.cat(tensor_list, dim=0)  # [B, 1, H, W] or [B, H, W]

    # Ensure 4D: [B, 1, H, W]
    if out.dim() == 3:
        out = out.unsqueeze(1)

    # Convert to 3-channel image [B, H, W, 3]
    out = out.squeeze(1)  # [B, H, W]
    out = out.unsqueeze(-1).repeat(1, 1, 1, 3).cpu().float()  # [B, H, W, 3]

    # Resize back to original dimensions (with even constraint) unless skip_resize is True
    if not skip_resize:
        final_H = (orig_H // 2) * 2
        final_W = (orig_W // 2) * 2

        if out.shape[1] != final_H or out.shape[2] != final_W:
            out = F.interpolate(
                out.permute(0, 3, 1, 2),
                size=(final_H, final_W),
                mode="bilinear"
            ).permute(0, 2, 3, 1)

    if normalize_output:
        return torch.clamp(out, 0, 1)
    return out


def process_tensor_to_mask(tensor_list, orig_H, orig_W, skip_resize=False):
    """Convert list of tensors to ComfyUI MASK format.

    Args:
        tensor_list: List of tensors with shape [1, H, W] or [H, W]
        orig_H: Original image height
        orig_W: Original image width
        skip_resize: If True, keep model's native output size instead of resizing back

    Returns:
        Tensor with shape [B, H, W] in ComfyUI MASK format
    """
    # Concatenate all tensors
    out = torch.cat(tensor_list, dim=0)  # [B, 1, H, W] or [B, H, W]

    # Ensure 3D: [B, H, W]
    if out.dim() == 4:
        out = out.squeeze(1)  # [B, H, W]

    out = out.cpu().float()

    # Resize back to original dimensions (with even constraint) unless skip_resize is True
    if not skip_resize:
        final_H = (orig_H // 2) * 2
        final_W = (orig_W // 2) * 2

        if out.shape[1] != final_H or out.shape[2] != final_W:
            out = F.interpolate(
                out.unsqueeze(1),  # [B, 1, H, W] for interpolation
                size=(final_H, final_W),
                mode="bilinear"
            ).squeeze(1)  # Back to [B, H, W]

    return torch.clamp(out, 0, 1)


def resize_to_patch_multiple(images_pt, patch_size=DEFAULT_PATCH_SIZE, method="resize"):
    """Resize images to be divisible by patch size.

    Args:
        images_pt: Tensor with shape [B, C, H, W]
        patch_size: Patch size to align to (default 14)
        method: How to handle non-divisible sizes:
            - "resize": Resize to nearest patch multiple (default, preserves content)
            - "crop": Center crop to floor patch multiple (loses edges)
            - "pad": Pad to ceiling patch multiple (adds black padding)

    Returns:
        Tuple of (resized_images, original_H, original_W)
    """
    _, _, H, W = images_pt.shape
    orig_H, orig_W = H, W

    if H % patch_size == 0 and W % patch_size == 0:
        return images_pt, orig_H, orig_W

    if method == "crop":
        # Center crop to floor of patch multiple
        new_H = (H // patch_size) * patch_size
        new_W = (W // patch_size) * patch_size

        if new_H == 0 or new_W == 0:
            raise ValueError(f"Image too small for patch size {patch_size}. Min size: {patch_size}x{patch_size}")

        # Calculate crop offsets (center crop)
        top = (H - new_H) // 2
        left = (W - new_W) // 2

        images_pt = images_pt[:, :, top:top+new_H, left:left+new_W]
        logger.debug(f"Cropped from {orig_H}x{orig_W} to {new_H}x{new_W} (center crop)")

    elif method == "pad":
        # Pad to ceiling of patch multiple
        new_H = ((H + patch_size - 1) // patch_size) * patch_size
        new_W = ((W + patch_size - 1) // patch_size) * patch_size

        # Calculate padding (pad bottom and right)
        pad_bottom = new_H - H
        pad_right = new_W - W

        # F.pad expects (left, right, top, bottom) for 4D tensor
        images_pt = F.pad(images_pt, (0, pad_right, 0, pad_bottom), mode='constant', value=0)
        logger.debug(f"Padded from {orig_H}x{orig_W} to {new_H}x{new_W} (zero padding)")

    else:  # method == "resize" (default)
        # Resize to nearest patch multiple
        def nearest_multiple(x, p):
            down = (x // p) * p
            up = down + p
            return up if abs(up - x) <= abs(x - down) else down

        new_H = nearest_multiple(H, patch_size)
        new_W = nearest_multiple(W, patch_size)

        if new_H == 0:
            new_H = patch_size
        if new_W == 0:
            new_W = patch_size

        images_pt = F.interpolate(images_pt, size=(new_H, new_W), mode="bilinear", align_corners=False)
        logger.debug(f"Resized from {orig_H}x{orig_W} to {new_H}x{new_W} (nearest multiple)")

    return images_pt, orig_H, orig_W


def save_gaussians_to_ply(gaussians, save_path, depth=None, extrinsics=None,
                           shift_and_scale=False, save_sh_dc_only=True,
                           prune_border=True, prune_depth_percent=0.9):
    """Save Gaussians to PLY file in standard 3DGS format.

    Ported from original DA3 gsply_helpers.export_ply + save_gaussian_ply.

    Args:
        gaussians: Object with .means, .scales, .rotations, .harmonics, .opacities
                   All tensors shape (B, N, ...) where B=1, N = V*H*W
        save_path: Output PLY file path (str or Path)
        depth: Optional depth tensor (V, H, W) or (V, H, W, 1) for spatial pruning.
               Provides shape info for border pruning + depth values for far pruning.
        extrinsics: Optional world-to-camera extrinsics tensor (4x4). If provided,
                    positions are transformed from world space to camera space via rigid
                    transform, preserving scale/position relationship.
        shift_and_scale: If True, normalize positions to ~[-1, 1] (default: False)
        save_sh_dc_only: If True, save only DC band SH coefficients (default: True)
        prune_border: If True, remove border Gaussians (requires depth for spatial shape)
        prune_depth_percent: Keep closest N% by depth (0.9 = keep 90%). Set 1.0 to skip.
    """
    import numpy as np
    from pathlib import Path

    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        logger.warning("plyfile not installed - cannot save Gaussians to PLY. "
                       "Install with: pip install plyfile")
        return ""

    # Extract tensors from Gaussians object (handle both object and dict access)
    if isinstance(gaussians, dict):
        means = gaussians['means']
        scales = gaussians['scales']
        rotations = gaussians['rotations']
        harmonics = gaussians['harmonics']
        opacities = gaussians['opacities']
    else:
        means = gaussians.means
        scales = gaussians.scales
        rotations = gaussians.rotations
        harmonics = gaussians.harmonics
        opacities = gaussians.opacities

    # Ensure batch dim exists, then take batch 0
    if means.dim() == 2:
        means = means.unsqueeze(0)
        scales = scales.unsqueeze(0)
        rotations = rotations.unsqueeze(0)
        harmonics = harmonics.unsqueeze(0)
        opacities = opacities.unsqueeze(0)

    # Work with batch=0: shapes are (N, 3), (N, 3), (N, 4), (N, 3, d_sh), (N,)
    world_means = means[0]
    gs_scales = scales[0]
    world_rotations = rotations[0]
    world_shs = harmonics[0]
    raw_opacities = opacities[0]
    if raw_opacities.dim() > 1:
        raw_opacities = raw_opacities.squeeze(-1)

    # Convert opacity to logit space: inverse_sigmoid(opacity)
    raw_opacities = raw_opacities.clamp(1e-6, 1.0 - 1e-6)
    gs_opacities = torch.log(raw_opacities / (1.0 - raw_opacities))

    # --- Pruning (requires depth for spatial layout) ---
    if depth is not None:
        depth_t = depth
        if depth_t.dim() == 4:
            depth_t = depth_t.squeeze(-1)  # (V, H, W)
        if depth_t.dim() == 2:
            depth_t = depth_t.unsqueeze(0)  # (1, H, W)
        src_v, out_h, out_w = depth_t.shape

        N = world_means.shape[0]
        expected_N = src_v * out_h * out_w
        if N == expected_N:
            # Reshape from flat (V*H*W, ...) to spatial (V, H, W, ...)
            spatial_means = world_means.reshape(src_v, out_h, out_w, -1)
            spatial_scales = gs_scales.reshape(src_v, out_h, out_w, -1)
            spatial_rots = world_rotations.reshape(src_v, out_h, out_w, -1)
            spatial_shs = world_shs.reshape(src_v, out_h, out_w, *world_shs.shape[1:])
            spatial_opac = gs_opacities.reshape(src_v, out_h, out_w)

            # --- Transform positions from world space to camera space ---
            # The GaussianAdapter computes positions AND scales in the same world
            # coordinate system. Applying extrinsics (world-to-camera rigid transform)
            # preserves the scale/position relationship — scales don't change under
            # rotation (same approach as Sharp's covariance transform).
            if extrinsics is not None:
                # Parse extrinsics: squeeze to (4, 4)
                E = extrinsics
                while E.dim() > 2:
                    E = E.squeeze(0)
                E = E.to(spatial_means.device, dtype=spatial_means.dtype)
                R = E[:3, :3]  # rotation (world-to-camera)
                t = E[:3, 3]   # translation

                # Transform all positions: cam_pos = R @ world_pos + t
                flat_means = spatial_means.reshape(-1, 3)
                cam_means = (flat_means @ R.T) + t
                # Flip Y and Z: OpenCV (Y-down, Z-forward) -> viewer (Y-up, Z-backward)
                cam_means[:, 1] *= -1
                cam_means[:, 2] *= -1
                spatial_means = cam_means.reshape(src_v, out_h, out_w, 3)

                # Scales are unchanged — rigid transform preserves singular values
                logger.info(f"Transformed Gaussian positions to camera space via extrinsics")

            # Build mask
            if prune_border:
                mask = torch.zeros(src_v, out_h, out_w, dtype=torch.bool,
                                   device=world_means.device)
                gstrim_h = max(int(8 / 256 * out_h), 1)
                gstrim_w = max(int(8 / 256 * out_w), 1)
                mask[:, gstrim_h:-gstrim_h, gstrim_w:-gstrim_w] = True
            else:
                mask = torch.ones(src_v, out_h, out_w, dtype=torch.bool,
                                  device=world_means.device)

            # Depth pruning
            if prune_depth_percent < 1.0:
                d_percentile = torch.quantile(
                    depth_t.reshape(src_v, -1).float(),
                    q=prune_depth_percent, dim=1
                ).reshape(-1, 1, 1)
                d_mask = depth_t <= d_percentile
                mask = mask & d_mask

            # Apply mask
            world_means = spatial_means[mask]
            gs_scales = spatial_scales[mask]
            world_rotations = spatial_rots[mask]
            world_shs = spatial_shs[mask]
            gs_opacities = spatial_opac[mask]

            logger.info(f"Pruned Gaussians: {N} -> {world_means.shape[0]} "
                        f"(border={prune_border}, depth_pct={prune_depth_percent})")
        else:
            logger.warning(f"Cannot prune: N={N} != V*H*W={expected_N}. Saving all.")

    # --- Optional shift and scale ---
    if shift_and_scale:
        median = world_means.median(dim=0).values
        world_means = world_means - median
        scale_factor = world_means.abs().quantile(0.95, dim=0).max()
        if scale_factor > 0:
            world_means = world_means / scale_factor
            gs_scales = gs_scales / scale_factor

    # --- Build PLY attributes (matching original DA3 format) ---
    f_dc = world_shs[..., 0]  # (N, 3) — DC band
    f_rest = world_shs[..., 1:].flatten(start_dim=1) if world_shs.shape[-1] > 1 else None

    # Build property list (includes nx/ny/nz dummy normals to match original DA3 format)
    attr_names = ["x", "y", "z", "nx", "ny", "nz"]
    if save_sh_dc_only:
        attr_names += ["f_dc_0", "f_dc_1", "f_dc_2"]
    else:
        attr_names += ["f_dc_0", "f_dc_1", "f_dc_2"]
        if f_rest is not None:
            for i in range(f_rest.shape[1]):
                attr_names.append(f"f_rest_{i}")
    attr_names.append("opacity")
    attr_names += ["scale_0", "scale_1", "scale_2"]
    attr_names += ["rot_0", "rot_1", "rot_2", "rot_3"]

    dtype_full = [(name, "f4") for name in attr_names]

    # Concatenate all attribute arrays
    means_np = world_means.detach().cpu().numpy()
    attr_arrays = [
        means_np,                                                    # x, y, z
        np.zeros_like(means_np),                                     # nx, ny, nz (dummy normals)
        f_dc.detach().cpu().contiguous().numpy(),                    # f_dc_0, f_dc_1, f_dc_2
    ]
    if not save_sh_dc_only and f_rest is not None:
        attr_arrays.append(f_rest.detach().cpu().contiguous().numpy())
    attr_arrays += [
        gs_opacities[..., None].detach().cpu().numpy(),              # opacity (logit)
        gs_scales.log().detach().cpu().numpy(),                      # scale_0,1,2 (log)
        world_rotations.detach().cpu().numpy(),                      # rot_0,1,2,3
    ]

    attributes = np.concatenate(attr_arrays, axis=1)

    N = world_means.shape[0]
    elements = np.empty(N, dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))

    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(save_path))
    logger.info(f"Saved Gaussians ({N} splats, {len(attr_names)} properties) to: {save_path}")
    return str(save_path)
