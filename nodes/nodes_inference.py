"""Basic inference nodes for DepthAnythingV3."""
import torch
import torch.nn.functional as F
from torchvision import transforms
from contextlib import nullcontext

import comfy.model_management as mm
from comfy.utils import ProgressBar

from .utils import (
    IMAGENET_MEAN, IMAGENET_STD, DEFAULT_PATCH_SIZE,
    format_camera_params, process_tensor_to_image, process_tensor_to_mask,
    resize_to_patch_multiple, logger, check_model_capabilities,
    get_or_create_da3_patcher,
)
from .normalization import (
    apply_edge_antialiasing,
    apply_standard_normalization,
    apply_v2_style_normalization,
    apply_raw_normalization,
)


class DepthAnything_V3:
    """
    Unified Depth Anything V3 node with multiple normalization modes.

    This consolidates all depth processing approaches into a single node:
    - Standard: Original V3 min-max normalization
    - V2-Style: Disparity-based normalization with content-aware contrast (by Ltamann/TBG)
    - Raw: No normalization, outputs metric depth for 3D reconstruction

    Always outputs all available data - connect what you need, ignore the rest.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "da3_model": ("DA3MODEL", ),
                "images": ("IMAGE", ),
                "normalization_mode": ([
                    "Standard",
                    "V2-Style",
                    "Raw"
                ], {"default": "V2-Style"}),
            },
            "optional": {
                "camera_params": ("CAMERA_PARAMS", {
                    "visible_when_connected": {
                        "input": "da3_model",
                        "source_widget": "model",
                        "contains": [
                            "da3_small", "da3_base", "da3_large", "da3_giant",
                            "da3nested",
                        ],
                    },
                }),
                "resize_method": (["resize", "crop", "pad"], {
                    "default": "resize",
                    "tooltip": "Model requires dimensions to be multiples of 14. resize: scale image (default), crop: center crop to multiple, pad: add black borders to multiple"
                }),
                "invert_depth": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "OFF (default): close=bright, far=dark. ON: far=bright, close=dark. Consistent across all normalization modes."
                }),
                "keep_model_size": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep model's native patch-aligned output size instead of resizing back to original dimensions"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING", "MASK", "EXTRINSICS", "INTRINSICS", "STRING")
    RETURN_NAMES = ("depth", "confidence", "resized_rgb_image", "ray_origin", "ray_direction", "extrinsics_json", "intrinsics_json", "sky_mask", "extrinsics", "intrinsics", "gaussian_ply_path")
    FUNCTION = "process"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Unified Depth Anything V3 node - all outputs, multiple normalization modes.

**Normalization Modes:**
- Standard: Original V3 min-max normalization (0-1 range, includes sky)
- V2-Style: Disparity-based with content-aware contrast (default, best for ControlNet)
  - Sky appears BLACK (like V2)
  - Content-only normalization with percentile-based contrast
  - Enhanced depth gradations via contrast boost
  - Subtle edge anti-aliasing for natural transitions
  - Contribution by Ltamann (TBG)
- Raw: No normalization, outputs metric depth (for 3D reconstruction/point clouds)

**Outputs (always available):**
- depth: Depth map (normalized or raw, depending on mode)
- confidence: Confidence map (normalized 0-1)
- ray_origin: Ray origin maps (for 3D, normalized for visualization)
- ray_direction: Ray direction maps (for 3D, normalized for visualization)
- extrinsics: Camera extrinsics (predicted camera pose)
- intrinsics: Camera intrinsics (predicted camera parameters)
- sky_mask: Sky segmentation (1=sky, 0=non-sky, Mono/Metric models only)
- gaussian_ply_path: Path to raw 3D Gaussians PLY (Giant model only, empty string if not supported)

**Optional Inputs:**
- camera_params: Connect DA3_CreateCameraParams for camera-conditioned estimation
- resize_method: How to handle patch size alignment (resize/crop/pad)
- invert_depth: Toggle output convention. OFF (default): close=bright. ON: far=bright.
- keep_model_size: Keep model's native output size instead of resizing back

**Note:** Ray maps and camera parameters only available for main series models.
Sky mask only available for Mono/Metric/Nested models.

Connect only the outputs you need - unused outputs are simply ignored.
"""

    def process(self, da3_model, images, normalization_mode="V2-Style", camera_params=None,
                resize_method="resize", invert_depth=False, keep_model_size=False):
        device = mm.get_torch_device()
        patcher = get_or_create_da3_patcher(self, da3_model)
        mm.load_models_gpu([patcher])
        model = patcher.model
        dtype = da3_model["dtype"]
        config = da3_model["config"]

        # Check model capabilities
        capabilities = check_model_capabilities(model)
        if not capabilities["has_sky_segmentation"] and normalization_mode == "V2-Style":
            logger.warning(
                "WARNING: This model does not support sky segmentation. "
                "V2-Style normalization will work but without sky masking. "
                "Use Mono/Metric/Nested models for best V2-Style results."
            )

        B, H, W, C = images.shape
        logger.info(f"Input image size: {H}x{W}")

        # Convert from ComfyUI format [B, H, W, C] to PyTorch [B, C, H, W]
        images_pt = images.permute(0, 3, 1, 2)

        # Resize to patch size multiple
        images_pt, orig_H, orig_W = resize_to_patch_multiple(images_pt, DEFAULT_PATCH_SIZE, resize_method)
        model_H, model_W = images_pt.shape[2], images_pt.shape[3]
        logger.info(f"Model input size (after resize): {model_H}x{model_W}")

        # Normalize with ImageNet stats
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        normalized_images = normalize(images_pt)

        # Prepare for model: add view dimension [B, N, 3, H, W] where N=1
        normalized_images = normalized_images.unsqueeze(1)

        # Prepare camera parameters if provided
        extrinsics_input = None
        intrinsics_input = None
        if camera_params is not None:
            if capabilities["has_camera_conditioning"]:
                extrinsics_input = camera_params["extrinsics"].to(device).to(dtype)
                intrinsics_input = camera_params["intrinsics"].to(device).to(dtype)
                if extrinsics_input.shape[0] == 1 and B > 1:
                    extrinsics_input = extrinsics_input.expand(B, -1, -1, -1)
                    intrinsics_input = intrinsics_input.expand(B, -1, -1, -1)
                logger.info("Using camera-conditioned depth estimation")
            else:
                logger.warning("Model does not support camera conditioning. Camera params ignored.")

        pbar = ProgressBar(B)
        depth_out = []
        conf_out = []
        sky_out = []
        ray_origin_out = []
        ray_dir_out = []
        extrinsics_list = []
        intrinsics_list = []
        gaussians_list = []

        # Check if model supports 3D Gaussians
        infer_gs = capabilities["has_3d_gaussians"]
        if infer_gs:
            logger.info("Model supports 3D Gaussians - will output raw Gaussians")

        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)

        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            for i in range(B):
                img = normalized_images[i:i+1].to(device)

                # Get camera params for this batch item
                ext_i = extrinsics_input[i:i+1] if extrinsics_input is not None else None
                int_i = intrinsics_input[i:i+1] if intrinsics_input is not None else None

                # Run model forward with optional camera conditioning and Gaussians
                output = model(img, extrinsics=ext_i, intrinsics=int_i, infer_gs=infer_gs)

                # Extract depth
                depth = None
                if hasattr(output, 'depth'):
                    depth = output.depth
                elif isinstance(output, dict) and 'depth' in output:
                    depth = output['depth']

                if depth is None or not torch.is_tensor(depth):
                    raise ValueError("Model output does not contain valid depth tensor")

                # Extract confidence
                conf = None
                if hasattr(output, 'depth_conf'):
                    conf = output.depth_conf
                elif isinstance(output, dict) and 'depth_conf' in output:
                    conf = output['depth_conf']

                if conf is None or not torch.is_tensor(conf):
                    conf = torch.ones_like(depth)

                # Extract sky mask
                sky = None
                if hasattr(output, 'sky'):
                    sky = output.sky
                elif isinstance(output, dict) and 'sky' in output:
                    sky = output['sky']

                if sky is None or not torch.is_tensor(sky):
                    sky = torch.zeros_like(depth)
                else:
                    # Normalize sky mask to 0-1 range
                    sky_min, sky_max = sky.min(), sky.max()
                    if sky_max > sky_min:
                        sky = (sky - sky_min) / (sky_max - sky_min)

                # ===== NORMALIZATION DISPATCH =====
                if normalization_mode == "Raw":
                    depth_processed = apply_raw_normalization(depth, invert_depth)
                elif normalization_mode == "V2-Style":
                    depth_processed = apply_v2_style_normalization(depth, sky, device, invert_depth)
                else:  # "Standard"
                    depth_processed = apply_standard_normalization(depth, invert_depth)

                # Normalize confidence
                conf_range = conf.max() - conf.min()
                if conf_range > 1e-8:
                    conf = (conf - conf.min()) / conf_range
                else:
                    conf = torch.ones_like(conf)

                depth_out.append(depth_processed.cpu())
                conf_out.append(conf.cpu())
                sky_out.append(sky.cpu())

                # Extract ray maps (if available)
                ray = None
                if hasattr(output, 'ray'):
                    ray = output.ray
                elif isinstance(output, dict) and 'ray' in output:
                    ray = output['ray']

                if ray is not None and torch.is_tensor(ray):
                    # ray shape: [B, S, 6, H, W] - first 3 channels are origin, last 3 are direction
                    ray = ray.squeeze(0)  # Remove batch dimension: [S, 6, H, W]
                    ray = ray.squeeze(0)  # Remove view dimension: [6, H, W]

                    ray_origin = ray[:3]  # [3, H, W]
                    ray_dir = ray[3:6]    # [3, H, W]

                    ray_origin_out.append(ray_origin.cpu())
                    ray_dir_out.append(ray_dir.cpu())
                else:
                    # Create dummy ray maps if not available
                    ray_origin_out.append(torch.zeros(3, depth.shape[-2], depth.shape[-1]))
                    ray_dir_out.append(torch.zeros(3, depth.shape[-2], depth.shape[-1]))

                # Extract camera parameters (if available)
                extr = None
                if hasattr(output, 'extrinsics'):
                    extr = output.extrinsics
                elif isinstance(output, dict) and 'extrinsics' in output:
                    extr = output['extrinsics']

                if extr is not None and torch.is_tensor(extr):
                    extrinsics_list.append(extr.cpu())
                else:
                    extrinsics_list.append(None)

                intr = None
                if hasattr(output, 'intrinsics'):
                    intr = output.intrinsics
                elif isinstance(output, dict) and 'intrinsics' in output:
                    intr = output['intrinsics']

                if intr is not None and torch.is_tensor(intr):
                    intr_cpu = intr.cpu()
                    logger.info(f"Model output intrinsics (batch {i}): shape={intr_cpu.shape}, values=\n{intr_cpu.squeeze()}")
                    intrinsics_list.append(intr_cpu)
                else:
                    intrinsics_list.append(None)

                # Extract 3D Gaussians (only if model supports them and we requested them)
                if infer_gs:
                    gs = None
                    if hasattr(output, 'gaussians'):
                        gs = output.gaussians
                    elif isinstance(output, dict) and 'gaussians' in output:
                        gs = output['gaussians']

                    # Validate that gs is actually a Gaussians object, not an empty addict.Dict
                    if gs is not None and hasattr(gs, 'means') and torch.is_tensor(gs.means):
                        gaussians_list.append(gs)

                pbar.update(1)

        # Process outputs based on normalization mode
        normalize_depth_output = (normalization_mode != "Raw")

        depth_final = process_tensor_to_image(depth_out, orig_H, orig_W,
                                               normalize_output=normalize_depth_output,
                                               skip_resize=keep_model_size)
        conf_final = process_tensor_to_image(conf_out, orig_H, orig_W,
                                              normalize_output=True,
                                              skip_resize=keep_model_size)
        sky_final = process_tensor_to_mask(sky_out, orig_H, orig_W, skip_resize=keep_model_size)
        ray_origin_final = self._process_ray_to_image(ray_origin_out, orig_H, orig_W,
                                                       normalize=True, skip_resize=keep_model_size)
        ray_dir_final = self._process_ray_to_image(ray_dir_out, orig_H, orig_W,
                                                    normalize=True, skip_resize=keep_model_size)

        # Process resized RGB image to match depth output dimensions
        rgb_resized = images_pt.permute(0, 2, 3, 1).float().cpu()  # [B, H, W, 3]
        if not keep_model_size:
            final_H = (orig_H // 2) * 2
            final_W = (orig_W // 2) * 2
            if rgb_resized.shape[1] != final_H or rgb_resized.shape[2] != final_W:
                rgb_resized = F.interpolate(
                    rgb_resized.permute(0, 3, 1, 2),
                    size=(final_H, final_W),
                    mode="bilinear"
                ).permute(0, 2, 3, 1)
        rgb_resized = torch.clamp(rgb_resized, 0, 1)

        # Scale intrinsics if we resized back to original dimensions
        if not keep_model_size:
            final_H = (orig_H // 2) * 2
            final_W = (orig_W // 2) * 2
            model_H, model_W = images_pt.shape[2], images_pt.shape[3]

            # Only scale if dimensions actually changed
            if final_H != model_H or final_W != model_W:
                scale_h = final_H / model_H
                scale_w = final_W / model_W
                logger.info(f"Resizing from {model_H}x{model_W} to {final_H}x{final_W}, scale: h={scale_h:.4f}, w={scale_w:.4f}")

                # Scale each intrinsics matrix
                for i, intr in enumerate(intrinsics_list):
                    if intr is not None and torch.is_tensor(intr):
                        # Squeeze to ensure [3, 3] shape (remove batch dimensions)
                        intr_scaled = intr.squeeze().clone()
                        # Scale focal lengths and principal points
                        intr_scaled[0, 0] *= scale_w  # fx
                        intr_scaled[1, 1] *= scale_h  # fy
                        intr_scaled[0, 2] *= scale_w  # cx
                        intr_scaled[1, 2] *= scale_h  # cy
                        logger.info(f"Scaled intrinsics (batch {i}):\n{intr_scaled}")
                        intrinsics_list[i] = intr_scaled

        # Format camera parameters as strings (for backward compatibility)
        extrinsics_str = format_camera_params(extrinsics_list, "extrinsics")
        intrinsics_str = format_camera_params(intrinsics_list, "intrinsics")

        # Prepare tensor outputs for direct connection to other nodes
        # Stack extrinsics: each should be 4x4, output shape [B, 4, 4]
        if extrinsics_list and extrinsics_list[0] is not None:
            extrinsics_tensor = torch.stack([e.squeeze() for e in extrinsics_list if e is not None], dim=0)
        else:
            # Return identity matrices if no extrinsics
            extrinsics_tensor = torch.eye(4).unsqueeze(0).expand(len(depth_out), -1, -1)

        # Stack intrinsics: each should be 3x3, output shape [B, 3, 3]
        if intrinsics_list and intrinsics_list[0] is not None:
            intrinsics_tensor = torch.stack([i.squeeze() if i.dim() > 2 else i for i in intrinsics_list if i is not None], dim=0)
        else:
            # Return default intrinsics if none available
            intrinsics_tensor = torch.eye(3).unsqueeze(0).expand(len(depth_out), -1, -1)

        # Save Gaussians to PLY file if available (Giant model only)
        gaussian_ply_path = ""
        if gaussians_list:
            gaussian_ply_path = self._save_gaussians_to_ply(gaussians_list)

        return (depth_final, conf_final, rgb_resized, ray_origin_final, ray_dir_final,
                extrinsics_str, intrinsics_str, sky_final, extrinsics_tensor, intrinsics_tensor, gaussian_ply_path)

    def _process_ray_to_image(self, ray_list, orig_H, orig_W, normalize=True, skip_resize=False):
        """Convert list of ray tensors to ComfyUI IMAGE format."""
        # Concatenate all ray tensors
        out = torch.cat([r.unsqueeze(0) for r in ray_list], dim=0)  # [B, 3, H, W]

        if normalize:
            # Normalize each batch independently for visualization
            for i in range(out.shape[0]):
                ray_batch = out[i]  # [3, H, W]
                ray_min = ray_batch.min()
                ray_max = ray_batch.max()
                if ray_max > ray_min:
                    out[i] = (ray_batch - ray_min) / (ray_max - ray_min)
                else:
                    out[i] = torch.zeros_like(ray_batch)

        # Convert to ComfyUI format [B, H, W, 3]
        out = out.permute(0, 2, 3, 1).float()  # [B, H, W, 3]

        # Resize back to original dimensions unless skip_resize is True
        if not skip_resize:
            final_H = (orig_H // 2) * 2
            final_W = (orig_W // 2) * 2

            if out.shape[1] != final_H or out.shape[2] != final_W:
                out = F.interpolate(
                    out.permute(0, 3, 1, 2),
                    size=(final_H, final_W),
                    mode="bilinear"
                ).permute(0, 2, 3, 1)

        if normalize:
            return torch.clamp(out, 0, 1)
        else:
            return out

    def _save_gaussians_to_ply(self, gaussians_list):
        """Save raw Gaussians to PLY file and return the path."""
        import numpy as np
        from pathlib import Path
        import folder_paths

        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            logger.warning("plyfile not installed - cannot save Gaussians to PLY")
            return ""

        # Concatenate all Gaussians
        means = torch.cat([g.means for g in gaussians_list], dim=0).cpu().numpy()
        scales = torch.cat([g.scales for g in gaussians_list], dim=0).cpu().numpy()
        rotations = torch.cat([g.rotations for g in gaussians_list], dim=0).cpu().numpy()
        harmonics = torch.cat([g.harmonics for g in gaussians_list], dim=0).cpu().numpy()
        opacities = torch.cat([g.opacities for g in gaussians_list], dim=0).cpu().numpy()

        B = means.shape[0]
        output_dir = Path(folder_paths.get_output_directory())
        output_dir.mkdir(parents=True, exist_ok=True)

        file_paths = []
        for b in range(B):
            xyz = means[b]
            scale = scales[b]
            rot = rotations[b]
            sh = harmonics[b]
            opacity = opacities[b] if opacities.ndim == 2 else opacities[b].squeeze()

            # Normalize coordinates to [-1, 1] range (shift_and_scale from original DA3)
            # This makes the PLY compatible with standard 3DGS viewers
            xyz_median = np.median(xyz, axis=0)
            xyz = xyz - xyz_median  # Center at origin
            scale_factor = np.quantile(np.abs(xyz), 0.95, axis=0).max()
            if scale_factor > 0:
                xyz = xyz / scale_factor
                scale = scale / scale_factor  # Scale Gaussian sizes proportionally
            logger.info(f"Normalized coordinates: center offset={xyz_median}, scale_factor={scale_factor:.4f}")

            N = xyz.shape[0]
            d_sh = sh.shape[-1]

            # Build dtype
            dtype_list = [
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ]
            if d_sh > 1:
                for i in range(1, d_sh):
                    for c in range(3):
                        dtype_list.append((f'f_rest_{(i-1)*3 + c}', 'f4'))
            dtype_list.append(('opacity', 'f4'))
            dtype_list.extend([('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4')])
            dtype_list.extend([('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')])

            vertices = np.zeros(N, dtype=dtype_list)
            vertices['x'] = xyz[:, 0]
            vertices['y'] = xyz[:, 1]
            vertices['z'] = xyz[:, 2]
            vertices['nx'] = 0
            vertices['ny'] = 0
            vertices['nz'] = 0
            vertices['f_dc_0'] = sh[:, 0, 0]
            vertices['f_dc_1'] = sh[:, 1, 0]
            vertices['f_dc_2'] = sh[:, 2, 0]
            if d_sh > 1:
                for i in range(1, d_sh):
                    for c in range(3):
                        vertices[f'f_rest_{(i-1)*3 + c}'] = sh[:, c, i]
            # 3DGS format: opacity in LOGIT space (viewers apply sigmoid)
            opacity_flat = opacity if len(opacity.shape) == 1 else opacity.squeeze()
            opacity_clamped = np.clip(opacity_flat, 1e-6, 1.0 - 1e-6)  # Avoid log(0) or log(inf)
            vertices['opacity'] = np.log(opacity_clamped / (1.0 - opacity_clamped))  # inverse sigmoid

            # 3DGS format: scales in LOG space (viewers apply exp)
            scale_clamped = np.maximum(scale, 1e-6)  # Avoid log(0)
            vertices['scale_0'] = np.log(scale_clamped[:, 0])
            vertices['scale_1'] = np.log(scale_clamped[:, 1])
            vertices['scale_2'] = np.log(scale_clamped[:, 2])
            vertices['rot_0'] = rot[:, 0]
            vertices['rot_1'] = rot[:, 1]
            vertices['rot_2'] = rot[:, 2]
            vertices['rot_3'] = rot[:, 3]

            el = PlyElement.describe(vertices, 'vertex')
            filepath = output_dir / f"gaussians_raw_{b:04d}.ply"
            PlyData([el]).write(str(filepath))
            file_paths.append(str(filepath))
            logger.info(f"Saved raw Gaussians ({N} points) to: {filepath}")

        return file_paths[0] if len(file_paths) == 1 else "\n".join(file_paths)


NODE_CLASS_MAPPINGS = {
    "DepthAnything_V3": DepthAnything_V3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthAnything_V3": "Depth Anything V3",
}
