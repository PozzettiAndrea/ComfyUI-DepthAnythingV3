"""Shared utilities for DepthAnythingV3 nodes."""
import json
import torch
import torch.nn.functional as F
import logging

# Configure logger
logger = logging.getLogger("DepthAnythingV3")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(name)s] %(levelname)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_PATCH_SIZE = 14


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


def process_tensor_to_image(tensor_list, orig_H, orig_W, normalize_output=False):
    """Convert list of depth/conf tensors to ComfyUI IMAGE format.

    Args:
        tensor_list: List of tensors with shape [1, H, W] or [H, W]
        orig_H: Original image height
        orig_W: Original image width
        normalize_output: If True, clamp output to 0-1 range

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

    # Resize back to original dimensions (with even constraint)
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


def process_tensor_to_mask(tensor_list, orig_H, orig_W):
    """Convert list of tensors to ComfyUI MASK format.

    Args:
        tensor_list: List of tensors with shape [1, H, W] or [H, W]
        orig_H: Original image height
        orig_W: Original image width

    Returns:
        Tensor with shape [B, H, W] in ComfyUI MASK format
    """
    # Concatenate all tensors
    out = torch.cat(tensor_list, dim=0)  # [B, 1, H, W] or [B, H, W]

    # Ensure 3D: [B, H, W]
    if out.dim() == 4:
        out = out.squeeze(1)  # [B, H, W]

    out = out.cpu().float()

    # Resize back to original dimensions (with even constraint)
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


def safe_model_to_device(model, device):
    """Safely move model to device, handling accelerate-loaded models.

    Args:
        model: The model to move
        device: Target device
    """
    try:
        model.to(device)
    except NotImplementedError:
        # Model might already be on device (via accelerate loading)
        pass
