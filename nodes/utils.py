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


def resize_to_patch_multiple(images_pt, patch_size=DEFAULT_PATCH_SIZE):
    """Resize images to be divisible by patch size.

    Args:
        images_pt: Tensor with shape [B, C, H, W]
        patch_size: Patch size to align to (default 14)

    Returns:
        Tuple of (resized_images, original_H, original_W)
    """
    _, _, H, W = images_pt.shape
    orig_H, orig_W = H, W

    if W % patch_size != 0:
        W = W - (W % patch_size)
    if H % patch_size != 0:
        H = H - (H % patch_size)

    if orig_H % patch_size != 0 or orig_W % patch_size != 0:
        images_pt = F.interpolate(images_pt, size=(H, W), mode="bilinear")

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
