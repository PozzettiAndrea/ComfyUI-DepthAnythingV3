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
    resize_to_patch_multiple, safe_model_to_device, logger, check_model_capabilities
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
                "camera_params": ("CAMERA_PARAMS", ),
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

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING", "MASK")
    RETURN_NAMES = ("depth", "confidence", "resized_rgb_image", "ray_origin", "ray_direction", "extrinsics", "intrinsics", "sky_mask")
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

**Optional Inputs:**
- camera_params: Connect DA3_CreateCameraParams for camera-conditioned estimation
- resize_method: How to handle patch size alignment (resize/crop/pad)
- invert_depth: Toggle output convention. OFF (default): close=bright. ON: far=bright.
- keep_model_size: Keep model's native output size instead of resizing back

**Note:** Ray maps and camera parameters only available for main series models.
Sky mask only available for Mono/Metric/Nested models.

Connect only the outputs you need - unused outputs are simply ignored.
"""

    def _apply_edge_antialiasing(self, mask):
        """Apply minimal anti-aliasing ONLY to border pixels (1-2px transition)."""
        # Ensure mask is in correct format [B, 1, H, W]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)

        # Very small 3x3 averaging kernel for minimal smoothing
        kernel = torch.ones((1, 1, 3, 3), device=mask.device, dtype=mask.dtype) / 9.0

        # Apply minimal blur
        mask_blurred = F.conv2d(mask, kernel, padding=1)

        # Detect edges: where original mask has transitions
        mask_dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        mask_eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)

        # Edge zone is where dilated and eroded differ
        edge_zone = (mask_dilated - mask_eroded).abs()
        edge_zone = (edge_zone > 0.01).float()

        # Apply anti-aliasing ONLY in edge zone
        mask_aa = mask * (1.0 - edge_zone) + mask_blurred * edge_zone

        return mask_aa

    def _apply_standard_normalization(self, depth, invert_depth):
        """
        Standard min-max normalization (original V3 approach).

        By default, inverts to match V2-Style convention (close=bright).
        Standard normalization naturally outputs far=bright, so we invert by default.
        """
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Invert by default to match V2-Style (close = bright)
        if not invert_depth:
            depth_norm = 1.0 - depth_norm

        return depth_norm

    def _apply_v2_style_normalization(self, depth, sky, device, invert_depth=False):
        """
        V2-Style disparity normalization (Ltamann/TBG approach).

        - Converts depth to disparity (1/depth) so sky becomes black
        - Uses content-only normalization (excludes sky)
        - Applies percentile-based contrast enhancement (1st-99th percentile)
        - Adds contrast boost via power transform (hardcoded to 2.0)
        - Applies edge anti-aliasing for natural transitions (hardcoded to True)
        - Optionally inverts the final output if invert_depth is True
        """
        epsilon = 1e-6
        contrast_boost = 2.0  # Hardcoded default
        edge_soften = True    # Hardcoded default

        # 1. Create HARD binary content mask
        if sky.max() > 0.1:
            # Threshold of 0.3 for aggressive sky detection
            content_mask_binary = (sky < 0.3).float()

            # Apply edge anti-aliasing (affects 1-2px border only)
            if edge_soften:
                content_mask_smooth = self._apply_edge_antialiasing(content_mask_binary)
            else:
                content_mask_smooth = content_mask_binary
        else:
            content_mask_binary = torch.ones_like(depth)
            content_mask_smooth = content_mask_binary

        # Ensure same shape as depth
        while content_mask_binary.dim() < depth.dim():
            content_mask_binary = content_mask_binary.unsqueeze(0)
        while content_mask_smooth.dim() < depth.dim():
            content_mask_smooth = content_mask_smooth.unsqueeze(0)

        # 2. Convert depth to disparity (inverse depth) like V2
        disparity = 1.0 / (depth + epsilon)

        # 3. Use HARD mask for normalization calculations
        disparity_masked = disparity * content_mask_binary

        # 4. Extract ONLY content pixels for normalization
        content_pixels = disparity_masked[content_mask_binary > 0.5]

        if content_pixels.numel() > 100:
            # Get min/max from CONTENT ONLY
            disp_min = content_pixels.min()
            disp_max = content_pixels.max()

            # Use percentile-based normalization for better contrast
            if content_pixels.numel() > 1000:
                sorted_pixels = torch.sort(content_pixels.flatten())[0]
                p1_idx = int(sorted_pixels.numel() * 0.01)
                p99_idx = int(sorted_pixels.numel() * 0.99)
                disp_min = sorted_pixels[p1_idx]
                disp_max = sorted_pixels[p99_idx]

            # Normalize using content-only range
            disparity_norm = (disparity - disp_min) / (disp_max - disp_min + epsilon)
            disparity_norm = torch.clamp(disparity_norm, 0.0, 1.0)
        else:
            # Fallback for very small content regions
            disp_min = disparity.min()
            disp_max = disparity.max()
            disparity_norm = (disparity - disp_min) / (disp_max - disp_min + epsilon)

        # 5. Apply contrast boost
        disparity_contrast = torch.pow(disparity_norm, 1.0 / contrast_boost)

        # 6. Apply SMOOTH mask for final output (with anti-aliased edges)
        disparity_final = disparity_contrast * content_mask_smooth

        # 7. Apply inversion if requested (do this AFTER all processing)
        if invert_depth:
            disparity_final = 1.0 - disparity_final

        return disparity_final

    def _apply_raw_normalization(self, depth, invert_depth):
        """Raw/metric depth - no normalization (for 3D reconstruction)."""
        if invert_depth:
            # For raw metric depth, invert as max - depth
            depth = depth.max() - depth

        return depth

    def process(self, da3_model, images, normalization_mode="V2-Style", camera_params=None,
                resize_method="resize", invert_depth=False, keep_model_size=False):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        model = da3_model['model']
        dtype = da3_model['dtype']
        config = da3_model['config']

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

        # Move model to device if not already there
        safe_model_to_device(model, device)

        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)

        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            for i in range(B):
                img = normalized_images[i:i+1].to(device)

                # Get camera params for this batch item
                ext_i = extrinsics_input[i:i+1] if extrinsics_input is not None else None
                int_i = intrinsics_input[i:i+1] if intrinsics_input is not None else None

                # Run model forward with optional camera conditioning
                output = model(img, extrinsics=ext_i, intrinsics=int_i)

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
                    depth_processed = self._apply_raw_normalization(depth, invert_depth)
                elif normalization_mode == "V2-Style":
                    depth_processed = self._apply_v2_style_normalization(depth, sky, device, invert_depth)
                else:  # "Standard"
                    depth_processed = self._apply_standard_normalization(depth, invert_depth)

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

                pbar.update(1)

        model.to(offload_device)
        mm.soft_empty_cache()

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

        # Format camera parameters as strings
        extrinsics_str = format_camera_params(extrinsics_list, "extrinsics")
        intrinsics_str = format_camera_params(intrinsics_list, "intrinsics")

        return (depth_final, conf_final, rgb_resized, ray_origin_final, ray_dir_final,
                extrinsics_str, intrinsics_str, sky_final)

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


NODE_CLASS_MAPPINGS = {
    "DepthAnything_V3": DepthAnything_V3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthAnything_V3": "Depth Anything V3",
}
