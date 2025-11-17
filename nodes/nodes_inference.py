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
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "da3_model": ("DA3MODEL", ),
                "images": ("IMAGE", ),
            },
            "optional": {
                "camera_params": ("CAMERA_PARAMS", ),
                "resize_method": (["resize", "crop", "pad"], {"default": "resize"}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "keep_model_size": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth",)
    FUNCTION = "process"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Depth Anything V3 - depth estimation from images.
Returns normalized depth maps.

Optional: Provide camera_params for camera-conditioned depth estimation.
Connect DA3_CreateCameraParams to improve depth accuracy with known camera pose.

resize_method controls how images are adjusted to patch size multiples:
- resize: Scale to nearest multiple (default, preserves all content)
- crop: Center crop to floor multiple (loses edges but sharp)
- pad: Pad to ceiling multiple (adds black borders)

invert_depth: If True, inverts depth output (closer = higher value, like disparity)

keep_model_size: If True, keeps the model's native output size (patch-aligned) instead of
resizing back to original dimensions. Useful when you want to preserve the exact model output.
"""

    def process(self, da3_model, images, camera_params=None, resize_method="resize", invert_depth=False, keep_model_size=False):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        model = da3_model['model']
        dtype = da3_model['dtype']
        config = da3_model['config']

        B, H, W, C = images.shape

        # Convert from ComfyUI format [B, H, W, C] to PyTorch [B, C, H, W]
        images_pt = images.permute(0, 3, 1, 2)

        # Resize to patch size multiple
        images_pt, orig_H, orig_W = resize_to_patch_multiple(images_pt, DEFAULT_PATCH_SIZE, resize_method)

        # Normalize with ImageNet stats
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        normalized_images = normalize(images_pt)

        # Prepare for model: add view dimension [B, N, 3, H, W] where N=1
        normalized_images = normalized_images.unsqueeze(1)

        # Prepare camera parameters if provided
        extrinsics_input = None
        intrinsics_input = None
        if camera_params is not None:
            has_cam_support = (
                hasattr(model, 'cam_enc') and model.cam_enc is not None and
                hasattr(model, 'cam_dec') and model.cam_dec is not None
            )
            if has_cam_support:
                extrinsics_input = camera_params["extrinsics"].to(device).to(dtype)
                intrinsics_input = camera_params["intrinsics"].to(device).to(dtype)
                if extrinsics_input.shape[0] == 1 and B > 1:
                    extrinsics_input = extrinsics_input.expand(B, -1, -1, -1)
                    intrinsics_input = intrinsics_input.expand(B, -1, -1, -1)
                logger.info("Using camera-conditioned depth estimation")
            else:
                logger.warning("Model does not support camera conditioning. Camera params ignored.")

        pbar = ProgressBar(B)
        out = []

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

                # Extract depth from output
                depth = None
                if hasattr(output, 'depth'):
                    depth = output.depth
                elif isinstance(output, dict) and 'depth' in output:
                    depth = output['depth']

                if depth is None or not torch.is_tensor(depth):
                    raise ValueError("Model output does not contain valid depth tensor")

                # Normalize depth
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

                # Apply inversion if requested (closer = higher value)
                if invert_depth:
                    depth = 1.0 - depth

                out.append(depth.cpu())
                pbar.update(1)

        model.to(offload_device)
        mm.soft_empty_cache()

        # Concatenate all depths
        depth_out = torch.cat(out, dim=0)

        # Convert to 3-channel image [B, H, W, 3]
        depth_out = depth_out.squeeze(1)  # [B, H, W]
        depth_out = depth_out.unsqueeze(-1).repeat(1, 1, 1, 3).cpu().float()  # [B, H, W, 3]

        # Resize back to original dimensions (with even constraint) unless keep_model_size is True
        if not keep_model_size:
            final_H = (orig_H // 2) * 2
            final_W = (orig_W // 2) * 2

            if depth_out.shape[1] != final_H or depth_out.shape[2] != final_W:
                depth_out = F.interpolate(
                    depth_out.permute(0, 3, 1, 2),
                    size=(final_H, final_W),
                    mode="bilinear"
                ).permute(0, 2, 3, 1)

        depth_out = torch.clamp(depth_out, 0, 1)

        return (depth_out,)


class DepthAnythingV3_3D:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "da3_model": ("DA3MODEL", ),
                "images": ("IMAGE", ),
            },
            "optional": {
                "camera_params": ("CAMERA_PARAMS", ),
                "resize_method": (["resize", "crop", "pad"], {"default": "resize"}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "keep_model_size": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("depth_raw", "confidence", "intrinsics", "sky_mask")
    FUNCTION = "process"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Depth Anything V3 node optimized for 3D reconstruction (point clouds, gaussian splats, etc).
Outputs the essential data needed for proper 3D reconstruction:
- Depth raw: Metric depth values (NOT normalized)
- Confidence: Confidence map
- Intrinsics: Camera intrinsic matrix (3x3) for geometric unprojection
- Sky mask: Sky segmentation mask (1=sky, 0=non-sky, only for Mono/Metric models)

These outputs can be directly connected to DA3_ToPointCloud or DA3_ToGaussianSplat nodes.

Uses the official DA3 approach: geometric unprojection with camera intrinsics,
NOT the model's auxiliary ray outputs.

Optional: Provide camera_params for camera-conditioned depth estimation.
Connect DA3_CreateCameraParams to improve depth accuracy with known camera pose.

Works with all model types (Small/Base/Large/Giant/Mono/Metric).
Note: Sky mask is only available for Mono/Metric models. Other models return zeros.

resize_method controls how images are adjusted to patch size multiples:
- resize: Scale to nearest multiple (default, preserves all content)
- crop: Center crop to floor multiple (loses edges but sharp)
- pad: Pad to ceiling multiple (adds black borders)

invert_depth: If True, inverts depth output (closer = higher value, like disparity)

keep_model_size: If True, keeps the model's native output size (patch-aligned) instead of
resizing back to original dimensions. Useful when you want to preserve the exact model output.
"""

    def process(self, da3_model, images, camera_params=None, resize_method="resize", invert_depth=False, keep_model_size=False):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        model = da3_model['model']
        dtype = da3_model['dtype']
        config = da3_model['config']

        # Check model capabilities
        capabilities = check_model_capabilities(model)
        if not capabilities["has_sky_segmentation"]:
            logger.warning(
                "WARNING: This model does not support sky segmentation. "
                "Sky mask output will be zeros. Use Mono/Metric/Nested models for sky segmentation."
            )

        B, H, W, C = images.shape

        # Convert from ComfyUI format [B, H, W, C] to PyTorch [B, C, H, W]
        images_pt = images.permute(0, 3, 1, 2)

        # Resize to patch size multiple
        images_pt, orig_H, orig_W = resize_to_patch_multiple(images_pt, DEFAULT_PATCH_SIZE, resize_method)

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
        depth_raw_out = []
        conf_out = []
        sky_out = []
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

                # Verify it's a tensor, otherwise create uniform confidence
                if conf is None or not torch.is_tensor(conf):
                    conf = torch.ones_like(depth)

                # Extract sky mask (if available - only for Mono/Metric models)
                sky = None
                if hasattr(output, 'sky'):
                    sky = output.sky
                elif isinstance(output, dict) and 'sky' in output:
                    sky = output['sky']

                if sky is None or not torch.is_tensor(sky):
                    # Create dummy sky mask (all zeros = no sky) for non-supported models
                    sky = torch.zeros_like(depth)
                else:
                    # Normalize sky mask to 0-1 range
                    sky_min, sky_max = sky.min(), sky.max()
                    if sky_max > sky_min:
                        sky = (sky - sky_min) / (sky_max - sky_min)

                # Apply inversion if requested (closer = higher value)
                if invert_depth:
                    # For raw metric depth, invert as max - depth
                    depth = depth.max() - depth

                # Store RAW depth (no normalization!)
                depth_raw_out.append(depth.cpu())

                # Normalize confidence only (but keep uniform confidence as 1.0)
                conf_range = conf.max() - conf.min()
                if conf_range > 1e-8:
                    conf = (conf - conf.min()) / conf_range
                else:
                    # Uniform confidence - keep as 1.0 (high confidence)
                    conf = torch.ones_like(conf)
                conf_out.append(conf.cpu())
                sky_out.append(sky.cpu())

                # Extract camera intrinsics (if available)
                intr = None
                if hasattr(output, 'intrinsics'):
                    intr = output.intrinsics
                elif isinstance(output, dict) and 'intrinsics' in output:
                    intr = output['intrinsics']

                if intr is not None and torch.is_tensor(intr):
                    intrinsics_list.append(intr.cpu())
                else:
                    intrinsics_list.append(None)

                pbar.update(1)

        model.to(offload_device)
        mm.soft_empty_cache()

        # Process outputs WITHOUT normalization
        depth_raw_final = process_tensor_to_image(depth_raw_out, orig_H, orig_W, normalize_output=False, skip_resize=keep_model_size)
        conf_final = process_tensor_to_image(conf_out, orig_H, orig_W, normalize_output=False, skip_resize=keep_model_size)
        sky_final = process_tensor_to_mask(sky_out, orig_H, orig_W, skip_resize=keep_model_size)

        # Format intrinsics as JSON string
        intrinsics_str = format_camera_params(intrinsics_list, "intrinsics")

        return (depth_raw_final, conf_final, intrinsics_str, sky_final)


class DepthAnythingV3_Advanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "da3_model": ("DA3MODEL", ),
                "images": ("IMAGE", ),
            },
            "optional": {
                "camera_params": ("CAMERA_PARAMS", ),
                "resize_method": (["resize", "crop", "pad"], {"default": "resize"}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "keep_model_size": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING", "MASK")
    RETURN_NAMES = ("depth", "confidence", "ray_origin", "ray_direction", "extrinsics", "intrinsics", "sky_mask")
    FUNCTION = "process"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Advanced Depth Anything V3 node that outputs all available data:
- Depth map (normalized 0-1 for visualization)
- Confidence map
- Ray origin maps (normalized 0-1 for visualization)
- Ray direction maps (normalized 0-1 for visualization)
- Camera extrinsics (predicted camera pose)
- Camera intrinsics (predicted camera parameters)
- Sky mask: Sky segmentation mask (1=sky, 0=non-sky, only for Mono/Metric models)

Optional: Provide camera_params for camera-conditioned depth estimation.
Connect DA3_CreateCameraParams to improve depth accuracy with known camera pose.

Note: Ray maps and camera parameters only available for main series models (Small/Base/Large/Giant).
Mono/Metric models output only depth and confidence (dummy zeros for rays).
Sky mask is only available for Mono/Metric models. Other models return zeros.

For point cloud generation, use the DepthAnythingV3_3D node instead which outputs raw metric depth.

resize_method controls how images are adjusted to patch size multiples:
- resize: Scale to nearest multiple (default, preserves all content)
- crop: Center crop to floor multiple (loses edges but sharp)
- pad: Pad to ceiling multiple (adds black borders)

invert_depth: If True, inverts depth output (closer = higher value, like disparity)

keep_model_size: If True, keeps the model's native output size (patch-aligned) instead of
resizing back to original dimensions. Useful when you want to preserve the exact model output.
"""

    def process(self, da3_model, images, camera_params=None, resize_method="resize", invert_depth=False, keep_model_size=False):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        model = da3_model['model']
        dtype = da3_model['dtype']
        config = da3_model['config']

        # Check model capabilities
        capabilities = check_model_capabilities(model)
        if not capabilities["has_sky_segmentation"]:
            logger.warning(
                "WARNING: This model does not support sky segmentation. "
                "Sky mask output will be zeros. Use Mono/Metric/Nested models for sky segmentation."
            )

        B, H, W, C = images.shape

        # Convert from ComfyUI format [B, H, W, C] to PyTorch [B, C, H, W]
        images_pt = images.permute(0, 3, 1, 2)

        # Resize to patch size multiple
        images_pt, orig_H, orig_W = resize_to_patch_multiple(images_pt, DEFAULT_PATCH_SIZE, resize_method)

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

                # Verify it's a tensor, otherwise create uniform confidence
                if conf is None or not torch.is_tensor(conf):
                    conf = torch.ones_like(depth)

                # Extract sky mask (if available - only for Mono/Metric models)
                sky = None
                if hasattr(output, 'sky'):
                    sky = output.sky
                elif isinstance(output, dict) and 'sky' in output:
                    sky = output['sky']

                if sky is None or not torch.is_tensor(sky):
                    # Create dummy sky mask (all zeros = no sky) for non-supported models
                    sky = torch.zeros_like(depth)
                else:
                    # Normalize sky mask to 0-1 range
                    sky_min, sky_max = sky.min(), sky.max()
                    if sky_max > sky_min:
                        sky = (sky - sky_min) / (sky_max - sky_min)

                # Normalize depth and confidence for visualization
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

                # Apply inversion if requested (closer = higher value)
                if invert_depth:
                    depth = 1.0 - depth

                conf_range = conf.max() - conf.min()
                if conf_range > 1e-8:
                    conf = (conf - conf.min()) / conf_range
                else:
                    # Uniform confidence - keep as 1.0 (high confidence)
                    conf = torch.ones_like(conf)

                depth_out.append(depth.cpu())
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

                    # Store unnormalized rays (for 3D reconstruction)
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
                    intrinsics_list.append(intr.cpu())
                else:
                    intrinsics_list.append(None)

                pbar.update(1)

        model.to(offload_device)
        mm.soft_empty_cache()

        # Process outputs
        depth_final = process_tensor_to_image(depth_out, orig_H, orig_W, normalize_output=True, skip_resize=keep_model_size)
        conf_final = process_tensor_to_image(conf_out, orig_H, orig_W, normalize_output=True, skip_resize=keep_model_size)
        sky_final = process_tensor_to_mask(sky_out, orig_H, orig_W, skip_resize=keep_model_size)
        ray_origin_final = self._process_ray_to_image(ray_origin_out, orig_H, orig_W, normalize=True, skip_resize=keep_model_size)
        ray_dir_final = self._process_ray_to_image(ray_dir_out, orig_H, orig_W, normalize=True, skip_resize=keep_model_size)

        # Format camera parameters as strings
        extrinsics_str = format_camera_params(extrinsics_list, "extrinsics")
        intrinsics_str = format_camera_params(intrinsics_list, "intrinsics")

        return (depth_final, conf_final, ray_origin_final, ray_dir_final, extrinsics_str, intrinsics_str, sky_final)

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
    "DepthAnythingV3_3D": DepthAnythingV3_3D,
    "DepthAnythingV3_Advanced": DepthAnythingV3_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthAnything_V3": "Depth Anything V3",
    "DepthAnythingV3_3D": "Depth Anything V3 (3D/Raw)",
    "DepthAnythingV3_Advanced": "Depth Anything V3 (Advanced)",
}
