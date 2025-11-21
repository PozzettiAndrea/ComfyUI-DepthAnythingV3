"""Multi-view processing nodes for DepthAnythingV3."""
import json
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


class DepthAnythingV3_MultiView:
    """Process multiple images together with cross-view attention for geometric consistency."""

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

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING", "MASK", "IMAGE")
    RETURN_NAMES = ("depth", "confidence", "ray_origin", "ray_direction", "extrinsics", "intrinsics", "sky_mask", "resized_rgb_image")
    FUNCTION = "process"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Multi-view Depth Anything V3 - processes multiple images TOGETHER with cross-view attention.

Key difference from standard nodes:
- Standard: Processes images one-by-one (sequential, independent)
- Multi-view: Processes all images together (cross-attention, geometrically consistent)

Use this for:
- Video frames (temporal consistency)
- Multiple angles of same scene (SfM/reconstruction)
- Stereo pairs (left/right cameras)

**Normalization Modes:**
- Standard: Original V3 min-max normalization (0-1 range)
- V2-Style: Disparity-based with content-aware contrast (default, best for ControlNet)
  - Sky appears BLACK, content-only normalization
  - Contribution by Ltamann (TBG)
- Raw: No normalization, outputs metric depth (for 3D reconstruction)

**Optional Inputs:**
- resize_method: How to handle patch size alignment (resize/crop/pad)
- invert_depth: Toggle output convention. OFF (default): close=bright. ON: far=bright.
- keep_model_size: Keep model's native output size instead of resizing back (intrinsics stay accurate)

Input: Batch of images [N, H, W, 3]
Outputs (all normalized across views together for consistency):
- depth: Batch of consistent depth maps [N, H, W, 3]
- confidence: Confidence maps [N, H, W, 3]
- ray_origin: Ray origin maps (for 3D, normalized for visualization)
- ray_direction: Ray direction maps (for 3D, normalized for visualization)
- extrinsics: Predicted camera poses for each view (JSON)
- intrinsics: Camera intrinsics for each view (JSON) - auto-scaled if resized
- sky_mask: Sky segmentation [N, H, W] (Mono/Metric/Nested only)
- resized_rgb_image: RGB images matching depth output dimensions

Note: All images must have the same resolution.
Higher N = more VRAM usage but better consistency.
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

    def process(self, da3_model, images, normalization_mode="V2-Style", resize_method="resize", invert_depth=False, keep_model_size=False):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        model = da3_model['model']
        dtype = da3_model['dtype']
        config = da3_model['config']

        N, H, W, C = images.shape
        logger.info(f"Multi-view input: {N} images, size: {H}x{W}")

        # Check model capabilities
        capabilities = check_model_capabilities(model)
        has_multiview_support = capabilities["has_multiview_attention"]

        if not has_multiview_support:
            logger.warning(
                "WARNING: Mono/Metric models do not have cross-view attention. "
                "Images will be processed together but without multi-view consistency benefits. "
                "For best multi-view results, use Main series models (Small/Base/Large/Giant) or Nested."
            )

        if not capabilities["has_sky_segmentation"] and normalization_mode == "V2-Style":
            logger.warning(
                "WARNING: This model does not support sky segmentation. "
                "V2-Style normalization will work but without sky masking. "
                "Use Mono/Metric/Nested models for best V2-Style results."
            )

        logger.info(f"Processing {N} images with multi-view attention")

        # Convert from ComfyUI format [N, H, W, C] to PyTorch [N, C, H, W]
        images_pt = images.permute(0, 3, 1, 2)

        # Resize to patch size multiple
        images_pt, orig_H, orig_W = resize_to_patch_multiple(images_pt, DEFAULT_PATCH_SIZE, resize_method)
        model_H, model_W = images_pt.shape[2], images_pt.shape[3]
        logger.info(f"Model input size (after resize): {model_H}x{model_W}")

        # Store resized RGB for output
        resized_rgb = images_pt.clone()

        # Normalize with ImageNet stats
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        normalized_images = normalize(images_pt)

        # Prepare for model: shape [B, N, 3, H, W] where B=1 (single batch of N views)
        # This is the key difference - we process all N images together
        normalized_images = normalized_images.unsqueeze(0)  # [1, N, C, H, W]

        # Move model to device
        safe_model_to_device(model, device)

        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)

        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            # Single forward pass with all N views
            output = model(normalized_images.to(device))

            # Extract depth - shape should be [1, N, H, W]
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

            # Extract ray maps
            ray_origin = None
            ray_direction = None
            if hasattr(output, 'ray_origin'):
                ray_origin = output.ray_origin
            elif isinstance(output, dict) and 'ray_origin' in output:
                ray_origin = output['ray_origin']

            if hasattr(output, 'ray_direction'):
                ray_direction = output.ray_direction
            elif isinstance(output, dict) and 'ray_direction' in output:
                ray_direction = output['ray_direction']

            # Extract camera parameters (extrinsics and intrinsics)
            extr = None
            intr = None
            if hasattr(output, 'extrinsics'):
                extr = output.extrinsics
            elif isinstance(output, dict) and 'extrinsics' in output:
                extr = output['extrinsics']

            if hasattr(output, 'intrinsics'):
                intr = output.intrinsics
            elif isinstance(output, dict) and 'intrinsics' in output:
                intr = output['intrinsics']

            if intr is not None and torch.is_tensor(intr):
                logger.info(f"Model output intrinsics: shape={intr.shape}, values=\n{intr.squeeze()}")

            # Remove batch dimension: [1, N, H, W] -> [N, H, W]
            depth = depth.squeeze(0)
            conf = conf.squeeze(0)
            sky = sky.squeeze(0)

            # Apply normalization mode (normalize across all views together for consistency)
            if normalization_mode == "Standard":
                depth = self._apply_standard_normalization(depth, invert_depth)
            elif normalization_mode == "V2-Style":
                depth = self._apply_v2_style_normalization(depth, sky, device, invert_depth)
            elif normalization_mode == "Raw":
                depth = self._apply_raw_normalization(depth, invert_depth)
            else:
                # Fallback to V2-Style
                depth = self._apply_v2_style_normalization(depth, sky, device, invert_depth)

            # Normalize confidence
            conf_range = conf.max() - conf.min()
            if conf_range > 1e-8:
                conf = (conf - conf.min()) / conf_range
            else:
                conf = torch.ones_like(conf)

            # Process ray maps (normalize for visualization)
            if ray_origin is not None and torch.is_tensor(ray_origin):
                ray_origin = ray_origin.squeeze(0)  # [N, 3, H, W]
                # Normalize each channel independently for visualization
                for c in range(ray_origin.shape[1]):
                    channel = ray_origin[:, c]
                    c_min, c_max = channel.min(), channel.max()
                    if c_max > c_min:
                        ray_origin[:, c] = (channel - c_min) / (c_max - c_min)
            else:
                # Create zeros with shape [N, 3, H, W]
                ray_origin = torch.zeros((depth.shape[0], 3, depth.shape[1], depth.shape[2]), device=device)

            if ray_direction is not None and torch.is_tensor(ray_direction):
                ray_direction = ray_direction.squeeze(0)  # [N, 3, H, W]
                # Normalize each channel independently for visualization
                for c in range(ray_direction.shape[1]):
                    channel = ray_direction[:, c]
                    c_min, c_max = channel.min(), channel.max()
                    if c_max > c_min:
                        ray_direction[:, c] = (channel - c_min) / (c_max - c_min)
            else:
                # Create zeros with shape [N, 3, H, W]
                ray_direction = torch.zeros((depth.shape[0], 3, depth.shape[1], depth.shape[2]), device=device)

        model.to(offload_device)
        mm.soft_empty_cache()

        # Convert to ComfyUI format [N, H, W, 3]
        depth_out = depth.unsqueeze(-1).repeat(1, 1, 1, 3).cpu().float()
        conf_out = conf.unsqueeze(-1).repeat(1, 1, 1, 3).cpu().float()
        sky_out = sky.cpu().float()  # Keep as [N, H, W] for MASK type

        # Convert ray maps from [N, 3, H, W] to [N, H, W, 3]
        ray_origin_out = ray_origin.permute(0, 2, 3, 1).cpu().float()  # [N, H, W, 3]
        ray_dir_out = ray_direction.permute(0, 2, 3, 1).cpu().float()  # [N, H, W, 3]

        # Process resized RGB image
        rgb_out = resized_rgb.permute(0, 2, 3, 1).cpu().float()  # [N, H, W, 3]

        # Resize back to original dimensions (unless keep_model_size is True)
        if not keep_model_size:
            final_H = (orig_H // 2) * 2
            final_W = (orig_W // 2) * 2

            if depth_out.shape[1] != final_H or depth_out.shape[2] != final_W:
                depth_out = F.interpolate(
                    depth_out.permute(0, 3, 1, 2),
                    size=(final_H, final_W),
                    mode="bilinear"
                ).permute(0, 2, 3, 1)
                conf_out = F.interpolate(
                    conf_out.permute(0, 3, 1, 2),
                    size=(final_H, final_W),
                    mode="bilinear"
                ).permute(0, 2, 3, 1)
                sky_out = F.interpolate(
                    sky_out.unsqueeze(1),
                    size=(final_H, final_W),
                    mode="bilinear"
                ).squeeze(1)
                ray_origin_out = F.interpolate(
                    ray_origin_out.permute(0, 3, 1, 2),
                    size=(final_H, final_W),
                    mode="bilinear"
                ).permute(0, 2, 3, 1)
                ray_dir_out = F.interpolate(
                    ray_dir_out.permute(0, 3, 1, 2),
                    size=(final_H, final_W),
                    mode="bilinear"
                ).permute(0, 2, 3, 1)
                rgb_out = F.interpolate(
                    rgb_out.permute(0, 3, 1, 2),
                    size=(final_H, final_W),
                    mode="bilinear"
                ).permute(0, 2, 3, 1)

        # Clamp outputs (except depth if in Raw mode)
        if normalization_mode != "Raw":
            depth_out = torch.clamp(depth_out, 0, 1)
        conf_out = torch.clamp(conf_out, 0, 1)
        sky_out = torch.clamp(sky_out, 0, 1)
        ray_origin_out = torch.clamp(ray_origin_out, 0, 1)
        ray_dir_out = torch.clamp(ray_dir_out, 0, 1)
        rgb_out = torch.clamp(rgb_out, 0, 1)

        # Format camera parameters as JSON strings
        # Extrinsics: [1, N, 4, 4] -> per-view matrices
        # Intrinsics: [1, N, 3, 3] -> per-view matrices
        extrinsics_list = []
        intrinsics_list = []

        if extr is not None and torch.is_tensor(extr):
            extr = extr.squeeze(0).cpu()  # [N, 4, 4]
            for i in range(N):
                extrinsics_list.append(extr[i])
        else:
            extrinsics_list = [None] * N

        if intr is not None and torch.is_tensor(intr):
            intr = intr.squeeze(0).cpu()  # [N, 3, 3]
            for i in range(N):
                intrinsics_list.append(intr[i])
        else:
            intrinsics_list = [None] * N

        # Scale intrinsics if we resized back to original dimensions
        if not keep_model_size:
            final_H = (orig_H // 2) * 2
            final_W = (orig_W // 2) * 2
            model_H, model_W = images_pt.shape[2], images_pt.shape[3]

            if final_H != model_H or final_W != model_W:
                scale_h = final_H / model_H
                scale_w = final_W / model_W
                logger.info(f"Resizing from {model_H}x{model_W} to {final_H}x{final_W}, scale: h={scale_h:.4f}, w={scale_w:.4f}")

                # Scale each intrinsics matrix
                for i, intr_mat in enumerate(intrinsics_list):
                    if intr_mat is not None and torch.is_tensor(intr_mat):
                        # Squeeze to ensure [3, 3] shape (remove batch dimensions)
                        intr_scaled = intr_mat.squeeze().clone()
                        # Scale focal lengths and principal points
                        intr_scaled[0, 0] *= scale_w  # fx
                        intr_scaled[1, 1] *= scale_h  # fy
                        intr_scaled[0, 2] *= scale_w  # cx
                        intr_scaled[1, 2] *= scale_h  # cy
                        logger.info(f"Scaled intrinsics (view {i}):\n{intr_scaled}")
                        intrinsics_list[i] = intr_scaled

        extrinsics_str = format_camera_params(extrinsics_list, "extrinsics")
        intrinsics_str = format_camera_params(intrinsics_list, "intrinsics")

        return (depth_out, conf_out, ray_origin_out, ray_dir_out, extrinsics_str, intrinsics_str, sky_out, rgb_out)


class DA3_MultiViewPointCloud:
    """Combine multi-view depth maps into a single world-space point cloud."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "depths": ("IMAGE", ),
                "images": ("IMAGE", ),
                "extrinsics": ("STRING", ),
                "intrinsics": ("STRING", ),
            },
            "optional": {
                "confidence": ("IMAGE", ),
                "sky_mask": ("MASK", ),
                "confidence_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "downsample": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1}),
                "use_icp": ("BOOLEAN", {"default": False}),
                "allow_around_1": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allow images with max depth value around 1"
                }),
                "filter_outliers": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Remove points far from point cloud center BEFORE ICP (reduces noise and improves alignment)"
                }),
                "outlier_percentage": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "tooltip": "Percent of furthest points to remove from center of each view"
                }),
            }
        }

    RETURN_TYPES = ("POINTCLOUD",)
    RETURN_NAMES = ("pointcloud",)
    FUNCTION = "fuse"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Fuse multi-view depth maps into a single world-space point cloud.

Uses predicted camera poses (extrinsics) to transform each view's depth
into a common world coordinate system, then combines all points.

Inputs:
- depths: Batch of depth maps [N, H, W, 3] from Multi-View 3D node
- images: Original images [N, H, W, 3] for RGB colors
- extrinsics: Camera poses JSON from Multi-View node
- intrinsics: Camera intrinsics JSON from Multi-View node
- confidence: Optional confidence maps to filter low-confidence points
- sky_mask: Optional sky segmentation to exclude sky pixels from point cloud
- use_icp: Refine alignment with ICP (slower but potentially more accurate)

Output: Single combined POINTCLOUD in world space.

Note: Requires Main series or Nested model (with camera pose prediction).
Mono/Metric models don't predict camera poses.
"""

    def _check_consistency(self, depths, images, confidence, sky_mask):
        """Validate that all views have matching spatial dimensions."""
        def get_hw(tensor):
            """Extract (height, width) from tensor of various shapes."""
            if tensor is None:
                return None
            dims = tensor.dim()
            if dims == 4:  # [N, H, W, C]
                return tensor.shape[1], tensor.shape[2]
            elif dims == 3:  # [N, H, W]
                return tensor.shape[1], tensor.shape[2]
            else:
                raise ValueError(f"Unsupported tensor dimensions: {tensor.shape}")

        # Get dimensions for all inputs
        ref_hw = get_hw(depths)
        inputs_to_check = [
            ("images", images),
            ("confidence", confidence),
            ("sky_mask", sky_mask),
        ]

        # Check each input against reference dimensions
        for name, tensor in inputs_to_check:
            if tensor is None:
                continue
            tensor_hw = get_hw(tensor)
            if tensor_hw != ref_hw:
                raise ValueError(
                    f"Shape mismatch: depths is {ref_hw} but {name} is {tensor_hw}. "
                    f"All inputs must have the same spatial resolution across all views. "
                    f"Make sure to use the resized_rgb_image output from the multi-view depth node."
                )

    def _parse_camera_params(self, param_str, param_name):
        """Parse camera parameters from JSON string."""
        if not param_str or param_str.strip() == "":
            return None

        try:
            data = json.loads(param_str)
            if param_name not in data:
                return None

            if isinstance(data[param_name], str):
                # Not available message
                return None

            params_list = data[param_name]
            matrices = []

            for item in params_list:
                for key, matrix in item.items():
                    if matrix is not None:
                        matrices.append(torch.tensor(matrix, dtype=torch.float32))
                    else:
                        matrices.append(None)

            return matrices
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Could not parse {param_name}: {e}")
            return None

    def _unproject_depth(self, depth, K):
        """Unproject depth map to 3D points in camera space."""
        H, W = depth.shape
        if K.dim() == 3:
            K = K[0]  # Take first view's intrinsics

        # Create pixel grid
        u = torch.arange(W, dtype=torch.float32)
        v = torch.arange(H, dtype=torch.float32)
        u, v = torch.meshgrid(u, v, indexing='xy')  # [H, W]

        # Unproject: P = K^(-1) * [u, v, 1]^T * depth
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Camera space coordinates
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        # Stack to [H, W, 3] then reshape to [H*W, 3]
        points = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        return points

    def _transform_points(self, points, extrinsics):
        """Transform points from camera space to world space."""
        # Handle both 3x4 and 4x4 extrinsics matrices
        if extrinsics.shape[-2] == 3:  # 3x4 matrix (rotation + translation only)
            # Convert to 4x4 by adding [0, 0, 0, 1] bottom row
            bottom_row = torch.tensor([[0., 0., 0., 1.]], dtype=extrinsics.dtype, device=extrinsics.device)
            extrinsics = torch.cat([extrinsics, bottom_row], dim=0)

        # extrinsics is typically world-to-camera (w2c)
        # We need camera-to-world (c2w) = inverse of w2c
        # Invert extrinsics (w2c -> c2w)
        c2w = torch.linalg.inv(extrinsics)

        # Convert to homogeneous coordinates
        ones = torch.ones((points.shape[0], 1), dtype=points.dtype)
        points_hom = torch.cat([points, ones], dim=1)  # [N, 4]

        # Transform: world_points = c2w @ points_hom
        world_points = (c2w @ points_hom.T).T  # [N, 4]
        world_points = world_points[:, :3]  # [N, 3]

        return world_points

    def _icp_align(self, source, target, max_iterations=50, tolerance=1e-6):
        """Align source point cloud to target using ICP (Iterative Closest Point).

        Args:
            source: [N, 3] source points to transform
            target: [M, 3] target (reference) points
            max_iterations: Maximum ICP iterations
            tolerance: Convergence tolerance (change in error)

        Returns:
            aligned_source: [N, 3] transformed source points
            transform: [4, 4] transformation matrix
        """
        # Subsample for efficiency (use max 10000 points for ICP)
        max_pts = 10000
        if source.shape[0] > max_pts:
            src_idx = torch.randperm(source.shape[0])[:max_pts]
            src_sample = source[src_idx]
        else:
            src_sample = source

        if target.shape[0] > max_pts:
            tgt_idx = torch.randperm(target.shape[0])[:max_pts]
            tgt_sample = target[tgt_idx]
        else:
            tgt_sample = target

        # Current transformation (identity)
        R_total = torch.eye(3, dtype=source.dtype)
        t_total = torch.zeros(3, dtype=source.dtype)

        # Transform source samples
        src_transformed = src_sample.clone()
        prev_error = float('inf')

        for iteration in range(max_iterations):
            # Find nearest neighbors (brute force for simplicity)
            # Compute pairwise distances (MPS-compatible implementation)
            src_expanded = src_transformed.unsqueeze(1)  # [N, 1, 3]
            tgt_expanded = tgt_sample.unsqueeze(0)  # [1, M, 3]
            dists = torch.sqrt(((src_expanded - tgt_expanded) ** 2).sum(dim=-1))  # [N, M]

            # Find nearest target point for each source point
            min_dists, nearest_idx = dists.min(dim=1)

            # Get corresponding points
            tgt_corr = tgt_sample[nearest_idx]  # [N, 3]

            # Compute centroids
            src_centroid = src_transformed.mean(dim=0)
            tgt_centroid = tgt_corr.mean(dim=0)

            # Center the points
            src_centered = src_transformed - src_centroid
            tgt_centered = tgt_corr - tgt_centroid

            # Compute optimal rotation using SVD
            H = src_centered.T @ tgt_centered  # [3, 3]
            # Use CPU fallback for SVD on MPS due to potential precision issues
            if H.device.type == 'mps':
                H_cpu = H.cpu()
                U, S, Vt = torch.linalg.svd(H_cpu)
                U, S, Vt = U.to(H.device), S.to(H.device), Vt.to(H.device)
            else:
                U, S, Vt = torch.linalg.svd(H)
            R = Vt.T @ U.T

            # Handle reflection case (ensure proper rotation)
            if torch.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            # Compute translation
            t = tgt_centroid - R @ src_centroid

            # Apply transformation to source samples
            src_transformed = (R @ src_transformed.T).T + t

            # Accumulate total transformation
            R_total = R @ R_total
            t_total = R @ t_total + t

            # Check convergence
            error = min_dists.mean().item()
            if abs(prev_error - error) < tolerance:
                logger.debug(f"ICP converged after {iteration + 1} iterations, error={error:.6f}")
                break
            prev_error = error

        # Apply total transformation to all source points
        aligned_source = (R_total @ source.T).T + t_total

        # Build 4x4 transformation matrix
        transform = torch.eye(4, dtype=source.dtype)
        transform[:3, :3] = R_total
        transform[:3, 3] = t_total

        return aligned_source, transform

    def _refine_with_icp(self, points_list, colors_list, conf_list, view_ids_list):
        """Refine multi-view point cloud alignment using ICP.

        Uses first view as reference and aligns all subsequent views to it.
        """
        if len(points_list) < 2:
            return points_list, colors_list, conf_list, view_ids_list

        logger.info(f"Refining alignment with ICP ({len(points_list)} views)")

        # First view is reference
        reference = points_list[0]

        refined_points = [reference]
        refined_colors = [colors_list[0]]
        refined_conf = [conf_list[0]]
        refined_view_ids = [view_ids_list[0]]

        # Align each subsequent view to accumulated reference
        accumulated_ref = reference

        for i in range(1, len(points_list)):
            source = points_list[i]

            logger.debug(f"ICP aligning view {i} ({source.shape[0]} pts) to reference ({accumulated_ref.shape[0]} pts)")

            # Align source to accumulated reference
            aligned, transform = self._icp_align(source, accumulated_ref)

            refined_points.append(aligned)
            refined_colors.append(colors_list[i])
            refined_conf.append(conf_list[i])
            refined_view_ids.append(view_ids_list[i])

            # Update accumulated reference (combine all aligned points so far)
            # Subsample to keep memory manageable
            if accumulated_ref.shape[0] + aligned.shape[0] > 100000:
                # Random subsample combined cloud
                combined = torch.cat([accumulated_ref, aligned], dim=0)
                subsample_idx = torch.randperm(combined.shape[0])[:100000]
                accumulated_ref = combined[subsample_idx]
            else:
                accumulated_ref = torch.cat([accumulated_ref, aligned], dim=0)

        logger.info("ICP refinement complete")
        return refined_points, refined_colors, refined_conf, refined_view_ids

    def fuse(self, depths, images, extrinsics, intrinsics, confidence=None, sky_mask=None,
             confidence_threshold=0.3, downsample=4, use_icp=False, allow_around_1=False,
             filter_outliers=False, outlier_percentage=5.0):
        """Fuse multi-view depth maps into world-space point cloud."""
        N = depths.shape[0]
        H, W = depths.shape[1], depths.shape[2]

        logger.info(f"Fusing {N} views into world-space point cloud")

        # Validate that depth is raw/metric, not normalized
        max_depth = depths.max().item()
        if 0.95 < max_depth < 1.05 and not allow_around_1:
            raise ValueError(
                f"Depth input appears to be normalized (max={max_depth:.4f}) instead of raw/metric depth. "
                f"Multi-view point cloud fusion requires raw metric depth values. "
                f"Please use DepthAnythingV3_MultiView node with normalization_mode='Raw' "
                f"and connect the depth output to this node's depths input. "
                f"If you think this is a mistake, feel free to toggle allow_around_1."
            )

        # Validate that all inputs have matching dimensions
        self._check_consistency(depths, images, confidence, sky_mask)

        # Parse camera parameters
        extr_list = self._parse_camera_params(extrinsics, "extrinsics")
        intr_list = self._parse_camera_params(intrinsics, "intrinsics")

        if extr_list is None or len(extr_list) != N:
            raise ValueError(f"Extrinsics not available or wrong count. Need {N} poses. "
                           "Make sure to use Main series or Nested model (not Mono/Metric).")

        if intr_list is None or len(intr_list) != N:
            raise ValueError(f"Intrinsics not available or wrong count. Need {N} matrices.")

        all_points = []
        all_colors = []
        all_confidences = []
        all_view_ids = []

        pbar = ProgressBar(N)

        for i in range(N):
            # Get depth for this view (take first channel, assuming grayscale)
            depth = depths[i, :, :, 0]

            # Get confidence mask
            if confidence is not None:
                conf = confidence[i, :, :, 0]
                valid_mask = conf > confidence_threshold
            else:
                conf = torch.ones_like(depth)
                valid_mask = torch.ones_like(depth, dtype=torch.bool)

            # Apply sky mask filtering
            points_before_sky = valid_mask.sum().item()
            if sky_mask is not None:
                sky = sky_mask[i]  # [H, W]
                # Filter out sky pixels (sky_mask < 0.5 means not sky)
                valid_mask = valid_mask & (sky < 0.5)
                points_after_sky = valid_mask.sum().item()
                logger.info(f"  View {i}: {points_before_sky} points → {points_after_sky} after sky filtering (removed {points_before_sky - points_after_sky})")
            else:
                points_after_sky = points_before_sky

            # Downsample
            if downsample > 1:
                valid_mask_ds = torch.zeros_like(valid_mask)
                valid_mask_ds[::downsample, ::downsample] = valid_mask[::downsample, ::downsample]
                valid_mask = valid_mask_ds
                points_after_downsample = valid_mask.sum().item()
                logger.info(f"  View {i}: {points_after_sky} points → {points_after_downsample} after {downsample}x downsampling")
            else:
                points_after_downsample = points_after_sky

            # Unproject to camera space
            K = intr_list[i]
            points_cam = self._unproject_depth(depth, K)  # [H*W, 3]

            # Get colors from original image
            colors = images[i].reshape(-1, 3)  # [H*W, 3]

            # Get confidence values
            conf_flat = conf.reshape(-1)  # [H*W]

            # Apply valid mask
            valid_flat = valid_mask.reshape(-1)
            points_cam = points_cam[valid_flat]
            colors = colors[valid_flat]
            conf_flat = conf_flat[valid_flat]

            # Create view ID array for this view
            num_points = points_cam.shape[0]
            view_id = torch.full((num_points,), i, dtype=torch.int32)

            # Transform to world space
            E = extr_list[i]
            points_world = self._transform_points(points_cam, E)

            # Apply outlier filtering if requested (BEFORE ICP for cleaner alignment)
            if filter_outliers and outlier_percentage > 0:
                original_count = points_world.shape[0]
                points_world, colors, conf_flat, view_id = self._filter_outliers(
                    points_world, colors, conf_flat, view_id, outlier_percentage
                )
                filtered_count = points_world.shape[0]
                logger.info(f"  View {i}: Outlier filtering: {original_count} → {filtered_count} points (removed {original_count - filtered_count}, {outlier_percentage}% furthest from center)")

            all_points.append(points_world)
            all_colors.append(colors)
            all_confidences.append(conf_flat)
            all_view_ids.append(view_id)

            logger.info(f"  View {i}: Contributing {points_world.shape[0]} points to combined cloud")

            pbar.update(1)

        # Optional: ICP refinement before combining
        if use_icp and N > 1:
            all_points, all_colors, all_confidences, all_view_ids = self._refine_with_icp(
                all_points, all_colors, all_confidences, all_view_ids
            )

        # Combine all views by concatenating (no deduplication - overlapping regions will have duplicate points)
        combined_points = torch.cat(all_points, dim=0)
        combined_colors = torch.cat(all_colors, dim=0)
        combined_conf = torch.cat(all_confidences, dim=0)
        combined_view_ids = torch.cat(all_view_ids, dim=0)

        # Log breakdown
        per_view_counts = [pts.shape[0] for pts in all_points]
        breakdown = ", ".join([f"view{i}={count}" for i, count in enumerate(per_view_counts)])
        logger.info(f"Combined point cloud: {combined_points.shape[0]} total points ({breakdown})")

        # Package as POINTCLOUD with view tracking
        pointcloud = {
            "points": combined_points.numpy(),
            "colors": combined_colors.numpy(),
            "confidence": combined_conf.numpy(),
            "view_id": combined_view_ids.numpy(),
        }

        return ([pointcloud],)

    def _filter_outliers(self, points, colors, confidence, view_id, percentage):
        """Remove points furthest from the point cloud center."""
        import torch

        # Calculate centroid
        centroid = points.mean(dim=0)

        # Calculate distances from centroid
        distances = torch.norm(points - centroid, dim=1)

        # Find threshold distance (keep (100-percentage)% closest points)
        threshold_idx = int(len(points) * (100 - percentage) / 100)
        sorted_indices = torch.argsort(distances)
        keep_indices = sorted_indices[:threshold_idx]

        # Filter all arrays
        filtered_points = points[keep_indices]
        filtered_colors = colors[keep_indices]
        filtered_confidence = confidence[keep_indices]
        filtered_view_id = view_id[keep_indices]

        return filtered_points, filtered_colors, filtered_confidence, filtered_view_id


NODE_CLASS_MAPPINGS = {
    "DepthAnythingV3_MultiView": DepthAnythingV3_MultiView,
    "DA3_MultiViewPointCloud": DA3_MultiViewPointCloud,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthAnythingV3_MultiView": "Depth Anything V3 (Multi-View)",
    "DA3_MultiViewPointCloud": "DA3 Multi-View Point Cloud",
}
