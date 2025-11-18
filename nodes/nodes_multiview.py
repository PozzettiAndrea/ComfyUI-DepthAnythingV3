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
            },
            "optional": {
                "resize_method": (["resize", "crop", "pad"], {"default": "resize"}),
                "invert_depth": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("depth", "confidence", "extrinsics", "intrinsics")
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

Input: Batch of images [N, H, W, 3]
Outputs:
- depth: Batch of consistent depth maps [N, H, W, 3] (normalized 0-1)
- confidence: Confidence maps [N, H, W, 3]
- extrinsics: Predicted camera poses for each view (JSON)
- intrinsics: Camera intrinsics for each view (JSON)

Note: All images must have the same resolution.
Higher N = more VRAM usage but better consistency.
Camera parameters only available for Main series/Nested models.
"""

    def process(self, da3_model, images, resize_method="resize", invert_depth=False):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        model = da3_model['model']
        dtype = da3_model['dtype']
        config = da3_model['config']

        N, H, W, C = images.shape

        # Check if model supports multi-view attention (has cam_enc/cam_dec)
        has_multiview_support = (
            hasattr(model, 'cam_enc') and model.cam_enc is not None and
            hasattr(model, 'cam_dec') and model.cam_dec is not None
        )

        if not has_multiview_support:
            logger.warning(
                "WARNING: Mono/Metric models do not have cross-view attention. "
                "Images will be processed together but without multi-view consistency benefits. "
                "For best multi-view results, use Main series models (Small/Base/Large/Giant) or Nested."
            )

        logger.info(f"Processing {N} images with multi-view attention")

        # Convert from ComfyUI format [N, H, W, C] to PyTorch [N, C, H, W]
        images_pt = images.permute(0, 3, 1, 2)

        # Resize to patch size multiple
        images_pt, orig_H, orig_W = resize_to_patch_multiple(images_pt, DEFAULT_PATCH_SIZE, resize_method)

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

            # Remove batch dimension: [1, N, H, W] -> [N, H, W]
            depth = depth.squeeze(0)
            conf = conf.squeeze(0)

            # Normalize depth for visualization
            # Normalize across all views together for consistency
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

            # Apply inversion if requested
            if invert_depth:
                depth = 1.0 - depth

            # Normalize confidence
            conf_range = conf.max() - conf.min()
            if conf_range > 1e-8:
                conf = (conf - conf.min()) / conf_range
            else:
                conf = torch.ones_like(conf)

        model.to(offload_device)
        mm.soft_empty_cache()

        # Convert to ComfyUI format [N, H, W, 3]
        depth_out = depth.unsqueeze(-1).repeat(1, 1, 1, 3).cpu().float()
        conf_out = conf.unsqueeze(-1).repeat(1, 1, 1, 3).cpu().float()

        # Resize back to original dimensions
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

        depth_out = torch.clamp(depth_out, 0, 1)
        conf_out = torch.clamp(conf_out, 0, 1)

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

        extrinsics_str = format_camera_params(extrinsics_list, "extrinsics")
        intrinsics_str = format_camera_params(intrinsics_list, "intrinsics")

        return (depth_out, conf_out, extrinsics_str, intrinsics_str)


class DepthAnythingV3_MultiView_3D:
    """Process multiple images together with cross-view attention, outputting raw metric depth for 3D reconstruction."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "da3_model": ("DA3MODEL", ),
                "images": ("IMAGE", ),
            },
            "optional": {
                "resize_method": (["resize", "crop", "pad"], {"default": "resize"}),
                "invert_depth": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "MASK")
    RETURN_NAMES = ("depth_raw", "confidence", "extrinsics", "intrinsics", "sky_mask")
    FUNCTION = "process"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Multi-view Depth Anything V3 (3D/Raw) - outputs raw metric depth for 3D reconstruction.

Same as Multi-View node but outputs RAW (non-normalized) depth values,
making it suitable for point cloud fusion and 3D reconstruction.

Use this for:
- Multi-view 3D reconstruction
- Point cloud fusion from multiple angles
- Consistent metric depth across views

Input: Batch of images [N, H, W, 3]
Outputs:
- depth_raw: Raw metric depth [N, H, W, 3] (NOT normalized, for 3D)
- confidence: Confidence maps [N, H, W, 3]
- extrinsics: Predicted camera poses for each view (JSON)
- intrinsics: Camera intrinsics for each view (JSON)
- sky_mask: Sky segmentation mask [N, H, W] (only for Mono/Metric/Nested)

Connect outputs to DA3_MultiViewPointCloud node for world-space fusion.
"""

    def process(self, da3_model, images, resize_method="resize", invert_depth=False):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        model = da3_model['model']
        dtype = da3_model['dtype']
        config = da3_model['config']

        N, H, W, C = images.shape

        # Check model capabilities
        capabilities = check_model_capabilities(model)
        has_multiview_support = capabilities["has_multiview_attention"]

        if not has_multiview_support:
            logger.warning(
                "WARNING: Mono/Metric models do not have cross-view attention. "
                "Images will be processed together but without multi-view consistency benefits."
            )

        if not capabilities["has_sky_segmentation"]:
            logger.warning(
                "WARNING: This model does not support sky segmentation. "
                "Sky mask output will be zeros. Use Mono/Metric/Nested models for sky segmentation."
            )

        logger.info(f"Processing {N} images with multi-view attention (3D/Raw mode)")

        # Convert from ComfyUI format [N, H, W, C] to PyTorch [N, C, H, W]
        images_pt = images.permute(0, 3, 1, 2)

        # Resize to patch size multiple
        images_pt, orig_H, orig_W = resize_to_patch_multiple(images_pt, DEFAULT_PATCH_SIZE, resize_method)

        # Normalize with ImageNet stats
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        normalized_images = normalize(images_pt)

        # Prepare for model: shape [B, N, 3, H, W] where B=1
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

            # Extract sky mask (if available)
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

            # Extract camera parameters
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

            # Remove batch dimension: [1, N, H, W] -> [N, H, W]
            depth = depth.squeeze(0)
            conf = conf.squeeze(0)
            sky = sky.squeeze(0)

            # Apply inversion if requested (for raw metric depth)
            if invert_depth:
                depth = depth.max() - depth

            # Normalize confidence only
            conf_range = conf.max() - conf.min()
            if conf_range > 1e-8:
                conf = (conf - conf.min()) / conf_range
            else:
                conf = torch.ones_like(conf)

        model.to(offload_device)
        mm.soft_empty_cache()

        # Convert to ComfyUI format [N, H, W, 3] for depth and confidence
        # Keep RAW depth values (no normalization!)
        depth_out = depth.unsqueeze(-1).repeat(1, 1, 1, 3).cpu().float()
        conf_out = conf.unsqueeze(-1).repeat(1, 1, 1, 3).cpu().float()
        sky_out = sky.cpu().float()  # Keep as [N, H, W] for MASK type

        # Resize back to original dimensions
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

        # Clamp confidence and sky (but NOT depth - keep raw values)
        conf_out = torch.clamp(conf_out, 0, 1)
        sky_out = torch.clamp(sky_out, 0, 1)

        # Format camera parameters as JSON strings
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

        extrinsics_str = format_camera_params(extrinsics_list, "extrinsics")
        intrinsics_str = format_camera_params(intrinsics_list, "intrinsics")

        return (depth_out, conf_out, extrinsics_str, intrinsics_str, sky_out)


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
                "confidence_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "downsample": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1}),
                "use_icp": ("BOOLEAN", {"default": False}),
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
- confidence: Optional confidence maps
- use_icp: Refine alignment with ICP (slower but potentially more accurate)

Output: Single combined POINTCLOUD in world space.

Note: Requires Main series or Nested model (with camera pose prediction).
Mono/Metric models don't predict camera poses.
"""

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
        c2w = torch.inverse(extrinsics)

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
            # Compute pairwise distances
            dists = torch.cdist(src_transformed, tgt_sample)  # [N, M]

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

    def _refine_with_icp(self, points_list, colors_list, conf_list):
        """Refine multi-view point cloud alignment using ICP.

        Uses first view as reference and aligns all subsequent views to it.
        """
        if len(points_list) < 2:
            return points_list, colors_list, conf_list

        logger.info(f"Refining alignment with ICP ({len(points_list)} views)")

        # First view is reference
        reference = points_list[0]

        refined_points = [reference]
        refined_colors = [colors_list[0]]
        refined_conf = [conf_list[0]]

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
        return refined_points, refined_colors, refined_conf

    def fuse(self, depths, images, extrinsics, intrinsics, confidence=None,
             confidence_threshold=0.3, downsample=4, use_icp=False):
        """Fuse multi-view depth maps into world-space point cloud."""
        N = depths.shape[0]
        H, W = depths.shape[1], depths.shape[2]

        logger.info(f"Fusing {N} views into world-space point cloud")

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

            # Downsample
            if downsample > 1:
                valid_mask_ds = torch.zeros_like(valid_mask)
                valid_mask_ds[::downsample, ::downsample] = valid_mask[::downsample, ::downsample]
                valid_mask = valid_mask_ds

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

            # Transform to world space
            E = extr_list[i]
            points_world = self._transform_points(points_cam, E)

            all_points.append(points_world)
            all_colors.append(colors)
            all_confidences.append(conf_flat)

            pbar.update(1)

        # Optional: ICP refinement before combining
        if use_icp and N > 1:
            all_points, all_colors, all_confidences = self._refine_with_icp(
                all_points, all_colors, all_confidences
            )

        # Combine all views
        combined_points = torch.cat(all_points, dim=0)
        combined_colors = torch.cat(all_colors, dim=0)
        combined_conf = torch.cat(all_confidences, dim=0)

        logger.info(f"Combined point cloud: {combined_points.shape[0]} points")

        # Package as POINTCLOUD
        pointcloud = {
            "points": combined_points.numpy(),
            "colors": combined_colors.numpy(),
            "confidence": combined_conf.numpy(),
        }

        return ([pointcloud],)


NODE_CLASS_MAPPINGS = {
    "DepthAnythingV3_MultiView": DepthAnythingV3_MultiView,
    "DepthAnythingV3_MultiView_3D": DepthAnythingV3_MultiView_3D,
    "DA3_MultiViewPointCloud": DA3_MultiViewPointCloud,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthAnythingV3_MultiView": "Depth Anything V3 (Multi-View)",
    "DepthAnythingV3_MultiView_3D": "Depth Anything V3 (Multi-View 3D/Raw)",
    "DA3_MultiViewPointCloud": "DA3 Multi-View Point Cloud",
}
