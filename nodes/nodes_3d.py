"""3D processing nodes (point clouds, Gaussians) for DepthAnythingV3."""
import torch
import torch.nn.functional as F
from torchvision import transforms
from contextlib import nullcontext

import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths

from .utils import (
    IMAGENET_MEAN, IMAGENET_STD, DEFAULT_PATCH_SIZE,
    resize_to_patch_multiple, safe_model_to_device, logger
)


class DA3_ToPointCloud:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "depth_raw": ("IMAGE", ),
                "confidence": ("IMAGE", ),
            },
            "optional": {
                "intrinsics": ("STRING", {"forceInput": True}),
                "sky_mask": ("MASK", ),
                "source_image": ("IMAGE", ),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Filter out points with confidence below this threshold (0-1)"
                }),
                "downsample": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Take every Nth pixel to reduce point cloud density. Higher = fewer points, faster processing. 1 = no downsampling (slowest, most detail)"
                }),
                "allow_around_1": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allow images with max depth value around 1"
                }),
                "filter_outliers": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Remove points far from point cloud center (reduces noise)"
                }),
                "outlier_percentage": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "tooltip": "Percent of furthest points to remove from center"
                }),

            }
        }

    RETURN_TYPES = ("POINTCLOUD",)
    RETURN_NAMES = ("pointcloud",)
    FUNCTION = "convert"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Convert DA3 depth map to 3D point cloud using proper camera geometry.
Uses geometric unprojection: P = K^(-1) * [u, v, 1]^T * depth

Inputs:
- depth_raw: Metric depth map (from DepthAnything_V3 with normalization_mode="Raw")
- confidence: Confidence map
- intrinsics: (Optional) Camera intrinsics JSON from DepthAnything_V3
  WARNING: If not provided, uses estimated intrinsics (may cause warping)
- sky_mask: (Optional but RECOMMENDED) Sky segmentation - excludes sky from point cloud
- source_image: (Optional) Source image for point colors

Parameters:
- confidence_threshold: Filter points below this confidence (0-1)
- downsample: Take every Nth pixel (5 = 1/25th of points, faster)

Output POINTCLOUD contains:
- points: Nx3 array of 3D coordinates
- colors: Nx3 array of RGB colors (if source_image provided)
- confidence: Nx1 array of confidence values
"""

    def _parse_intrinsics(self, intrinsics_str, batch_idx=0):
        """Parse camera intrinsics from JSON string."""
        import json
        import numpy as np

        if not intrinsics_str or intrinsics_str.strip() == "":
            return None

        try:
            data = json.loads(intrinsics_str)
            if "intrinsics" not in data:
                return None

            intrinsics_list = data["intrinsics"]
            if batch_idx >= len(intrinsics_list):
                return None

            intrinsics_data = intrinsics_list[batch_idx]
            img_key = f"image_{batch_idx}"

            if img_key not in intrinsics_data or intrinsics_data[img_key] is None:
                return None

            # Convert to tensor
            K = torch.tensor(intrinsics_data[img_key], dtype=torch.float32)
            return K
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Could not parse intrinsics: {e}")
            return None

    def _check_consistency(self, depth, conf, sky, img):
        """Validate that all inputs have matching spatial dimensions."""
        def get_hw(tensor):
            """Extract (height, width) from tensor of various shapes."""
            if tensor is None:
                return None
            dims = tensor.dim()
            if dims == 4:  # [B, H, W, C]
                return tensor.shape[1], tensor.shape[2]
            elif dims == 3:  # [H, W, C]
                return tensor.shape[0], tensor.shape[1]
            elif dims == 2:  # [H, W]
                return tensor.shape
            else:
                raise ValueError(f"Unsupported tensor dimensions: {tensor.shape}")

        # Get dimensions for all inputs
        ref_hw = get_hw(depth)
        inputs_to_check = [
            ("confidence", conf),
            ("sky_mask", sky),
            ("source_image", img),
        ]

        # Check each input against reference dimensions
        for name, tensor in inputs_to_check:
            if tensor is None:
                continue
            tensor_hw = get_hw(tensor)
            if tensor_hw != ref_hw:
                raise ValueError(
                    f"Shape mismatch: depth_raw is {ref_hw} but {name} is {tensor_hw}. "
                    f"All inputs must have the same spatial resolution. "
                    f"Make sure to use the resized_rgb_image output from the depth node."
                )


    def _create_default_intrinsics(self, H, W):
        """
        Create default pinhole camera intrinsics.

        WARNING: These are rough estimates! For accurate 3D reconstruction,
        provide actual camera intrinsics from the depth model or calibration.

        Assumes ~60 degree horizontal FOV (common for consumer cameras).
        """
        # For ~60 deg horizontal FOV: fx = W / (2 * tan(30 deg)) ~= 0.866 * W
        # Using a slightly wider assumption for better general results
        fx = fy = float(W) * 0.7  # Assumes ~70° FOV
        cx = (W - 1) / 2.0  # Principal point at image center (0-indexed)
        cy = (H - 1) / 2.0

        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32)

        logger.warning(
            f"Using default camera intrinsics (fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}). "
            "For accurate 3D reconstruction, connect intrinsics output from DepthAnything_V3 node."
        )

        return K

    def convert(self, depth_raw, confidence, allow_around_1=False, intrinsics=None, sky_mask=None, source_image=None, confidence_threshold=0.5, downsample=1, filter_outliers=False, outlier_percentage=5.0):
        """Convert depth map to point cloud using geometric unprojection."""
        # Validate that depth is raw/metric, not normalized
        max_depth = depth_raw.max().item()
        if 0.95 < max_depth < 1.05 and not allow_around_1:
            raise ValueError(
                f"Depth input appears to be normalized (max={max_depth:.4f}) instead of raw/metric depth. "
                f"Point cloud generation requires raw metric depth values. "
                f"Please use DepthAnything_V3 node with normalization_mode='Raw' "
                f"and connect the depth output to this node's depth_raw input. "
                f"If you think this is a mistake, feel free to toggle allow_around_1."
            )
        
        B = depth_raw.shape[0]
        point_clouds = []
        
        for b in range(B):
            self._check_consistency(
                depth_raw[b],
                confidence[b],
                sky_mask[b] if sky_mask is not None else None,
                source_image[b] if source_image is not None else None,
            )

            # Extract single image
            depth_map = depth_raw[b, :, :, 0]  # [H, W] - use first channel only
            conf_map = confidence[b, :, :, 0]  # [H, W] - use first channel only

            H, W = depth_map.shape

            # Get camera intrinsics - REQUIRED for accurate 3D reconstruction
            K = self._parse_intrinsics(intrinsics, b)
            if K is None:
                raise ValueError(
                    f"Camera intrinsics are required for point cloud generation.\n\n"
                    f"To get intrinsics:\n"
                    f"  1. Use a Main series model (Small/Base/Large/Giant) or Nested model\n"
                    f"  2. Connect the 'intrinsics' output from DepthAnything_V3 node\n"
                    f"     to this node's 'intrinsics' input\n\n"
                    f"Note: Mono/Metric models don't output intrinsics.\n"
                    f"For those models, either:\n"
                    f"  - Use a Nested model (has both metric depth + camera)\n"
                    f"  - Or run a separate Main model to get intrinsics"
                )
            intrinsics_source = "DA3 model"

            # Extract sky mask if provided
            if sky_mask is not None:
                sky_map = sky_mask[b]  # [H, W]
            else:
                sky_map = None

            # Downsample if needed
            if downsample > 1:
                depth_map = depth_map[::downsample, ::downsample]
                conf_map = conf_map[::downsample, ::downsample]

                if sky_map is not None:
                    sky_map = sky_map[::downsample, ::downsample]

                # Scale intrinsics for downsampling
                K = K.clone()
                K[0, 0] /= downsample  # fx
                K[1, 1] /= downsample  # fy
                K[0, 2] /= downsample  # cx
                K[1, 2] /= downsample  # cy

                if source_image is not None:
                    colors = source_image[b, ::downsample, ::downsample]  # [H', W', 3]
                else:
                    colors = None
            else:
                if source_image is not None:
                    colors = source_image[b]  # [H, W, 3]
                else:
                    colors = None

            # Resize colors to match depth_map dimensions if needed
            if colors is not None:
                if colors.shape[0] != depth_map.shape[0] or colors.shape[1] != depth_map.shape[1]:
                    # Convert to [1, 3, H, W] for interpolation
                    colors = colors.permute(2, 0, 1).unsqueeze(0)
                    colors = F.interpolate(colors, size=depth_map.shape, mode='bilinear', align_corners=False)
                    # Convert back to [H, W, 3]
                    colors = colors.squeeze(0).permute(1, 2, 0)

            # Generate pixel grid coordinates
            H_final, W_final = depth_map.shape
            u, v = torch.meshgrid(
                torch.arange(W_final, dtype=torch.float32, device=depth_map.device),
                torch.arange(H_final, dtype=torch.float32, device=depth_map.device),
                indexing='xy'
            )

            # Create homogeneous pixel coordinates [u, v, 1]
            pix_coords = torch.stack([u, v, torch.ones_like(u)], dim=-1)  # (H, W, 3)

            # Unproject using camera intrinsics: K^(-1) @ [u, v, 1]^T
            K = K.to(depth_map.device)
            K_inv = torch.linalg.inv(K)
            rays = torch.einsum('ij,hwj->hwi', K_inv, pix_coords)  # (H, W, 3)

            # Multiply by depth to get 3D points in camera space
            points_3d = rays * depth_map.unsqueeze(-1)  # (H, W, 3)

            # Transform from OpenCV to standard 3D convention
            # OpenCV: X-right, Y-down, Z-forward
            # Standard 3D (Three.js/OpenGL): X-right, Y-up, Z-backward
            points_3d[..., 1] *= -1  # Flip Y: down -> up
            points_3d[..., 2] *= -1  # Flip Z: forward -> backward

            # Flatten arrays
            points_flat = points_3d.reshape(-1, 3)  # (N, 3)
            conf_flat = conf_map.flatten()  # (N,)

            if colors is not None:
                colors_flat = colors.reshape(-1, 3)  # (N, 3)
            else:
                colors_flat = None

            # Filter by confidence
            mask = conf_flat >= confidence_threshold

            # ALWAYS filter out sky pixels if sky mask is provided
            if sky_map is not None:
                sky_flat = sky_map.flatten()  # (N,)
                # Sky mask: 1=sky, 0=non-sky, so we keep pixels where sky < 0.5
                mask = mask & (sky_flat < 0.5)

            points_3d = points_flat[mask]
            conf_flat = conf_flat[mask]

            if colors_flat is not None:
                colors_flat = colors_flat[mask]

            # Apply outlier filtering if requested
            if filter_outliers and outlier_percentage > 0:
                original_count = points_3d.shape[0]
                points_3d, colors_flat, conf_flat = self._filter_outliers(
                    points_3d, colors_flat, conf_flat, outlier_percentage
                )
                filtered_count = points_3d.shape[0]
                logger.info(f"Outlier filtering: {original_count} -> {filtered_count} points (removed {original_count - filtered_count}, {outlier_percentage}% furthest from center)")

            # Debug logs
            logger.debug(f"Point Cloud (batch {b}): intrinsics={intrinsics_source}, "
                        f"fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
            logger.debug(f"Depth range: [{depth_map.min():.4f}, {depth_map.max():.4f}], "
                        f"points after filtering: {points_3d.shape[0]}")

            # Check if we have any valid points
            if points_3d.shape[0] == 0:
                raise ValueError(f"No valid points after filtering (batch {b}). This may indicate the depth map is invalid or all depths were filtered out. Try adjusting min_depth/max_depth parameters or checking the input image.")

            logger.debug(f"Points 3D range: X[{points_3d[:, 0].min():.4f}, {points_3d[:, 0].max():.4f}], "
                        f"Y[{points_3d[:, 1].min():.4f}, {points_3d[:, 1].max():.4f}], "
                        f"Z[{points_3d[:, 2].min():.4f}, {points_3d[:, 2].max():.4f}]")

            # Create point cloud dict
            pc = {
                'points': points_3d.cpu().numpy(),
                'confidence': conf_flat.cpu().numpy(),
                'colors': colors_flat.cpu().numpy() if colors_flat is not None else None,
            }

            point_clouds.append(pc)

        # Return as tuple containing list of point clouds
        return (point_clouds,)

    def _filter_outliers(self, points, colors, confidence, percentage):
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
        filtered_colors = colors[keep_indices] if colors is not None else None
        filtered_confidence = confidence[keep_indices]

        return filtered_points, filtered_colors, filtered_confidence


class DA3_SavePointCloud:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pointcloud": ("POINTCLOUD", ),
                "filename_prefix": ("STRING", {"default": "pointcloud"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Save point cloud to PLY file.

Always saves:
- Original RGB colors (if available)
- view_id as custom property (if available from multi-view fusion)
- Confidence values (if available)

Use DA3 Preview Point Cloud to visualize with different color modes.

Output directory: ComfyUI/output/
Returns file path for use with ComfyUI 3D viewer.
"""

    def save(self, pointcloud, filename_prefix):
        """Save point cloud(s) to PLY file. Always saves view_id if available."""
        import numpy as np
        from pathlib import Path

        # Get output directory
        output_dir = folder_paths.get_output_directory()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []
        file_paths = []
        for idx, pc in enumerate(pointcloud):
            points = pc['points']
            confidence = pc.get('confidence', None)
            colors = pc.get('colors', None)
            view_id = pc.get('view_id', None)

            # Generate filename
            filename = f"{filename_prefix}_{idx:04d}.ply"
            filepath = output_path / filename

            # Ensure parent directory exists (for subfolder prefixes like "subfolder/pointcloud")
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Write PLY file (saves original RGB + view_id as custom property)
            self._write_ply(filepath, points, colors, confidence, view_id)

            # Extract subfolder from filename if present
            subfolder = str(Path(filename).parent) if Path(filename).parent != Path(".") else ""
            results.append({
                "filename": Path(filename).name,
                "subfolder": subfolder,
                "type": "output"
            })
            file_paths.append(str(filepath))
            logger.info(f"Saved point cloud to: {filepath}")

        # Return first file path (or all paths joined by newline if multiple)
        output_file_path = file_paths[0] if len(file_paths) == 1 else "\n".join(file_paths)

        return {
            "ui": {
                "pointclouds": results,
                "file_path": [output_file_path]
            },
            "result": (output_file_path,)
        }

    def _write_ply(self, filepath, points, colors=None, confidence=None, view_id=None):
        """Write point cloud to PLY file."""
        import numpy as np

        N = len(points)

        # Prepare header
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {N}",
            "property float x",
            "property float y",
            "property float z",
        ]

        if colors is not None:
            header.extend([
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ])

        if confidence is not None:
            header.append("property float confidence")

        if view_id is not None:
            header.append("property int view_id")

        header.append("end_header")

        # Write file
        with open(filepath, 'w') as f:
            # Write header
            f.write('\n'.join(header) + '\n')

            # Write points
            for i in range(N):
                x, y, z = points[i]
                line = f"{x} {y} {z}"

                if colors is not None:
                    r, g, b = (colors[i] * 255).astype(np.uint8)
                    line += f" {r} {g} {b}"

                if confidence is not None:
                    line += f" {confidence[i]}"

                if view_id is not None:
                    line += f" {int(view_id[i])}"

                f.write(line + '\n')


class DA3_FilterGaussians:
    """Load raw Gaussians PLY, apply filters, and save filtered PLY."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gaussian_ply_path": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "gaussians_filtered"}),
            },
            "optional": {
                "sky_mask": ("MASK", ),
                "filter_sky": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Filter out Gaussians in sky regions using sky_mask"
                }),
                "depth_prune_percent": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Prune Gaussians with depth above this percentile (0.9 = keep closest 90%)"
                }),
                "opacity_threshold": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Remove Gaussians with opacity below this threshold"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ply_path",)
    FUNCTION = "process"
    OUTPUT_NODE = True
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Filter 3D Gaussians from a PLY file and save filtered result.

Connect 'gaussian_ply_path' from DepthAnything_V3 node.

Filtering options:
- filter_sky: Remove Gaussians in sky regions (requires sky_mask from DepthAnything_V3)
- depth_prune_percent: Keep only closest X% of Gaussians by depth (0.9 = keep 90%)
- opacity_threshold: Remove low-opacity Gaussians

Output: Path to filtered PLY file (compatible with SuperSplat, gsplat.js, 3DGS viewers)
"""

    def process(self, gaussian_ply_path, filename_prefix, sky_mask=None, filter_sky=True,
                depth_prune_percent=0.9, opacity_threshold=0.0):
        """Load, filter, and save Gaussians."""
        import numpy as np
        from pathlib import Path

        if not gaussian_ply_path or gaussian_ply_path.strip() == "":
            raise ValueError(
                "No Gaussian PLY path provided. Make sure you're using the Giant model."
            )

        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            raise ImportError(
                "plyfile is required. Install with: pip install plyfile"
            )

        # Load the raw PLY
        ply_path = Path(gaussian_ply_path)
        if not ply_path.exists():
            raise ValueError(f"PLY file not found: {ply_path}")

        logger.info(f"Loading Gaussians from: {ply_path}")
        plydata = PlyData.read(str(ply_path))
        vertices = plydata['vertex'].data

        N = len(vertices)
        logger.info(f"Loaded {N} Gaussians")

        # Extract data
        xyz = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1)
        opacity = vertices['opacity']

        # Create valid mask
        valid_mask = np.ones(N, dtype=bool)

        # Apply sky mask filtering
        if filter_sky and sky_mask is not None:
            sky_np = sky_mask.cpu().numpy() if hasattr(sky_mask, 'cpu') else sky_mask
            sky_flat = sky_np.flatten()

            # Match sizes
            if len(sky_flat) >= N:
                sky_flat = sky_flat[:N]
            else:
                # Repeat to match
                repeats = (N // len(sky_flat)) + 1
                sky_flat = np.tile(sky_flat, repeats)[:N]

            sky_filter = sky_flat < 0.5
            valid_mask = valid_mask & sky_filter
            logger.info(f"After sky filtering: {valid_mask.sum()} / {N} Gaussians")

        # Apply opacity threshold
        # Note: PLY stores opacity in LOGIT space, convert back for comparison
        if opacity_threshold > 0:
            # Convert logit to actual opacity: sigmoid(x) = 1 / (1 + exp(-x))
            actual_opacity = 1.0 / (1.0 + np.exp(-opacity))
            opacity_filter = actual_opacity >= opacity_threshold
            valid_mask = valid_mask & opacity_filter
            logger.info(f"After opacity filtering: {valid_mask.sum()} / {N} Gaussians")

        # Apply depth percentile pruning
        if depth_prune_percent < 1.0:
            valid_depths = xyz[valid_mask, 2]  # Z coordinate
            if len(valid_depths) > 0:
                threshold = np.percentile(valid_depths, depth_prune_percent * 100)
                depth_filter = xyz[:, 2] <= threshold
                valid_mask = valid_mask & depth_filter
                logger.info(f"After depth pruning ({depth_prune_percent*100:.0f}%): {valid_mask.sum()} / {N} Gaussians")

        # Filter vertices
        filtered_vertices = vertices[valid_mask]
        N_filtered = len(filtered_vertices)

        if N_filtered == 0:
            raise ValueError("No Gaussians remaining after filtering")

        logger.info(f"Saving {N_filtered} filtered Gaussians")

        # Save filtered PLY
        output_dir = Path(folder_paths.get_output_directory())
        output_dir.mkdir(parents=True, exist_ok=True)

        el = PlyElement.describe(filtered_vertices, 'vertex')
        filename = f"{filename_prefix}_0000.ply"
        filepath = output_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        PlyData([el]).write(str(filepath))

        logger.info(f"Saved filtered Gaussians to: {filepath}")

        subfolder = str(Path(filename).parent) if Path(filename).parent != Path(".") else ""
        return {
            "ui": {"gaussians": [{"filename": Path(filename).name, "subfolder": subfolder, "type": "output"}]},
            "result": (str(filepath),)
        }


class DA3_ToMesh:
    """Convert depth map to textured 3D mesh using grid-based triangulation."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "depth_raw": ("IMAGE",),
                "confidence": ("IMAGE",),
            },
            "optional": {
                "intrinsics": ("STRING", {"forceInput": True}),
                "sky_mask": ("MASK",),
                "source_image": ("IMAGE",),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Filter out vertices with confidence below this threshold"
                }),
                "depth_edge_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Skip triangles across depth discontinuities (relative threshold)"
                }),
                "downsample": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Downsample factor for mesh density"
                }),
                "filename_prefix": ("STRING", {"default": "mesh"}),
                "allow_around_1": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allow images with max depth value around 1"
                }),
                "use_draco_compression": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use Draco compression for smaller file size and faster export"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Convert DA3 depth map to textured 3D mesh (GLB format).

Uses grid-based triangulation to create a clean mesh from the depth map.
Automatically filters invalid regions (sky, low confidence, depth discontinuities).

Inputs:
- depth_raw: Metric depth map (from DepthAnything_V3 with normalization_mode="Raw")
- confidence: Confidence map
- intrinsics: Camera intrinsics (REQUIRED - connect from DepthAnything_V3)
- sky_mask: Sky segmentation (recommended - excludes sky from mesh)
- source_image: Source image for mesh texture

Parameters:
- confidence_threshold: Filter vertices below this confidence
- depth_edge_threshold: Skip triangles across large depth jumps (prevents artifacts)
- downsample: Reduce mesh density (higher = fewer triangles, faster)
- filename_prefix: Output filename prefix

Output: GLB file path
"""

    def _parse_intrinsics(self, intrinsics_str, batch_idx=0):
        """Parse camera intrinsics from JSON string."""
        import json

        if not intrinsics_str or intrinsics_str.strip() == "":
            return None

        try:
            data = json.loads(intrinsics_str)
            if "intrinsics" not in data:
                return None

            intrinsics_list = data["intrinsics"]
            if batch_idx >= len(intrinsics_list):
                return None

            intrinsics_data = intrinsics_list[batch_idx]
            img_key = f"image_{batch_idx}"

            if img_key not in intrinsics_data or intrinsics_data[img_key] is None:
                return None

            # Convert to tensor
            K = torch.tensor(intrinsics_data[img_key], dtype=torch.float32)
            return K
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Could not parse intrinsics: {e}")
            return None

    def _unproject_grid(self, depth_map, K):
        """Unproject depth map to 3D points while preserving grid structure."""
        H, W = depth_map.shape

        # Create pixel grid
        u = torch.arange(W, dtype=torch.float32, device=depth_map.device)
        v = torch.arange(H, dtype=torch.float32, device=depth_map.device)
        u, v = torch.meshgrid(u, v, indexing='xy')

        # Create homogeneous pixel coordinates [u, v, 1]
        pix_coords = torch.stack([u, v, torch.ones_like(u)], dim=-1)  # (H, W, 3)

        # Unproject using camera intrinsics
        K = K.to(depth_map.device)
        K_inv = torch.linalg.inv(K)
        rays = torch.einsum('ij,hwj->hwi', K_inv, pix_coords)  # (H, W, 3)

        # Multiply by depth to get 3D points
        points_3d = rays * depth_map.unsqueeze(-1)  # (H, W, 3)

        # Transform from OpenCV to standard 3D convention
        points_3d[..., 1] *= -1  # Flip Y
        points_3d[..., 2] *= -1  # Flip Z

        return points_3d

    def _create_mesh_from_grid(self, points_3d, colors, valid_mask, depth_map, depth_edge_threshold):
        """Create triangular mesh from grid of 3D points (vectorized)."""
        import numpy as np

        H, W = points_3d.shape[:2]

        # Convert to numpy
        points_np = points_3d.cpu().numpy()
        colors_np = colors.cpu().numpy() if colors is not None else None
        valid_np = valid_mask.cpu().numpy()
        depth_np = depth_map.cpu().numpy()

        # Build vertex list using vectorized boolean indexing
        vertices = points_np[valid_np]
        vertex_colors = colors_np[valid_np] if colors_np is not None else None

        # Create UV coordinates for valid vertices
        i_coords, j_coords = np.where(valid_np)
        uvs = np.stack([j_coords / (W - 1), 1.0 - i_coords / (H - 1)], axis=1)

        # Create vertex index map (2D array: -1 for invalid, vertex index for valid)
        vertex_map = np.full((H, W), -1, dtype=np.int32)
        vertex_map[valid_np] = np.arange(len(vertices))

        # Build faces using vectorized operations
        # Create all potential quads
        i_range = np.arange(H - 1)
        j_range = np.arange(W - 1)
        ii, jj = np.meshgrid(i_range, j_range, indexing='ij')

        # Get vertex indices for all quad corners (vectorized)
        v00 = vertex_map[ii, jj]
        v10 = vertex_map[ii + 1, jj]
        v01 = vertex_map[ii, jj + 1]
        v11 = vertex_map[ii + 1, jj + 1]

        # Check if all corners are valid
        all_valid = (v00 >= 0) & (v10 >= 0) & (v01 >= 0) & (v11 >= 0)

        # Check for depth discontinuities (vectorized)
        d00 = depth_np[ii, jj]
        d10 = depth_np[ii + 1, jj]
        d01 = depth_np[ii, jj + 1]
        d11 = depth_np[ii + 1, jj + 1]

        depths_quad = np.stack([d00, d10, d01, d11], axis=-1)
        depth_range = depths_quad.max(axis=-1) - depths_quad.min(axis=-1)
        avg_depth = depths_quad.mean(axis=-1)

        # Skip quads with large depth discontinuities
        no_discontinuity = (depth_range / (avg_depth + 1e-6)) <= depth_edge_threshold

        # Combine all validity checks
        valid_quads = all_valid & no_discontinuity

        # Extract valid quad indices
        valid_i, valid_j = np.where(valid_quads)

        # Build faces for valid quads
        n_valid = len(valid_i)
        faces = np.empty((n_valid * 2, 3), dtype=np.int32)

        # First triangle of each quad
        faces[0::2, 0] = v00[valid_i, valid_j]
        faces[0::2, 1] = v10[valid_i, valid_j]
        faces[0::2, 2] = v01[valid_i, valid_j]

        # Second triangle of each quad
        faces[1::2, 0] = v10[valid_i, valid_j]
        faces[1::2, 1] = v11[valid_i, valid_j]
        faces[1::2, 2] = v01[valid_i, valid_j]

        return vertices, faces, vertex_colors, uvs

    def _compute_vertex_normals(self, vertices, faces):
        """Compute smooth vertex normals (vectorized)."""
        import numpy as np

        normals = np.zeros_like(vertices)

        # Get vertices for all faces at once (vectorized)
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        # Compute all face normals at once
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normals = np.cross(edge1, edge2)

        # Accumulate face normals to vertices using np.add.at
        # This efficiently handles duplicate indices
        np.add.at(normals, faces[:, 0], face_normals)
        np.add.at(normals, faces[:, 1], face_normals)
        np.add.at(normals, faces[:, 2], face_normals)

        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, norms, where=norms > 1e-10)

        return normals

    def _export_to_glb(self, filepath, vertices, faces, vertex_colors, uvs, normals, texture_image=None, use_draco_compression=True):
        """Export mesh to GLB format using trimesh."""
        try:
            import trimesh
        except ImportError:
            raise ImportError(
                "trimesh is required for mesh export. Install with: pip install trimesh"
            )

        import numpy as np

        # Create trimesh object
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=normals,
            process=False  # Don't auto-process
        )

        # Add vertex colors if available
        if vertex_colors is not None:
            mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)

        # Add UV coordinates and texture if available
        if uvs is not None and texture_image is not None:
            from PIL import Image

            # Convert texture to PIL Image
            texture_np = (texture_image.cpu().numpy() * 255).astype(np.uint8)
            texture_pil = Image.fromarray(texture_np)

            # Create textured visual
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=uvs,
                image=texture_pil
            )

        # Export to GLB with optional Draco compression
        # Note: Draco compression requires pygltflib
        if use_draco_compression:
            try:
                # Export with Draco compression using pygltflib backend
                mesh.export(filepath, file_type='glb',
                          extras={'compress': True, 'compressor': 'draco'})
            except Exception as e:
                # Fall back to uncompressed if Draco is not available
                logger.warning(f"Draco compression failed ({e}), exporting uncompressed")
                mesh.export(filepath, file_type='glb')
        else:
            mesh.export(filepath, file_type='glb')

        # Post-process GLB to enable double-sided rendering
        self._make_glb_double_sided(filepath)

    def _make_glb_double_sided(self, filepath):
        """Modify GLB file to make all materials double-sided."""
        try:
            import pygltflib
        except ImportError:
            logger.warning(
                "pygltflib is required for double-sided materials. "
                "Install with: pip install pygltflib. "
                "Meshes will only be visible from front face."
            )
            return

        try:
            # Load the GLB file
            gltf = pygltflib.GLTF2().load(filepath)

            # Set doubleSided = True for all materials
            if gltf.materials:
                for material in gltf.materials:
                    material.doubleSided = True
            else:
                # If no materials exist, create a default double-sided material
                gltf.materials = [pygltflib.Material(doubleSided=True)]
                # Link all meshes to this material
                if gltf.meshes:
                    for mesh in gltf.meshes:
                        for primitive in mesh.primitives:
                            primitive.material = 0

            # Save the modified GLB
            gltf.save(filepath)
            logger.debug(f"Set double-sided materials in {filepath}")

        except Exception as e:
            logger.warning(f"Failed to set double-sided materials: {e}")

    def convert(self, depth_raw, confidence, intrinsics=None, sky_mask=None, source_image=None,
                confidence_threshold=0.5, depth_edge_threshold=0.1, downsample=2, filename_prefix="mesh", allow_around_1=False, use_draco_compression=True):
        """Convert depth map to mesh and save as GLB."""
        from pathlib import Path

        # Validate depth
        max_depth = depth_raw.max().item()
        if 0.95 < max_depth < 1.05 and not allow_around_1:
            raise ValueError(
                f"Depth input appears to be normalized (max={max_depth:.4f}) instead of raw/metric depth. "
                f"Mesh generation requires raw metric depth values. "
                f"Please use DepthAnything_V3 node with normalization_mode='Raw'. "
                f"If you think this is a mistake, feel free to toggle allow_around_1."
            )

        B = depth_raw.shape[0]
        if B > 1:
            logger.warning(f"Batch size {B} > 1, only processing first image")

        # Extract single image
        depth_map = depth_raw[0, :, :, 0]  # [H, W]
        conf_map = confidence[0, :, :, 0]  # [H, W]

        # Get camera intrinsics
        K = self._parse_intrinsics(intrinsics, 0)
        if K is None:
            raise ValueError(
                f"Camera intrinsics are required for mesh generation.\n\n"
                f"Connect the 'intrinsics' output from DepthAnything_V3 node to this node's 'intrinsics' input.\n"
                f"Note: Mono/Metric models don't output intrinsics - use Main/Nested models."
            )

        # Get sky mask
        sky_map = sky_mask[0] if sky_mask is not None else None

        # Get source image
        colors = source_image[0] if source_image is not None else None

        # Downsample if needed
        if downsample > 1:
            depth_map = depth_map[::downsample, ::downsample]
            conf_map = conf_map[::downsample, ::downsample]
            if sky_map is not None:
                sky_map = sky_map[::downsample, ::downsample]
            if colors is not None:
                colors = colors[::downsample, ::downsample]

            # Scale intrinsics
            K = K.clone()
            K[0, 0] /= downsample  # fx
            K[1, 1] /= downsample  # fy
            K[0, 2] /= downsample  # cx
            K[1, 2] /= downsample  # cy

        H, W = depth_map.shape
        logger.info(f"Creating mesh from {H}x{W} depth map")

        # Create valid mask
        valid_mask = conf_map >= confidence_threshold
        if sky_map is not None:
            valid_mask = valid_mask & (sky_map < 0.5)

        # Unproject to 3D
        points_3d = self._unproject_grid(depth_map, K)

        # Create mesh
        vertices, faces, vertex_colors, uvs = self._create_mesh_from_grid(
            points_3d, colors, valid_mask, depth_map, depth_edge_threshold
        )

        logger.info(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")

        # Compute normals
        normals = self._compute_vertex_normals(vertices, faces)

        # Get output directory
        output_dir = folder_paths.get_output_directory()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"{filename_prefix}_0000.glb"
        filepath = output_path / filename

        # Ensure parent directory exists (for subfolder prefixes like "subfolder/mesh")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Export to GLB
        self._export_to_glb(
            str(filepath),
            vertices,
            faces,
            vertex_colors,
            uvs,
            normals,
            texture_image=colors,
            use_draco_compression=use_draco_compression
        )

        logger.info(f"Saved mesh to: {filepath}")

        # Extract subfolder from filename if present
        subfolder = str(Path(filename).parent) if Path(filename).parent != Path(".") else ""
        return {
            "ui": {"meshes": [{"filename": Path(filename).name, "subfolder": subfolder, "type": "output"}]},
            "result": (str(filepath),)
        }


NODE_CLASS_MAPPINGS = {
    "DA3_ToPointCloud": DA3_ToPointCloud,
    "DA3_SavePointCloud": DA3_SavePointCloud,
    "DA3_FilterGaussians": DA3_FilterGaussians,
    "DA3_ToMesh": DA3_ToMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DA3_ToPointCloud": "DA3 to Point Cloud",
    "DA3_SavePointCloud": "DA3 Save Point Cloud",
    "DA3_FilterGaussians": "DA3 Filter Gaussians",
    "DA3_ToMesh": "DA3 to Mesh",
}
