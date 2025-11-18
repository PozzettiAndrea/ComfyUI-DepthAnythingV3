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
                "intrinsics": ("STRING", {"default": ""}),
                "source_image": ("IMAGE", ),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "downsample": ("INT", {"default": 10, "min": 1, "max": 16, "step": 1}),
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
- depth_raw: Metric depth map (from DA3_3D or DA3_Advanced)
- confidence: Confidence map
- intrinsics: (Optional) Camera intrinsics JSON from DA3_Advanced
- source_image: (Optional) Source image for point colors

If intrinsics not provided, uses default pinhole camera model.

Parameters:
- confidence_threshold: Filter points below this confidence (0-1)
- downsample: Reduce point density by factor N (1 = no downsampling)

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

    def _create_default_intrinsics(self, H, W):
        """Create default pinhole camera intrinsics."""
        # Assume focal length = image width (reasonable default)
        fx = fy = float(W)
        cx = W / 2.0
        cy = H / 2.0

        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32)

        return K

    def convert(self, depth_raw, confidence, intrinsics="", source_image=None, confidence_threshold=0.5, downsample=1):
        """Convert depth map to point cloud using geometric unprojection."""
        B = depth_raw.shape[0]
        point_clouds = []

        for b in range(B):
            # Extract single image
            depth_map = depth_raw[b, :, :, 0]  # [H, W] - use first channel only
            conf_map = confidence[b, :, :, 0]  # [H, W] - use first channel only

            H, W = depth_map.shape

            # Get camera intrinsics
            K = self._parse_intrinsics(intrinsics, b)
            if K is None:
                K = self._create_default_intrinsics(H, W)
                intrinsics_source = "default"
            else:
                intrinsics_source = "DA3 model"

            # Downsample if needed
            if downsample > 1:
                depth_map = depth_map[::downsample, ::downsample]
                conf_map = conf_map[::downsample, ::downsample]

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
            K_inv = torch.inverse(K)
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
            points_3d = points_flat[mask]
            conf_flat = conf_flat[mask]

            if colors_flat is not None:
                colors_flat = colors_flat[mask]

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
Output directory: ComfyUI/output/
Returns file path for use with ComfyUI 3D viewer.
"""

    def save(self, pointcloud, filename_prefix):
        """Save point cloud(s) to PLY file."""
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

            # Generate filename
            filename = f"{filename_prefix}_{idx:04d}.ply"
            filepath = output_path / filename

            # Write PLY file
            self._write_ply(filepath, points, colors, confidence)

            results.append({
                "filename": filename,
                "subfolder": "",
                "type": "output"
            })
            file_paths.append(str(filepath))
            logger.info(f"Saved point cloud to: {filepath}")

        # Return first file path (or all paths joined by newline if multiple)
        output_file_path = file_paths[0] if len(file_paths) == 1 else "\n".join(file_paths)

        return {
            "ui": {"pointclouds": results},
            "result": (output_file_path,)
        }

    def _write_ply(self, filepath, points, colors=None, confidence=None):
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

                f.write(line + '\n')


class DA3_To3DGaussians:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "da3_model": ("DA3MODEL", ),
                "images": ("IMAGE", ),
            },
            "optional": {
                "enable_gs": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("GAUSSIANS",)
    RETURN_NAMES = ("gaussians",)
    FUNCTION = "process"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Extract 3D Gaussian Splats from DA3 model.

NOTE: This requires a fine-tuned DA3 model with GS-DPT head.
Base models (Small/Base/Large/Giant) do NOT include the GS head by default.

If your model supports 3DGS, this will output:
- Gaussian means (3D positions)
- Gaussian scales
- Gaussian rotations (quaternions)
- Spherical harmonics (appearance)
- Opacities

Output is a GAUSSIANS type that can be saved to PLY format.
"""

    def process(self, da3_model, images, enable_gs=True):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        model = da3_model['model']
        dtype = da3_model['dtype']
        config = da3_model['config']

        B, H, W, C = images.shape

        # Check if model has GS capability
        if not hasattr(model, 'gs_head') or model.gs_head is None:
            raise ValueError(
                "This model does not have a 3D Gaussian Splatting head. "
                "Please use a fine-tuned model with GS support (e.g., DA3-Giant with GS)."
            )

        # Convert from ComfyUI format [B, H, W, C] to PyTorch [B, C, H, W]
        images_pt = images.permute(0, 3, 1, 2)

        # Resize to patch size multiple
        images_pt, orig_H, orig_W = resize_to_patch_multiple(images_pt, DEFAULT_PATCH_SIZE)

        # Normalize with ImageNet stats
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        normalized_images = normalize(images_pt)

        # Prepare for model: add view dimension [B, N, 3, H, W] where N=1
        normalized_images = normalized_images.unsqueeze(1)

        pbar = ProgressBar(B)
        gaussians_list = []

        # Move model to device
        safe_model_to_device(model, device)

        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)

        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            for i in range(B):
                img = normalized_images[i:i+1].to(device)

                # Run model forward with GS inference enabled
                output = model(img, infer_gs=enable_gs)

                # Extract Gaussians
                if hasattr(output, 'gaussians'):
                    gaussians = output.gaussians
                elif isinstance(output, dict) and 'gaussians' in output:
                    gaussians = output['gaussians']
                else:
                    raise ValueError(
                        "Model output does not contain Gaussians. "
                        "Make sure your model has GS support and enable_gs=True."
                    )

                # Convert to dict format for serialization
                gs_dict = {
                    'means': gaussians.means.cpu(),
                    'scales': gaussians.scales.cpu(),
                    'rotations': gaussians.rotations.cpu(),
                    'harmonics': gaussians.harmonics.cpu(),
                    'opacities': gaussians.opacities.cpu(),
                }

                gaussians_list.append(gs_dict)
                pbar.update(1)

        model.to(offload_device)
        mm.soft_empty_cache()

        # Return as tuple containing list of Gaussians
        return (gaussians_list,)


class DA3_Save3DGaussians:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gaussians": ("GAUSSIANS", ),
                "filename_prefix": ("STRING", {"default": "gaussians"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Save 3D Gaussian Splats to PLY file.
Output directory: ComfyUI/output/
Returns file path for use with ComfyUI 3D viewer.

The saved PLY file can be viewed in:
- ComfyUI 3D Viewer
- SuperSplat (https://supersplat.io/)
- 3D Gaussian Splatting viewers
- Blender with appropriate plugins
"""

    def save(self, gaussians, filename_prefix):
        """Save Gaussians to PLY file."""
        import numpy as np
        from pathlib import Path

        # Get output directory
        output_dir = folder_paths.get_output_directory()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []
        file_paths = []
        for idx, gs in enumerate(gaussians):
            means = gs['means'].numpy() if hasattr(gs['means'], 'numpy') else gs['means']
            scales = gs['scales'].numpy() if hasattr(gs['scales'], 'numpy') else gs['scales']
            rotations = gs['rotations'].numpy() if hasattr(gs['rotations'], 'numpy') else gs['rotations']
            harmonics = gs['harmonics'].numpy() if hasattr(gs['harmonics'], 'numpy') else gs['harmonics']
            opacities = gs['opacities'].numpy() if hasattr(gs['opacities'], 'numpy') else gs['opacities']

            # Generate filename
            filename = f"{filename_prefix}_{idx:04d}.ply"
            filepath = output_path / filename

            # Write PLY file
            self._write_gaussian_ply(filepath, means, scales, rotations, harmonics, opacities)

            results.append({
                "filename": filename,
                "subfolder": "",
                "type": "output"
            })
            file_paths.append(str(filepath))
            logger.info(f"Saved Gaussians to: {filepath}")

        # Return first file path (or all paths joined by newline if multiple)
        output_file_path = file_paths[0] if len(file_paths) == 1 else "\n".join(file_paths)

        return {
            "ui": {"gaussians": results},
            "result": (output_file_path,)
        }

    def _write_gaussian_ply(self, filepath, means, scales, rotations, harmonics, opacities):
        """Write Gaussians to PLY file in standard 3DGS format."""
        import numpy as np

        # Flatten batch dimension if present
        if means.ndim == 3:  # [batch, N, 3]
            means = means.reshape(-1, 3)
            scales = scales.reshape(-1, 3)
            rotations = rotations.reshape(-1, 4)
            harmonics = harmonics.reshape(-1, harmonics.shape[-2], harmonics.shape[-1])
            if opacities.ndim > 1:
                opacities = opacities.reshape(-1)

        N = len(means)

        # Convert SH coefficients to RGB for DC component (first SH coefficient)
        # SH DC component: C_0 = 0.28209479177387814 * sh[0]
        sh_dc = harmonics[..., 0]  # [N, 3]
        colors = sh_dc * 0.28209479177387814  # Convert from SH to RGB
        colors = np.clip(colors * 255, 0, 255).astype(np.uint8)

        # Prepare header
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {N}",
            "property float x",
            "property float y",
            "property float z",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property float opacity",
            "end_header"
        ]

        # Write file
        with open(filepath, 'w') as f:
            # Write header
            f.write('\n'.join(header) + '\n')

            # Write Gaussians
            for i in range(N):
                x, y, z = means[i]
                sx, sy, sz = scales[i]
                qw, qx, qy, qz = rotations[i]  # Note: rotations are in wxyz format
                r, g, b = colors[i]
                opacity = opacities[i] if opacities.ndim == 1 else opacities[i].mean()

                line = f"{x} {y} {z} {sx} {sy} {sz} {qw} {qx} {qy} {qz} {r} {g} {b} {opacity}"
                f.write(line + '\n')


NODE_CLASS_MAPPINGS = {
    "DA3_ToPointCloud": DA3_ToPointCloud,
    "DA3_SavePointCloud": DA3_SavePointCloud,
    "DA3_To3DGaussians": DA3_To3DGaussians,
    "DA3_Save3DGaussians": DA3_Save3DGaussians,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DA3_ToPointCloud": "DA3 to Point Cloud",
    "DA3_SavePointCloud": "DA3 Save Point Cloud",
    "DA3_To3DGaussians": "DA3 to 3D Gaussians",
    "DA3_Save3DGaussians": "DA3 Save 3D Gaussians",
}
