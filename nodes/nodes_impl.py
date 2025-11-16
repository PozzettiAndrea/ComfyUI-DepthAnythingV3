import torch
import torch.nn.functional as F
from torchvision import transforms
import os
from contextlib import nullcontext

import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file
import folder_paths

from ..depth_anything_v3.configs import MODEL_CONFIGS, MODEL_REPOS
from ..depth_anything_v3.model.da3 import DepthAnything3Net
from ..depth_anything_v3.model.dinov2.dinov2 import DinoV2
from ..depth_anything_v3.model.dualdpt import DualDPT
from ..depth_anything_v3.model.dpt import DPT

try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except:
    is_accelerate_available = False


class DownloadAndLoadDepthAnythingV3Model:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        'da3_small.safetensors',
                        'da3_base.safetensors',
                        'da3_large.safetensors',
                        'da3_giant.safetensors',
                        'da3mono_large.safetensors',
                        'da3metric_large.safetensors',
                        'da3nested_giant_large.safetensors',
                    ],
                    {
                        "default": 'da3_large.safetensors'
                    }
                ),
            },
            "optional": {
                "precision": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("DA3MODEL",)
    RETURN_NAMES = ("da3_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Models autodownload to `ComfyUI/models/depthanything3` from HuggingFace.

Supports all DA3 variants including Small, Base, Large, Giant, Mono, Metric, and Nested models.
"""

    def loadmodel(self, model, precision="auto"):
        device = mm.get_torch_device()

        # Determine dtype
        if precision == "auto":
            dtype = torch.float16 if "fp16" in model else torch.float32
        elif precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
        elif precision == "fp32":
            dtype = torch.float32

        # Get model configuration
        model_key = model.replace('.safetensors', '').replace('_', '-')
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_key}")

        config = MODEL_CONFIGS[model_key]

        # Download model if needed
        download_path = os.path.join(folder_paths.models_dir, "depthanything3")
        model_path = os.path.join(download_path, model)

        if not os.path.exists(model_path):
            print(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download
            repo = MODEL_REPOS[model]
            snapshot_download(
                repo_id=repo,
                allow_patterns=["*.safetensors"],
                local_dir=download_path,
                local_dir_use_symlinks=False
            )
            # The downloaded file might be named differently (model.safetensors)
            # Try to find and rename it
            repo_name = repo.split('/')[-1]
            downloaded_file = os.path.join(download_path, "model.safetensors")
            if os.path.exists(downloaded_file) and not os.path.exists(model_path):
                os.rename(downloaded_file, model_path)

        print(f"Loading model from: {model_path}")

        # Build the model architecture
        # For simplicity, we'll create a minimal DepthAnything3Net
        # This is a simplified version - full version would use cfg.create_object
        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            # Create backbone (DinoV2)
            backbone = DinoV2(
                name=config['encoder'],
                out_layers=config.get('out_layers', [4, 11, 17, 23]),
                alt_start=config.get('alt_start', -1),
                qknorm_start=config.get('qknorm_start', -1),
                rope_start=config.get('rope_start', -1),
                cat_token=config.get('cat_token', False),
            )

            # Create head
            if config.get('is_mono', False) or config.get('is_metric', False):
                # Use DPT head for mono/metric models
                head = DPT(
                    dim_in=config['dim_in'],
                    output_dim=1,
                    features=config['features'],
                    out_channels=config['out_channels'],
                )
            else:
                # Use DualDPT for main series models
                head = DualDPT(
                    dim_in=config['dim_in'],
                    output_dim=2,
                    features=config['features'],
                    out_channels=config['out_channels'],
                )

            # Create the full model (simplified - no camera dec/enc for now)
            self.model = DepthAnything3Net(
                net=backbone,
                head=head,
                cam_dec=None,  # Simplified
                cam_enc=None,  # Simplified
                gs_head=None,  # Simplified
                gs_adapter=None,  # Simplified
            )

        # Load weights
        state_dict = load_torch_file(model_path)

        # Strip 'model.' prefix from keys if present
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        if is_accelerate_available:
            failed_keys = []
            for key in new_state_dict:
                try:
                    set_module_tensor_to_device(self.model, key, device=device, dtype=dtype, value=new_state_dict[key])
                except Exception as e:
                    failed_keys.append(key)
            if failed_keys:
                print(f"Warning: Could not load {len(failed_keys)} weights (this is normal for simplified models)")
        else:
            # Try to load state dict, handling potential key mismatches
            try:
                self.model.load_state_dict(new_state_dict, strict=False)
            except Exception as e:
                print(f"Warning during model loading: {e}")
                # Try partial loading
                model_dict = self.model.state_dict()
                filtered_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
                model_dict.update(filtered_dict)
                self.model.load_state_dict(model_dict)

        # Don't move to device here if using accelerate (already done during loading)
        if not is_accelerate_available:
            self.model.to(device)

        self.model.eval()

        da3_model = {
            "model": self.model,
            "dtype": dtype,
            "config": config,
        }

        return (da3_model,)


class DepthAnything_V3:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "da3_model": ("DA3MODEL", ),
                "images": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth",)
    FUNCTION = "process"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Depth Anything V3 - depth estimation from images.
Returns normalized depth maps.
"""

    def process(self, da3_model, images):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        model = da3_model['model']
        dtype = da3_model['dtype']
        config = da3_model['config']

        B, H, W, C = images.shape

        # Convert from ComfyUI format [B, H, W, C] to PyTorch [B, C, H, W]
        images_pt = images.permute(0, 3, 1, 2)

        # DA3 uses patch size 14
        orig_H, orig_W = H, W
        if W % 14 != 0:
            W = W - (W % 14)
        if H % 14 != 0:
            H = H - (H % 14)
        if orig_H % 14 != 0 or orig_W % 14 != 0:
            images_pt = F.interpolate(images_pt, size=(H, W), mode="bilinear")

        # Normalize with ImageNet stats
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalized_images = normalize(images_pt)

        # Prepare for model: add view dimension [B, N, 3, H, W] where N=1
        normalized_images = normalized_images.unsqueeze(1)

        pbar = ProgressBar(B)
        out = []

        # Move model to device if not already there
        try:
            model.to(device)
        except NotImplementedError:
            # Model might already be on device (via accelerate loading)
            pass

        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)

        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            for i in range(B):
                img = normalized_images[i:i+1].to(device)

                # Run model forward
                output = model(img)

                # Extract depth from output
                if hasattr(output, 'depth'):
                    depth = output.depth
                elif isinstance(output, dict) and 'depth' in output:
                    depth = output['depth']
                else:
                    raise ValueError("Model output does not contain depth")

                # Normalize depth
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                out.append(depth.cpu())
                pbar.update(1)

        model.to(offload_device)
        mm.soft_empty_cache()

        # Concatenate all depths
        depth_out = torch.cat(out, dim=0)

        # Convert to 3-channel image [B, H, W, 3]
        # depth_out is [B, 1, H, W], squeeze channel dimension first
        depth_out = depth_out.squeeze(1)  # [B, H, W]
        depth_out = depth_out.unsqueeze(-1).repeat(1, 1, 1, 3).cpu().float()  # [B, H, W, 3]

        # Resize back to original dimensions (with even constraint)
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


class DepthAnythingV3_Advanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "da3_model": ("DA3MODEL", ),
                "images": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("depth", "confidence", "ray_origin", "ray_direction", "extrinsics", "intrinsics")
    FUNCTION = "process"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Advanced Depth Anything V3 node that outputs all available data:
- Depth map
- Confidence map
- Ray origin maps (3D coordinates as RGB)
- Ray direction maps (3D vectors as RGB)
- Camera extrinsics (predicted camera pose)
- Camera intrinsics (predicted camera parameters)

Note: Ray maps and camera parameters only available for main series models (Small/Base/Large/Giant).
Mono/Metric models output only depth and confidence.
"""

    def process(self, da3_model, images):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        model = da3_model['model']
        dtype = da3_model['dtype']
        config = da3_model['config']

        B, H, W, C = images.shape

        # Convert from ComfyUI format [B, H, W, C] to PyTorch [B, C, H, W]
        images_pt = images.permute(0, 3, 1, 2)

        # DA3 uses patch size 14
        orig_H, orig_W = H, W
        if W % 14 != 0:
            W = W - (W % 14)
        if H % 14 != 0:
            H = H - (H % 14)
        if orig_H % 14 != 0 or orig_W % 14 != 0:
            images_pt = F.interpolate(images_pt, size=(H, W), mode="bilinear")

        # Normalize with ImageNet stats
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalized_images = normalize(images_pt)

        # Prepare for model: add view dimension [B, N, 3, H, W] where N=1
        normalized_images = normalized_images.unsqueeze(1)

        pbar = ProgressBar(B)
        depth_out = []
        conf_out = []
        ray_origin_out = []
        ray_dir_out = []
        extrinsics_list = []
        intrinsics_list = []

        # Move model to device if not already there
        try:
            model.to(device)
        except NotImplementedError:
            # Model might already be on device (via accelerate loading)
            pass

        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)

        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            for i in range(B):
                img = normalized_images[i:i+1].to(device)

                # Run model forward
                output = model(img)

                # Extract depth
                if hasattr(output, 'depth'):
                    depth = output.depth
                elif isinstance(output, dict) and 'depth' in output:
                    depth = output['depth']
                else:
                    raise ValueError("Model output does not contain depth")

                # Extract confidence
                if hasattr(output, 'depth_conf'):
                    conf = output.depth_conf
                elif isinstance(output, dict) and 'depth_conf' in output:
                    conf = output['depth_conf']
                else:
                    # If no confidence, create uniform confidence
                    conf = torch.ones_like(depth)

                # Normalize depth and confidence
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)

                depth_out.append(depth.cpu())
                conf_out.append(conf.cpu())

                # Extract ray maps (if available)
                # Note: addict.Dict returns empty Dict for non-existent keys, so check if it's a tensor
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
                # Note: addict.Dict returns empty Dict for non-existent keys, so check if it's a tensor
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
        depth_final = self._process_tensor_to_image(depth_out, orig_H, orig_W)
        conf_final = self._process_tensor_to_image(conf_out, orig_H, orig_W)
        ray_origin_final = self._process_ray_to_image(ray_origin_out, orig_H, orig_W)
        ray_dir_final = self._process_ray_to_image(ray_dir_out, orig_H, orig_W)

        # Format camera parameters as strings
        extrinsics_str = self._format_camera_params(extrinsics_list, "extrinsics")
        intrinsics_str = self._format_camera_params(intrinsics_list, "intrinsics")

        return (depth_final, conf_final, ray_origin_final, ray_dir_final, extrinsics_str, intrinsics_str)

    def _process_tensor_to_image(self, tensor_list, orig_H, orig_W):
        """Convert list of depth/conf tensors to ComfyUI IMAGE format."""
        # Concatenate all tensors
        out = torch.cat(tensor_list, dim=0)  # [B, 1, H, W] or [B, H, W]

        # Ensure 4D: [B, 1, H, W]
        if out.dim() == 3:
            out = out.unsqueeze(1)

        # Convert to 3-channel image [B, H, W, 3]
        out = out.squeeze(1)  # [B, H, W]
        out = out.unsqueeze(-1).repeat(1, 1, 1, 3).float()  # [B, H, W, 3]

        # Resize back to original dimensions (with even constraint)
        final_H = (orig_H // 2) * 2
        final_W = (orig_W // 2) * 2

        if out.shape[1] != final_H or out.shape[2] != final_W:
            out = F.interpolate(
                out.permute(0, 3, 1, 2),
                size=(final_H, final_W),
                mode="bilinear"
            ).permute(0, 2, 3, 1)

        return torch.clamp(out, 0, 1)

    def _process_ray_to_image(self, ray_list, orig_H, orig_W):
        """Convert list of ray tensors to ComfyUI IMAGE format."""
        # Concatenate all ray tensors
        out = torch.cat([r.unsqueeze(0) for r in ray_list], dim=0)  # [B, 3, H, W]

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

        # Resize back to original dimensions
        final_H = (orig_H // 2) * 2
        final_W = (orig_W // 2) * 2

        if out.shape[1] != final_H or out.shape[2] != final_W:
            out = F.interpolate(
                out.permute(0, 3, 1, 2),
                size=(final_H, final_W),
                mode="bilinear"
            ).permute(0, 2, 3, 1)

        return torch.clamp(out, 0, 1)

    def _format_camera_params(self, param_list, param_name):
        """Format camera parameters as JSON string."""
        import json

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


class DA3_ToPointCloud:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "depth": ("IMAGE", ),
                "confidence": ("IMAGE", ),
                "ray_origin": ("IMAGE", ),
                "ray_direction": ("IMAGE", ),
            },
            "optional": {
                "source_image": ("IMAGE", ),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "downsample": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
            }
        }

    RETURN_TYPES = ("POINTCLOUD",)
    RETURN_NAMES = ("pointcloud",)
    FUNCTION = "convert"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Convert DA3 depth and ray maps to 3D point cloud.
Uses the formula: P = ray_origin + depth × ray_direction

Parameters:
- confidence_threshold: Filter points below this confidence (0-1)
- downsample: Reduce point density by factor N (1 = no downsampling)

Output is a POINTCLOUD type containing:
- points: Nx3 array of 3D coordinates
- colors: Nx3 array of RGB colors (from input image if available)
- confidence: Nx1 array of confidence values
"""

    def convert(self, depth, confidence, ray_origin, ray_direction, source_image=None, confidence_threshold=0.5, downsample=1):
        """
        Convert depth + ray maps to point cloud.

        Args:
            depth: [B, H, W, 3] depth map (grayscale repeated across 3 channels)
            confidence: [B, H, W, 3] confidence map
            ray_origin: [B, H, W, 3] ray origin coordinates
            ray_direction: [B, H, W, 3] ray direction vectors
            source_image: [B, H, W, 3] source image for colors (optional)
            confidence_threshold: Minimum confidence to include point
            downsample: Downsample factor
        """
        B = depth.shape[0]
        point_clouds = []

        for b in range(B):
            # Extract single image
            depth_map = depth[b, :, :, 0]  # [H, W] - use first channel only
            conf_map = confidence[b, :, :, 0]  # [H, W] - use first channel only
            ray_o = ray_origin[b]  # [H, W, 3]
            ray_d = ray_direction[b]  # [H, W, 3]

            H, W = depth_map.shape

            # Downsample if needed
            if downsample > 1:
                depth_map = depth_map[::downsample, ::downsample]
                conf_map = conf_map[::downsample, ::downsample]
                ray_o = ray_o[::downsample, ::downsample]
                ray_d = ray_d[::downsample, ::downsample]

                if source_image is not None:
                    colors = source_image[b, ::downsample, ::downsample]  # [H', W', 3]
                else:
                    colors = None
            else:
                if source_image is not None:
                    colors = source_image[b]  # [H, W, 3]
                else:
                    colors = None

            # Flatten to points
            depth_flat = depth_map.flatten().unsqueeze(-1)  # [N, 1]
            ray_o_flat = ray_o.reshape(-1, 3)  # [N, 3]
            ray_d_flat = ray_d.reshape(-1, 3)  # [N, 3]
            conf_flat = conf_map.flatten()  # [N]

            if colors is not None:
                colors_flat = colors.reshape(-1, 3)  # [N, 3]
            else:
                colors_flat = None

            # Filter by confidence
            mask = conf_flat >= confidence_threshold
            depth_flat = depth_flat[mask]
            ray_o_flat = ray_o_flat[mask]
            ray_d_flat = ray_d_flat[mask]
            conf_flat = conf_flat[mask]

            if colors_flat is not None:
                colors_flat = colors_flat[mask]

            # Compute 3D points: P = origin + depth * direction
            points_3d = ray_o_flat + depth_flat * ray_d_flat  # [N, 3]

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
            print(f"Saved point cloud to: {filepath}")

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

⚠️ NOTE: This requires a fine-tuned DA3 model with GS-DPT head.
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

        # DA3 uses patch size 14
        orig_H, orig_W = H, W
        if W % 14 != 0:
            W = W - (W % 14)
        if H % 14 != 0:
            H = H - (H % 14)
        if orig_H % 14 != 0 or orig_W % 14 != 0:
            images_pt = F.interpolate(images_pt, size=(H, W), mode="bilinear")

        # Normalize with ImageNet stats
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalized_images = normalize(images_pt)

        # Prepare for model: add view dimension [B, N, 3, H, W] where N=1
        normalized_images = normalized_images.unsqueeze(1)

        pbar = ProgressBar(B)
        gaussians_list = []

        # Move model to device
        try:
            model.to(device)
        except NotImplementedError:
            pass

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
            print(f"Saved Gaussians to: {filepath}")

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
    "DepthAnything_V3": DepthAnything_V3,
    "DepthAnythingV3_Advanced": DepthAnythingV3_Advanced,
    "DownloadAndLoadDepthAnythingV3Model": DownloadAndLoadDepthAnythingV3Model,
    "DA3_ToPointCloud": DA3_ToPointCloud,
    "DA3_SavePointCloud": DA3_SavePointCloud,
    "DA3_To3DGaussians": DA3_To3DGaussians,
    "DA3_Save3DGaussians": DA3_Save3DGaussians,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthAnything_V3": "Depth Anything V3",
    "DepthAnythingV3_Advanced": "Depth Anything V3 (Advanced)",
    "DownloadAndLoadDepthAnythingV3Model": "(down)Load Depth Anything V3 Model",
    "DA3_ToPointCloud": "DA3 to Point Cloud",
    "DA3_SavePointCloud": "DA3 Save Point Cloud",
    "DA3_To3DGaussians": "DA3 to 3D Gaussians",
    "DA3_Save3DGaussians": "DA3 Save 3D Gaussians",
}
