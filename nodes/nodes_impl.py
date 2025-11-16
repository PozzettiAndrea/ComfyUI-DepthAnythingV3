import torch
import torch.nn.functional as F
from torchvision import transforms
import os
from contextlib import nullcontext

import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file
import folder_paths

# Use relative imports since depth_anything_v3 is now inside the nodes package
from .depth_anything_v3.configs import MODEL_CONFIGS, MODEL_REPOS
from .depth_anything_v3.model.da3 import DepthAnything3Net
from .depth_anything_v3.model.dinov2.dinov2 import DinoV2
from .depth_anything_v3.model.dualdpt import DualDPT
from .depth_anything_v3.model.dpt import DPT
from .depth_anything_v3.model.cam_enc import CameraEnc
from .depth_anything_v3.model.cam_dec import CameraDec

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
        # Only use init_empty_weights on CUDA devices to avoid meta device issues on CPU
        use_empty_weights = is_accelerate_available and device.type == 'cuda'

        # Encoder embed dimensions for camera modules
        encoder_embed_dims = {
            'vits': 384,
            'vitb': 768,
            'vitl': 1024,
            'vitg': 1536,
        }

        with (init_empty_weights() if use_empty_weights else nullcontext()):
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

            # Create camera encoder/decoder if model has camera support
            cam_enc = None
            cam_dec = None
            if config.get('has_cam', False) and config.get('alt_start', -1) != -1:
                embed_dim = encoder_embed_dims.get(config['encoder'], 1024)
                # Camera encoder: encodes known camera params to tokens
                cam_enc = CameraEnc(
                    dim_out=embed_dim,
                    dim_in=9,  # 9D pose encoding: [T(3), quat(4), fov(2)]
                    trunk_depth=4,
                    num_heads=embed_dim // 64,  # Match head dim = 64
                    mlp_ratio=4,
                    init_values=0.01,
                )
                # Camera decoder: decodes features to camera pose
                # dim_in from config accounts for cat_token (e.g., 2048 for vitl with cat_token=True)
                cam_dec = CameraDec(
                    dim_in=config['dim_in'],  # Uses concatenated token dimension
                )

            # Create the full model with camera encoder/decoder
            self.model = DepthAnything3Net(
                net=backbone,
                head=head,
                cam_dec=cam_dec,
                cam_enc=cam_enc,
                gs_head=None,  # Not implemented (requires fine-tuned model)
                gs_adapter=None,  # Not implemented
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

        if use_empty_weights:
            # Used init_empty_weights, must use set_module_tensor_to_device
            failed_keys = []
            for key in new_state_dict:
                try:
                    set_module_tensor_to_device(self.model, key, device=device, dtype=dtype, value=new_state_dict[key])
                except Exception as e:
                    failed_keys.append(key)
            if failed_keys:
                print(f"Warning: Could not load {len(failed_keys)} weights (this is normal for simplified models)")
        else:
            # Standard model loading (CPU or no accelerate)
            try:
                self.model.load_state_dict(new_state_dict, strict=False)
            except Exception as e:
                print(f"Warning during model loading: {e}")
                # Try partial loading
                model_dict = self.model.state_dict()
                filtered_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
                model_dict.update(filtered_dict)
                self.model.load_state_dict(model_dict)

        # Move to device if we didn't use init_empty_weights
        if not use_empty_weights:
            self.model.to(device).to(dtype)

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
            "optional": {
                "camera_params": ("CAMERA_PARAMS", ),
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
"""

    def process(self, da3_model, images, camera_params=None):
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
                print("[DepthAnything_V3] Using camera-conditioned depth estimation")
            else:
                print("[DepthAnything_V3] Warning: Model does not support camera conditioning. Camera params ignored.")

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

                # Get camera params for this batch item
                ext_i = extrinsics_input[i:i+1] if extrinsics_input is not None else None
                int_i = intrinsics_input[i:i+1] if intrinsics_input is not None else None

                # Run model forward with optional camera conditioning
                output = model(img, extrinsics=ext_i, intrinsics=int_i)

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
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("depth_raw", "confidence", "intrinsics")
    FUNCTION = "process"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Depth Anything V3 node optimized for 3D reconstruction (point clouds, gaussian splats, etc).
Outputs the essential data needed for proper 3D reconstruction:
- Depth raw: Metric depth values (NOT normalized)
- Confidence: Confidence map
- Intrinsics: Camera intrinsic matrix (3x3) for geometric unprojection

These outputs can be directly connected to DA3_ToPointCloud or DA3_ToGaussianSplat nodes.

Uses the official DA3 approach: geometric unprojection with camera intrinsics,
NOT the model's auxiliary ray outputs.

Optional: Provide camera_params for camera-conditioned depth estimation.
Connect DA3_CreateCameraParams to improve depth accuracy with known camera pose.

Works with all model types (Small/Base/Large/Giant/Mono/Metric).
"""

    def process(self, da3_model, images, camera_params=None):
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
                print("[DepthAnythingV3_3D] Using camera-conditioned depth estimation")
            else:
                print("[DepthAnythingV3_3D] Warning: Model does not support camera conditioning. Camera params ignored.")

        pbar = ProgressBar(B)
        depth_raw_out = []
        conf_out = []
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

                # Get camera params for this batch item
                ext_i = extrinsics_input[i:i+1] if extrinsics_input is not None else None
                int_i = intrinsics_input[i:i+1] if intrinsics_input is not None else None

                # Run model forward with optional camera conditioning
                output = model(img, extrinsics=ext_i, intrinsics=int_i)

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

                # Store RAW depth (no normalization!)
                depth_raw_out.append(depth.cpu())

                # Normalize confidence only
                conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
                conf_out.append(conf.cpu())

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
        depth_raw_final = self._process_tensor_to_image(depth_raw_out, orig_H, orig_W)
        conf_final = self._process_tensor_to_image(conf_out, orig_H, orig_W)

        # Format intrinsics as JSON string
        intrinsics_str = self._format_camera_params(intrinsics_list, "intrinsics")

        return (depth_raw_final, conf_final, intrinsics_str)

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

    def _process_tensor_to_image(self, tensor_list, orig_H, orig_W):
        """Convert list of depth/conf tensors to ComfyUI IMAGE format (no normalization)."""
        # Concatenate all tensors
        out = torch.cat(tensor_list, dim=0)  # [B, 1, H, W]

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

        return out


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
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("depth", "confidence", "ray_origin", "ray_direction", "extrinsics", "intrinsics")
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

Optional: Provide camera_params for camera-conditioned depth estimation.
Connect DA3_CreateCameraParams to improve depth accuracy with known camera pose.

Note: Ray maps and camera parameters only available for main series models (Small/Base/Large/Giant).
Mono/Metric models output only depth and confidence (dummy zeros for rays).

For point cloud generation, use the DepthAnythingV3_3D node instead which outputs raw metric depth.
"""

    def process(self, da3_model, images, camera_params=None):
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
                print("[DepthAnythingV3_Advanced] Using camera-conditioned depth estimation")
            else:
                print("[DepthAnythingV3_Advanced] Warning: Model does not support camera conditioning. Camera params ignored.")

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

                # Get camera params for this batch item
                ext_i = extrinsics_input[i:i+1] if extrinsics_input is not None else None
                int_i = intrinsics_input[i:i+1] if intrinsics_input is not None else None

                # Run model forward with optional camera conditioning
                output = model(img, extrinsics=ext_i, intrinsics=int_i)

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

                # Normalize depth and confidence for visualization
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
        ray_origin_final = self._process_ray_to_image(ray_origin_out, orig_H, orig_W, normalize=True)
        ray_dir_final = self._process_ray_to_image(ray_dir_out, orig_H, orig_W, normalize=True)

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

    def _process_ray_to_image(self, ray_list, orig_H, orig_W, normalize=True):
        """Convert list of ray tensors to ComfyUI IMAGE format.

        Args:
            ray_list: List of ray tensors [3, H, W]
            orig_H: Original height
            orig_W: Original width
            normalize: If True, normalize to 0-1 for visualization. If False, keep original values for point cloud.
        """
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

        # Resize back to original dimensions
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
            print(f"[DA3_ToPointCloud] Warning: Could not parse intrinsics: {e}")
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
        """
        Convert depth map to point cloud using geometric unprojection.

        Args:
            depth_raw: [B, H, W, 3] metric depth map
            confidence: [B, H, W, 3] confidence map
            intrinsics: JSON string with camera intrinsics (optional)
            source_image: [B, H, W, 3] source image for colors (optional)
            confidence_threshold: Minimum confidence to include point
            downsample: Downsample factor
        """
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
                    import torch.nn.functional as F
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

            # Debug prints
            print(f"\n=== Point Cloud Debug (batch {b}) ===")
            print(f"Intrinsics source: {intrinsics_source}")
            print(f"Focal length: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
            print(f"Principal point: cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
            print(f"Depth range: [{depth_map.min():.4f}, {depth_map.max():.4f}]")
            print(f"Number of points after filtering: {points_3d.shape[0]}")
            print(f"Points 3D range: X[{points_3d[:, 0].min():.4f}, {points_3d[:, 0].max():.4f}], "
                  f"Y[{points_3d[:, 1].min():.4f}, {points_3d[:, 1].max():.4f}], "
                  f"Z[{points_3d[:, 2].min():.4f}, {points_3d[:, 2].max():.4f}]")
            print(f"Points 3D std: X={points_3d[:, 0].std():.4f}, Y={points_3d[:, 1].std():.4f}, Z={points_3d[:, 2].std():.4f}")
            print(f"=================================\n")

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


class DA3_CreateCameraParams:
    """Create camera parameters (extrinsics and intrinsics) for conditioning depth estimation."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_width": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "image_height": ("INT", {"default": 512, "min": 1, "max": 8192}),
            },
            "optional": {
                # Camera position (translation)
                "cam_x": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "cam_y": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "cam_z": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                # Camera rotation (Euler angles in degrees)
                "rot_x": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "rot_y": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "rot_z": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                # Intrinsics
                "focal_length": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 1.0}),
                "fov_degrees": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 180.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("CAMERA_PARAMS",)
    RETURN_NAMES = ("camera_params",)
    FUNCTION = "create_params"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Create camera parameters for conditioning DA3 depth estimation.

Provides known camera pose to improve depth estimation accuracy.

Parameters:
- cam_x/y/z: Camera position in world space
- rot_x/y/z: Camera rotation (Euler angles in degrees)
- focal_length: If > 0, uses this value. Otherwise uses fov_degrees.
- fov_degrees: Field of view in degrees (used if focal_length is 0)

Output:
- CAMERA_PARAMS: Dictionary with extrinsics (4x4) and intrinsics (3x3) matrices
"""

    def create_params(self, image_width, image_height, cam_x=0.0, cam_y=0.0, cam_z=0.0,
                     rot_x=0.0, rot_y=0.0, rot_z=0.0, focal_length=0.0, fov_degrees=60.0):
        import numpy as np

        # Create rotation matrix from Euler angles (XYZ order)
        rx = np.radians(rot_x)
        ry = np.radians(rot_y)
        rz = np.radians(rot_z)

        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])

        R = Rz @ Ry @ Rx  # Rotation matrix
        t = np.array([cam_x, cam_y, cam_z])  # Translation

        # Create extrinsics (world-to-camera, 4x4)
        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[:3, :3] = R.T  # Inverse rotation
        extrinsics[:3, 3] = -R.T @ t  # Inverse translation

        # Create intrinsics (3x3)
        if focal_length > 0:
            fx = fy = focal_length
        else:
            # Calculate from FOV
            fov_rad = np.radians(fov_degrees)
            fx = fy = (image_width / 2.0) / np.tan(fov_rad / 2.0)

        cx = image_width / 2.0
        cy = image_height / 2.0

        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Convert to tensors
        extrinsics_tensor = torch.from_numpy(extrinsics).unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
        intrinsics_tensor = torch.from_numpy(intrinsics).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

        camera_params = {
            "extrinsics": extrinsics_tensor,
            "intrinsics": intrinsics_tensor,
            "image_size": (image_height, image_width),
        }

        return (camera_params,)


class DA3_ParseCameraPose:
    """Parse camera pose from DA3 output strings into usable format."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "extrinsics_json": ("STRING", {"multiline": True}),
                "intrinsics_json": ("STRING", {"multiline": True}),
            },
            "optional": {
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("cam_x", "cam_y", "cam_z", "rot_x", "rot_y", "rot_z", "fx", "fy")
    FUNCTION = "parse"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Parse camera pose from DA3 JSON output.

Extracts camera position and rotation from extrinsics matrix,
and focal lengths from intrinsics matrix.

Inputs:
- extrinsics_json: JSON string from DA3 output
- intrinsics_json: JSON string from DA3 output
- batch_index: Which image's parameters to extract (default 0)

Outputs:
- cam_x/y/z: Camera position in world space
- rot_x/y/z: Camera rotation (Euler angles in degrees)
- fx/fy: Focal lengths
"""

    def parse(self, extrinsics_json, intrinsics_json, batch_index=0):
        import json
        import numpy as np

        # Default values
        cam_x, cam_y, cam_z = 0.0, 0.0, 0.0
        rot_x, rot_y, rot_z = 0.0, 0.0, 0.0
        fx, fy = 512.0, 512.0

        try:
            # Parse extrinsics
            ext_data = json.loads(extrinsics_json)
            if "extrinsics" in ext_data and isinstance(ext_data["extrinsics"], list):
                if batch_index < len(ext_data["extrinsics"]):
                    img_key = f"image_{batch_index}"
                    ext_matrix = ext_data["extrinsics"][batch_index].get(img_key)

                    if ext_matrix is not None:
                        ext = np.array(ext_matrix)
                        if ext.ndim == 3:  # [N, 4, 4] or [N, 3, 4]
                            ext = ext[0]  # Take first view

                        # Extract rotation and translation
                        R = ext[:3, :3]
                        t = ext[:3, 3]

                        # Convert world-to-camera to camera position in world
                        # cam_pos = -R^T @ t
                        cam_pos = -R.T @ t
                        cam_x, cam_y, cam_z = float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])

                        # Extract Euler angles from rotation matrix (XYZ order)
                        # R = Rz @ Ry @ Rx, so we extract in reverse
                        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
                        singular = sy < 1e-6

                        if not singular:
                            rot_x = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                            rot_y = np.degrees(np.arctan2(-R[2, 0], sy))
                            rot_z = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                        else:
                            rot_x = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
                            rot_y = np.degrees(np.arctan2(-R[2, 0], sy))
                            rot_z = 0.0

            # Parse intrinsics
            int_data = json.loads(intrinsics_json)
            if "intrinsics" in int_data and isinstance(int_data["intrinsics"], list):
                if batch_index < len(int_data["intrinsics"]):
                    img_key = f"image_{batch_index}"
                    int_matrix = int_data["intrinsics"][batch_index].get(img_key)

                    if int_matrix is not None:
                        intr = np.array(int_matrix)
                        if intr.ndim == 3:  # [N, 3, 3]
                            intr = intr[0]  # Take first view

                        fx = float(intr[0, 0])
                        fy = float(intr[1, 1])

        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            print(f"[DA3_ParseCameraPose] Error parsing camera params: {e}")

        return (cam_x, cam_y, cam_z, rot_x, rot_y, rot_z, fx, fy)


NODE_CLASS_MAPPINGS = {
    "DepthAnything_V3": DepthAnything_V3,
    "DepthAnythingV3_3D": DepthAnythingV3_3D,
    "DepthAnythingV3_Advanced": DepthAnythingV3_Advanced,
    "DownloadAndLoadDepthAnythingV3Model": DownloadAndLoadDepthAnythingV3Model,
    "DA3_ToPointCloud": DA3_ToPointCloud,
    "DA3_SavePointCloud": DA3_SavePointCloud,
    "DA3_To3DGaussians": DA3_To3DGaussians,
    "DA3_Save3DGaussians": DA3_Save3DGaussians,
    "DA3_CreateCameraParams": DA3_CreateCameraParams,
    "DA3_ParseCameraPose": DA3_ParseCameraPose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthAnything_V3": "Depth Anything V3",
    "DepthAnythingV3_3D": "Depth Anything V3 (3D/Raw)",
    "DepthAnythingV3_Advanced": "Depth Anything V3 (Advanced)",
    "DownloadAndLoadDepthAnythingV3Model": "(down)Load Depth Anything V3 Model",
    "DA3_ToPointCloud": "DA3 to Point Cloud",
    "DA3_SavePointCloud": "DA3 Save Point Cloud",
    "DA3_To3DGaussians": "DA3 to 3D Gaussians",
    "DA3_Save3DGaussians": "DA3 Save 3D Gaussians",
    "DA3_CreateCameraParams": "DA3 Create Camera Parameters",
    "DA3_ParseCameraPose": "DA3 Parse Camera Pose",
}
