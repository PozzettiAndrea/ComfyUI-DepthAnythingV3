"""Model loading and configuration nodes for DepthAnythingV3."""
import torch
import os
from contextlib import nullcontext

import comfy.model_management as mm
from comfy.utils import load_torch_file
import folder_paths

from .depth_anything_v3.configs import MODEL_CONFIGS, MODEL_REPOS
from .depth_anything_v3.model.da3 import DepthAnything3Net, NestedDepthAnything3Net
from .depth_anything_v3.model.dinov2.dinov2 import DinoV2
from .depth_anything_v3.model.dualdpt import DualDPT
from .depth_anything_v3.model.dpt import DPT
from .depth_anything_v3.model.cam_enc import CameraEnc
from .depth_anything_v3.model.cam_dec import CameraDec
from .utils import DEFAULT_PATCH_SIZE, logger

try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except (ImportError, ModuleNotFoundError):
    is_accelerate_available = False


class DA3ModelWrapper(torch.nn.Module):
    """Wrapper to match checkpoint parameter naming (da3.backbone... etc)"""
    def __init__(self, model):
        super().__init__()
        self.da3 = model

    def forward(self, *args, **kwargs):
        return self.da3(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.da3 = self.da3.to(*args, **kwargs)
        return self

    # Pass-through properties to access inner model attributes
    @property
    def cam_enc(self):
        return self.da3.cam_enc if hasattr(self.da3, 'cam_enc') else None

    @property
    def cam_dec(self):
        return self.da3.cam_dec if hasattr(self.da3, 'cam_dec') else None

    @property
    def gs_head(self):
        return self.da3.gs_head if hasattr(self.da3, 'gs_head') else None


class NestedModelWrapper(torch.nn.Module):
    """Wrapper for nested DA3 model with two branches (main + metric).

    This wrapper directly holds two DepthAnything3Net instances and delegates
    to NestedDepthAnything3Net's forward logic for metric scaling/alignment.
    """
    def __init__(self, da3_main, da3_metric):
        super().__init__()
        self.da3 = da3_main
        self.da3_metric = da3_metric

    def forward(self, *args, **kwargs):
        # Import alignment utilities lazily to avoid circular imports
        from .depth_anything_v3.utils.alignment import (
            apply_metric_scaling, compute_sky_mask, compute_alignment_mask,
            sample_tensor_for_quantile, least_squares_scale_scalar
        )

        # Get predictions from both branches
        output = self.da3(*args, **kwargs)
        # Metric branch doesn't use camera parameters
        x = args[0] if args else kwargs.get('x')
        infer_gs = kwargs.get('infer_gs', False)
        metric_output = self.da3_metric(x, infer_gs=infer_gs)

        # Apply metric scaling to depth
        metric_output.depth = apply_metric_scaling(
            metric_output.depth,
            output.intrinsics,
        )

        # Compute non-sky mask and alignment
        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)

        if non_sky_mask.sum() > 10:
            # Sample depth confidence for quantile computation
            depth_conf_ns = output.depth_conf[non_sky_mask]
            depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100000)
            median_conf = torch.quantile(depth_conf_sampled, 0.5)

            # Compute alignment mask
            align_mask = compute_alignment_mask(
                output.depth_conf, non_sky_mask, output.depth, metric_output.depth, median_conf
            )

            # Compute scale factor using least squares
            valid_depth = output.depth[align_mask]
            valid_metric_depth = metric_output.depth[align_mask]
            scale_factor = least_squares_scale_scalar(valid_metric_depth, valid_depth)

            # Apply scaling to depth and extrinsics
            output.depth *= scale_factor
            if hasattr(output, 'extrinsics') and output.extrinsics is not None:
                output.extrinsics[:, :, :3, 3] *= scale_factor
            output.is_metric = 1
            output.scale_factor = scale_factor.item()

            # Handle sky regions
            non_sky_depth = output.depth[non_sky_mask]
            if non_sky_depth.numel() > 100000:
                idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
                sampled_depth = non_sky_depth.flatten()[idx]
            else:
                sampled_depth = non_sky_depth
            max_depth = torch.quantile(sampled_depth, 0.99)
            sky_depth = max(200.0, max_depth.item() * 2.0)
            output.depth[~non_sky_mask] = sky_depth
            output.sky = metric_output.sky
        else:
            logger.warning("Insufficient non-sky pixels for metric alignment")
            output.sky = metric_output.sky

        return output

    def to(self, *args, **kwargs):
        self.da3 = self.da3.to(*args, **kwargs)
        self.da3_metric = self.da3_metric.to(*args, **kwargs)
        return self

    # Pass-through properties to main branch
    @property
    def cam_enc(self):
        return self.da3.cam_enc if hasattr(self.da3, 'cam_enc') else None

    @property
    def cam_dec(self):
        return self.da3.cam_dec if hasattr(self.da3, 'cam_dec') else None

    @property
    def gs_head(self):
        return self.da3.gs_head if hasattr(self.da3, 'gs_head') else None


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
            logger.info(f"Downloading model to: {model_path}")
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

        logger.info(f"Loading model from: {model_path}")

        # Build the model architecture
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
            # Check if this is a nested model (requires two branches)
            is_nested = config.get('is_nested', False)

            if is_nested:
                logger.info("Creating nested model with main (Giant) and metric (Large) branches")

                # Main branch: DA3-Giant with camera support
                backbone_main = DinoV2(
                    name=config['encoder'],  # vitg
                    out_layers=config.get('out_layers', [19, 27, 33, 39]),
                    alt_start=config.get('alt_start', 13),
                    qknorm_start=config.get('qknorm_start', 13),
                    rope_start=config.get('rope_start', 13),
                    cat_token=config.get('cat_token', True),
                )
                head_main = DualDPT(
                    dim_in=config['dim_in'],  # 3072 (vitg with cat_token)
                    output_dim=2,
                    features=config['features'],
                    out_channels=config['out_channels'],
                )
                embed_dim = encoder_embed_dims.get(config['encoder'], 1536)
                cam_enc_main = CameraEnc(
                    dim_out=embed_dim,
                    dim_in=9,
                    trunk_depth=4,
                    num_heads=embed_dim // 64,
                    mlp_ratio=4,
                    init_values=0.01,
                )
                cam_dec_main = CameraDec(dim_in=config['dim_in'])
                da3_main = DepthAnything3Net(
                    net=backbone_main,
                    head=head_main,
                    cam_dec=cam_dec_main,
                    cam_enc=cam_enc_main,
                    gs_head=None,
                    gs_adapter=None,
                )

                # Metric branch: DA3Metric-Large (no camera support, DPT head)
                metric_config = MODEL_CONFIGS.get('da3metric-large', {
                    'encoder': 'vitl',
                    'features': 256,
                    'out_channels': [256, 512, 1024, 1024],
                    'dim_in': 1024,
                    'out_layers': [4, 11, 17, 23],
                })
                backbone_metric = DinoV2(
                    name=metric_config.get('encoder', 'vitl'),
                    out_layers=metric_config.get('out_layers', [4, 11, 17, 23]),
                    alt_start=-1,
                    qknorm_start=-1,
                    rope_start=-1,
                    cat_token=False,
                )
                head_metric = DPT(
                    dim_in=metric_config.get('dim_in', 1024),
                    output_dim=1,
                    features=metric_config.get('features', 256),
                    out_channels=metric_config.get('out_channels', [256, 512, 1024, 1024]),
                )
                da3_metric = DepthAnything3Net(
                    net=backbone_metric,
                    head=head_metric,
                    cam_dec=None,
                    cam_enc=None,
                    gs_head=None,
                    gs_adapter=None,
                )

                inner_model = NestedModelWrapper(da3_main, da3_metric)
            else:
                # Standard single-branch model
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
                    cam_dec = CameraDec(
                        dim_in=config['dim_in'],  # Uses concatenated token dimension
                    )

                # Create the full model with camera encoder/decoder
                inner_model = DepthAnything3Net(
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
        stripped_count = 0
        for key, value in state_dict.items():
            new_key = key
            # Strip model. prefix only
            if new_key.startswith('model.'):
                new_key = new_key[6:]  # Remove 'model.' prefix
                stripped_count += 1
            new_state_dict[new_key] = value

        if stripped_count > 0:
            logger.debug(f"Stripped 'model.' prefix from {stripped_count} keys")
        # Show example keys
        sample_keys = list(new_state_dict.keys())[:3]
        logger.debug(f"Sample checkpoint keys: {sample_keys}")
        # Show head keys to understand structure
        head_keys = [k for k in new_state_dict.keys() if 'head.' in k]
        logger.debug(f"Checkpoint head keys ({len(head_keys)} total): {head_keys[:10]}")

        # Check if checkpoint uses da3. prefix (nested model format)
        has_da3_prefix = any(k.startswith('da3.') for k in new_state_dict.keys())

        if is_nested:
            # Nested model already has da3. and da3_metric. structure
            logger.debug("Using nested model wrapper (da3 + da3_metric branches)")
            self.model = inner_model
        elif has_da3_prefix:
            # Wrap model to match nested checkpoint structure (da3.backbone... etc)
            logger.debug("Detected nested model checkpoint format (da3. prefix)")
            self.model = DA3ModelWrapper(inner_model)
        else:
            # Use model directly (keys match backbone.*, head.*)
            logger.debug("Detected standard model checkpoint format (no prefix)")
            self.model = inner_model

        if use_empty_weights:
            # Used init_empty_weights, must use set_module_tensor_to_device
            failed_keys = []
            loaded_keys = []
            for key in new_state_dict:
                try:
                    set_module_tensor_to_device(self.model, key, device=device, dtype=dtype, value=new_state_dict[key])
                    loaded_keys.append(key)
                except Exception as e:
                    failed_keys.append((key, str(e)))
            if failed_keys:
                logger.warning(f"Could not load {len(failed_keys)} weights (this is normal for simplified models)")
                # Debug: show first few failed keys to understand the pattern
                logger.debug(f"First 10 failed keys: {[k for k, e in failed_keys[:10]]}")
                # Show head-specific failures
                head_failures = [(k, e) for k, e in failed_keys if k.startswith('head.')]
                if head_failures:
                    logger.debug(f"Head failures ({len(head_failures)}): {head_failures[:5]}")

            # Materialize any remaining meta tensors (parameters not in checkpoint)
            meta_params = []
            for name, param in self.model.named_parameters():
                if param.device.type == 'meta':
                    meta_params.append(name)
                    # Check if this key exists in checkpoint but wasn't loaded (shape mismatch?)
                    if name in new_state_dict:
                        ckpt_shape = new_state_dict[name].shape
                        model_shape = param.shape
                        if ckpt_shape != model_shape:
                            logger.debug(f"Shape mismatch for {name}: checkpoint {ckpt_shape} vs model {model_shape}")
                    # Initialize with zeros and move to correct device
                    set_module_tensor_to_device(
                        self.model, name, device=device, dtype=dtype,
                        value=torch.zeros(param.shape, dtype=dtype)
                    )

            if meta_params:
                logger.warning(f"Initialized {len(meta_params)} missing parameters with zeros (not in checkpoint)")
                # Debug: show first few meta params to understand the pattern
                logger.debug(f"First 10 missing params: {meta_params[:10]}")
        else:
            # Standard model loading (CPU or no accelerate)
            try:
                self.model.load_state_dict(new_state_dict, strict=False)
            except Exception as e:
                logger.warning(f"Exception during model loading: {e}")
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


class DA3_EnableTiledProcessing:
    """Configure model for tiled processing to handle high-resolution images."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "da3_model": ("DA3MODEL", ),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 14}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 256, "step": 14}),
            },
        }

    RETURN_TYPES = ("DA3MODEL",)
    RETURN_NAMES = ("da3_model",)
    FUNCTION = "configure"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Enable tiled processing for memory-efficient inference on high-resolution images.

This node configures the model to process images in tiles with overlapping regions,
then blends the results for seamless output.

Parameters:
- tile_size: Size of each tile (should be multiple of 14 for patch alignment)
- overlap: Overlap between adjacent tiles for smooth blending

Use this when:
- Processing 4K+ resolution images
- GPU memory is limited
- Getting out-of-memory errors

Note: Tiled processing may produce slightly different results at tile boundaries,
but the overlap and blending minimize artifacts.
"""

    def configure(self, da3_model, tile_size=512, overlap=64):
        # Ensure tile_size is multiple of patch size
        patch_size = DEFAULT_PATCH_SIZE
        tile_size = (tile_size // patch_size) * patch_size
        if tile_size < patch_size:
            tile_size = patch_size

        # Ensure overlap is multiple of patch size
        overlap = (overlap // patch_size) * patch_size

        # Create a copy of the model dict with tiled config
        tiled_model = {
            "model": da3_model["model"],
            "dtype": da3_model["dtype"],
            "config": da3_model["config"],
            "tiled_config": {
                "enabled": True,
                "tile_size": tile_size,
                "overlap": overlap,
            }
        }

        logger.info(f"Enabled tiled processing: tile_size={tile_size}, overlap={overlap}")

        return (tiled_model,)


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadDepthAnythingV3Model": DownloadAndLoadDepthAnythingV3Model,
    "DA3_EnableTiledProcessing": DA3_EnableTiledProcessing,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadDepthAnythingV3Model": "(down)Load Depth Anything V3 Model",
    "DA3_EnableTiledProcessing": "DA3 Enable Tiled Processing",
}
