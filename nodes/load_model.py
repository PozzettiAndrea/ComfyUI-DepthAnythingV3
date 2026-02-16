"""Model loading and configuration nodes for DepthAnythingV3."""
import torch
import os

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
from .depth_anything_v3.model.gsdpt import GSDPT
from .depth_anything_v3.model.gs_adapter import GaussianAdapter
from .utils import DEFAULT_PATCH_SIZE, logger
from .depth_anything_v3.model.attention_dispatch import set_backend as set_attention_backend, auto_detect_precision

try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except (ImportError, ModuleNotFoundError):
    is_accelerate_available = False


def _build_gs_modules(config):
    """Build GS head and adapter for Giant model.

    Only Giant model has gs_head/gs_adapter in the checkpoint.
    Config from da3-giant.yaml: gs_head output_dim=38, gs_adapter sh_degree=2.
    """
    # GS head: GSDPT with Giant config
    gs_head = GSDPT(
        dim_in=config['dim_in'],  # 3072 for Giant
        output_dim=38,  # matches GaussianAdapter.d_in with sh_degree=2
        features=config['features'],  # 256
        out_channels=config['out_channels'],  # [256, 512, 1024, 1024]
    )

    # GS adapter: converts raw GS output to Gaussians
    gs_adapter = GaussianAdapter(
        sh_degree=2,
        pred_color=False,  # predict SH coefficients
        pred_offset_depth=True,
        pred_offset_xy=True,
        gaussian_scale_min=1e-5,
        gaussian_scale_max=30.0,
    )

    return gs_head, gs_adapter


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

    @property
    def gs_adapter(self):
        return self.da3.gs_adapter if hasattr(self.da3, 'gs_adapter') else None


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

    @property
    def gs_adapter(self):
        return self.da3.gs_adapter if hasattr(self.da3, 'gs_adapter') else None


def _build_da3_model(model_path, model_key, dtype, attention):
    """Build and load the DA3 model from checkpoint.

    Called lazily by inference nodes on first use.
    Returns a loaded nn.Module on CPU with the given dtype.
    """
    config = MODEL_CONFIGS[model_key]

    # Encoder embed dimensions for camera modules
    encoder_embed_dims = {
        'vits': 384,
        'vitb': 768,
        'vitl': 1024,
        'vitg': 1536,
    }

    is_nested = config.get('is_nested', False)

    with torch.device("meta"):
        if is_nested:
            logger.info("Creating nested model with main (Giant) and metric (Large) branches")

            backbone_main = DinoV2(
                name=config['encoder'],
                out_layers=config.get('out_layers', [19, 27, 33, 39]),
                alt_start=config.get('alt_start', 13),
                qknorm_start=config.get('qknorm_start', 13),
                rope_start=config.get('rope_start', 13),
                cat_token=config.get('cat_token', True),
            )
            head_main = DualDPT(
                dim_in=config['dim_in'],
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
            gs_head_main, gs_adapter_main = _build_gs_modules(config)

            da3_main = DepthAnything3Net(
                net=backbone_main,
                head=head_main,
                cam_dec=cam_dec_main,
                cam_enc=cam_enc_main,
                gs_head=gs_head_main,
                gs_adapter=gs_adapter_main,
            )

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
            backbone = DinoV2(
                name=config['encoder'],
                out_layers=config.get('out_layers', [4, 11, 17, 23]),
                alt_start=config.get('alt_start', -1),
                qknorm_start=config.get('qknorm_start', -1),
                rope_start=config.get('rope_start', -1),
                cat_token=config.get('cat_token', False),
            )

            if config.get('is_mono', False) or config.get('is_metric', False):
                head = DPT(
                    dim_in=config['dim_in'],
                    output_dim=1,
                    features=config['features'],
                    out_channels=config['out_channels'],
                )
            else:
                head = DualDPT(
                    dim_in=config['dim_in'],
                    output_dim=2,
                    features=config['features'],
                    out_channels=config['out_channels'],
                )

            cam_enc = None
            cam_dec = None
            if config.get('has_cam', False) and config.get('alt_start', -1) != -1:
                embed_dim = encoder_embed_dims.get(config['encoder'], 1024)
                cam_enc = CameraEnc(
                    dim_out=embed_dim,
                    dim_in=9,
                    trunk_depth=4,
                    num_heads=embed_dim // 64,
                    mlp_ratio=4,
                    init_values=0.01,
                )
                cam_dec = CameraDec(dim_in=config['dim_in'])

            gs_head = None
            gs_adapter = None
            if model_key == 'da3-giant':
                gs_head, gs_adapter = _build_gs_modules(config)
                logger.info("Built GS head and adapter for Giant model (Gaussian splatting enabled)")

            inner_model = DepthAnything3Net(
                net=backbone,
                head=head,
                cam_dec=cam_dec,
                cam_enc=cam_enc,
                gs_head=gs_head,
                gs_adapter=gs_adapter,
            )

    # Load weights
    logger.info(f"Loading model from: {model_path}")
    state_dict = load_torch_file(model_path)

    # Strip 'model.' prefix from keys if present
    new_state_dict = {}
    stripped_count = 0
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith('model.'):
            new_key = new_key[6:]
            stripped_count += 1
        new_state_dict[new_key] = value

    if stripped_count > 0:
        logger.debug(f"Stripped 'model.' prefix from {stripped_count} keys")
    sample_keys = list(new_state_dict.keys())[:3]
    logger.debug(f"Sample checkpoint keys: {sample_keys}")
    head_keys = [k for k in new_state_dict.keys() if 'head.' in k]
    logger.debug(f"Checkpoint head keys ({len(head_keys)} total): {head_keys[:10]}")

    # Check if checkpoint uses da3. prefix (nested model format)
    has_da3_prefix = any(k.startswith('da3.') for k in new_state_dict.keys())

    if is_nested:
        logger.debug("Using nested model wrapper (da3 + da3_metric branches)")
        model = inner_model
    elif has_da3_prefix:
        logger.debug("Detected nested model checkpoint format (da3. prefix)")
        model = DA3ModelWrapper(inner_model)
    else:
        logger.debug("Detected standard model checkpoint format (no prefix)")
        model = inner_model

    # Load weights -- meta->real via assign=True (PyTorch 2.1+) or accelerate fallback
    try:
        model.load_state_dict(new_state_dict, strict=False, assign=True)
        model.to(dtype=dtype)  # set dtype, stay on CPU
    except TypeError:
        if is_accelerate_available:
            logger.info("Using accelerate fallback for weight loading (PyTorch < 2.1)")
            offload_device = mm.unet_offload_device()
            for key in new_state_dict:
                try:
                    set_module_tensor_to_device(model, key, device=offload_device, dtype=dtype, value=new_state_dict[key])
                except Exception:
                    pass
            for name, param in model.named_parameters():
                if param.device.type == 'meta':
                    set_module_tensor_to_device(
                        model, name, device=offload_device, dtype=dtype,
                        value=torch.zeros(param.shape, dtype=dtype)
                    )
        else:
            raise RuntimeError(
                "Model loading requires PyTorch >= 2.1 (for assign=True) or the accelerate package. "
                "Please upgrade PyTorch or install accelerate: pip install accelerate"
            )

    model.eval()
    set_attention_backend(attention)
    logger.info(f"Model ready ({dtype})")
    return model


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
                "attention": (["auto", "sdpa", "flash_attn", "sage"], {
                    "default": "auto",
                    "tooltip": "Attention backend. auto: best available (sage > flash_attn > sdpa). sdpa: PyTorch native. flash_attn: Tri Dao's FlashAttention (FA2/FA3, requires flash-attn package). sage: SageAttention (auto-detects v3 for Blackwell or v2, requires sageattention/sageattn3 package)."
                }),
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

    def loadmodel(self, model, precision="auto", attention="auto"):
        """Resolve config and download model if needed. No weight loading."""
        device = mm.get_torch_device()

        # Determine dtype
        if precision == "auto":
            if mm.should_use_bf16(device):
                dtype = torch.bfloat16
            elif mm.should_use_fp16(device):
                dtype = torch.float16
            else:
                dtype = torch.float32
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
            downloaded_file = os.path.join(download_path, "model.safetensors")
            if os.path.exists(downloaded_file) and not os.path.exists(model_path):
                os.rename(downloaded_file, model_path)

        return ({
            "model_path": model_path,
            "model_key": model_key,
            "dtype": dtype,
            "attention": attention,
            "config": config,
        },)


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

        # Shallow copy config dict so tiled config doesn't affect original
        tiled = dict(da3_model)
        tiled["tiled_config"] = {
            "enabled": True,
            "tile_size": tile_size,
            "overlap": overlap,
        }

        logger.info(f"Enabled tiled processing: tile_size={tile_size}, overlap={overlap}")

        return (tiled,)


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadDepthAnythingV3Model": DownloadAndLoadDepthAnythingV3Model,
    "DA3_EnableTiledProcessing": DA3_EnableTiledProcessing,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadDepthAnythingV3Model": "(down)Load Depth Anything V3 Model",
    "DA3_EnableTiledProcessing": "DA3 Enable Tiled Processing",
}
