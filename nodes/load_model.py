"""Model loading and configuration nodes for DepthAnythingV3."""
import logging
import torch
import os

import comfy.ops
import comfy.model_management as mm
import comfy.model_patcher
from comfy.utils import load_torch_file
import folder_paths
from comfy_api.latest import io

from .depth_anything_v3.configs import MODEL_CONFIGS, MODEL_REPOS
from .depth_anything_v3.model import (
    DepthAnything3Net, NestedDepthAnything3Net,
    DinoV2, DualDPT, DPT,
)
from .depth_anything_v3.camera import CameraEnc, CameraDec
from .depth_anything_v3.gs import GSDPT, GaussianAdapter
from .utils import DEFAULT_PATCH_SIZE, logger, check_model_capabilities

# Register model folder with ComfyUI's folder_paths system
_da3_model_dir = os.path.join(folder_paths.models_dir, "depthanything3")
os.makedirs(_da3_model_dir, exist_ok=True)
folder_paths.add_model_folder_path("depth_anything_v3", _da3_model_dir)


def _get_da3_model_list():
    """Get combined model list: known HuggingFace models + locally-placed files."""
    local_models = folder_paths.get_filename_list("depth_anything_v3")
    known_models = list(MODEL_REPOS.keys())
    # Known models first, then any extra local files, deduplicated
    return list(dict.fromkeys(known_models + local_models))


# Encoder embed dimensions for camera modules
ENCODER_EMBED_DIMS = {
    'vits': 384,
    'vitb': 768,
    'vitl': 1024,
    'vitg': 1536,
}


def detect_da3_variant(state_dict):
    """Auto-detect DA3 model variant from state_dict keys and weight shapes.

    Returns the model config key (e.g. 'da3-large', 'da3mono-large', etc.).
    """
    keys = set(state_dict.keys())

    # Strip 'model.' prefix for analysis
    stripped_keys = set()
    for k in keys:
        stripped_keys.add(k[6:] if k.startswith('model.') else k)

    # 1. Check for nested model (has both da3. and da3_metric. branches)
    has_da3_metric = any(k.startswith('da3_metric.') for k in stripped_keys)
    if has_da3_metric:
        return 'da3nested-giant-large'

    # Determine effective key prefix (some checkpoints use da3. prefix)
    has_da3_prefix = any(k.startswith('da3.') for k in stripped_keys)
    prefix = 'da3.' if has_da3_prefix else ''

    # 2. Detect encoder type from backbone embed_dim
    # The patch_embed projection weight shape tells us the embed_dim
    patch_key = f'{prefix}net.patch_embed.proj.weight'
    if patch_key in stripped_keys:
        embed_dim = state_dict.get(patch_key, state_dict.get(f'model.{patch_key}')).shape[0]
    else:
        # Try alternative key patterns
        embed_dim = None
        for k in stripped_keys:
            if 'patch_embed.proj.weight' in k:
                sd_key = k if k in state_dict else f'model.{k}'
                if sd_key in state_dict:
                    embed_dim = state_dict[sd_key].shape[0]
                break

    # Map embed_dim to encoder name
    dim_to_encoder = {384: 'vits', 768: 'vitb', 1024: 'vitl', 1536: 'vitg'}
    encoder = dim_to_encoder.get(embed_dim)

    # 3. Check for camera modules (main series vs mono/metric)
    has_cam = any('cam_enc' in k for k in stripped_keys)

    # 4. Check for GS head (giant model)
    has_gs = any('gs_head' in k for k in stripped_keys)

    # 5. Check head type: DPT (1-ch, mono/metric) vs DualDPT (2-ch, main series)
    # DualDPT has output_conv1 with 2 output channels, DPT has 1
    head_output_key = f'{prefix}head.output_conv1.2.weight'
    is_dual_head = False
    if head_output_key in stripped_keys:
        sd_key = head_output_key if head_output_key in state_dict else f'model.{head_output_key}'
        if sd_key in state_dict:
            out_channels = state_dict[sd_key].shape[0]
            is_dual_head = (out_channels == 2)

    # Map to config key
    if encoder == 'vitg' and has_cam:
        if has_gs:
            return 'da3-giant'
        return 'da3-giant'  # giant always has GS potential
    elif encoder == 'vitl':
        if has_cam and is_dual_head:
            return 'da3-large'
        elif not has_cam and not is_dual_head:
            # Could be mono or metric - architecturally identical
            # Default to mono, user can override via filename hint
            return 'da3mono-large'
        elif has_cam:
            return 'da3-large'
        else:
            return 'da3mono-large'
    elif encoder == 'vitb':
        return 'da3-base'
    elif encoder == 'vits':
        return 'da3-small'

    # Fallback: try to guess from key count or structure
    logger.warning(f"Could not auto-detect model variant (embed_dim={embed_dim}, "
                   f"has_cam={has_cam}, has_gs={has_gs}). Defaulting to da3-large.")
    return 'da3-large'


def detect_da3_variant_with_filename_hint(state_dict, filename):
    """Auto-detect variant, using filename as a hint for ambiguous cases."""
    variant = detect_da3_variant(state_dict)

    # For mono/metric disambiguation (same architecture), check filename
    if variant == 'da3mono-large' and filename:
        fname_lower = filename.lower()
        if 'metric' in fname_lower:
            return 'da3metric-large'

    return variant


def _build_gs_modules(config, operations):
    """Build GS head and adapter for Giant model."""
    gs_head = GSDPT(
        dim_in=config['dim_in'],
        output_dim=38,
        features=config['features'],
        out_channels=config['out_channels'],
        operations=operations,
    )

    gs_adapter = GaussianAdapter(
        sh_degree=2,
        pred_color=False,
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
    """Wrapper for nested DA3 model with two branches (main + metric)."""
    def __init__(self, da3_main, da3_metric):
        super().__init__()
        self.da3 = da3_main
        self.da3_metric = da3_metric

    def forward(self, *args, **kwargs):
        from .depth_anything_v3.alignment import (
            apply_metric_scaling, compute_sky_mask, compute_alignment_mask,
            sample_tensor_for_quantile, least_squares_scale_scalar
        )

        output = self.da3(*args, **kwargs)
        x = args[0] if args else kwargs.get('x')
        infer_gs = kwargs.get('infer_gs', False)
        metric_output = self.da3_metric(x, infer_gs=infer_gs)

        metric_output.depth = apply_metric_scaling(
            metric_output.depth,
            output.intrinsics,
        )

        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)

        if non_sky_mask.sum() > 10:
            depth_conf_ns = output.depth_conf[non_sky_mask]
            depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100000)
            median_conf = torch.quantile(depth_conf_sampled, 0.5)

            align_mask = compute_alignment_mask(
                output.depth_conf, non_sky_mask, output.depth, metric_output.depth, median_conf
            )

            valid_depth = output.depth[align_mask]
            valid_metric_depth = metric_output.depth[align_mask]
            scale_factor = least_squares_scale_scalar(valid_metric_depth, valid_depth)

            output.depth *= scale_factor
            if hasattr(output, 'extrinsics') and output.extrinsics is not None:
                output.extrinsics[:, :, :3, 3] *= scale_factor
            output.is_metric = 1
            output.scale_factor = scale_factor.item()

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

    Returns a loaded nn.Module on CPU with the given dtype.
    """
    logger.info(f"[build] model_key={model_key}, dtype={dtype}, attention={attention}")
    config = MODEL_CONFIGS[model_key]
    logger.info(f"[build] config: encoder={config.get('encoder')}, dim_in={config.get('dim_in')}, "
                f"features={config.get('features')}, out_channels={config.get('out_channels')}")

    is_nested = config.get('is_nested', False)
    operations = comfy.ops.manual_cast

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
                operations=operations,
            )
            head_main = DualDPT(
                dim_in=config['dim_in'],
                output_dim=2,
                features=config['features'],
                out_channels=config['out_channels'],
                operations=operations,
            )
            embed_dim = ENCODER_EMBED_DIMS.get(config['encoder'], 1536)
            cam_enc_main = CameraEnc(
                dim_out=embed_dim,
                dim_in=9,
                trunk_depth=4,
                num_heads=embed_dim // 64,
                mlp_ratio=4,
                init_values=0.01,
                operations=operations,
            )
            cam_dec_main = CameraDec(dim_in=config['dim_in'], operations=operations)
            gs_head_main, gs_adapter_main = _build_gs_modules(config, operations)

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
                operations=operations,
            )
            head_metric = DPT(
                dim_in=metric_config.get('dim_in', 1024),
                output_dim=1,
                features=metric_config.get('features', 256),
                out_channels=metric_config.get('out_channels', [256, 512, 1024, 1024]),
                operations=operations,
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
                operations=operations,
            )

            if config.get('is_mono', False) or config.get('is_metric', False):
                head = DPT(
                    dim_in=config['dim_in'],
                    output_dim=1,
                    features=config['features'],
                    out_channels=config['out_channels'],
                    operations=operations,
                )
            else:
                head = DualDPT(
                    dim_in=config['dim_in'],
                    output_dim=2,
                    features=config['features'],
                    out_channels=config['out_channels'],
                    operations=operations,
                )

            cam_enc = None
            cam_dec = None
            if config.get('has_cam', False) and config.get('alt_start', -1) != -1:
                embed_dim = ENCODER_EMBED_DIMS.get(config['encoder'], 1024)
                cam_enc = CameraEnc(
                    dim_out=embed_dim,
                    dim_in=9,
                    trunk_depth=4,
                    num_heads=embed_dim // 64,
                    mlp_ratio=4,
                    init_values=0.01,
                    operations=operations,
                )
                cam_dec = CameraDec(dim_in=config['dim_in'], operations=operations)

            gs_head = None
            gs_adapter = None
            if config.get('has_3d_gaussians', model_key == 'da3-giant'):
                gs_head, gs_adapter = _build_gs_modules(config, operations)
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
    logger.info(f"[build] Loading model from: {model_path}")
    state_dict = load_torch_file(model_path)
    logger.info(f"[build] Checkpoint: {len(state_dict)} keys loaded")

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
        logger.info(f"[build] Stripped 'model.' prefix from {stripped_count} keys")

    # Check if checkpoint uses da3. prefix (nested model format)
    has_da3_prefix = any(k.startswith('da3.') for k in new_state_dict.keys())

    if is_nested:
        model = inner_model
    elif has_da3_prefix:
        model = DA3ModelWrapper(inner_model)
    else:
        model = inner_model

    # Expand shared-weight checkpoint keys (e.g. output_conv2_aux.0.2 -> 1.2, 2.2, 3.2)
    expanded = {}
    for key in list(new_state_dict.keys()):
        if 'output_conv2_aux.0.' in key:
            prefix, suffix = key.split('output_conv2_aux.0.')
            for idx in range(1, 4):
                clone_key = f"{prefix}output_conv2_aux.{idx}.{suffix}"
                if clone_key not in new_state_dict:
                    expanded[clone_key] = new_state_dict[key].clone()
    if expanded:
        logger.info(f"[build] Expanded {len(expanded)} shared-weight keys for output_conv2_aux")
        new_state_dict.update(expanded)

    # Load weights via assign=True (PyTorch 2.1+)
    result = model.load_state_dict(new_state_dict, strict=False, assign=True)
    if result.missing_keys:
        logger.info(f"[build] Missing keys ({len(result.missing_keys)}): {result.missing_keys[:10]}")
    if result.unexpected_keys:
        logger.info(f"[build] Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:10]}")
    model.to(dtype=dtype)

    # Safety: replace any remaining meta parameters with zeros
    for name, param in list(model.named_parameters()):
        if param.device.type == 'meta':
            logger.warning(f"[build] Meta param remaining: {name}, initializing with zeros")
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], torch.nn.Parameter(
                torch.zeros(param.shape, dtype=dtype), requires_grad=False
            ))

    # Safety: replace any remaining meta buffers with zeros
    for name, buf in list(model.named_buffers()):
        if buf.device.type == 'meta':
            logger.warning(f"[build] Meta buffer remaining: {name}, initializing with zeros")
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            parent.register_buffer(parts[-1], torch.zeros(buf.shape, dtype=buf.dtype))

    model.eval()
    logger.info(f"Model ready ({dtype})")
    return model


class DownloadAndLoadDepthAnythingV3Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DownloadAndLoadDepthAnythingV3Model",
            display_name="Load Depth Anything V3 Model",
            category="loaders",
            description="Load a Depth Anything V3 model from ComfyUI/models/depthanything3/. "
                        "Model variant is auto-detected from weights. Supports Small, Base, Large, Giant, Mono, Metric, and Nested.",
            inputs=[
                io.Combo.Input("model", options=_get_da3_model_list(), default='da3_large.safetensors',
                               tooltip="Select a model. Auto-downloads from HuggingFace if not found locally in ComfyUI/models/depthanything3/."),
                io.Combo.Input("precision", options=["auto", "bf16", "fp16", "fp32"], default="auto", optional=True),
                io.Combo.Input("attention", options=["auto", "sdpa", "flash_attn", "sage"], default="auto", optional=True,
                               tooltip="Attention backend. auto: best available (sage > flash_attn > sdpa)."),
            ],
            outputs=[
                io.Custom("DA3MODEL").Output(display_name="da3_model"),
            ],
        )

    @classmethod
    def execute(cls, model, precision="auto", attention="auto"):
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
        else:
            dtype = torch.float32

        # Resolve full path — auto-download from HuggingFace if not found locally
        model_path = folder_paths.get_full_path("depth_anything_v3", model)
        if model_path is None and model in MODEL_REPOS:
            download_dir = os.path.join(folder_paths.models_dir, "depthanything3")
            os.makedirs(download_dir, exist_ok=True)
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                raise ImportError(
                    "huggingface_hub is required to auto-download models. "
                    "Install with: pip install huggingface_hub\n"
                    "Or manually download and place in ComfyUI/models/depthanything3/"
                )
            logger.info(f"Auto-downloading {model} from HuggingFace ({MODEL_REPOS[model]})...")
            snapshot_download(
                repo_id=MODEL_REPOS[model],
                allow_patterns=["*.safetensors"],
                local_dir=download_dir,
                local_dir_use_symlinks=False,
            )
            # HuggingFace may save as model.safetensors — rename to expected name
            hf_default = os.path.join(download_dir, "model.safetensors")
            target_path = os.path.join(download_dir, model)
            if os.path.exists(hf_default) and not os.path.exists(target_path):
                os.rename(hf_default, target_path)
            model_path = target_path
            logger.info(f"Downloaded model to: {model_path}")

        if model_path is None:
            raise FileNotFoundError(
                f"Model '{model}' not found in ComfyUI/models/depthanything3/ and not a known HuggingFace model."
            )

        # Auto-detect model variant from state_dict
        sd = load_torch_file(model_path)
        model_key = detect_da3_variant_with_filename_hint(sd, model)
        logger.info(f"Auto-detected model variant: {model_key}")
        del sd  # Free memory before building model

        config = MODEL_CONFIGS[model_key]

        # Build and load model immediately
        loaded_model = _build_da3_model(model_path, model_key, dtype, attention)

        # Wrap in ModelPatcher for ComfyUI memory management
        patcher = comfy.model_patcher.ModelPatcher(
            loaded_model,
            load_device=mm.get_torch_device(),
            offload_device=mm.unet_offload_device(),
        )

        # Store metadata on patcher for use by inference nodes
        patcher.model_options["da3_capabilities"] = check_model_capabilities(loaded_model)
        patcher.model_options["da3_config"] = config
        patcher.model_options["da3_dtype"] = dtype
        patcher.model_options["da3_model_key"] = model_key

        return io.NodeOutput(patcher)


class DA3_EnableTiledProcessing(io.ComfyNode):
    """Configure model for tiled processing to handle high-resolution images."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DA3_EnableTiledProcessing",
            display_name="DA3 Enable Tiled Processing",
            category="DepthAnythingV3",
            description="Enable tiled processing for memory-efficient inference on high-resolution images. "
                        "tile_size should be a multiple of 14 for patch alignment.",
            inputs=[
                io.Custom("DA3MODEL").Input("da3_model"),
                io.Int.Input("tile_size", default=512, min=256, max=2048, step=14),
                io.Int.Input("overlap", default=64, min=0, max=256, step=14),
            ],
            outputs=[
                io.Custom("DA3MODEL").Output(display_name="da3_model"),
            ],
        )

    @classmethod
    def execute(cls, da3_model, tile_size=512, overlap=64):
        patch_size = DEFAULT_PATCH_SIZE
        tile_size = (tile_size // patch_size) * patch_size
        if tile_size < patch_size:
            tile_size = patch_size
        overlap = (overlap // patch_size) * patch_size

        # Store tiled config in model_options on the ModelPatcher
        da3_model.model_options["da3_tiled_config"] = {
            "enabled": True,
            "tile_size": tile_size,
            "overlap": overlap,
        }

        logger.info(f"Enabled tiled processing: tile_size={tile_size}, overlap={overlap}")

        return io.NodeOutput(da3_model)


class DA3_DownloadModel(io.ComfyNode):
    """Download a DA3 model from HuggingFace if not already present."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DA3_DownloadModel",
            display_name="DA3 Download Model (HuggingFace)",
            category="DepthAnythingV3",
            is_output_node=True,
            description="Download a DA3 model from HuggingFace to ComfyUI/models/depthanything3/. "
                        "Only downloads if not already present. After downloading, use the Load node to load the model.",
            inputs=[
                io.Combo.Input("model", options=list(MODEL_REPOS.keys()), default='da3_large.safetensors'),
            ],
            outputs=[
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, model):
        download_path = os.path.join(folder_paths.models_dir, "depthanything3")
        os.makedirs(download_path, exist_ok=True)
        model_path = os.path.join(download_path, model)

        if os.path.exists(model_path):
            return io.NodeOutput(f"Model already exists: {model_path}")

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            return io.NodeOutput("Error: huggingface_hub not installed. Install with: pip install huggingface_hub")

        logger.info(f"Downloading model to: {model_path}")
        repo = MODEL_REPOS[model]
        snapshot_download(
            repo_id=repo,
            allow_patterns=["*.safetensors"],
            local_dir=download_path,
            local_dir_use_symlinks=False
        )
        # The downloaded file might be named differently (model.safetensors)
        downloaded_file = os.path.join(download_path, "model.safetensors")
        if os.path.exists(downloaded_file) and not os.path.exists(model_path):
            os.rename(downloaded_file, model_path)

        return io.NodeOutput(f"Downloaded: {model_path}")


SALAD_CKPT_URL = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
SALAD_CKPT_NAME = "dino_salad.ckpt"


def _build_salad_model(ckpt_path):
    """Build and load the SALAD VPR model from checkpoint."""
    from .salad.model import VPRModel

    model = VPRModel(operations=comfy.ops.manual_cast)
    state_dict = load_torch_file(str(ckpt_path))
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"SALAD model loaded from: {ckpt_path}")
    return model


class LoadSALADModel(io.ComfyNode):
    """Download and load the SALAD model for visual place recognition / loop closure."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadSALADModel",
            display_name="Load SALAD Model",
            category="DepthAnythingV3",
            description="Load the SALAD model for loop closure detection. "
                        "Uses DINOv2 ViT-B14 backbone + SALAD aggregator (~340MB). "
                        "Auto-downloads from GitHub on first use.",
            inputs=[],
            outputs=[
                io.Custom("SALAD_MODEL").Output(display_name="salad_model"),
            ],
        )

    @classmethod
    def execute(cls):
        download_dir = os.path.join(folder_paths.models_dir, "salad")
        os.makedirs(download_dir, exist_ok=True)
        ckpt_path = os.path.join(download_dir, SALAD_CKPT_NAME)

        if not os.path.exists(ckpt_path):
            logger.info(f"Downloading SALAD model to: {ckpt_path}")
            torch.hub.download_url_to_file(SALAD_CKPT_URL, ckpt_path)

        # Build and wrap in ModelPatcher for ComfyUI memory management
        salad_nn = _build_salad_model(ckpt_path)
        patcher = comfy.model_patcher.ModelPatcher(
            salad_nn,
            load_device=mm.get_torch_device(),
            offload_device=mm.unet_offload_device(),
        )

        return io.NodeOutput(patcher)
