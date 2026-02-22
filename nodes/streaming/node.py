"""DA3 Streaming node — Chunked depth processing with Sim(3) alignment for long videos."""
import logging
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import comfy.model_management as mm
import folder_paths
from comfy.utils import ProgressBar
from comfy_api.latest import io

from ..utils import (
    IMAGENET_MEAN, IMAGENET_STD, DEFAULT_PATCH_SIZE,
    resize_to_patch_multiple, imagenet_normalize, check_model_capabilities,
)
from .pipeline import StreamingConfig, StreamingPipeline

logger = logging.getLogger("DA3Streaming")


class DepthAnythingV3_Streaming(io.ComfyNode):
    """Process long video sequences with chunked DA3 inference and Sim(3) alignment.

    Accepts VIDEO input, saves per-frame NPZ data to disk, and returns a
    grayscale depth VIDEO plus the NPZ folder path.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DepthAnythingV3_Streaming",
            display_name="DA3 Streaming",
            category="DepthAnythingV3",
            description="DA3 Streaming \u2014 Process long videos with chunked inference and Sim(3) alignment.\n\n"
                "**How it works:**\n"
                "1. Splits video into overlapping chunks (e.g., 30 frames, 8 overlap)\n"
                "2. Runs DA3 multi-view inference on each chunk\n"
                "3. Estimates Sim(3) alignment between chunks using overlap-region point clouds\n"
                "4. Blends overlap regions with linear interpolation\n"
                "5. If SALAD model is connected: runs loop closure for drift correction\n\n"
                "**Memory:** VRAM bounded to ~1 chunk at a time. Per-frame results saved to NPZ files on disk.\n\n"
                "**Outputs:**\n"
                "- depth_video: Grayscale depth visualization (connect to SaveVideo)\n"
                "- npz_folder: Path to folder with per-frame .npz files (depth, conf, intrinsics, extrinsics)\n"
                "- pointcloud_path: PLY file path (if save_pointcloud enabled)\n\n"
                "**Loop closure:** Connect a \"Load SALAD Model\" node to the salad_model input to enable automatic loop closure detection.",
            inputs=[
                io.Custom("DA3MODEL").Input("da3_model"),
                io.Video.Input("video", tooltip="Video input from LoadVideo node"),
                io.Combo.Input("normalization_mode", options=["Standard", "V2-Style", "Raw"], default="V2-Style"),
                io.Custom("SALAD_MODEL").Input("salad_model", optional=True,
                    tooltip="SALAD model for loop closure detection. Connect a Load SALAD Model node to enable loop closure."),
                io.Int.Input("chunk_size", default=30, min=4, max=256, step=1, optional=True,
                    tooltip="Frames per chunk. Lower = less VRAM. 30 for 24GB, 15 for 12GB VRAM."),
                io.Int.Input("overlap", default=8, min=2, max=64, step=1, optional=True,
                    tooltip="Overlap frames between chunks for Sim(3) alignment. 4-12 typical."),
                io.Combo.Input("align_lib", options=["auto", "torch", "triton", "numba", "numpy"], default="auto", optional=True,
                    tooltip="Alignment backend. auto selects fastest available (triton > torch > numba > numpy)."),
                io.Combo.Input("align_method", options=["sim3", "se3", "scale+se3"], default="sim3", optional=True,
                    tooltip="Alignment method. sim3: full 7-DOF. se3: 6-DOF (no scale). scale+se3: precompute scale then SE(3)."),
                io.Combo.Input("resize_method", options=["resize", "crop", "pad"], default="resize", optional=True,
                    tooltip="How to handle non-patch-aligned dimensions."),
                io.Boolean.Input("invert_depth", default=False, optional=True,
                    tooltip="Invert depth output (far=bright)."),
                io.Boolean.Input("save_pointcloud", default=False, optional=True,
                    tooltip="Export aligned point cloud as PLY file."),
                io.Float.Input("sample_ratio", default=0.015, min=0.001, max=1.0, step=0.005, optional=True,
                    tooltip="Point cloud downsampling ratio (lower = fewer points)."),
                io.Float.Input("conf_threshold_coef", default=0.75, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="Confidence threshold coefficient for point cloud filtering."),
            ],
            outputs=[
                io.Video.Output(display_name="depth_video"),
                io.String.Output(display_name="npz_folder"),
                io.String.Output(display_name="pointcloud_path"),
            ],
        )

    @staticmethod
    def _apply_standard_normalization(depth, invert_depth):
        d_min = depth.min()
        d_max = depth.max()
        d_range = d_max - d_min
        if d_range > 1e-8:
            depth = (depth - d_min) / d_range
        else:
            depth = torch.zeros_like(depth)
        if invert_depth:
            depth = 1.0 - depth
        return depth

    @staticmethod
    def _apply_v2_style_normalization(depth, sky, device, invert_depth):
        d_min = depth.min()
        d_max = depth.max()
        d_range = d_max - d_min
        if d_range > 1e-8:
            depth = (depth - d_min) / d_range
        else:
            depth = torch.zeros_like(depth)

        disparity = 1.0 / (depth + 1e-6)

        sky_mask = sky > 0.5 if sky is not None else torch.zeros_like(depth, dtype=torch.bool)
        non_sky = disparity[~sky_mask].flatten() if sky_mask.any() else disparity.flatten()

        if non_sky.numel() > 0:
            # Subsample for quantile — torch.quantile has element count limits
            if non_sky.numel() > 1_000_000:
                idx = torch.randint(0, non_sky.numel(), (1_000_000,), device=non_sky.device)
                sampled = non_sky[idx]
            else:
                sampled = non_sky
            p_low = torch.quantile(sampled, 0.02)
            p_high = torch.quantile(sampled, 0.98)
            disp_range = p_high - p_low
            if disp_range > 1e-8:
                depth = torch.clamp((disparity - p_low) / disp_range, 0, 1)
            else:
                depth = torch.clamp(disparity / (disparity.max() + 1e-6), 0, 1)
        else:
            depth = torch.clamp(disparity / (disparity.max() + 1e-6), 0, 1)

        if sky_mask.any():
            depth[sky_mask] = 0.0

        if invert_depth:
            depth = 1.0 - depth

        return depth

    @staticmethod
    def _apply_raw_normalization(depth, invert_depth):
        if invert_depth:
            d_min = depth.min()
            d_max = depth.max()
            d_range = d_max - d_min
            if d_range > 1e-8:
                depth = d_max - depth + d_min
        return depth

    @classmethod
    def execute(cls, da3_model, video, normalization_mode="V2-Style",
                salad_model=None,
                chunk_size=30, overlap=8,
                align_lib="auto", align_method="sim3",
                resize_method="resize", invert_depth=False,
                save_pointcloud=False, sample_ratio=0.015,
                conf_threshold_coef=0.75):

        def _vram_debug(tag, device):
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated(device) / (1024**3)
                reserved = torch.cuda.memory_reserved(device) / (1024**3)
                free_cuda = torch.cuda.mem_get_info(device)[0] / (1024**3)
                total_cuda = torch.cuda.mem_get_info(device)[1] / (1024**3)
                logger.debug(f"[VRAM {tag}] allocated={alloc:.2f}GB reserved={reserved:.2f}GB free={free_cuda:.2f}GB total={total_cuda:.2f}GB")

        def _tensor_debug(name, t):
            if torch.is_tensor(t):
                size_mb = t.nelement() * t.element_size() / (1024**2)
                logger.debug(f"[TENSOR {name}] shape={list(t.shape)} dtype={t.dtype} device={t.device} size={size_mb:.1f}MB")

        # Extract frames from VIDEO input
        components = video.get_components()
        images = components.images  # [N, H, W, C] float32 0-1
        fps = components.frame_rate  # Fraction

        if images is None or images.shape[0] == 0:
            raise ValueError("Video contains no frames.")

        num_views = images.shape[0]
        orig_H, orig_W = images.shape[1], images.shape[2]

        logger.info(f"Streaming: {num_views} frames, {orig_H}x{orig_W}, {float(fps):.2f} fps")

        # Get model and device
        device = mm.get_torch_device()

        _vram_debug("before_model_load", device)
        _tensor_debug("raw_video_frames", images)

        # Load all models in a single call so ComfyUI can manage memory holistically
        models_to_load = [da3_model]
        if salad_model is not None:
            models_to_load.append(salad_model)
        mm.load_models_gpu(models_to_load)

        _vram_debug("after_model_load", device)

        model = da3_model.model
        dtype = da3_model.model_options.get("da3_dtype", torch.float16)
        salad_nn_model = salad_model.model if salad_model is not None else None
        logger.debug(f"[DEBUG] model dtype={dtype}, salad={'loaded' if salad_nn_model else 'None'}")

        pbar = ProgressBar(num_views)

        # Preprocessing
        images_pt = images.permute(0, 3, 1, 2)  # [N, C, H, W]
        _tensor_debug("images_pt_after_permute", images_pt)

        images_pt, resize_orig_H, resize_orig_W = resize_to_patch_multiple(
            images_pt, DEFAULT_PATCH_SIZE, resize_method
        )
        model_H, model_W = images_pt.shape[2], images_pt.shape[3]
        logger.info(f"Model input size: {model_H}x{model_W}")
        _tensor_debug("images_pt_after_resize", images_pt)

        # Normalize and cast to model dtype early to halve CPU memory
        normalized_images = imagenet_normalize(images_pt).to(dtype=dtype)
        _tensor_debug("normalized_images", normalized_images)
        del images_pt

        # Add batch dim: [N, C, H, W] -> [1, N, C, H, W]
        normalized_images = normalized_images.unsqueeze(0)
        _tensor_debug("normalized_images_batched", normalized_images)
        _vram_debug("after_preprocessing", device)

        # Create streaming config
        # Enable loop closure if SALAD model is connected
        loop_enable = salad_model is not None

        config = StreamingConfig(
            chunk_size=chunk_size,
            overlap=overlap,
            align_lib=align_lib,
            align_method=align_method,
            loop_enable=loop_enable,
            save_pointcloud=save_pointcloud,
            sample_ratio=sample_ratio,
            conf_threshold_coef=conf_threshold_coef,
        )

        # Run streaming pipeline
        pipeline = StreamingPipeline(model, config, device, dtype)
        pipeline._vram_debug = _vram_debug
        pipeline._tensor_debug = _tensor_debug
        result = pipeline.run(normalized_images, pbar=pbar,
                              salad_model=salad_nn_model, video_frames=images)

        # --- Save per-frame NPZ files ---
        output_dir = Path(folder_paths.get_output_directory())
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        npz_dir = output_dir / "da3_streaming" / timestamp
        npz_dir.mkdir(parents=True, exist_ok=True)

        raw_depth = result.depth  # [N, H, W] CPU tensor
        raw_conf = result.conf    # [N, H, W] CPU tensor

        for i in range(num_views):
            frame_data = {
                "depth": raw_depth[i].numpy().astype(np.float32),
                "conf": raw_conf[i].numpy().astype(np.float32),
            }
            if result.intrinsics is not None:
                frame_data["intrinsics"] = result.intrinsics[i].astype(np.float32) if isinstance(result.intrinsics, np.ndarray) else result.intrinsics[i].cpu().numpy().astype(np.float32)
            if result.extrinsics is not None:
                frame_data["extrinsics"] = result.extrinsics[i].astype(np.float32) if isinstance(result.extrinsics, np.ndarray) else result.extrinsics[i].cpu().numpy().astype(np.float32)
            np.savez_compressed(npz_dir / f"frame_{i:06d}.npz", **frame_data)

        logger.info(f"Saved {num_views} NPZ files to {npz_dir}")

        # --- Build depth VIDEO (all on CPU to avoid OOM) ---
        depth = raw_depth.float()
        sky = result.sky.float()

        # Apply normalization for visualization
        if normalization_mode == "Standard":
            depth = cls._apply_standard_normalization(depth, invert_depth)
        elif normalization_mode == "V2-Style":
            depth = cls._apply_v2_style_normalization(depth, sky, "cpu", invert_depth)
        elif normalization_mode == "Raw":
            depth = cls._apply_raw_normalization(depth, invert_depth)

        # Grayscale -> RGB: [N, H, W] -> [N, H, W, 3]
        depth_frames = depth.unsqueeze(-1).repeat(1, 1, 1, 3).cpu().float()

        # Resize to original dimensions
        final_H = (orig_H // 2) * 2
        final_W = (orig_W // 2) * 2

        if depth_frames.shape[1] != final_H or depth_frames.shape[2] != final_W:
            depth_frames = F.interpolate(
                depth_frames.permute(0, 3, 1, 2), size=(final_H, final_W), mode="bilinear"
            ).permute(0, 2, 3, 1)

        if normalization_mode != "Raw":
            depth_frames = torch.clamp(depth_frames, 0, 1)

        # Create VIDEO object
        from comfy_api.latest._input_impl.video_types import VideoFromComponents
        from comfy_api.latest._util.video_types import VideoComponents

        depth_video = VideoFromComponents(VideoComponents(
            images=depth_frames,
            frame_rate=Fraction(fps) if not isinstance(fps, Fraction) else fps,
        ))

        pointcloud_path = result.pointcloud_path or ""

        return io.NodeOutput(depth_video, str(npz_dir), pointcloud_path)
