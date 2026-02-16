"""DA3 Streaming node — Chunked depth processing with Sim(3) alignment for long videos."""
import json
import logging
import torch
import torch.nn.functional as F
from torchvision import transforms
from contextlib import nullcontext

import comfy.model_management as mm
from comfy.utils import ProgressBar

from ..utils import (
    IMAGENET_MEAN, IMAGENET_STD, DEFAULT_PATCH_SIZE,
    format_camera_params, process_tensor_to_image, process_tensor_to_mask,
    resize_to_patch_multiple, logger as utils_logger, check_model_capabilities,
    get_or_create_da3_patcher,
)
from .pipeline import StreamingConfig, StreamingPipeline

logger = logging.getLogger("DA3Streaming")


class DepthAnythingV3_Streaming:
    """Process long video sequences with chunked DA3 inference and Sim(3) alignment.

    Splits video into overlapping chunks, runs DA3 multi-view inference per chunk,
    then aligns chunks using Sim(3) estimation from overlapping depth-derived point clouds.
    Optional SALAD-based loop closure detection for globally consistent results.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "da3_model": ("DA3MODEL", ),
                "normalization_mode": ([
                    "Standard",
                    "V2-Style",
                    "Raw"
                ], {"default": "V2-Style"}),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Batch of frames [N, H, W, C] from LoadImage, VHS_LoadVideo, etc."
                }),
                "video": ("VIDEO", {
                    "tooltip": "Video from ComfyUI LoadVideo node"
                }),
                "chunk_size": ("INT", {
                    "default": 30,
                    "min": 4,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Frames per chunk. Lower = less VRAM. 30 for 24GB, 15 for 12GB VRAM."
                }),
                "overlap": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Overlap frames between chunks for Sim(3) alignment. 4-12 typical."
                }),
                "align_lib": (["auto", "torch", "triton", "numba", "numpy"], {
                    "default": "auto",
                    "tooltip": "Alignment backend. auto selects fastest available (triton > torch > numba > numpy)."
                }),
                "align_method": (["sim3", "se3", "scale+se3"], {
                    "default": "sim3",
                    "tooltip": "Alignment method. sim3: full 7-DOF. se3: 6-DOF (no scale). scale+se3: precompute scale then SE(3)."
                }),
                "resize_method": (["resize", "crop", "pad"], {
                    "default": "resize",
                    "tooltip": "How to handle non-patch-aligned dimensions."
                }),
                "invert_depth": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert depth output (far=bright)."
                }),
                "loop_enable": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable SALAD loop closure (requires SALAD weights + faiss + pypose)."
                }),
                "salad_model_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to dino_salad.ckpt for loop closure detection."
                }),
                "save_pointcloud": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Export aligned point cloud as PLY file."
                }),
                "sample_ratio": ("FLOAT", {
                    "default": 0.015,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.005,
                    "tooltip": "Point cloud downsampling ratio (lower = fewer points)."
                }),
                "conf_threshold_coef": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Confidence threshold coefficient for point cloud filtering."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("depth", "confidence", "extrinsics", "intrinsics", "sky_mask", "resized_rgb", "pointcloud_path")
    FUNCTION = "process"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
DA3 Streaming — Process long videos with chunked inference and Sim(3) alignment.

**How it works:**
1. Splits video into overlapping chunks (e.g., 30 frames, 8 overlap)
2. Runs DA3 multi-view inference on each chunk
3. Estimates Sim(3) alignment between chunks using overlap-region point clouds
4. Blends overlap regions with linear interpolation
5. Optionally runs SALAD loop closure for drift correction

**Memory:** VRAM bounded to ~1 chunk at a time. Each chunk's output is offloaded to CPU.

**Inputs:** Connect either images (IMAGE batch) or video (VIDEO from LoadVideo).

**Alignment backends:** auto tries triton → torch → numba → numpy (fastest available).

Ported from the official DA3 Streaming pipeline (VGGT-Long).
"""

    def _apply_standard_normalization(self, depth, invert_depth):
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

    def _apply_v2_style_normalization(self, depth, sky, device, invert_depth):
        d_min = depth.min()
        d_max = depth.max()
        d_range = d_max - d_min
        if d_range > 1e-8:
            depth = (depth - d_min) / d_range
        else:
            depth = torch.zeros_like(depth)

        disparity = 1.0 / (depth + 1e-6)

        sky_mask = sky > 0.5 if sky is not None else torch.zeros_like(depth, dtype=torch.bool)
        non_sky = disparity[~sky_mask] if sky_mask.any() else disparity

        if non_sky.numel() > 0:
            p_low = torch.quantile(non_sky, 0.02)
            p_high = torch.quantile(non_sky, 0.98)
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

    def _apply_raw_normalization(self, depth, invert_depth):
        if invert_depth:
            d_min = depth.min()
            d_max = depth.max()
            d_range = d_max - d_min
            if d_range > 1e-8:
                depth = d_max - depth + d_min
        return depth

    def process(self, da3_model, normalization_mode="V2-Style",
                images=None, video=None,
                chunk_size=30, overlap=8,
                align_lib="auto", align_method="sim3",
                resize_method="resize", invert_depth=False,
                loop_enable=False, salad_model_path="",
                save_pointcloud=False, sample_ratio=0.015,
                conf_threshold_coef=0.75):

        # Resolve images from VIDEO or IMAGE input
        if video is not None:
            components = video.get_components()
            video_images = components.images  # [B, H, W, C] float32 0-1
            if images is not None:
                images = torch.cat([images, video_images], dim=0)
            else:
                images = video_images

        if images is None or images.shape[0] == 0:
            raise ValueError("No input provided. Connect either 'images' or 'video' input.")

        num_views = images.shape[0]
        orig_H, orig_W = images.shape[1], images.shape[2]

        logger.info(f"Streaming: {num_views} frames, {orig_H}x{orig_W}")

        # Get model and device
        device = mm.get_torch_device()
        patcher = get_or_create_da3_patcher(self, da3_model)
        mm.load_models_gpu([patcher])
        model = patcher.model
        dtype = da3_model["dtype"]

        pbar = ProgressBar(num_views)

        # Preprocessing (same as MultiView)
        images_pt = images.permute(0, 3, 1, 2)  # [N, C, H, W]
        images_pt, resize_orig_H, resize_orig_W = resize_to_patch_multiple(
            images_pt, DEFAULT_PATCH_SIZE, resize_method
        )
        model_H, model_W = images_pt.shape[2], images_pt.shape[3]
        logger.info(f"Model input size: {model_H}x{model_W}")

        # Store resized RGB
        resized_rgb = images_pt.clone()

        # Normalize
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        normalized_images = normalize(images_pt)

        # Add batch dim: [N, C, H, W] → [1, N, C, H, W]
        normalized_images = normalized_images.unsqueeze(0)

        # Create streaming config
        config = StreamingConfig(
            chunk_size=chunk_size,
            overlap=overlap,
            align_lib=align_lib,
            align_method=align_method,
            loop_enable=loop_enable,
            salad_model_path=salad_model_path,
            save_pointcloud=save_pointcloud,
            sample_ratio=sample_ratio,
            conf_threshold_coef=conf_threshold_coef,
        )

        # Run streaming pipeline
        pipeline = StreamingPipeline(model, config, device, dtype)
        result = pipeline.run(normalized_images, pbar=pbar)

        # Post-process on device
        depth = result.depth.to(device)
        conf = result.conf.to(device)
        sky = result.sky.to(device)

        # Apply normalization
        if normalization_mode == "Standard":
            depth = self._apply_standard_normalization(depth, invert_depth)
        elif normalization_mode == "V2-Style":
            depth = self._apply_v2_style_normalization(depth, sky, device, invert_depth)
        elif normalization_mode == "Raw":
            depth = self._apply_raw_normalization(depth, invert_depth)

        # Normalize confidence to 0-1
        conf_range = conf.max() - conf.min()
        if conf_range > 1e-8:
            conf = (conf - conf.min()) / conf_range
        else:
            conf = torch.ones_like(conf)

        # Normalize sky to 0-1
        sky_min, sky_max = sky.min(), sky.max()
        if sky_max > sky_min:
            sky = (sky - sky_min) / (sky_max - sky_min)

        # Convert to ComfyUI output format
        depth_out = depth.unsqueeze(-1).repeat(1, 1, 1, 3).cpu().float()  # [N, H, W, 3]
        conf_out = conf.unsqueeze(-1).repeat(1, 1, 1, 3).cpu().float()
        sky_out = sky.cpu().float()  # [N, H, W] MASK type

        rgb_out = resized_rgb.permute(0, 2, 3, 1).cpu().float()  # [N, H, W, 3]

        # Resize to original dimensions
        final_H = (orig_H // 2) * 2
        final_W = (orig_W // 2) * 2

        if depth_out.shape[1] != final_H or depth_out.shape[2] != final_W:
            depth_out = F.interpolate(
                depth_out.permute(0, 3, 1, 2), size=(final_H, final_W), mode="bilinear"
            ).permute(0, 2, 3, 1)
            conf_out = F.interpolate(
                conf_out.permute(0, 3, 1, 2), size=(final_H, final_W), mode="bilinear"
            ).permute(0, 2, 3, 1)
            sky_out = F.interpolate(
                sky_out.unsqueeze(1), size=(final_H, final_W), mode="bilinear"
            ).squeeze(1)
            rgb_out = F.interpolate(
                rgb_out.permute(0, 3, 1, 2), size=(final_H, final_W), mode="bilinear"
            ).permute(0, 2, 3, 1)

        # Clamp
        if normalization_mode != "Raw":
            depth_out = torch.clamp(depth_out, 0, 1)
        conf_out = torch.clamp(conf_out, 0, 1)
        sky_out = torch.clamp(sky_out, 0, 1)
        rgb_out = torch.clamp(rgb_out, 0, 1)

        # Format camera params as JSON
        extrinsics_json = format_camera_params(result.extrinsics, "extrinsics")
        intrinsics_json = format_camera_params(result.intrinsics, "intrinsics")

        pointcloud_path = result.pointcloud_path or ""

        return (
            depth_out,
            conf_out,
            extrinsics_json,
            intrinsics_json,
            sky_out,
            rgb_out,
            pointcloud_path,
        )
