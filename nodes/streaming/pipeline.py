# DA3 Streaming Pipeline — Chunked depth processing with Sim(3) alignment
# Adapted from Depth-Anything-3 DA3 Streaming (Apache 2.0)
# Original: https://github.com/DepthAnything/Depth-Anything-3

import gc
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import comfy.model_management

logger = logging.getLogger("DA3Streaming")


@dataclass
class StreamingConfig:
    """Configuration for the streaming pipeline."""
    chunk_size: int = 30
    overlap: int = 8
    align_lib: str = "auto"  # auto, torch, triton, numba, numpy
    align_method: str = "sim3"  # sim3, se3, scale+se3
    scale_compute_method: str = "auto"  # auto, ransac, weighted
    loop_enable: bool = False
    loop_chunk_size: int = 20
    depth_threshold: float = 15.0
    save_pointcloud: bool = False
    sample_ratio: float = 0.015
    conf_threshold_coef: float = 0.75
    irls_delta: float = 0.1
    irls_max_iters: int = 5
    irls_tol: float = 1e-9
    ref_view_strategy: str = "saddle_balanced"

    def to_legacy_config(self):
        """Convert to the dict format expected by original alignment functions."""
        return {
            "Model": {
                "align_lib": self._resolve_align_lib(),
                "align_method": self.align_method,
                "scale_compute_method": self.scale_compute_method,
                "depth_threshold": self.depth_threshold,
                "IRLS": {
                    "delta": self.irls_delta,
                    "max_iters": self.irls_max_iters,
                    "tol": str(self.irls_tol),
                },
                "Pointcloud_Save": {
                    "sample_ratio": self.sample_ratio,
                    "conf_threshold_coef": self.conf_threshold_coef,
                },
                "loop_enable": self.loop_enable,
                "loop_chunk_size": self.loop_chunk_size,
                "ref_view_strategy": self.ref_view_strategy,
                "ref_view_strategy_loop": self.ref_view_strategy,
            },
            "Weights": {},
            "Loop": {
                "SALAD": {
                    "image_size": [336, 336],
                    "batch_size": 32,
                    "similarity_threshold": 0.85,
                    "top_k": 5,
                    "use_nms": True,
                    "nms_threshold": 25,
                },
                "SIM3_Optimizer": {
                    "lang_version": "python",
                    "max_iterations": 30,
                    "lambda_init": "1e-6",
                },
            },
        }

    def _resolve_align_lib(self):
        """Resolve 'auto' to the best available backend."""
        if self.align_lib != "auto":
            return self.align_lib

        # Try triton first (fastest GPU)
        try:
            from .loop_utils.alignment_triton import HAS_TRITON
            if HAS_TRITON:
                return "triton"
        except ImportError:
            pass

        # Then torch (GPU, always available with CUDA)
        if comfy.model_management.get_torch_device().type == "cuda":
            return "torch"

        # Then numba (JIT CPU)
        try:
            from .loop_utils.sim3utils import HAS_NUMBA
            if HAS_NUMBA:
                return "numba"
        except ImportError:
            pass

        # Fallback to numpy (pure CPU)
        return "numpy"


@dataclass
class ChunkResult:
    """Results from processing a single chunk."""
    depth: np.ndarray  # [C, H, W] float32
    conf: np.ndarray  # [C, H, W] float32
    sky: np.ndarray  # [C, H, W] float32
    extrinsics: np.ndarray  # [C, 3, 4] float32
    intrinsics: np.ndarray  # [C, 3, 3] float32
    start_idx: int
    end_idx: int


@dataclass
class StreamingResult:
    """Final results from the streaming pipeline."""
    depth: torch.Tensor  # [N, H, W]
    conf: torch.Tensor  # [N, H, W]
    sky: torch.Tensor  # [N, H, W]
    extrinsics: np.ndarray  # [N, 3, 4] or [N, 4, 4]
    intrinsics: np.ndarray  # [N, 3, 3]
    pointcloud_path: str = ""


def get_chunk_indices(num_frames, chunk_size, overlap):
    """Compute (start, end) tuples for chunking with overlap.

    Args:
        num_frames: Total number of frames
        chunk_size: Frames per chunk
        overlap: Overlap frames between chunks

    Returns:
        List of (start_idx, end_idx) tuples
    """
    if num_frames <= chunk_size:
        return [(0, num_frames)]

    step = chunk_size - overlap
    chunks = []
    for i in range((num_frames - overlap + step - 1) // step):
        start_idx = i * step
        end_idx = min(start_idx + chunk_size, num_frames)
        chunks.append((start_idx, end_idx))
    return chunks


def extract_output_field(output, field_name, default=None):
    """Extract a field from model output (handles both dict and object-style)."""
    if hasattr(output, field_name):
        val = getattr(output, field_name)
    elif isinstance(output, dict) and field_name in output:
        val = output[field_name]
    else:
        return default
    return val if torch.is_tensor(val) else default


def depth_to_point_cloud(depth, intrinsics, extrinsics, device=None):
    """Unproject depth maps to 3D point clouds in world coordinates.

    Args:
        depth: [N, H, W] numpy array or torch tensor
        intrinsics: [N, 3, 3] camera intrinsics
        extrinsics: [N, 3, 4] w2c camera poses

    Returns:
        [N, H, W, 3] point cloud in world coordinates
    """
    input_is_numpy = isinstance(depth, np.ndarray)

    if input_is_numpy:
        depth_t = torch.from_numpy(depth).float()
        intr_t = torch.from_numpy(intrinsics).float()
        extr_t = torch.from_numpy(extrinsics).float()
    else:
        depth_t = depth.float()
        intr_t = intrinsics.float()
        extr_t = extrinsics.float()

    if device is not None:
        depth_t = depth_t.to(device)
        intr_t = intr_t.to(device)
        extr_t = extr_t.to(device)

    N, H, W = depth_t.shape
    dev = depth_t.device

    u = torch.arange(W, device=dev, dtype=torch.float32).view(1, 1, W)
    v = torch.arange(H, device=dev, dtype=torch.float32).view(1, H, 1)
    u_exp = u.expand(N, H, W)
    v_exp = v.expand(N, H, W)
    ones = torch.ones((N, H, W), device=dev)
    pixel_coords = torch.stack([u_exp, v_exp, ones], dim=-1)  # [N, H, W, 3]

    K_inv = torch.inverse(intr_t)  # [N, 3, 3]
    cam_coords = torch.einsum("nij,nhwj->nhwi", K_inv, pixel_coords)
    cam_coords = cam_coords * depth_t.unsqueeze(-1)

    cam_homo = torch.cat([cam_coords, torch.ones((N, H, W, 1), device=dev)], dim=-1)

    extr_4x4 = torch.zeros(N, 4, 4, device=dev)
    extr_4x4[:, :3, :4] = extr_t
    extr_4x4[:, 3, 3] = 1.0
    c2w = torch.inverse(extr_4x4)  # [N, 4, 4]

    world_homo = torch.einsum("nij,nhwj->nhwi", c2w, cam_homo)
    points = world_homo[..., :3]

    if input_is_numpy:
        return points.cpu().numpy()
    return points


class StreamingPipeline:
    """DA3 Streaming pipeline for processing long video sequences.

    Processes video in overlapping chunks, aligns them with Sim(3) transforms,
    and optionally applies loop closure optimization.
    """

    def __init__(self, model, config: StreamingConfig, device, dtype):
        self.model = model
        self.config = config
        self.device = device
        self.dtype = dtype
        self.legacy_config = config.to_legacy_config()

    def run(self, normalized_images, pbar=None, salad_model=None, video_frames=None):
        """Run the full streaming pipeline.

        Args:
            normalized_images: [1, N, C, H, W] preprocessed tensor (ImageNet normalized)
            pbar: Optional ComfyUI ProgressBar
            salad_model: Optional loaded SALAD model dict {"model": nn.Module, "device": device}
            video_frames: Optional [N, H, W, C] raw video frames for SALAD descriptor extraction

        Returns:
            StreamingResult with aligned depth, conf, camera params
        """
        num_views = normalized_images.shape[1]
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap

        if overlap >= chunk_size:
            raise ValueError(
                f"Overlap ({overlap}) must be less than chunk size ({chunk_size})"
            )

        chunks = get_chunk_indices(num_views, chunk_size, overlap)
        logger.info(
            f"Processing {num_views} frames in {len(chunks)} chunks "
            f"(size={chunk_size}, overlap={overlap})"
        )

        if pbar:
            pbar.update_absolute(0, len(chunks) * 2 + 1)

        # Phase 1: Chunked inference
        chunk_results = []
        for i, (start, end) in enumerate(chunks):
            logger.info(f"Chunk {i+1}/{len(chunks)}: frames [{start}, {end})")
            result = self._process_chunk(normalized_images, start, end)
            chunk_results.append(result)
            comfy.model_management.soft_empty_cache()
            if pbar:
                pbar.update_absolute(i + 1)

        # Phase 2: Pairwise Sim(3) alignment
        sim3_list = []
        if len(chunk_results) > 1:
            for i in range(len(chunk_results) - 1):
                logger.info(f"Aligning chunk {i} → {i+1}")
                s, R, t = self._align_chunks(chunk_results[i], chunk_results[i + 1])
                sim3_list.append((s, R, t))
                if pbar:
                    pbar.update_absolute(len(chunks) + i + 1)

        # Phase 3: Loop closure (optional, requires SALAD model)
        if self.config.loop_enable and len(sim3_list) > 0 and salad_model is not None:
            sim3_list = self._run_loop_closure(
                sim3_list, chunk_results, chunks, salad_model, video_frames
            )

        # Phase 4: Accumulate transforms and blend
        result = self._apply_and_blend(chunk_results, chunks, sim3_list)

        if pbar:
            pbar.update_absolute(len(chunks) * 2 + 1)

        return result

    def _process_chunk(self, normalized_images, start, end):
        """Run model inference on a single chunk.

        Args:
            normalized_images: [1, N, C, H, W] full sequence
            start: Start frame index
            end: End frame index

        Returns:
            ChunkResult with outputs on CPU
        """
        from contextlib import nullcontext
        import comfy.model_management as mm

        chunk_input = normalized_images[:, start:end, ...].to(self.device)

        autocast_condition = (self.dtype != torch.float32) and not mm.is_device_mps(self.device)

        with torch.autocast(
            mm.get_autocast_device(self.device), dtype=self.dtype
        ) if autocast_condition else nullcontext():
            output = self.model(chunk_input)

        # Extract outputs and move to CPU/numpy
        depth = extract_output_field(output, 'depth')
        conf = extract_output_field(output, 'depth_conf')
        sky = extract_output_field(output, 'sky')
        extr = extract_output_field(output, 'extrinsics')
        intr = extract_output_field(output, 'intrinsics')

        if depth is None:
            raise ValueError("Model output does not contain depth tensor")

        # Squeeze batch dim: [1, C, H, W] → [C, H, W]
        depth = depth.squeeze(0).cpu().numpy()
        conf = conf.squeeze(0).cpu().numpy() if conf is not None else np.ones_like(depth)
        sky = sky.squeeze(0).cpu().numpy() if sky is not None else np.zeros_like(depth)

        # Handle extrinsics/intrinsics shapes
        if extr is not None:
            extr = extr.squeeze(0).cpu().numpy()  # [C, 3, 4] or [C, 4, 4]
            if extr.shape[-2] == 4 and extr.shape[-1] == 4:
                extr = extr[:, :3, :]  # [C, 3, 4]
        else:
            # Create identity extrinsics if not provided
            C = depth.shape[0]
            extr = np.zeros((C, 3, 4), dtype=np.float32)
            for j in range(C):
                extr[j, :3, :3] = np.eye(3)

        if intr is not None:
            intr = intr.squeeze(0).cpu().numpy()  # [C, 3, 3]
        else:
            C, H, W = depth.shape
            intr = np.zeros((C, 3, 3), dtype=np.float32)
            for j in range(C):
                intr[j] = np.array([
                    [W, 0, W / 2],
                    [0, W, H / 2],
                    [0, 0, 1]
                ], dtype=np.float32)

        # Shift confidence (match original DA3 streaming)
        conf = conf - 1.0

        del output, chunk_input
        comfy.model_management.soft_empty_cache()

        return ChunkResult(
            depth=depth,
            conf=conf,
            sky=sky,
            extrinsics=extr,
            intrinsics=intr,
            start_idx=start,
            end_idx=end,
        )

    def _align_chunks(self, prev_result, curr_result):
        """Estimate Sim(3) alignment between two consecutive chunks.

        Uses overlapping frames to unproject depth to point clouds,
        then estimates Sim(3) with robust Huber-weighted IRLS.

        Returns:
            (s, R, t) tuple: scale, rotation [3,3], translation [3,]
        """
        from .loop_utils.sim3utils import weighted_align_point_maps

        overlap = self.config.overlap

        # Extract overlap regions
        prev_depth = prev_result.depth[-overlap:]
        prev_conf = prev_result.conf[-overlap:]
        prev_extr = prev_result.extrinsics[-overlap:]
        prev_intr = prev_result.intrinsics[-overlap:]

        curr_depth = curr_result.depth[:overlap]
        curr_conf = curr_result.conf[:overlap]
        curr_extr = curr_result.extrinsics[:overlap]
        curr_intr = curr_result.intrinsics[:overlap]

        # Unproject to point clouds
        point_map1 = depth_to_point_cloud(prev_depth, prev_intr, prev_extr)
        point_map2 = depth_to_point_cloud(curr_depth, curr_intr, curr_extr)

        # Confidence threshold
        conf_threshold = min(np.median(prev_conf), np.median(curr_conf)) * 0.1

        # Precompute scale if using scale+se3
        precompute_scale = None
        if self.config.align_method == "scale+se3":
            from .loop_utils.sim3utils import precompute_scale_chunks_with_depth
            scale_factor, _, _ = precompute_scale_chunks_with_depth(
                prev_depth, prev_conf, curr_depth, curr_conf,
                method=self.config.scale_compute_method,
            )
            precompute_scale = scale_factor

        s, R, t = weighted_align_point_maps(
            point_map1, prev_conf, point_map2, curr_conf,
            conf_threshold=conf_threshold,
            config=self.legacy_config,
            precompute_scale=precompute_scale,
        )

        logger.info(f"Sim(3) alignment: scale={s:.4f}")
        return s, R, t

    def _run_loop_closure(self, sim3_list, chunk_results, chunks, salad_model, video_frames):
        """Run loop closure detection and optimization using SALAD model.

        Args:
            sim3_list: List of pairwise (s, R, t) transforms
            chunk_results: List of ChunkResult objects
            chunks: List of (start, end) chunk index tuples
            salad_model: Dict with "model" (nn.Module) and "device"
            video_frames: [N, H, W, C] raw video frames tensor

        Returns:
            Optimized sim3_list
        """
        try:
            import faiss
        except ImportError:
            logger.warning("faiss not installed, skipping loop closure. pip install faiss-cpu")
            return sim3_list

        try:
            from .loop_utils.sim3loop import Sim3LoopOptimizer
            from .loop_utils.sim3utils import (
                compute_sim3_ab, process_loop_list, weighted_align_point_maps
            )
        except Exception as e:
            logger.warning(f"Loop closure dependencies not available: {e}")
            return sim3_list

        if video_frames is None:
            logger.warning("No video frames available for SALAD descriptor extraction")
            return sim3_list

        model = salad_model["model"]
        device = salad_model["device"]

        # --- Extract SALAD descriptors from video frames ---
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize([336, 336], interpolation=T.InterpolationMode.BILINEAR),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        num_frames = video_frames.shape[0]
        batch_size = 32
        descriptors = []

        logger.info(f"Extracting SALAD descriptors for {num_frames} frames...")
        for i in range(0, num_frames, batch_size):
            batch = video_frames[i:i+batch_size]  # [B, H, W, C]
            batch = batch.permute(0, 3, 1, 2)  # [B, C, H, W]
            batch = transform(batch).to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    desc = model(batch).cpu()
            descriptors.append(desc)

        descriptors = torch.cat(descriptors)  # [N, 8448]
        logger.info(f"SALAD descriptors: {descriptors.shape}")

        # --- Find loop closures via faiss ---
        embed_size = descriptors.shape[1]
        faiss_index = faiss.IndexFlatIP(embed_size)
        desc_np = descriptors.numpy()
        faiss_index.add(desc_np)

        top_k = 5
        similarity_threshold = 0.85
        nms_threshold = 25

        similarities, indices = faiss_index.search(desc_np, top_k + 1)

        loop_closures = []
        for i in range(num_frames):
            for j in range(1, top_k + 1):
                neighbor_idx = indices[i, j]
                similarity = similarities[i, j]
                if similarity > similarity_threshold and abs(i - neighbor_idx) > 10:
                    pair = (min(i, neighbor_idx), max(i, neighbor_idx), similarity)
                    loop_closures.append(pair)

        loop_closures = list(set(loop_closures))
        loop_closures.sort(key=lambda x: x[2], reverse=True)

        # Simple NMS
        filtered = []
        suppressed = set()
        for idx1, idx2, sim in loop_closures:
            if idx1 not in suppressed and idx2 not in suppressed:
                filtered.append((idx1, idx2, sim))
                for k in range(max(0, idx1 - nms_threshold), min(idx1 + nms_threshold + 1, idx2)):
                    suppressed.add(k)
                for k in range(max(idx1 + 1, idx2 - nms_threshold), min(idx2 + nms_threshold + 1, num_frames)):
                    suppressed.add(k)

        if not filtered:
            logger.info("No loop closures detected")
            return sim3_list

        loop_pairs = [(idx1, idx2) for idx1, idx2, _ in filtered]
        logger.info(f"Found {len(loop_pairs)} loop closures")
        for idx1, idx2, sim in filtered[:5]:
            logger.info(f"  Loop: frame {idx1} <-> {idx2} (similarity={sim:.4f})")

        # --- Compute loop constraint Sim(3) transforms ---
        try:
            loop_constraints = []
            for frame_i, frame_j in loop_pairs:
                # Find which chunks these frames belong to
                chunk_i = next((c for c, (s, e) in enumerate(chunks) if s <= frame_i < e), None)
                chunk_j = next((c for c, (s, e) in enumerate(chunks) if s <= frame_j < e), None)
                if chunk_i is not None and chunk_j is not None and chunk_i != chunk_j:
                    # Compute Sim(3) between these chunks using overlap point clouds
                    s, R, t = compute_sim3_ab(
                        chunk_results[chunk_i], chunk_results[chunk_j],
                        self.legacy_config
                    )
                    loop_constraints.append((chunk_i, chunk_j, (s, R, t)))

            if loop_constraints:
                optimizer = Sim3LoopOptimizer(device=self.device)
                sim3_list = optimizer.optimize(sim3_list, loop_constraints)
                logger.info(f"Loop closure optimization applied with {len(loop_constraints)} constraints")
            else:
                logger.info("No cross-chunk loop constraints found")
        except Exception as e:
            logger.warning(f"Loop closure optimization failed: {e}")

        return sim3_list

    def _apply_and_blend(self, chunk_results, chunks, sim3_list):
        """Apply cumulative Sim(3) transforms and blend overlap regions.

        Args:
            chunk_results: List of ChunkResult objects
            chunks: List of (start, end) index tuples
            sim3_list: List of pairwise (s, R, t) transforms

        Returns:
            StreamingResult with aligned outputs
        """
        from .loop_utils.sim3utils import accumulate_sim3_transforms

        num_chunks = len(chunk_results)
        overlap = self.config.overlap

        if num_chunks == 1:
            # Single chunk — no alignment needed
            r = chunk_results[0]
            return StreamingResult(
                depth=torch.from_numpy(r.depth).float(),
                conf=torch.from_numpy(r.conf).float(),
                sky=torch.from_numpy(r.sky).float(),
                extrinsics=r.extrinsics,
                intrinsics=r.intrinsics,
            )

        # Accumulate transforms
        cumulative = accumulate_sim3_transforms(sim3_list)

        # Apply cumulative Sim(3) to chunks 1..N-1
        # Chunk 0 stays as-is (reference frame)
        for i in range(1, num_chunks):
            s_cum, R_cum, t_cum = cumulative[i - 1]

            # Scale depth
            chunk_results[i].depth = chunk_results[i].depth * s_cum

            # Transform extrinsics: apply Sim(3) to camera poses
            chunk_results[i].extrinsics = self._transform_extrinsics(
                chunk_results[i].extrinsics, s_cum, R_cum, t_cum
            )

        # Determine total number of frames (accounting for overlaps)
        total_frames = chunks[-1][1]  # end of last chunk = total frames
        H, W = chunk_results[0].depth.shape[1], chunk_results[0].depth.shape[2]

        # Allocate output arrays
        depth_out = np.zeros((total_frames, H, W), dtype=np.float32)
        conf_out = np.zeros((total_frames, H, W), dtype=np.float32)
        sky_out = np.zeros((total_frames, H, W), dtype=np.float32)
        weight_out = np.zeros((total_frames, H, W), dtype=np.float32)
        extr_out = np.zeros((total_frames, 3, 4), dtype=np.float32)
        intr_out = np.zeros((total_frames, 3, 3), dtype=np.float32)

        for i, result in enumerate(chunk_results):
            start, end = chunks[i]
            chunk_len = end - start

            for local_idx in range(chunk_len):
                global_idx = start + local_idx

                # Compute blending weight for overlap regions
                w = 1.0
                if i > 0 and local_idx < overlap:
                    # Overlap with previous chunk: ramp weight from 0 to 1
                    w = (local_idx + 1) / (overlap + 1)
                elif i < num_chunks - 1 and local_idx >= chunk_len - overlap:
                    # Overlap with next chunk: ramp weight from 1 to 0
                    pos_from_end = chunk_len - local_idx
                    w = pos_from_end / (overlap + 1)

                depth_out[global_idx] += w * result.depth[local_idx]
                conf_out[global_idx] += w * result.conf[local_idx]
                sky_out[global_idx] += w * result.sky[local_idx]
                weight_out[global_idx] += w

                # For extrinsics/intrinsics, use whichever chunk has higher weight
                # weight_out[global_idx] is (H, W); check if this is the first write
                if w >= 0.5 or np.allclose(weight_out[global_idx], w):
                    extr_out[global_idx] = result.extrinsics[local_idx]
                    intr_out[global_idx] = result.intrinsics[local_idx]

        # Normalize by total weight
        mask = weight_out > 0
        depth_out[mask] /= weight_out[mask]
        conf_out[mask] /= weight_out[mask]
        sky_out[mask] /= weight_out[mask]

        # Optional point cloud export
        pointcloud_path = ""
        if self.config.save_pointcloud:
            pointcloud_path = self._save_pointcloud(
                depth_out, conf_out, extr_out, intr_out, chunk_results
            )

        return StreamingResult(
            depth=torch.from_numpy(depth_out).float(),
            conf=torch.from_numpy(conf_out).float(),
            sky=torch.from_numpy(sky_out).float(),
            extrinsics=extr_out,
            intrinsics=intr_out,
            pointcloud_path=pointcloud_path,
        )

    def _transform_extrinsics(self, extrinsics, s, R, t):
        """Apply Sim(3) transform to camera extrinsics (w2c).

        Sim(3) aligns chunk's world frame to the reference frame.
        c2w_aligned = S @ c2w_original, then convert back to w2c.

        Args:
            extrinsics: [C, 3, 4] w2c camera poses
            s: scale factor
            R: [3, 3] rotation matrix
            t: [3,] translation vector

        Returns:
            [C, 3, 4] aligned w2c camera poses
        """
        C = extrinsics.shape[0]
        S = np.eye(4, dtype=np.float32)
        S[:3, :3] = s * R
        S[:3, 3] = t

        aligned = np.zeros_like(extrinsics)
        for i in range(C):
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :] = extrinsics[i]
            c2w = np.linalg.inv(w2c)

            c2w_aligned = S @ c2w
            # Normalize rotation (remove scale from rotation part)
            c2w_aligned[:3, :3] /= s

            w2c_aligned = np.linalg.inv(c2w_aligned)
            aligned[i] = w2c_aligned[:3, :]

        return aligned

    def _save_pointcloud(self, depth, conf, extrinsics, intrinsics, chunk_results):
        """Save merged point cloud as PLY file.

        Returns:
            Path to the saved PLY file
        """
        try:
            from .loop_utils.sim3utils import save_confident_pointcloud_batch
        except ImportError:
            logger.warning("Cannot save point cloud: missing dependencies")
            return ""

        output_dir = tempfile.mkdtemp(prefix="da3_streaming_pcd_")
        ply_path = os.path.join(output_dir, "merged_pointcloud.ply")

        points = depth_to_point_cloud(depth, intrinsics, extrinsics)

        # Use first chunk's processed images for colors (or create dummy)
        N, H, W = depth.shape
        colors = np.full((N, H, W, 3), 128, dtype=np.uint8)

        conf_threshold = np.mean(conf) * self.config.conf_threshold_coef

        save_confident_pointcloud_batch(
            points=points,
            colors=colors,
            confs=conf,
            output_path=ply_path,
            conf_threshold=conf_threshold,
            sample_ratio=self.config.sample_ratio,
        )

        logger.info(f"Point cloud saved to {ply_path}")
        return ply_path
