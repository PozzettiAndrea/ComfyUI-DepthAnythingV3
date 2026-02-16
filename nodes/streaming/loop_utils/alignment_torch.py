# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long)
# Ported from Depth-Anything-3 DA3 Streaming (Apache 2.0)
# Original: https://github.com/DepthAnything/Depth-Anything-3

import logging

import numpy as np
import torch

logger = logging.getLogger("DA3Streaming")


def weighted_estimate_se3_torch(source_points, target_points, weights):
    source_points = torch.from_numpy(source_points).cuda().float()
    target_points = torch.from_numpy(target_points).cuda().float()
    weights = torch.from_numpy(weights).cuda().float()

    total_weight = torch.sum(weights)
    if total_weight < 1e-6:
        return (
            1.0,
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32),
        )

    normalized_weights = weights / total_weight

    mu_src = torch.sum(normalized_weights[:, None] * source_points, dim=0)
    mu_tgt = torch.sum(normalized_weights[:, None] * target_points, dim=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    weighted_src = src_centered * torch.sqrt(normalized_weights)[:, None]
    weighted_tgt = tgt_centered * torch.sqrt(normalized_weights)[:, None]

    H = weighted_src.T @ weighted_tgt

    return 1.0, mu_src.cpu().numpy(), mu_tgt.cpu().numpy(), H.cpu().numpy()


def weighted_estimate_sim3_torch(source_points, target_points, weights):

    source_points = torch.from_numpy(source_points).cuda().float()
    target_points = torch.from_numpy(target_points).cuda().float()
    weights = torch.from_numpy(weights).cuda().float()

    total_weight = torch.sum(weights)
    if total_weight < 1e-6:
        return (
            -1.0,
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32),
        )

    normalized_weights = weights / total_weight

    mu_src = torch.sum(normalized_weights[:, None] * source_points, dim=0)
    mu_tgt = torch.sum(normalized_weights[:, None] * target_points, dim=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    scale_src = torch.sqrt(torch.sum(normalized_weights * torch.sum(src_centered**2, dim=1)))
    scale_tgt = torch.sqrt(torch.sum(normalized_weights * torch.sum(tgt_centered**2, dim=1)))
    s = scale_tgt / scale_src

    weighted_src = (s * src_centered) * torch.sqrt(normalized_weights)[:, None]
    weighted_tgt = tgt_centered * torch.sqrt(normalized_weights)[:, None]

    H = weighted_src.T @ weighted_tgt

    return s.cpu().numpy(), mu_src.cpu().numpy(), mu_tgt.cpu().numpy(), H.cpu().numpy()


def weighted_estimate_sim3_numba_torch(source_points, target_points, weights, align_method="sim3"):

    if align_method == "sim3":
        s, mu_src, mu_tgt, H = weighted_estimate_sim3_torch(source_points, target_points, weights)
    elif align_method == "se3" or align_method == "scale+se3":
        s, mu_src, mu_tgt, H = weighted_estimate_se3_torch(source_points, target_points, weights)

    if s < 0:
        raise ValueError("Total weight too small for meaningful estimation")

    H_torch = torch.from_numpy(H).cuda().float()
    U, _, Vt = torch.linalg.svd(H_torch)

    U = U.cpu().numpy()
    Vt = Vt.cpu().numpy()

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    mu_src = mu_src.astype(np.float32)
    mu_tgt = mu_tgt.astype(np.float32)
    R = R.astype(np.float32)

    if align_method == "se3" or align_method == "scale+se3":
        t = mu_tgt - R @ mu_src
    else:
        t = mu_tgt - s * R @ mu_src

    return s, R, t.astype(np.float32)


def huber_loss_torch(r, delta):

    r_torch = torch.from_numpy(r).cuda().float()
    delta_torch = torch.tensor(delta, device="cuda", dtype=torch.float32)

    abs_r = torch.abs(r_torch)
    result = torch.where(
        abs_r <= delta_torch, 0.5 * r_torch**2, delta_torch * (abs_r - 0.5 * delta_torch)
    )

    return result.cpu().numpy()


def compute_residuals_torch(tgt, transformed):

    tgt_torch = torch.from_numpy(tgt).cuda().float()
    transformed_torch = torch.from_numpy(transformed).cuda().float()

    residuals = torch.sqrt(torch.sum((tgt_torch - transformed_torch) ** 2, dim=1))
    return residuals.cpu().numpy()


def compute_huber_weights_torch(residuals, delta):

    residuals_torch = torch.from_numpy(residuals).cuda().float()
    delta_torch = torch.tensor(delta, device="cuda", dtype=torch.float32)

    weights = torch.ones_like(residuals_torch)
    mask = residuals_torch > delta_torch
    weights[mask] = delta_torch / residuals_torch[mask]

    return weights.cpu().numpy()


def apply_transformation_torch(src, s, R, t):

    src_torch = torch.from_numpy(src).cuda().float()
    R_torch = torch.from_numpy(R).cuda().float()
    t_torch = torch.from_numpy(t).cuda().float()
    s_torch = torch.tensor(s, device="cuda", dtype=torch.float32)

    transformed = s_torch * (src_torch @ R_torch.T) + t_torch
    return transformed.cpu().numpy()


def robust_weighted_estimate_sim3_torch(
    src, tgt, init_weights, delta=0.1, max_iters=20, tol=1e-9, align_method="sim3"
):

    src = src.astype(np.float32)
    tgt = tgt.astype(np.float32)
    init_weights = init_weights.astype(np.float32)

    s, R, t = weighted_estimate_sim3_numba_torch(src, tgt, init_weights, align_method=align_method)

    prev_error = float("inf")

    for iter in range(max_iters):
        transformed = apply_transformation_torch(src, s, R, t)
        residuals = compute_residuals_torch(tgt, transformed)

        logger.debug(f"Iter {iter}: Mean residual = {np.mean(residuals):.6f}")

        huber_weights = compute_huber_weights_torch(residuals, delta)
        combined_weights = init_weights * huber_weights
        combined_weights /= np.sum(combined_weights) + 1e-12

        s_new, R_new, t_new = weighted_estimate_sim3_numba_torch(
            src, tgt, combined_weights, align_method=align_method
        )

        param_change = np.abs(s_new - s) + np.linalg.norm(t_new - t)
        rot_angle = np.arccos(min(1.0, max(-1.0, (np.trace(R_new @ R.T) - 1) / 2)))

        current_error = np.sum(huber_loss_torch(residuals, delta) * init_weights)

        if (param_change < tol and rot_angle < np.radians(0.1)) or (
            abs(prev_error - current_error) < tol * prev_error
        ):
            logger.debug(f"Converged at iteration {iter}")
            break

        s, R, t = s_new, R_new, t_new
        prev_error = current_error

    return s, R, t


def apply_sim3_direct_torch(point_maps, s, R, t, device=None):
    """
    PyTorch SIM3
    point_maps: (b, h, w, 3) numpy array
    s: scalar or (b,) array
    R: (3, 3) or (b, 3, 3) numpy array
    t: (3,) or (b, 3) numpy array
    """
    if isinstance(point_maps, np.ndarray):
        point_maps_torch = torch.from_numpy(point_maps).float()
        R_torch = torch.from_numpy(R).float()
        t_torch = torch.from_numpy(t).float()
        s_torch = torch.tensor(s).float() if np.isscalar(s) else torch.from_numpy(s).float()
    else:
        point_maps_torch = point_maps
        R_torch = R
        t_torch = t
        s_torch = s

    if device is not None:
        point_maps_torch = point_maps_torch.to(device)
        R_torch = R_torch.to(device)
        t_torch = t_torch.to(device)
        s_torch = s_torch.to(device)

    b, h, w, c = point_maps_torch.shape

    points_flat = point_maps_torch.reshape(b, -1, 3)  # (b, h*w, 3)

    if R_torch.dim() == 2:
        R_torch = R_torch.unsqueeze(0).expand(b, 3, 3)  # (b, 3, 3)

    if t_torch.dim() == 1:
        t_torch = t_torch.unsqueeze(0).expand(b, 3)  # (b, 3)

    if s_torch.dim() == 0:
        s_torch = s_torch.unsqueeze(0).expand(b)  # (b,)

    rotated_flat = torch.bmm(points_flat, R_torch.transpose(1, 2))  # (b, h*w, 3)

    transformed_flat = s_torch[:, None, None] * rotated_flat + t_torch[:, None, :]

    transformed = transformed_flat.reshape(b, h, w, 3)

    if isinstance(point_maps, np.ndarray):
        return transformed.cpu().numpy()
    return transformed


def depth_to_point_cloud_optimized_torch(depth, intrinsics, extrinsics, device=None):

    input_is_numpy = isinstance(depth, np.ndarray)

    if input_is_numpy:
        depth_tensor = torch.from_numpy(depth).float()
        intrinsics_tensor = torch.from_numpy(intrinsics).float()
        extrinsics_tensor = torch.from_numpy(extrinsics).float()
    else:
        depth_tensor = depth
        intrinsics_tensor = intrinsics
        extrinsics_tensor = extrinsics

    if device is not None:
        depth_tensor = depth_tensor.to(device)
        intrinsics_tensor = intrinsics_tensor.to(device)
        extrinsics_tensor = extrinsics_tensor.to(device)

    N, H, W = depth_tensor.shape
    device = depth_tensor.device

    u = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
    v = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)

    u_expanded = u.expand(N, H, W)
    v_expanded = v.expand(N, H, W)

    ones = torch.ones((N, H, W), device=device)
    pixel_coords = torch.stack([u_expanded, v_expanded, ones], dim=-1)  # [N, H, W, 3]

    intrinsics_inv = torch.inverse(intrinsics_tensor)  # [N, 3, 3]

    camera_coords = torch.einsum("nij,nhwj->nhwi", intrinsics_inv, pixel_coords)

    camera_coords = camera_coords * depth_tensor.unsqueeze(-1)  # [N, H, W, 3]

    camera_coords_homo = torch.cat(
        [camera_coords, torch.ones((N, H, W, 1), device=device)], dim=-1
    )

    extrinsics_4x4 = torch.zeros(N, 4, 4, device=device)
    extrinsics_4x4[:, :3, :4] = extrinsics_tensor
    extrinsics_4x4[:, 3, 3] = 1.0

    c2w = torch.inverse(extrinsics_4x4)  # [N, 4, 4]

    world_coords_homo = torch.einsum("nij,nhwj->nhwi", c2w, camera_coords_homo)
    point_cloud_world = world_coords_homo[..., :3]  # [N, H, W, 3]

    if input_is_numpy:
        return point_cloud_world.cpu().numpy()
    return point_cloud_world
