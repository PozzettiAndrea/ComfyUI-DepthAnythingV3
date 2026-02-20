# Depth Anything V3 — ComfyUI-native camera modules
#
# Consolidates: CameraEnc, CameraDec, camera-specific attention/block/mlp,
# and transform utilities (quaternion conversions, pose encoding).
#
# Original sources:
#   model/cam_enc.py, model/cam_dec.py
#   model/utils/attention.py, model/utils/block.py
#   model/utils/transform.py

from __future__ import annotations

from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import comfy.ops
from comfy.ldm.modules.attention import optimized_attention

from .geometry import affine_inverse


# =============================================================================
# Transform utilities (from model/utils/transform.py)
# =============================================================================


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Returns torch.sqrt(torch.max(0, x)) with zero subgradient where x is 0."""
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert a unit quaternion to standard form (real part non-negative). Order: XYZW."""
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert XYZW quaternions to 3x3 rotation matrices."""
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrices to XYZW quaternions."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = _sqrt_positive_part(
        torch.stack([
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ], dim=-1)
    )
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    out = out[..., [1, 2, 3, 0]]
    out = standardize_quaternion(out)
    return out


def extri_intri_to_pose_encoding(extrinsics, intrinsics, image_size_hw=None):
    """Convert camera extrinsics (BxSx3x4) and intrinsics (BxSx3x3) to 9D pose encoding."""
    R = extrinsics[:, :, :3, :3]
    T = extrinsics[:, :, :3, 3]
    quat = mat_to_quat(R)
    H, W = image_size_hw
    fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
    fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
    pose_encoding = torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()
    return pose_encoding


def pose_encoding_to_extri_intri(pose_encoding, image_size_hw=None):
    """Convert 9D pose encoding back to camera extrinsics and intrinsics."""
    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    H, W = image_size_hw
    # Swap FOV indices for portrait images to compensate ViT spatial bug
    if H > W:
        fov_w = pose_encoding[..., 7]
        fov_h = pose_encoding[..., 8]
    else:
        fov_h = pose_encoding[..., 7]
        fov_w = pose_encoding[..., 8]
    R = quat_to_mat(quat)
    extrinsics = torch.cat([R, T[..., None]], dim=-1)
    fy_raw = (H / 2.0) / torch.clamp(torch.tan(fov_h / 2.0), 1e-6)
    fx_raw = (W / 2.0) / torch.clamp(torch.tan(fov_w / 2.0), 1e-6)
    f = torch.sqrt(fx_raw * fy_raw)
    fx = f
    fy = f
    intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = W / 2
    intrinsics[..., 1, 2] = H / 2
    intrinsics[..., 2, 2] = 1.0
    return extrinsics, intrinsics


def cam_quat_xyzw_to_world_quat_wxyz(cam_quat_xyzw, c2w):
    """Transform camera-space XYZW quaternions to world-space WXYZ quaternions."""
    b, n = cam_quat_xyzw.shape[:2]
    cam_quat_wxyz = torch.cat([
        cam_quat_xyzw[..., 3:4],
        cam_quat_xyzw[..., 0:1],
        cam_quat_xyzw[..., 1:2],
        cam_quat_xyzw[..., 2:3],
    ], dim=-1)
    cam_quat_wxyz_flat = cam_quat_wxyz.reshape(-1, 4)
    rotmat_cam = quat_to_mat(cam_quat_wxyz_flat).reshape(b, n, 3, 3)
    rotmat_c2w = c2w[..., :3, :3]
    rotmat_world = torch.matmul(rotmat_c2w, rotmat_cam)
    rotmat_world_flat = rotmat_world.reshape(-1, 3, 3)
    world_quat_wxyz_flat = mat_to_quat(rotmat_world_flat)
    world_quat_wxyz = world_quat_wxyz_flat.reshape(b, n, 4)
    return world_quat_wxyz


# =============================================================================
# Camera-specific Attention / Mlp / Block (from model/utils/attention.py, block.py)
# =============================================================================


class CameraLayerScale(nn.Module):
    def __init__(self, dim: int, init_values: Union[float, Tensor] = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class CameraMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        dtype=None, device=None, operations=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = operations.Linear(in_features, hidden_features, bias=bias, dtype=dtype, device=device)
        self.act = act_layer()
        self.fc2 = operations.Linear(hidden_features, out_features, bias=bias, dtype=dtype, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CameraAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        rope=None,
        dtype=None, device=None, operations=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.q_norm = operations.LayerNorm(self.head_dim, dtype=dtype, device=device) if qk_norm else nn.Identity()
        self.k_norm = operations.LayerNorm(self.head_dim, dtype=dtype, device=device) if qk_norm else nn.Identity()
        self.proj = operations.Linear(dim, dim, bias=proj_bias, dtype=dtype, device=device)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if attn_mask is not None:
            attn_mask_expanded = attn_mask if attn_mask.dim() == 4 else attn_mask[:, None].expand(-1, self.num_heads, -1, -1)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_expanded)
            x = x.transpose(1, 2).reshape(B, N, C)
        else:
            x = optimized_attention(q, k, v, heads=self.num_heads, skip_reshape=True)

        x = self.proj(x)
        return x


class CameraBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        attn_class: Callable[..., nn.Module] = CameraAttention,
        ffn_layer: Callable[..., nn.Module] = CameraMlp,
        qk_norm: bool = False,
        rope=None,
        dtype=None, device=None, operations=None,
    ) -> None:
        super().__init__()
        self.norm1 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            rope=rope,
            dtype=dtype, device=device, operations=operations,
        )
        self.ls1 = CameraLayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = operations.LayerNorm(dim, dtype=dtype, device=device)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            dtype=dtype, device=device, operations=operations,
        )
        self.ls2 = CameraLayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


# =============================================================================
# CameraEnc and CameraDec
# =============================================================================


class CameraEnc(nn.Module):
    """Encodes camera parameters (extrinsics/intrinsics) into token representations."""

    def __init__(
        self,
        dim_out: int = 1024,
        dim_in: int = 9,
        trunk_depth: int = 4,
        target_dim: int = 9,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        dtype=None, device=None, operations=None,
        **kwargs,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.trunk_depth = trunk_depth
        self.trunk = nn.Sequential(
            *[
                CameraBlock(
                    dim=dim_out,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    dtype=dtype, device=device, operations=operations,
                )
                for _ in range(trunk_depth)
            ]
        )
        self.token_norm = operations.LayerNorm(dim_out, dtype=dtype, device=device)
        self.trunk_norm = operations.LayerNorm(dim_out, dtype=dtype, device=device)
        self.pose_branch = CameraMlp(
            in_features=dim_in,
            hidden_features=dim_out // 2,
            out_features=dim_out,
            drop=0,
            dtype=dtype, device=device, operations=operations,
        )

    def forward(self, ext, ixt, image_size):
        c2ws = affine_inverse(ext)
        pose_encoding = extri_intri_to_pose_encoding(c2ws, ixt, image_size)
        pose_tokens = self.pose_branch(pose_encoding)
        pose_tokens = self.token_norm(pose_tokens)
        pose_tokens = self.trunk(pose_tokens)
        pose_tokens = self.trunk_norm(pose_tokens)
        return pose_tokens


class CameraDec(nn.Module):
    """Decodes visual features from backbone into camera pose parameters (9D)."""

    def __init__(self, dim_in=1536, dtype=None, device=None, operations=None):
        super().__init__()
        output_dim = dim_in
        self.backbone = nn.Sequential(
            operations.Linear(output_dim, output_dim, dtype=dtype, device=device),
            nn.ReLU(),
            operations.Linear(output_dim, output_dim, dtype=dtype, device=device),
            nn.ReLU(),
        )
        self.fc_t = operations.Linear(output_dim, 3, dtype=dtype, device=device)
        self.fc_qvec = operations.Linear(output_dim, 4, dtype=dtype, device=device)
        self.fc_fov = nn.Sequential(
            operations.Linear(output_dim, 2, dtype=dtype, device=device),
            nn.ReLU(),
        )

    def forward(self, feat, camera_encoding=None, *args, **kwargs):
        B, N = feat.shape[:2]
        feat = feat.reshape(B * N, -1)
        feat = self.backbone(feat)
        out_t = self.fc_t(feat.float()).reshape(B, N, 3)
        if camera_encoding is None:
            out_qvec = self.fc_qvec(feat.float()).reshape(B, N, 4)
            out_fov = self.fc_fov(feat.float()).reshape(B, N, 2)
        else:
            out_qvec = camera_encoding[..., 3:7]
            out_fov = camera_encoding[..., -2:]
        pose_enc = torch.cat([out_t, out_qvec, out_fov], dim=-1)
        return pose_enc
