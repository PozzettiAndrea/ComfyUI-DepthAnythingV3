# Depth Anything V3 — ComfyUI-native model definitions
#
# Consolidates: DINOv2 backbone, DPT/DualDPT decoders, DA3 top-level models.
# All nn.Linear/Conv/Norm layers use operations= for ComfyUI weight management.
#
# Original sources:
#   model/dinov2/layers/  (attention, block, mlp, swiglu_ffn, patch_embed, rope, drop_path, layer_scale)
#   model/dinov2/vision_transformer.py
#   model/dpt.py, model/dualdpt.py
#   model/da3.py
#   model/utils/head_utils.py

from __future__ import annotations

import math
from typing import Callable, Dict as TyDict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import comfy.ops
from comfy.ldm.modules.attention import optimized_attention_for_device

from .alignment import (
    apply_metric_scaling,
    compute_alignment_mask,
    compute_sky_mask,
    least_squares_scale_scalar,
    sample_tensor_for_quantile,
    set_sky_regions_to_max_depth,
)
from .geometry import affine_inverse, as_homogeneous


class ModelOutput(dict):
    """Dict subclass with attribute access, replacing addict.Dict dependency."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


# =============================================================================
# Utility helpers (from model/utils/head_utils.py)
# =============================================================================


class Permute(nn.Module):
    """nn.Module wrapper around Tensor.permute for cleaner nn.Sequential usage."""
    def __init__(self, dims: Tuple[int, ...]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


def activate_head_gs(out, activation="norm_exp", conf_activation="expp1", conf_dim=None):
    """Process network output to extract GS params and density values."""
    fmap = out.permute(0, 2, 3, 1)
    conf_dim = 1 if conf_dim is None else conf_dim
    xyz = fmap[:, :, :, :-conf_dim]
    conf = fmap[:, :, :, -1] if conf_dim == 1 else fmap[:, :, :, -conf_dim:]

    if activation == "norm_exp":
        d = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        xyz_normed = xyz / d
        pts3d = xyz_normed * torch.expm1(d)
    elif activation == "norm":
        pts3d = xyz / xyz.norm(dim=-1, keepdim=True)
    elif activation == "exp":
        pts3d = torch.exp(xyz)
    elif activation == "relu":
        pts3d = F.relu(xyz)
    elif activation == "sigmoid":
        pts3d = torch.sigmoid(xyz)
    elif activation == "linear":
        pts3d = xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")

    if conf_activation == "expp1":
        conf_out = 1 + conf.exp()
    elif conf_activation == "expp0":
        conf_out = conf.exp()
    elif conf_activation == "sigmoid":
        conf_out = torch.sigmoid(conf)
    elif conf_activation == "linear":
        conf_out = conf
    else:
        raise ValueError(f"Unknown conf_activation: {conf_activation}")

    return pts3d, conf_out


def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    """Generate 1D positional embedding from positions using sine/cosine functions."""
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb.float()


def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    """Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)."""
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)
    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)
    emb = torch.cat([emb_x, emb_y], dim=-1)
    return emb.view(H, W, embed_dim)


def create_uv_grid(
    width: int,
    height: int,
    aspect_ratio: float = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Create a normalized UV grid of shape (width, height, 2)."""
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)
    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)
    return uv_grid


def custom_interpolate(
    x: torch.Tensor,
    size: Union[Tuple[int, int], None] = None,
    scale_factor: Union[float, None] = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """Safe interpolation to avoid INT_MAX overflow in torch.nn.functional.interpolate."""
    if size is None:
        assert scale_factor is not None, "Either size or scale_factor must be provided."
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    INT_MAX = 1610612736
    total = size[0] * size[1] * x.shape[0] * x.shape[1]
    if total > INT_MAX:
        chunks = torch.chunk(x, chunks=(total // INT_MAX) + 1, dim=0)
        outs = [
            F.interpolate(c, size=size, mode=mode, align_corners=align_corners)
            for c in chunks
        ]
        return torch.cat(outs, dim=0).contiguous()
    return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)


# =============================================================================
# DINOv2 layers — DropPath, LayerScale, Mlp, SwiGLUFFN, RoPE, PatchEmbed, Attention, Block
# =============================================================================


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: Union[float, Tensor] = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.inplace = inplace
        self.init_values = init_values
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        gamma = self.gamma.to(device=x.device, dtype=x.dtype)
        return x.mul_(gamma) if self.inplace else x * gamma


class Mlp(nn.Module):
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


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network with fused gate/value projection."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
        dtype=None, device=None, operations=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        self.w12 = operations.Linear(in_features, 2 * hidden_features, bias=bias, dtype=dtype, device=device)
        self.w3 = operations.Linear(hidden_features, out_features, bias=bias, dtype=dtype, device=device)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


# Alias for backward compatibility
SwiGLUFFNFused = SwiGLUFFN


# =============================================================================
# 2D Rotary Position Embeddings (no learnable params)
# =============================================================================


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid."""
    def __init__(self):
        self.position_cache = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            positions = torch.stack([yy.flatten(), xx.flatten()], dim=1)
            self.position_cache[height, width] = positions
        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding implementation."""
    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache = {}

    def _compute_frequency_components(self, dim, seq_len, device, dtype):
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)
        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(self, tokens, positions, cos_comp, sin_comp):
        positions_int = positions.to(torch.int32)
        cos = F.embedding(positions_int, cos_comp)[:, None, :, :]
        sin = F.embedding(positions_int, sin_comp)[:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2
        feature_dim = tokens.size(-1) // 2
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, max_position, tokens.device, tokens.dtype
        )
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)
        return torch.cat((vertical_features, horizontal_features), dim=-1)


# =============================================================================
# PatchEmbed
# =============================================================================


def _make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B,C,H,W) -> (B,N,D)"""
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
        dtype=None, device=None, operations=None,
    ) -> None:
        super().__init__()
        image_HW = _make_2tuple(img_size)
        patch_HW = _make_2tuple(patch_size)
        patch_grid_size = (image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1])

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.proj = operations.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW,
                                      dtype=dtype, device=device)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size
        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"
        x = self.proj(x)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)
        return x


# =============================================================================
# DINOv2 Attention
# =============================================================================


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
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
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each [B, H, N, D]
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if attn_mask is not None:
            attn_mask = attn_mask[:, None].expand(-1, self.num_heads, -1, -1)
            fn = optimized_attention_for_device(q.device, mask=True)
            x = fn(q, k, v, heads=self.num_heads, mask=attn_mask, skip_reshape=True)
        else:
            x = optimized_attention_for_device(q.device)(q, k, v, heads=self.num_heads, skip_reshape=True)

        x = self.proj(x)
        return x


# =============================================================================
# DINOv2 Block
# =============================================================================


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        rope=None,
        ln_eps: float = 1e-6,
        dtype=None, device=None, operations=None,
    ) -> None:
        super().__init__()
        self.norm1 = operations.LayerNorm(dim, eps=ln_eps, dtype=dtype, device=device)
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
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = operations.LayerNorm(dim, eps=ln_eps, dtype=dtype, device=device)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            dtype=dtype, device=device, operations=operations,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        def attn_residual_func(x, pos=None, attn_mask=None):
            return self.ls1(self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask))

        def ffn_residual_func(x):
            return self.ls2(self.mlp(self.norm2(x)))

        x = x + self.drop_path1(attn_residual_func(x, pos=pos, attn_mask=attn_mask))
        x = x + self.drop_path2(ffn_residual_func(x))
        return x


# =============================================================================
# DinoVisionTransformer
# =============================================================================


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=1.0,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        alt_start=-1,
        qknorm_start=-1,
        rope_start=-1,
        rope_freq=100,
        plus_cam_token=False,
        cat_token=True,
        dtype=None, device=None, operations=None,
    ):
        super().__init__()
        self.patch_start_idx = 1
        self.num_features = self.embed_dim = embed_dim
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            dtype=dtype, device=device, operations=operations,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.alt_start != -1:
            self.camera_token = nn.Parameter(torch.randn(1, 2, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]

        if ffn_layer == "mlp":
            ffn_layer_cls = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            ffn_layer_cls = SwiGLUFFN
        elif ffn_layer == "identity":
            def ffn_layer_cls(*args, **kwargs):
                return nn.Identity()
        else:
            raise NotImplementedError

        if self.rope_start != -1:
            self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
            self.position_getter = PositionGetter() if self.rope is not None else None
        else:
            self.rope = None

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                act_layer=act_layer,
                ffn_layer=ffn_layer_cls,
                init_values=init_values,
                qk_norm=i >= qknorm_start if qknorm_start != -1 else False,
                rope=self.rope if i >= rope_start and rope_start != -1 else None,
                dtype=dtype, device=device, operations=operations,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = operations.LayerNorm(embed_dim, dtype=dtype, device=device)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_cls_token(self, B, S):
        cls_token = self.cls_token.expand(B, S, -1)
        cls_token = cls_token.reshape(B * S, -1, self.embed_dim)
        return cls_token

    def prepare_tokens_with_masks(self, x, masks=None, cls_token=None, **kwargs):
        B, S, nc, w, h = x.shape
        x = x.flatten(0, 1)  # [B*S, C, H, W]
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        cls_token = self.prepare_cls_token(B, S).to(device=x.device, dtype=x.dtype)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h).to(device=x.device)
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        x = x.reshape(B, S, *x.shape[1:])  # [B, S, N, C]
        return x

    def _prepare_rope(self, B, S, H, W, device):
        pos = None
        pos_nodiff = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=device
            )
            pos = pos.reshape(B, S, *pos.shape[1:])  # [B, S, N, C]
            pos_nodiff = torch.zeros_like(pos).to(pos.dtype)
            if self.patch_start_idx > 0:
                pos = pos + 1
                pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(device).to(pos.dtype)
                pos_special = pos_special.reshape(B, S, *pos_special.shape[1:])  # [B, S, N, C]
                pos = torch.cat([pos_special, pos], dim=2)
                pos_nodiff = pos_nodiff + 1
                pos_nodiff = torch.cat([pos_special, pos_nodiff], dim=2)
        return pos, pos_nodiff

    def _get_intermediate_layers_not_chunked(self, x, n=1, export_feat_layers=[], **kwargs):
        import logging as _logging
        _dbg = _logging.getLogger("DA3Streaming")

        B, S, _, H, W = x.shape
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len, aux_output = [], len(self.blocks), []
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)

        _dbg.debug(f"[DEBUG backbone] B={B} S={S} H={H} W={W} tokens_shape={list(x.shape)} "
                  f"blocks={total_block_len} alt_start={self.alt_start}")

        for i, blk in enumerate(self.blocks):
            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos
            if self.alt_start != -1 and i == self.alt_start:
                if kwargs.get("cam_token", None) is not None:
                    cam_token = kwargs.get("cam_token")
                else:
                    _cam_tok = self.camera_token.to(device=x.device, dtype=x.dtype)
                    ref_token = _cam_tok[:, :1].expand(B, -1, -1)
                    src_token = _cam_tok[:, 1:].expand(B, S - 1, -1)
                    cam_token = torch.cat([ref_token, src_token], dim=1)
                x[:, :, 0] = cam_token

            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                x = self.process_attention(
                    x, blk, "global", pos=g_pos, attn_mask=kwargs.get("attn_mask", None)
                )
                if i == self.alt_start + 1:
                    a = torch.cuda.memory_allocated(x.device) / (1024**3) if x.is_cuda else 0
                    r = torch.cuda.memory_reserved(x.device) / (1024**3) if x.is_cuda else 0
                    _dbg.debug(f"[VRAM backbone] after first GLOBAL attn (layer {i}): "
                              f"allocated={a:.2f}GB reserved={r:.2f}GB seq_len={S}*N tokens={S * x.shape[2]}")
            else:
                x = self.process_attention(x, blk, "local", pos=l_pos)
                local_x = x
                if i == 0:
                    a = torch.cuda.memory_allocated(x.device) / (1024**3) if x.is_cuda else 0
                    r = torch.cuda.memory_reserved(x.device) / (1024**3) if x.is_cuda else 0
                    _dbg.debug(f"[VRAM backbone] after first LOCAL attn (layer {i}): "
                              f"allocated={a:.2f}GB reserved={r:.2f}GB")

            if x.is_cuda and i % 4 == 3:
                a = torch.cuda.memory_allocated(x.device) / (1024**3)
                r = torch.cuda.memory_reserved(x.device) / (1024**3)
                attn_type = "global" if (self.alt_start != -1 and i >= self.alt_start and i % 2 == 1) else "local"
                _dbg.debug(f"[VRAM backbone] layer {i}/{total_block_len} ({attn_type}): "
                          f"allocated={a:.2f}GB reserved={r:.2f}GB")

            if i in blocks_to_take:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                output.append((out_x[:, :, 0], out_x))
            if i in export_feat_layers:
                aux_output.append(x)

        if x.is_cuda:
            a = torch.cuda.memory_allocated(x.device) / (1024**3)
            r = torch.cuda.memory_reserved(x.device) / (1024**3)
            _dbg.info(f"[VRAM backbone] DONE: allocated={a:.2f}GB reserved={r:.2f}GB")

        return output, aux_output

    def process_attention(self, x, block, attn_type="global", pos=None, attn_mask=None):
        b, s, n = x.shape[:3]
        if attn_type == "local":
            x = x.flatten(0, 1)  # [B*S, N, C]
            if pos is not None:
                pos = pos.flatten(0, 1)
        elif attn_type == "global":
            x = x.reshape(b, s * n, x.shape[-1])  # [B, S*N, C]
            if pos is not None:
                pos = pos.reshape(b, s * n, pos.shape[-1])
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")

        x = block(x, pos=pos, attn_mask=attn_mask)

        if attn_type == "local":
            x = x.reshape(b, s, *x.shape[1:])  # [B, S, N, C]
        elif attn_type == "global":
            x = x.reshape(b, s, -1, x.shape[-1])  # [B, S, N, C]
        return x

    def get_intermediate_layers(self, x, n=1, export_feat_layers=[], **kwargs):
        outputs, aux_outputs = self._get_intermediate_layers_not_chunked(
            x, n, export_feat_layers=export_feat_layers, **kwargs
        )
        camera_tokens = [out[0] for out in outputs]
        if outputs[0][1].shape[-1] == self.embed_dim:
            outputs = [self.norm(out[1]) for out in outputs]
        elif outputs[0][1].shape[-1] == (self.embed_dim * 2):
            outputs = [
                torch.cat(
                    [out[1][..., : self.embed_dim], self.norm(out[1][..., self.embed_dim :])],
                    dim=-1,
                )
                for out in outputs
            ]
        else:
            raise ValueError(f"Invalid output shape: {outputs[0][1].shape}")
        aux_outputs = [self.norm(out) for out in aux_outputs]
        outputs = [out[..., 1 + self.num_register_tokens :, :] for out in outputs]
        aux_outputs = [out[..., 1 + self.num_register_tokens :, :] for out in aux_outputs]
        return tuple(zip(outputs, camera_tokens)), aux_outputs


# Factory functions for standard configurations

def vit_small(patch_size=16, num_register_tokens=0, depth=12, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=depth, num_heads=6,
        mlp_ratio=4, num_register_tokens=num_register_tokens, **kwargs,
    )

def vit_base(patch_size=16, num_register_tokens=0, depth=12, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=depth, num_heads=12,
        mlp_ratio=4, num_register_tokens=num_register_tokens, **kwargs,
    )

def vit_large(patch_size=16, num_register_tokens=0, depth=24, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=depth, num_heads=16,
        mlp_ratio=4, num_register_tokens=num_register_tokens, **kwargs,
    )

def vit_giant2(patch_size=16, num_register_tokens=0, depth=40, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size, embed_dim=1536, depth=depth, num_heads=24,
        mlp_ratio=4, num_register_tokens=num_register_tokens, **kwargs,
    )


class DinoV2(nn.Module):
    """Thin wrapper that selects a ViT variant by name and exposes intermediate layers."""

    def __init__(
        self,
        name: str,
        out_layers: list,
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        cat_token: bool = True,
        dtype=None, device=None, operations=None,
        **kwargs,
    ):
        super().__init__()
        assert name in {"vits", "vitb", "vitl", "vitg"}
        self.name = name
        self.out_layers = out_layers
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        encoder_map = {
            "vits": vit_small,
            "vitb": vit_base,
            "vitl": vit_large,
            "vitg": vit_giant2,
        }
        encoder_fn = encoder_map[self.name]
        ffn_layer = "swiglufused" if self.name == "vitg" else "mlp"
        self.pretrained = encoder_fn(
            img_size=518,
            patch_size=14,
            ffn_layer=ffn_layer,
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            cat_token=cat_token,
            dtype=dtype, device=device, operations=operations,
        )

    def forward(self, x, **kwargs):
        return self.pretrained.get_intermediate_layers(
            x,
            self.out_layers,
            **kwargs,
        )


# =============================================================================
# DPT Decoder — Dense Prediction Transformer head
# =============================================================================


class ResidualConvUnit(nn.Module):
    """Lightweight residual convolution block for fusion."""
    def __init__(self, features: int, activation: nn.Module, bn: bool, groups: int = 1,
                 dtype=None, device=None, operations=None) -> None:
        super().__init__()
        self.bn = bn
        self.groups = groups
        self.conv1 = operations.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups,
                                       dtype=dtype, device=device)
        self.conv2 = operations.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups,
                                       dtype=dtype, device=device)
        self.norm1 = None
        self.norm2 = None
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Top-down fusion block: (optional) residual merge + upsampling + 1x1 contraction."""
    def __init__(
        self,
        features: int,
        activation: nn.Module,
        deconv: bool = False,
        bn: bool = False,
        expand: bool = False,
        align_corners: bool = True,
        size: Tuple[int, int] = None,
        has_residual: bool = True,
        groups: int = 1,
        dtype=None, device=None, operations=None,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.size = size
        self.has_residual = has_residual

        self.resConfUnit1 = (
            ResidualConvUnit(features, activation, bn, groups=groups,
                             dtype=dtype, device=device, operations=operations)
            if has_residual else None
        )
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=groups,
                                             dtype=dtype, device=device, operations=operations)
        out_features = (features // 2) if expand else features
        self.out_conv = operations.Conv2d(features, out_features, 1, 1, 0, bias=True, groups=groups,
                                          dtype=dtype, device=device)

    def forward(self, *xs: torch.Tensor, size: Tuple[int, int] = None) -> torch.Tensor:
        y = xs[0]
        if self.has_residual and len(xs) > 1 and self.resConfUnit1 is not None:
            y = y + self.resConfUnit1(xs[1])
        y = self.resConfUnit2(y)
        if (size is None) and (self.size is None):
            up_kwargs = {"scale_factor": 2}
        elif size is None:
            up_kwargs = {"size": self.size}
        else:
            up_kwargs = {"size": size}
        y = custom_interpolate(y, **up_kwargs, mode="bilinear", align_corners=self.align_corners)
        y = self.out_conv(y)
        return y


def _make_fusion_block(
    features: int,
    size: Tuple[int, int] = None,
    has_residual: bool = True,
    groups: int = 1,
    inplace: bool = False,
    dtype=None, device=None, operations=None,
) -> nn.Module:
    return FeatureFusionBlock(
        features=features,
        activation=nn.ReLU(inplace=inplace),
        deconv=False, bn=False, expand=False, align_corners=True,
        size=size, has_residual=has_residual, groups=groups,
        dtype=dtype, device=device, operations=operations,
    )


def _make_scratch(
    in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False,
    dtype=None, device=None, operations=None,
) -> nn.Module:
    scratch = nn.Module()
    c1 = out_shape
    c2 = out_shape * (2 if expand else 1)
    c3 = out_shape * (4 if expand else 1)
    c4 = out_shape * (8 if expand else 1)
    scratch.layer1_rn = operations.Conv2d(in_shape[0], c1, 3, 1, 1, bias=False, groups=groups,
                                          dtype=dtype, device=device)
    scratch.layer2_rn = operations.Conv2d(in_shape[1], c2, 3, 1, 1, bias=False, groups=groups,
                                          dtype=dtype, device=device)
    scratch.layer3_rn = operations.Conv2d(in_shape[2], c3, 3, 1, 1, bias=False, groups=groups,
                                          dtype=dtype, device=device)
    scratch.layer4_rn = operations.Conv2d(in_shape[3], c4, 3, 1, 1, bias=False, groups=groups,
                                          dtype=dtype, device=device)
    return scratch


class DPT(nn.Module):
    """DPT for dense prediction (main head + optional sky head)."""

    def __init__(
        self,
        dim_in: int,
        *,
        patch_size: int = 14,
        output_dim: int = 1,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = False,
        down_ratio: int = 1,
        head_name: str = "depth",
        use_sky_head: bool = True,
        sky_name: str = "sky",
        sky_activation: str = "relu",
        use_ln_for_heads: bool = False,
        norm_type: str = "idt",
        fusion_block_inplace: bool = False,
        dtype=None, device=None, operations=None,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.head_main = head_name
        self.sky_name = sky_name
        self.out_dim = output_dim
        self.has_conf = output_dim > 1
        self.use_sky_head = use_sky_head
        self.sky_activation = sky_activation
        self.intermediate_layer_idx: Tuple[int, int, int, int] = (0, 1, 2, 3)

        # Token pre-norm + per-stage projection
        if norm_type == "layer":
            self.norm = operations.LayerNorm(dim_in, dtype=dtype, device=device)
        elif norm_type == "idt":
            self.norm = nn.Identity()
        else:
            raise Exception(f"Unknown norm_type {norm_type}, should be 'layer' or 'idt'.")

        self.projects = nn.ModuleList([
            operations.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0,
                              dtype=dtype, device=device)
            for oc in out_channels
        ])

        # Spatial re-size
        self.resize_layers = nn.ModuleList([
            operations.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0,
                                       dtype=dtype, device=device),
            operations.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0,
                                       dtype=dtype, device=device),
            nn.Identity(),
            operations.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1,
                              dtype=dtype, device=device),
        ])

        # Scratch: stage adapters + main fusion chain
        self.scratch = _make_scratch(list(out_channels), features, expand=False,
                                     dtype=dtype, device=device, operations=operations)

        self.scratch.refinenet1 = _make_fusion_block(features, inplace=fusion_block_inplace,
                                                     dtype=dtype, device=device, operations=operations)
        self.scratch.refinenet2 = _make_fusion_block(features, inplace=fusion_block_inplace,
                                                     dtype=dtype, device=device, operations=operations)
        self.scratch.refinenet3 = _make_fusion_block(features, inplace=fusion_block_inplace,
                                                     dtype=dtype, device=device, operations=operations)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False, inplace=fusion_block_inplace,
                                                     dtype=dtype, device=device, operations=operations)

        # Heads
        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = operations.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1,
            dtype=dtype, device=device)

        ln_seq = (
            [Permute((0, 2, 3, 1)),
             operations.LayerNorm(head_features_2, dtype=dtype, device=device),
             Permute((0, 3, 1, 2))]
            if use_ln_for_heads else []
        )

        # Main head
        self.scratch.output_conv2 = nn.Sequential(
            operations.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1,
                              dtype=dtype, device=device),
            *ln_seq,
            nn.ReLU(inplace=True),
            operations.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0,
                              dtype=dtype, device=device),
        )

        # Sky head
        if self.use_sky_head:
            self.scratch.sky_output_conv2 = nn.Sequential(
                operations.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1,
                                  dtype=dtype, device=device),
                *ln_seq,
                nn.ReLU(inplace=True),
                operations.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0,
                                  dtype=dtype, device=device),
            )

    def forward(self, feats, H, W, patch_start_idx, chunk_size=1, **kwargs):
        B, S, N, C = feats[0][0].shape
        feats = [feat[0].reshape(B * S, N, C) for feat in feats]
        extra_kwargs = {}
        if "images" in kwargs:
            images = kwargs["images"]
            extra_kwargs.update({"images": images.flatten(0, 1)})
        if chunk_size is None or chunk_size >= S:
            out_dict = self._forward_impl(feats, H, W, patch_start_idx, **extra_kwargs)
            out_dict = {k: v.view(B, S, *v.shape[1:]) for k, v in out_dict.items()}
            return ModelOutput(out_dict)
        out_dicts = []
        for s0 in range(0, S, chunk_size):
            s1 = min(s0 + chunk_size, S)
            kw = {}
            if "images" in extra_kwargs:
                kw.update({"images": extra_kwargs["images"][s0:s1]})
            out_dicts.append(
                self._forward_impl([f[s0:s1] for f in feats], H, W, patch_start_idx, **kw)
            )
        out_dict = {k: torch.cat([od[k] for od in out_dicts], dim=0) for k in out_dicts[0].keys()}
        out_dict = {k: v.view(B, S, *v.shape[1:]) for k, v in out_dict.items()}
        return ModelOutput(out_dict)

    def _forward_impl(self, feats, H, W, patch_start_idx, **kwargs):
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(B, C, ph, pw)
            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)
            resized_feats.append(x)

        fused = self._fuse(resized_feats)
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

        fused = self.scratch.output_conv1(fused)
        fused = custom_interpolate(fused, (h_out, w_out), mode="bilinear", align_corners=True)
        if self.pos_embed:
            fused = self._add_pos_embed(fused, W, H)

        feat = fused
        main_logits = self.scratch.output_conv2(feat)
        outs = {}
        if self.has_conf:
            fmap = main_logits.permute(0, 2, 3, 1)
            pred = self._apply_activation_single(fmap[..., :-1], self.activation)
            conf = self._apply_activation_single(fmap[..., -1], self.conf_activation)
            outs[self.head_main] = pred.squeeze(1)
            outs[f"{self.head_main}_conf"] = conf.squeeze(1)
        else:
            outs[self.head_main] = self._apply_activation_single(
                main_logits, self.activation
            ).squeeze(1)

        if self.use_sky_head:
            sky_logits = self.scratch.sky_output_conv2(feat)
            outs[self.sky_name] = self._apply_sky_activation(sky_logits).squeeze(1)

        return outs

    def _fuse(self, feats):
        l1, l2, l3, l4 = feats
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)
        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        out = self.scratch.refinenet1(out, l1_rn)
        return out

    def _apply_activation_single(self, x, activation="linear"):
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp":
            return torch.exp(x)
        if act == "expp1":
            return torch.exp(x) + 1
        if act == "expm1":
            return torch.expm1(x)
        if act == "relu":
            return torch.relu(x)
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "softplus":
            return F.softplus(x)
        if act == "tanh":
            return torch.tanh(x)
        return x

    def _apply_sky_activation(self, x):
        act = self.sky_activation.lower() if isinstance(self.sky_activation, str) else self.sky_activation
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "relu":
            return torch.relu(x)
        return x

    def _add_pos_embed(self, x, W, H, ratio=0.1):
        pw, ph = x.shape[-1], x.shape[-2]
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pe


# =============================================================================
# DualDPT — Dual-head DPT
# =============================================================================


class DualDPT(nn.Module):
    """Dual-head DPT for dense prediction with an always-on auxiliary head."""

    def __init__(
        self,
        dim_in: int,
        *,
        patch_size: int = 14,
        output_dim: int = 2,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        down_ratio: int = 1,
        aux_pyramid_levels: int = 4,
        aux_out1_conv_num: int = 5,
        head_names: Tuple[str, str] = ("depth", "ray"),
        dtype=None, device=None, operations=None,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.aux_levels = aux_pyramid_levels
        self.aux_out1_conv_num = aux_out1_conv_num
        self.head_main, self.head_aux = head_names
        self.intermediate_layer_idx: Tuple[int, int, int, int] = (0, 1, 2, 3)

        self.norm = operations.LayerNorm(dim_in, dtype=dtype, device=device)
        self.projects = nn.ModuleList([
            operations.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0,
                              dtype=dtype, device=device)
            for oc in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            operations.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0,
                                       dtype=dtype, device=device),
            operations.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0,
                                       dtype=dtype, device=device),
            nn.Identity(),
            operations.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1,
                              dtype=dtype, device=device),
        ])

        self.scratch = _make_scratch(list(out_channels), features, expand=False,
                                     dtype=dtype, device=device, operations=operations)

        # Main fusion chain
        self.scratch.refinenet1 = _make_fusion_block(features, dtype=dtype, device=device, operations=operations)
        self.scratch.refinenet2 = _make_fusion_block(features, dtype=dtype, device=device, operations=operations)
        self.scratch.refinenet3 = _make_fusion_block(features, dtype=dtype, device=device, operations=operations)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False,
                                                     dtype=dtype, device=device, operations=operations)

        # Primary head
        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = operations.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1,
            dtype=dtype, device=device)
        self.scratch.output_conv2 = nn.Sequential(
            operations.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1,
                              dtype=dtype, device=device),
            nn.ReLU(inplace=True),
            operations.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0,
                              dtype=dtype, device=device),
        )

        # Auxiliary fusion chain
        self.scratch.refinenet1_aux = _make_fusion_block(features, dtype=dtype, device=device, operations=operations)
        self.scratch.refinenet2_aux = _make_fusion_block(features, dtype=dtype, device=device, operations=operations)
        self.scratch.refinenet3_aux = _make_fusion_block(features, dtype=dtype, device=device, operations=operations)
        self.scratch.refinenet4_aux = _make_fusion_block(features, has_residual=False,
                                                         dtype=dtype, device=device, operations=operations)

        # Aux pre-head per level
        self.scratch.output_conv1_aux = nn.ModuleList([
            self._make_aux_out1_block(head_features_1, dtype=dtype, device=device, operations=operations)
            for _ in range(self.aux_levels)
        ])

        # Aux final projection per level (each gets its own LayerNorm instance)
        self.scratch.output_conv2_aux = nn.ModuleList([
            nn.Sequential(
                operations.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1,
                                  dtype=dtype, device=device),
                Permute((0, 2, 3, 1)),
                operations.LayerNorm(head_features_2, dtype=dtype, device=device),
                Permute((0, 3, 1, 2)),
                nn.ReLU(inplace=True),
                operations.Conv2d(head_features_2, 7, kernel_size=1, stride=1, padding=0,
                                  dtype=dtype, device=device),
            )
            for _ in range(self.aux_levels)
        ])

    def forward(self, feats, H, W, patch_start_idx, chunk_size=1):
        import logging as _logging
        _dbg = _logging.getLogger("DA3Streaming")

        def _vram(tag):
            dev = feats[0][0].device if feats[0][0].is_cuda else None
            if dev is not None:
                a = torch.cuda.memory_allocated(dev) / (1024**3)
                r = torch.cuda.memory_reserved(dev) / (1024**3)
                _dbg.debug(f"[VRAM DPTHead {tag}] allocated={a:.2f}GB reserved={r:.2f}GB")

        B, S, N, C = feats[0][0].shape
        _dbg.debug(f"[DEBUG DPTHead] B={B} S={S} N={N} C={C} chunk_size={chunk_size} H={H} W={W}")
        feats = [feat[0].reshape(B * S, N, C) for feat in feats]

        total_mb = sum(f.nelement() * f.element_size() / (1024**2) for f in feats)
        _dbg.debug(f"[DEBUG DPTHead] {len(feats)} reshaped feats total: {total_mb:.1f}MB")
        _vram("entry")

        if chunk_size is None or chunk_size >= S:
            _dbg.debug(f"[DEBUG DPTHead] processing all {S} frames at once (no sub-chunking)")
            out_dict = self._forward_impl(feats, H, W, patch_start_idx)
            _vram("after_forward_impl_all")
            out_dict = {k: v.reshape(B, S, *v.shape[1:]) for k, v in out_dict.items()}
            return ModelOutput(out_dict)

        out_dicts = []
        num_sub_chunks = (S + chunk_size - 1) // chunk_size
        for s0 in range(0, S, chunk_size):
            s1 = min(s0 + chunk_size, S)
            sub_idx = s0 // chunk_size + 1
            _dbg.debug(f"[DEBUG DPTHead] sub-chunk {sub_idx}/{num_sub_chunks}: frames [{s0}:{s1}]")
            _vram(f"sub_chunk_{sub_idx}_before")
            out_dicts.append(
                self._forward_impl([feat[s0:s1] for feat in feats], H, W, patch_start_idx)
            )
            _vram(f"sub_chunk_{sub_idx}_after")

        out_dict = {
            k: torch.cat([od[k] for od in out_dicts], dim=0)
            for k in out_dicts[0].keys()
        }
        out_dict = {k: v.view(B, S, *v.shape[1:]) for k, v in out_dict.items()}
        _vram("exit")
        return ModelOutput(out_dict)

    def _forward_impl(self, feats, H, W, patch_start_idx):
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(B, C, ph, pw)
            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)
            resized_feats.append(x)

        fused_main, fused_aux_pyr = self._fuse(resized_feats)
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

        fused_main = custom_interpolate(fused_main, (h_out, w_out), mode="bilinear", align_corners=True)
        if self.pos_embed:
            fused_main = self._add_pos_embed(fused_main, W, H)

        main_logits = self.scratch.output_conv2(fused_main)
        fmap = main_logits.permute(0, 2, 3, 1)
        main_pred = self._apply_activation_single(fmap[..., :-1], self.activation)
        main_conf = self._apply_activation_single(fmap[..., -1], self.conf_activation)

        # Auxiliary head
        last_aux = fused_aux_pyr[-1]
        if self.pos_embed:
            last_aux = self._add_pos_embed(last_aux, W, H)
        last_aux_logits = self.scratch.output_conv2_aux[-1](last_aux)
        fmap_last = last_aux_logits.permute(0, 2, 3, 1)
        aux_pred = self._apply_activation_single(fmap_last[..., :-1], "linear")
        aux_conf = self._apply_activation_single(fmap_last[..., -1], self.conf_activation)

        return {
            self.head_main: main_pred.squeeze(-1),
            f"{self.head_main}_conf": main_conf,
            self.head_aux: aux_pred,
            f"{self.head_aux}_conf": aux_conf,
        }

    def _fuse(self, feats):
        l1, l2, l3, l4 = feats
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        aux_out = self.scratch.refinenet4_aux(l4_rn, size=l3_rn.shape[2:])
        aux_list = []
        if self.aux_levels >= 4:
            aux_list.append(aux_out)

        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        aux_out = self.scratch.refinenet3_aux(aux_out, l3_rn, size=l2_rn.shape[2:])
        if self.aux_levels >= 3:
            aux_list.append(aux_out)

        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        aux_out = self.scratch.refinenet2_aux(aux_out, l2_rn, size=l1_rn.shape[2:])
        if self.aux_levels >= 2:
            aux_list.append(aux_out)

        out = self.scratch.refinenet1(out, l1_rn)
        aux_out = self.scratch.refinenet1_aux(aux_out, l1_rn)
        aux_list.append(aux_out)

        out = self.scratch.output_conv1(out)
        aux_list = [self.scratch.output_conv1_aux[i](aux) for i, aux in enumerate(aux_list)]

        return out, aux_list

    def _add_pos_embed(self, x, W, H, ratio=0.1):
        pw, ph = x.shape[-1], x.shape[-2]
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pe

    def _make_aux_out1_block(self, in_ch, dtype=None, device=None, operations=None):
        if self.aux_out1_conv_num == 5:
            return nn.Sequential(
                operations.Conv2d(in_ch, in_ch // 2, 3, 1, 1, dtype=dtype, device=device),
                operations.Conv2d(in_ch // 2, in_ch, 3, 1, 1, dtype=dtype, device=device),
                operations.Conv2d(in_ch, in_ch // 2, 3, 1, 1, dtype=dtype, device=device),
                operations.Conv2d(in_ch // 2, in_ch, 3, 1, 1, dtype=dtype, device=device),
                operations.Conv2d(in_ch, in_ch // 2, 3, 1, 1, dtype=dtype, device=device),
            )
        if self.aux_out1_conv_num == 3:
            return nn.Sequential(
                operations.Conv2d(in_ch, in_ch // 2, 3, 1, 1, dtype=dtype, device=device),
                operations.Conv2d(in_ch // 2, in_ch, 3, 1, 1, dtype=dtype, device=device),
                operations.Conv2d(in_ch, in_ch // 2, 3, 1, 1, dtype=dtype, device=device),
            )
        if self.aux_out1_conv_num == 1:
            return nn.Sequential(
                operations.Conv2d(in_ch, in_ch // 2, 3, 1, 1, dtype=dtype, device=device),
            )
        raise ValueError(f"aux_out1_conv_num {self.aux_out1_conv_num} not supported")

    def _apply_activation_single(self, x, activation="linear"):
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp":
            return torch.exp(x)
        if act == "expm1":
            return torch.expm1(x)
        if act == "expp1":
            return torch.exp(x) + 1
        if act == "relu":
            return torch.relu(x)
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "softplus":
            return F.softplus(x)
        if act == "tanh":
            return torch.tanh(x)
        return x


# =============================================================================
# DA3 Top-level Models
# =============================================================================


class DepthAnything3Net(nn.Module):
    """Depth Anything 3 network for depth estimation and camera pose estimation."""

    PATCH_SIZE = 14

    def __init__(self, net, head, cam_dec=None, cam_enc=None, gs_head=None, gs_adapter=None,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.backbone = net
        self.head = head
        self.cam_dec = cam_dec
        self.cam_enc = cam_enc
        self.gs_adapter = gs_adapter
        self.gs_head = gs_head

    def forward(self, x, extrinsics=None, intrinsics=None, export_feat_layers=[], infer_gs=False):
        import logging as _logging
        _dbg = _logging.getLogger("DA3Streaming")

        def _vram(tag):
            if x.device.type == 'cuda':
                a = torch.cuda.memory_allocated(x.device) / (1024**3)
                r = torch.cuda.memory_reserved(x.device) / (1024**3)
                _dbg.debug(f"[VRAM model.forward {tag}] allocated={a:.2f}GB reserved={r:.2f}GB")

        _vram("entry")
        _dbg.debug(f"[DEBUG model.forward] input x: shape={list(x.shape)} dtype={x.dtype} device={x.device}")

        if extrinsics is not None:
            cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])
        else:
            cam_token = None

        _vram("before_backbone")
        feats, aux_feats = self.backbone(
            x, cam_token=cam_token, export_feat_layers=export_feat_layers
        )
        _vram("after_backbone")

        total_feat_mb = 0
        for fi, feat in enumerate(feats):
            for fj, f in enumerate(feat):
                if torch.is_tensor(f):
                    mb = f.nelement() * f.element_size() / (1024**2)
                    total_feat_mb += mb
                    _dbg.debug(f"[DEBUG backbone] feats[{fi}][{fj}]: shape={list(f.shape)} dtype={f.dtype} size={mb:.1f}MB")
        _dbg.debug(f"[DEBUG backbone] total feature size: {total_feat_mb:.1f}MB")

        H, W = x.shape[-2], x.shape[-1]

        _vram("before_depth_head")
        output = self._process_depth_head(feats, H, W)
        _vram("after_depth_head")

        output = self._process_camera_estimation(feats, H, W, output)
        _vram("after_camera_est")

        if infer_gs:
            output = self._process_gs_head(feats, H, W, output, x, extrinsics, intrinsics)

        output.aux = self._extract_auxiliary_features(aux_feats, export_feat_layers, H, W)
        _vram("exit")
        return output

    def _process_depth_head(self, feats, H, W):
        return self.head(feats, H, W, patch_start_idx=0)

    def _process_camera_estimation(self, feats, H, W, output):
        if self.cam_dec is not None:
            from .camera import pose_encoding_to_extri_intri
            pose_enc = self.cam_dec(feats[-1][1])
            c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))
            output.extrinsics = affine_inverse(c2w)
            output.intrinsics = ixt
        return output

    def _process_gs_head(self, feats, H, W, output, in_images, extrinsics=None, intrinsics=None):
        if self.gs_head is None or self.gs_adapter is None:
            return output
        assert output.get("depth", None) is not None, "must provide MV depth for the GS head."

        if extrinsics is not None and intrinsics is not None:
            ctx_extr = extrinsics
            ctx_intr = intrinsics
        else:
            ctx_extr = output.get("extrinsics", None)
            ctx_intr = output.get("intrinsics", None)
            assert ctx_extr is not None and ctx_intr is not None, \
                "must process camera info first if GT is not available"
        gt_extr = extrinsics
        ctx_extr = as_homogeneous(ctx_extr)
        if gt_extr is not None:
            gt_extr = as_homogeneous(gt_extr)

        gs_outs = self.gs_head(feats=feats, H=H, W=W, patch_start_idx=0, images=in_images)
        raw_gaussians = gs_outs.raw_gs
        densities = gs_outs.raw_gs_conf

        from .geometry import map_pdf_to_opacity
        gs_world = self.gs_adapter(
            extrinsics=ctx_extr, intrinsics=ctx_intr, depths=output.depth,
            opacities=map_pdf_to_opacity(densities), raw_gaussians=raw_gaussians,
            image_shape=(H, W), gt_extrinsics=gt_extr,
        )
        output.gaussians = gs_world
        return output

    def _extract_auxiliary_features(self, feats, feat_layers, H, W):
        aux_features = ModelOutput()
        assert len(feats) == len(feat_layers)
        for feat, feat_layer in zip(feats, feat_layers):
            feat_reshaped = feat.reshape([
                feat.shape[0], feat.shape[1],
                H // self.PATCH_SIZE, W // self.PATCH_SIZE,
                feat.shape[-1],
            ])
            aux_features[f"feat_layer_{feat_layer}"] = feat_reshaped
        return aux_features


class NestedDepthAnything3Net(nn.Module):
    """Nested DA3 with metric scaling: combines two DepthAnything3Net branches."""

    def __init__(self, da3_main, da3_metric, dtype=None, device=None, operations=None):
        super().__init__()
        self.da3 = da3_main
        self.da3_metric = da3_metric

    def forward(self, x, extrinsics=None, intrinsics=None, export_feat_layers=[], infer_gs=False):
        output = self.da3(x, extrinsics, intrinsics,
                          export_feat_layers=export_feat_layers, infer_gs=infer_gs)
        metric_output = self.da3_metric(x, infer_gs=infer_gs)
        output = self._apply_metric_scaling(output, metric_output)
        output = self._apply_depth_alignment(output, metric_output)
        output = self._handle_sky_regions(output, metric_output)
        return output

    def _apply_metric_scaling(self, output, metric_output):
        metric_output.depth = apply_metric_scaling(metric_output.depth, output.intrinsics)
        return output

    def _apply_depth_alignment(self, output, metric_output):
        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)
        assert non_sky_mask.sum() > 10, "Insufficient non-sky pixels for alignment"
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
        output.extrinsics[:, :, :3, 3] *= scale_factor
        output.is_metric = 1
        output.scale_factor = scale_factor.item()
        return output

    def _handle_sky_regions(self, output, metric_output, sky_depth_def=200.0):
        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)
        non_sky_depth = output.depth[non_sky_mask]
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth
        non_sky_max = min(torch.quantile(sampled_depth, 0.99), sky_depth_def)
        output.depth, output.depth_conf = set_sky_regions_to_max_depth(
            output.depth, output.depth_conf, non_sky_mask, max_depth=non_sky_max
        )
        return output
