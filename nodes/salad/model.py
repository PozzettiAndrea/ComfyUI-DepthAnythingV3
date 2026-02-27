"""SALAD (Sinkhorn Algorithm for Locally-Aggregated Descriptors) model.

Inline implementation so we don't depend on the external serizba/salad package.
Architecture: DINOv2 ViT-B14 backbone + SALAD aggregator.
All layers use comfy.ops for VRAM management and dtype casting.

Original: https://github.com/serizba/salad (MIT License)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.ops
from comfy.ldm.modules.attention import optimized_attention_for_device


# ---------------------------------------------------------------------------
# DINOv2 ViT-B14 Backbone (inline, comfy-native)
# ---------------------------------------------------------------------------
# Matches Facebook's dinov2_vitb14 attribute names so existing SALAD
# checkpoints load without key remapping.

class _Attention(nn.Module):
    """Multi-head self-attention with combined qkv projection."""

    def __init__(self, dim, num_heads=12, qkv_bias=True,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.proj = operations.Linear(dim, dim, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = optimized_attention_for_device(q.device)(q, k, v, heads=self.num_heads, skip_reshape=True)
        x = self.proj(x)
        return x


class _Mlp(nn.Module):
    """MLP with GELU activation."""

    def __init__(self, dim, mlp_ratio=4.0,
                 dtype=None, device=None, operations=None):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = operations.Linear(dim, hidden, bias=True, dtype=dtype, device=device)
        self.fc2 = operations.Linear(hidden, dim, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class _LayerScale(nn.Module):
    """Per-channel learnable scale (dinov2 uses init_values=1.0)."""

    def __init__(self, dim, dtype=None, device=None):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))

    def forward(self, x):
        return x * comfy.model_management.cast_to_device(self.gamma, x.device, x.dtype)


class _Block(nn.Module):
    """Transformer block matching Facebook dinov2 attribute names."""

    def __init__(self, dim, num_heads=12,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.LayerNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.attn = _Attention(dim, num_heads=num_heads,
                               dtype=dtype, device=device, operations=operations)
        self.ls1 = _LayerScale(dim, dtype=dtype, device=device)
        self.norm2 = operations.LayerNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.mlp = _Mlp(dim, dtype=dtype, device=device, operations=operations)
        self.ls2 = _LayerScale(dim, dtype=dtype, device=device)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class _PatchEmbed(nn.Module):
    """Patch embedding: Conv2d projection."""

    def __init__(self, in_chans=3, embed_dim=768, patch_size=14,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Conv2d(in_chans, embed_dim,
                                      kernel_size=patch_size, stride=patch_size,
                                      dtype=dtype, device=device)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, N, D)
        return self.proj(x).flatten(2).transpose(1, 2)


class DINOv2(nn.Module):
    """Inline DINOv2 ViT-B14 using comfy.ops.

    Attribute names match Facebook's dinov2 so existing checkpoints
    load via state_dict without key remapping.
    """

    def __init__(self, model_name="dinov2_vitb14", num_trainable_blocks=2,
                 norm_layer=False, return_token=False,
                 dtype=None, device=None, operations=None):
        super().__init__()
        # ViT-B14 config
        ARCHS = {
            "dinov2_vits14": {"embed_dim": 384, "num_heads": 6, "depth": 12},
            "dinov2_vitb14": {"embed_dim": 768, "num_heads": 12, "depth": 12},
            "dinov2_vitl14": {"embed_dim": 1024, "num_heads": 16, "depth": 24},
            "dinov2_vitg14": {"embed_dim": 1536, "num_heads": 24, "depth": 40},
        }
        assert model_name in ARCHS, f"Unknown model name {model_name}"
        cfg = ARCHS[model_name]
        embed_dim = cfg["embed_dim"]
        num_heads = cfg["num_heads"]
        depth = cfg["depth"]
        patch_size = 14
        image_size = 518  # dinov2 default
        num_patches = (image_size // patch_size) ** 2

        self.num_channels = embed_dim
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer_enabled = norm_layer
        self.return_token = return_token
        self.patch_size = patch_size

        # Model layers (attribute names match Facebook dinov2)
        self.model = nn.Module()
        self.model.patch_embed = _PatchEmbed(
            embed_dim=embed_dim, patch_size=patch_size,
            dtype=dtype, device=device, operations=operations,
        )
        self.model.cls_token = nn.Parameter(
            torch.empty(1, 1, embed_dim, dtype=dtype, device=device)
        )
        self.model.mask_token = nn.Parameter(
            torch.empty(1, embed_dim, dtype=dtype, device=device)
        )
        self.model.pos_embed = nn.Parameter(
            torch.empty(1, num_patches + 1, embed_dim, dtype=dtype, device=device)
        )
        self.model.blocks = nn.ModuleList([
            _Block(embed_dim, num_heads=num_heads,
                   dtype=dtype, device=device, operations=operations)
            for _ in range(depth)
        ])
        self.model.norm = operations.LayerNorm(embed_dim, eps=1e-6, dtype=dtype, device=device)

    def _prepare_tokens(self, x):
        """Patch embed + CLS token + positional embedding."""
        B, C, H, W = x.shape
        x = self.model.patch_embed(x)  # (B, N, D)
        cls_tokens = comfy.model_management.cast_to_device(
            self.model.cls_token, x.device, x.dtype
        ).expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # Interpolate pos_embed if input size differs from training size
        pos_embed = comfy.model_management.cast_to_device(
            self.model.pos_embed, x.device, x.dtype
        )
        if x.shape[1] != pos_embed.shape[1]:
            # Separate CLS and patch embeddings
            cls_pos = pos_embed[:, :1, :]
            patch_pos = pos_embed[:, 1:, :]
            # Interpolate patch pos embeddings
            orig_n = int(patch_pos.shape[1] ** 0.5)
            new_h, new_w = H // self.patch_size, W // self.patch_size
            patch_pos = patch_pos.reshape(1, orig_n, orig_n, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(new_h, new_w),
                                      mode="bicubic", align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, pos_embed.shape[-1])
            pos_embed = torch.cat((cls_pos, patch_pos), dim=1)
        x = x + pos_embed
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        x = self._prepare_tokens(x)

        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x)

        if self.norm_layer_enabled:
            x = self.model.norm(x)

        t = x[:, 0]
        f = x[:, 1:]
        f = f.reshape((B, H // self.patch_size, W // self.patch_size,
                        self.num_channels)).permute(0, 3, 1, 2)

        if self.return_token:
            return f, t
        return f


# ---------------------------------------------------------------------------
# SALAD Aggregator
# ---------------------------------------------------------------------------

def _log_otp_solver(log_a, log_b, M, num_iters=20, reg=1.0):
    """Sinkhorn matrix scaling for differentiable optimal transport."""
    M = M / reg
    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()
    return M + u.unsqueeze(2) + v.unsqueeze(1)


def _get_matching_probs(S, dustbin_score=1.0, num_iters=3, reg=1.0):
    batch_size, m, n = S.size()
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a = norm.expand(m + 1).contiguous()
    log_b = norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n - m)
    log_a = log_a.expand(batch_size, -1)
    log_b = log_b.expand(batch_size, -1)

    log_P = _log_otp_solver(log_a, log_b, S_aug, num_iters=num_iters, reg=reg)
    return log_P - norm


class SALAD(nn.Module):
    def __init__(self, num_channels=1536, num_clusters=64, cluster_dim=128,
                 token_dim=256, dropout=0.3,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim

        drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.token_features = nn.Sequential(
            operations.Linear(self.num_channels, 512, dtype=dtype, device=device),
            nn.ReLU(),
            operations.Linear(512, self.token_dim, dtype=dtype, device=device),
        )
        self.cluster_features = nn.Sequential(
            operations.Conv2d(self.num_channels, 512, 1, dtype=dtype, device=device),
            drop,
            nn.ReLU(),
            operations.Conv2d(512, self.cluster_dim, 1, dtype=dtype, device=device),
        )
        self.score = nn.Sequential(
            operations.Conv2d(self.num_channels, 512, 1, dtype=dtype, device=device),
            drop,
            nn.ReLU(),
            operations.Conv2d(512, self.num_clusters, 1, dtype=dtype, device=device),
        )
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x, t = x
        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(t)

        p = _get_matching_probs(p, self.dust_bin, 3)
        p = torch.exp(p)
        p = p[:, :-1, :]

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        f = torch.cat([
            F.normalize(t, p=2, dim=-1),
            F.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1),
        ], dim=-1)

        return F.normalize(f, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Full VPR Model (backbone + aggregator)
# ---------------------------------------------------------------------------

class VPRModel(nn.Module):
    """Visual Place Recognition model: DINOv2 backbone + SALAD aggregator."""

    def __init__(self, dtype=None, device=None, operations=None):
        super().__init__()
        self.backbone = DINOv2(
            model_name="dinov2_vitb14",
            num_trainable_blocks=4,
            return_token=True,
            norm_layer=True,
            dtype=dtype, device=device, operations=operations,
        )
        self.aggregator = SALAD(
            num_channels=768,
            num_clusters=64,
            cluster_dim=128,
            token_dim=256,
            dtype=dtype, device=device, operations=operations,
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x
