"""SALAD (Sinkhorn Algorithm for Locally-Aggregated Descriptors) model.

Inline implementation so we don't depend on the external serizba/salad package.
Architecture: DINOv2 ViT-B14 backbone + SALAD aggregator.

Original: https://github.com/serizba/salad (MIT License)
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# DINOv2 Backbone
# ---------------------------------------------------------------------------

DINOV2_ARCHS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class DINOv2(nn.Module):
    def __init__(self, model_name="dinov2_vitb14", num_trainable_blocks=2,
                 norm_layer=False, return_token=False):
        super().__init__()
        assert model_name in DINOV2_ARCHS, f"Unknown model name {model_name}"
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.model.prepare_tokens_with_masks(x)

        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)

        t = x[:, 0]
        f = x[:, 1:]
        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

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
                 token_dim=256, dropout=0.3):
        super().__init__()
        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim

        drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim),
        )
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            drop,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1),
        )
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            drop,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
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
            nn.functional.normalize(t, p=2, dim=-1),
            nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1),
        ], dim=-1)

        return nn.functional.normalize(f, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Full VPR Model (backbone + aggregator)
# ---------------------------------------------------------------------------

class VPRModel(nn.Module):
    """Visual Place Recognition model: DINOv2 backbone + SALAD aggregator."""

    def __init__(self):
        super().__init__()
        self.backbone = DINOv2(
            model_name="dinov2_vitb14",
            num_trainable_blocks=4,
            return_token=True,
            norm_layer=True,
        )
        self.aggregator = SALAD(
            num_channels=768,
            num_clusters=64,
            cluster_dim=128,
            token_dim=256,
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x
