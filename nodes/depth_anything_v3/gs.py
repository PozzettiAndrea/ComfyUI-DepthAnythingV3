# Depth Anything V3 — ComfyUI-native Gaussian Splatting modules
#
# Consolidates: GSDPT, GaussianAdapter, Gaussians dataclass.
#
# Original sources:
#   model/gsdpt.py, model/gs_adapter.py, specs.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict as TyDict, List, Optional, Sequence

import torch
import torch.nn as nn

import comfy.ops

from .camera import cam_quat_xyzw_to_world_quat_wxyz
from .model import DPT, activate_head_gs, custom_interpolate
from .geometry import affine_inverse, get_world_rays, sample_image_grid
from .sh_helpers import rotate_sh


# ---------------------------------------------------------------------------
# Gaussians dataclass
# ---------------------------------------------------------------------------

@dataclass
class Gaussians:
    """3DGS parameters, all in world space"""

    means: torch.Tensor  # world points, "batch gaussian dim"
    scales: torch.Tensor  # scales_std, "batch gaussian 3"
    rotations: torch.Tensor  # world_quat_wxyz, "batch gaussian 4"
    harmonics: torch.Tensor  # world SH, "batch gaussian 3 d_sh"
    opacities: torch.Tensor  # opacity | opacity SH, "batch gaussian" | "batch gaussian 1 d_sh"


# ---------------------------------------------------------------------------
# GaussianAdapter
# ---------------------------------------------------------------------------

class GaussianAdapter(nn.Module):

    def __init__(
        self,
        sh_degree: int = 0,
        pred_color: bool = False,
        pred_offset_depth: bool = False,
        pred_offset_xy: bool = True,
        gaussian_scale_min: float = 1e-5,
        gaussian_scale_max: float = 30.0,
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.pred_color = pred_color
        self.pred_offset_depth = pred_offset_depth
        self.pred_offset_xy = pred_offset_xy
        self.gaussian_scale_min = gaussian_scale_min
        self.gaussian_scale_max = gaussian_scale_max

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        if not pred_color:
            # Computed constant — force CPU to avoid meta device context
            sh_mask = torch.ones((self.d_sh,), dtype=torch.float32, device="cpu")
            for degree in range(1, sh_degree + 1):
                sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree
            self.register_buffer("sh_mask", sh_mask, persistent=False)

    def forward(
        self,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        depths: torch.Tensor,
        opacities: torch.Tensor,
        raw_gaussians: torch.Tensor,
        image_shape: tuple[int, int],
        eps: float = 1e-8,
        gt_extrinsics: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Gaussians:
        device = extrinsics.device
        dtype = raw_gaussians.dtype
        H, W = image_shape
        b, v = raw_gaussians.shape[:2]

        # get cam2worlds and intr_normed to adapt to 3DGS codebase
        cam2worlds = affine_inverse(extrinsics)
        intr_normed = intrinsics.clone().detach()
        intr_normed[..., 0, :] /= W
        intr_normed[..., 1, :] /= H

        # 1. compute 3DGS means
        # 1.1) offset the predicted depth if needed
        if self.pred_offset_depth:
            gs_depths = depths + raw_gaussians[..., -1]
            raw_gaussians = raw_gaussians[..., :-1]
        else:
            gs_depths = depths

        # 1.2) align predicted poses with GT if needed
        if gt_extrinsics is not None and not torch.equal(extrinsics, gt_extrinsics):
            try:
                from .pose_align import batch_align_poses_umeyama
                _, _, pose_scales = batch_align_poses_umeyama(
                    gt_extrinsics.detach().float(),
                    extrinsics.detach().float(),
                )
                pose_scales = torch.clamp(pose_scales, min=1 / 3.0, max=3.0)
                cam2worlds[:, :, :3, 3] = cam2worlds[:, :, :3, 3] * pose_scales[:, None, None]
                gs_depths = gs_depths * pose_scales[:, None, None, None]
            except ImportError:
                pass

        # 1.3) casting xy in image space
        xy_ray, _ = sample_image_grid((H, W), device)
        xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)
        if self.pred_offset_xy:
            pixel_size = 1 / torch.tensor((W, H), dtype=xy_ray.dtype, device=device)
            offset_xy = raw_gaussians[..., :2]
            xy_ray = xy_ray + offset_xy * pixel_size
            raw_gaussians = raw_gaussians[..., 2:]

        # 1.4) unproject depth + xy to world ray
        origins, directions = get_world_rays(
            xy_ray,
            cam2worlds[:, :, None, None, :, :].expand(-1, -1, H, W, -1, -1),
            intr_normed[:, :, None, None, :, :].expand(-1, -1, H, W, -1, -1),
        )
        gs_means_world = origins + directions * gs_depths[..., None]
        gs_means_world = gs_means_world.flatten(1, 3)  # [B, V*H*W, D]

        # 2. compute other GS attributes
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

        # 2.1) 3DGS scales
        scale_min = self.gaussian_scale_min
        scale_max = self.gaussian_scale_max
        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        pixel_size = 1 / torch.tensor((W, H), dtype=dtype, device=device)
        multiplier = self.get_scale_multiplier(intr_normed, pixel_size)
        gs_scales = scales * gs_depths[..., None] * multiplier[..., None, None, None]
        gs_scales = gs_scales.flatten(1, 3)  # [B, V*H*W, D]

        # 2.2) 3DGS quaternion (world space)
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
        cam_quat_xyzw = rotations.flatten(1, 3)  # [B, V*H*W, C]
        c2w_mat = cam2worlds[:, :, None, None, :, :].expand(-1, -1, H, W, -1, -1).flatten(1, 3)  # [B, V*H*W, I, J]
        world_quat_wxyz = cam_quat_xyzw_to_world_quat_wxyz(cam_quat_xyzw, c2w_mat)
        gs_rotations_world = world_quat_wxyz

        # 2.3) 3DGS color / SH coefficient (world space)
        sh = sh.unflatten(-1, (3, -1))  # [..., 3*d_sh] -> [..., 3, d_sh]
        if not self.pred_color:
            sh = sh * self.sh_mask

        if self.pred_color or self.sh_degree == 0:
            gs_sh_world = sh
        else:
            gs_sh_world = rotate_sh(sh, cam2worlds[:, :, None, None, None, :3, :3])
        gs_sh_world = gs_sh_world.flatten(1, 3)  # [B, V*H*W, XYZ, D_SH]

        # 2.4) 3DGS opacity
        gs_opacities = opacities.flatten(1, 3)  # [B, V*H*W, ...]

        return Gaussians(
            means=gs_means_world,
            harmonics=gs_sh_world,
            opacities=gs_opacities,
            scales=gs_scales,
            rotations=gs_rotations_world,
        )

    def get_scale_multiplier(
        self,
        intrinsics: torch.Tensor,
        pixel_size: torch.Tensor,
        multiplier: float = 0.1,
    ) -> torch.Tensor:
        xy_multipliers = multiplier * torch.einsum(
            "...ij,j->...i",
            intrinsics[..., :2, :2].float().inverse().to(intrinsics),
            pixel_size,
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return 1 if self.pred_color else (self.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        raw_gs_dim = 0
        if self.pred_offset_xy:
            raw_gs_dim += 2
        raw_gs_dim += 3  # scales
        raw_gs_dim += 4  # quaternion
        raw_gs_dim += 3 * self.d_sh  # color
        if self.pred_offset_depth:
            raw_gs_dim += 1
        return raw_gs_dim


# ---------------------------------------------------------------------------
# GSDPT — DPT head for Gaussian Splatting output
# ---------------------------------------------------------------------------

class GSDPT(DPT):
    """
    GS-DPT head for Gaussian Splatting output.
    Extends DPT with additional image feature merging.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "linear",
        conf_activation: str = "sigmoid",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
        conf_dim: int = 1,
        norm_type: str = "idt",
        fusion_block_inplace: bool = False,
        dtype=None, device=None, operations=None,
    ) -> None:
        super().__init__(
            dim_in=dim_in,
            patch_size=patch_size,
            output_dim=output_dim,
            activation=activation,
            conf_activation=conf_activation,
            features=features,
            out_channels=out_channels,
            pos_embed=pos_embed,
            down_ratio=down_ratio,
            head_name="raw_gs",
            use_sky_head=False,
            norm_type=norm_type,
            fusion_block_inplace=fusion_block_inplace,
            dtype=dtype, device=device, operations=operations,
        )
        self.conf_dim = conf_dim
        if conf_dim and conf_dim > 1:
            assert (
                conf_activation == "linear"
            ), "use linear prediction when using view-dependent opacity"

        merger_out_dim = features if feature_only else features // 2
        self.images_merger = nn.Sequential(
            operations.Conv2d(3, merger_out_dim // 4, 3, 1, 1, dtype=dtype, device=device),
            nn.GELU(),
            operations.Conv2d(merger_out_dim // 4, merger_out_dim // 2, 3, 1, 1, dtype=dtype, device=device),
            nn.GELU(),
            operations.Conv2d(merger_out_dim // 2, merger_out_dim, 3, 1, 1, dtype=dtype, device=device),
            nn.GELU(),
        )

    # -------------------------------------------------------------------------
    # Internal forward (single chunk)
    # -------------------------------------------------------------------------
    def _forward_impl(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
        images: torch.Tensor,
    ) -> TyDict[str, torch.Tensor]:
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

        # Fusion pyramid
        fused = self._fuse(resized_feats)
        fused = self.scratch.output_conv1(fused)

        # Upsample to target resolution
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

        fused = custom_interpolate(fused, (h_out, w_out), mode="bilinear", align_corners=True)

        # inject the image information
        fused = fused + self.images_merger(images)

        if self.pos_embed:
            fused = self._add_pos_embed(fused, W, H)

        feat = fused

        # Main head: logits -> activate_head or single channel activation
        main_logits = self.scratch.output_conv2(feat)
        outs: TyDict[str, torch.Tensor] = {}
        if self.has_conf:
            pred, conf = activate_head_gs(
                main_logits,
                activation=self.activation,
                conf_activation=self.conf_activation,
                conf_dim=self.conf_dim,
            )
            outs[self.head_main] = pred.squeeze(1)
            outs[f"{self.head_main}_conf"] = conf.squeeze(1)
        else:
            outs[self.head_main] = self._apply_activation_single(main_logits).squeeze(1)

        return outs
