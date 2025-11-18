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

import torch
import torch.nn as nn


class CameraDec(nn.Module):
    """
    CameraDec decodes visual features from the backbone into camera pose parameters.

    Takes features from the last backbone layer and predicts:
    - Translation (3D)
    - Quaternion rotation (4D, XYZW format)
    - Field of view (2D)

    Output is a 9D pose encoding that can be converted to extrinsics/intrinsics.
    """

    def __init__(self, dim_in=1536):
        super().__init__()
        output_dim = dim_in
        self.backbone = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )
        self.fc_t = nn.Linear(output_dim, 3)
        self.fc_qvec = nn.Linear(output_dim, 4)
        self.fc_fov = nn.Sequential(nn.Linear(output_dim, 2), nn.ReLU())

    def forward(self, feat, camera_encoding=None, *args, **kwargs):
        """
        Decode features into camera pose encoding.

        Args:
            feat: Features from backbone, shape (B, N, dim_in)
            camera_encoding: Optional pre-computed encoding. If provided,
                            uses its qvec and fov values instead of predicting them.

        Returns:
            pose_enc: 9D pose encoding (B, N, 9) = [t_x, t_y, t_z, qx, qy, qz, qw, fov_h, fov_w]
        """
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
