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

import numpy as np
import torch

try:
    from scipy.sparse import coo_matrix, csc_matrix
    from scipy.sparse.linalg import spsolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def solve_sparse(A: "csc_matrix", b: np.ndarray, freen: int) -> np.ndarray:
    """Solve linear system A * delta = b, supports submatrix solving"""
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for solve_sparse but is not installed. "
            "Install with: pip install scipy"
        )
    if freen < 0:
        return spsolve(A, b)
    else:
        A_sub = A[:freen, :freen].tocsc()
        b_sub = b[:freen]
        delta_sub = spsolve(A_sub, b_sub)
        delta = np.zeros_like(b)
        delta[:freen] = delta_sub
        return delta


def solve_system_py(
    J_Ginv_i: torch.Tensor,
    J_Ginv_j: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    res: torch.Tensor,
    ep: float,
    lm: float,
    freen: int,
) -> torch.Tensor:
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for solve_system_py but is not installed. "
            "Install with: pip install scipy"
        )
    # Ensure all tensors are on CPU
    device = res.device
    J_Ginv_i = J_Ginv_i.cpu()
    J_Ginv_j = J_Ginv_j.cpu()
    ii = ii.cpu()
    jj = jj.cpu()
    res = res.clone().cpu()

    r = res.size(0)  # Number of edges
    n = max(ii.max().item(), jj.max().item()) + 1  # Number of nodes

    res_vec = res.view(-1).numpy().astype(np.float64)

    rows, cols, data = [], [], []
    ii_np = ii.numpy()
    jj_np = jj.numpy()
    J_Ginv_i_np = J_Ginv_i.numpy()
    J_Ginv_j_np = J_Ginv_j.numpy()

    for x in range(r):
        i = ii_np[x]
        j = jj_np[x]
        if i == j:
            raise ValueError("Self-edges are not allowed")

        for k in range(7):
            for l in range(7):
                row_idx = x * 7 + k
                col_idx_i = i * 7 + l
                val_i = J_Ginv_i_np[x, k, l]
                rows.append(row_idx)
                cols.append(col_idx_i)
                data.append(val_i)

                col_idx_j = j * 7 + l
                val_j = J_Ginv_j_np[x, k, l]
                rows.append(row_idx)
                cols.append(col_idx_j)
                data.append(val_j)

    J = coo_matrix((data, (rows, cols)), shape=(r * 7, n * 7)).tocsc()

    b_vec = -J.T @ res_vec

    A_mat = J.T @ J

    diag = A_mat.diagonal()
    new_diag = diag * (1.0 + lm) + ep
    A_mat.setdiag(new_diag)

    freen_total = freen * 7
    delta = solve_sparse(A_mat.tocsc(), b_vec, freen_total)

    delta_tensor = torch.from_numpy(delta.astype(np.float32)).view(n, 7).to(device)
    return delta_tensor
