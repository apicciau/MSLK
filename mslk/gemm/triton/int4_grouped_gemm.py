# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
BF16 x INT4 grouped GEMM for ROCm/AMD GPUs.

Delegates to matmul_bf16i4_rowwise from int4_gemm.py for each group.
On ROCm, bf16i4bf16_shuffled_grouped routes through this path — no CUTLASS
shuffle layout exists on AMD, so weights are already in rowwise format.

Weight layout:
  WQ          : [G, N, K//2]        int8
  w_scale_group: [G, num_groups, N]  float32 or bfloat16
  w_zero_group : [G, num_groups, N]  float32 or bfloat16
  M_sizes      : [G]                 int32 or int64 — rows per group

Output: [M_total, N] bfloat16, where M_total = sum(M_sizes).
"""

import torch

from mslk.gemm.triton.int4_gemm import matmul_bf16i4_rowwise


def matmul_bf16i4_rowwise_grouped(
    X: torch.Tensor,
    WQ: torch.Tensor,
    w_scale_group: torch.Tensor,
    w_zero_group: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    """
    Grouped BF16 x INT4 rowwise GEMM.

    Args:
        X             : [M_total, K]         bfloat16
        WQ            : [G, N, K//2]         int8
        w_scale_group : [G, num_groups, N]   float32 or bfloat16
        w_zero_group  : [G, num_groups, N]   float32 or bfloat16
        M_sizes       : [G]                  rows per group (must sum to M_total)

    Returns:
        Y : [M_total, N]  bfloat16
    """
    outputs = []
    m_offset = 0
    for g, m in enumerate(M_sizes.tolist()):
        m = int(m)
        outputs.append(
            matmul_bf16i4_rowwise(
                X[m_offset : m_offset + m],
                WQ[g],
                w_scale_group[g],
                w_zero_group[g],
            )
        )
        m_offset += m
    return torch.cat(outputs, dim=0)


def _register_rocm_ops() -> None:
    if not (torch.version.hip is not None and hasattr(torch.ops, "mslk")):
        return

    if hasattr(torch.ops.mslk, "bf16i4bf16_shuffled_grouped"):

        @torch.library.impl("mslk::bf16i4bf16_shuffled_grouped", "CUDA")
        def _bf16i4bf16_shuffled_grouped_rocm(
            X: torch.Tensor,
            WQ: torch.Tensor,
            w_scale_group: torch.Tensor,
            w_zero_group: torch.Tensor,
            M_sizes: torch.Tensor,
        ) -> torch.Tensor:
            return matmul_bf16i4_rowwise_grouped(
                X, WQ, w_scale_group, w_zero_group, M_sizes
            )


try:
    _register_rocm_ops()
except Exception:
    pass
