# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for the FlyDSL f8f8bf16_blockwise kernel via torch.ops.mslk dispatch."""

import unittest

import torch
import torch.nn.functional as F

import mslk.gemm  # noqa: F401 — triggers op registration
from mslk.flydsl.common import is_flydsl_available


def _run_torch_blockscale_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 128,
) -> torch.Tensor:
    """Pure-torch reference for blockscale GEMM."""
    m, k = x.shape
    n = weight.shape[0]
    scale_m = (m + block_m - 1) // block_m
    scale_n = (n + block_n - 1) // block_n
    scale_k = (k + block_k - 1) // block_k

    x_scale_expanded = (
        x_scale.view(-1, 1)
        .repeat(1, block_m * block_k)
        .view(scale_m, scale_k, block_m, block_k)
        .permute(0, 2, 1, 3)
        .reshape(scale_m * block_m, scale_k * block_k)
    )
    x_scale_expanded = x_scale_expanded[:m, :k]
    x_f32 = x.to(torch.float32) * x_scale_expanded.to(torch.float32)

    w_scale_expanded = (
        w_scale.view(-1, 1)
        .repeat(1, block_n * block_k)
        .view(scale_n, scale_k, block_n, block_k)
        .permute(0, 2, 1, 3)
        .reshape(scale_n * block_n, scale_k * block_k)
    )
    w_scale_expanded = w_scale_expanded[:n, :k]
    w_f32 = weight.to(torch.float32) * w_scale_expanded.to(torch.float32)

    return F.linear(x_f32, w_f32).to(torch.bfloat16)


_HAS_OP = hasattr(torch.ops, "mslk") and hasattr(torch.ops.mslk, "f8f8bf16_blockwise")
_HAS_PRESHUFFLE_OP = _HAS_OP and hasattr(torch.ops.mslk, "f8f8bf16_blockwise_preshuffle")

_REQUIRES_FLYDSL = unittest.skipUnless(
    torch.version.hip is not None
    and torch.cuda.is_available()
    and is_flydsl_available()
    and _HAS_OP,
    "requires ROCm GPU with FlyDSL and mslk C++ ops",
)


@_REQUIRES_FLYDSL
class FlyDSLBlockscaleGemmTest(unittest.TestCase):

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128

    SHAPES = [
        (1, 7168, 2304),
        (16, 7168, 2304),
        (33, 7168, 2304),
        (64, 7168, 2304),
        (256, 7168, 2304),
        (1, 3072, 1536),
        (64, 3072, 1536),
    ]

    def _run_one(self, M: int, N: int, K: int) -> None:
        device = torch.device("cuda")

        scale_m = (M + self.BLOCK_M - 1) // self.BLOCK_M
        scale_n = (N + self.BLOCK_N - 1) // self.BLOCK_N
        scale_k = (K + self.BLOCK_K - 1) // self.BLOCK_K

        fp8_dtype = (
            torch.float8_e4m3fn
            if hasattr(torch, "float8_e4m3fn")
            and "gfx95" in torch.cuda.get_device_properties(0).gcnArchName
            else torch.float8_e4m3fnuz
        )

        x = (torch.rand((M, K), dtype=torch.float16, device=device) / 10).to(fp8_dtype)
        w = (torch.rand((N, K), dtype=torch.float16, device=device) / 10).to(fp8_dtype)

        x_scale = torch.rand([scale_m, scale_k], dtype=torch.float32, device=device)
        w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device=device)

        ref = _run_torch_blockscale_ref(
            x, w, x_scale, w_scale,
            self.BLOCK_M, self.BLOCK_N, self.BLOCK_K,
        )

        out = torch.ops.mslk.f8f8bf16_blockwise(
            x, w, x_scale, w_scale,
            self.BLOCK_M, self.BLOCK_N, self.BLOCK_K,
        )

        torch.testing.assert_close(
            out.to(torch.float32),
            ref.to(torch.float32),
            rtol=1e-2,
            atol=0.01,
        )

    def test_shapes(self) -> None:
        for M, N, K in self.SHAPES:
            with self.subTest(M=M, N=N, K=K):
                self._run_one(M, N, K)


_REQUIRES_PRESHUFFLE = unittest.skipUnless(
    torch.version.hip is not None
    and torch.cuda.is_available()
    and is_flydsl_available()
    and _HAS_PRESHUFFLE_OP,
    "requires ROCm GPU with FlyDSL and mslk C++ ops (preshuffle schema)",
)


@_REQUIRES_PRESHUFFLE
class FlyDSLBlockscalePreshuffleGemmTest(unittest.TestCase):

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128

    SHAPES = FlyDSLBlockscaleGemmTest.SHAPES

    def _run_one(self, M: int, N: int, K: int) -> None:
        from mslk.quantize.shuffle import preshuffle_b_mfma

        device = torch.device("cuda")

        scale_m = (M + self.BLOCK_M - 1) // self.BLOCK_M
        scale_n = (N + self.BLOCK_N - 1) // self.BLOCK_N
        scale_k = (K + self.BLOCK_K - 1) // self.BLOCK_K

        fp8_dtype = (
            torch.float8_e4m3fn
            if hasattr(torch, "float8_e4m3fn")
            and "gfx95" in torch.cuda.get_device_properties(0).gcnArchName
            else torch.float8_e4m3fnuz
        )

        x = (torch.rand((M, K), dtype=torch.float16, device=device) / 10).to(fp8_dtype)
        w = (torch.rand((N, K), dtype=torch.float16, device=device) / 10).to(fp8_dtype)

        x_scale = torch.rand([scale_m, scale_k], dtype=torch.float32, device=device)
        w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device=device)

        w_shuf = preshuffle_b_mfma(w)

        ref = _run_torch_blockscale_ref(
            x, w, x_scale, w_scale,
            self.BLOCK_M, self.BLOCK_N, self.BLOCK_K,
        )

        out = torch.ops.mslk.f8f8bf16_blockwise_preshuffle(
            x, w_shuf, x_scale, w_scale,
            self.BLOCK_M, self.BLOCK_N, self.BLOCK_K,
        )

        torch.testing.assert_close(
            out.to(torch.float32),
            ref.to(torch.float32),
            rtol=1e-2,
            atol=0.01,
        )

    def test_shapes(self) -> None:
        for M, N, K in self.SHAPES:
            with self.subTest(M=M, N=N, K=K):
                self._run_one(M, N, K)


if __name__ == "__main__":
    unittest.main()
