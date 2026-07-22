# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""FlyDSL implementation of f8f8bf16_blockwise for ROCm.

Replaces the CK-based implementation with a FlyDSL blockscale preshuffle
GEMM kernel. The B matrix is preshuffled on every call via ``ck_preshuffle``
(PyTorch strided copy on the GPU). Scale layout matches CK convention:
scale_a [scale_m, scale_k] M-outer row-major, scale_b [scale_n, scale_k].
"""

import torch

from mslk.utils.flydsl import is_flydsl_available, run_compiled

if is_flydsl_available():
    import flydsl.compiler as flyc

    from mslk.gemm.flydsl.blockscale_preshuffle_gemm import (
        compile_blockscale_preshuffle_gemm,
    )
    from mslk.quantize.shuffle import ck_preshuffle

    _kernel_cache: dict = {}

    def _get_compiled_kernel(M, N, K, tile_m, tile_n, tile_k, scale_block_k):
        key = (M, N, K, tile_m, tile_n, tile_k, scale_block_k)
        if key not in _kernel_cache:
            _kernel_cache[key] = compile_blockscale_preshuffle_gemm(
                M=M, N=N, K=K,
                tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
                scale_block_k=scale_block_k,
            )
        return _kernel_cache[key]

    def _select_tile_config(M, N, K, scale_block_k=128):
        candidates = [
            (16, 64, 256), (16, 128, 256),
            (32, 64, 128), (32, 64, 256), (32, 128, 128), (32, 128, 256),
            (64, 64, 128), (64, 64, 256), (64, 128, 128), (64, 128, 256),
            (64, 256, 128),
        ]

        def _valid(tm, tn, tk):
            return (
                N % tn == 0
                and K % tk == 0
                and tk % scale_block_k == 0
                and tm * tk // 256 >= 16
            )

        valid = [(tm, tn, tk) for tm, tn, tk in candidates if _valid(tm, tn, tk)]
        if not valid:
            return (64, 128, 128)

        def _score(tm, tn, tk):
            s = 0
            total_blocks = ((M + tm - 1) // tm) * (N // tn)
            if total_blocks >= 256:
                s += 15
            elif total_blocks >= 128:
                s += 10
            elif total_blocks >= 64:
                s += 5

            if M <= 48:
                s += 12 if tm == 16 else (8 if tm == 32 else 0)
            elif M <= 128:
                s += 10 if tm == 32 else (6 if tm == 16 else (4 if tm == 64 else 0))
            elif M <= 512:
                s += 12 if tm == 64 else (8 if tm == 32 else 0)
            else:
                s += 12 if tm == 64 else 0

            if M <= 128:
                s += 6 if tn == 64 else (4 if tn == 128 else (2 if tn == 256 else 0))
            else:
                s += 8 if tn == 128 else (4 if tn == 64 else (4 if tn == 256 else 0))

            s += 6 if tk == 128 else 3
            return s

        return max(valid, key=lambda t: _score(*t))

    @torch.library.impl("mslk::f8f8bf16_blockwise", "CUDA")
    def _f8f8bf16_blockwise_flydsl(
        XQ: torch.Tensor,
        WQ: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        block_m: int = 128,
        block_n: int = 128,
        block_k: int = 128,
    ) -> torch.Tensor:
        assert block_m == 128 and block_n == 128 and block_k == 128, (
            "Only block_size=128 is supported"
        )

        M = XQ.shape[0] if XQ.dim() == 2 else XQ[..., 0:1].numel()
        K = XQ.shape[-1]
        N = WQ.shape[0] if WQ.dim() == 2 else WQ[..., 0:1].numel()

        out_sizes = list(XQ.shape)
        out_sizes[-1] = N
        Y = torch.empty(out_sizes, dtype=torch.bfloat16, device=XQ.device)

        if M == 0 or N == 0 or K == 0:
            return Y

        XQ_2d = XQ.view(-1, K)
        WQ_2d = WQ.view(-1, K)
        Y_2d = Y.view(-1, N)
        M = XQ_2d.shape[0]

        WQ_shuf = ck_preshuffle(WQ_2d)

        scale_block_k = block_k
        tile_m, tile_n, tile_k = _select_tile_config(M, N, K, scale_block_k)

        launcher = _get_compiled_kernel(M, N, K, tile_m, tile_n, tile_k, scale_block_k)

        x_scale_flat = x_scale.contiguous().view(-1)
        w_scale_flat = w_scale.contiguous().view(-1)

        run_compiled(
            launcher,
            Y_2d, XQ_2d, WQ_shuf, x_scale_flat, w_scale_flat,
            M, N, torch.cuda.current_stream(),
        )

        return Y
