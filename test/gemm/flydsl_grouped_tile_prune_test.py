# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""The FlyDSL grouped GEMM autotune prune, checked without a GPU.

The prune asks the kernel factory's LDS validators which tiles fit the device,
so the device is chosen here by patching the arch query rather than by running
on it.
"""

import unittest
from unittest import mock

from mslk.flydsl.common import is_flydsl_available
from parameterized import parameterized

# Divisible by every candidate tile_n and tile_k, so the divisibility prune
# keeps everything and only the LDS filter acts.
_EVERY_TILE_DIVIDES = 4096

# The arch query _fits_lds imports at call time, patched where it is defined.
_ARCH_QUERY = "flydsl.runtime.device.get_rocm_arch"


def _candidates(tiles):
    from flydsl.autotune import Config

    return [Config(**tile) for tile in tiles]


def _named_args(
    *, b_preshuffled, in_dtype, n=_EVERY_TILE_DIVIDES, k=_EVERY_TILE_DIVIDES
):
    return {"n": n, "k": k, "b_preshuffled": b_preshuffled, "in_dtype": in_dtype}


def _fits_factory_budget(config, *, b_preshuffled, in_dtype, arch):
    from mslk.flydsl.kernels.gemm.fp8_grouped_gemm_common import (
        ELEM_BYTES,
        validate_lds_budget_plain,
        validate_lds_budget_preshuffle,
    )

    validate_lds_budget = (
        validate_lds_budget_preshuffle if b_preshuffled else validate_lds_budget_plain
    )
    try:
        validate_lds_budget(
            tile_m=config.kwargs["tile_m"],
            tile_n=config.kwargs["tile_n"],
            tile_k=config.kwargs["tile_k"],
            elem_bytes=ELEM_BYTES[in_dtype],
            arch=arch,
        )
    except ValueError:
        return False
    return True


@unittest.skipUnless(is_flydsl_available(), "requires FlyDSL")
class GroupedTilePruneTest(unittest.TestCase):
    @parameterized.expand(
        [
            # gfx942 has 64 KiB of LDS per workgroup, gfx950 160 KiB.
            (
                "gfx942_rowwise_preshuffled_fp8",
                "gfx942",
                "ROWWISE_TILES",
                True,
                "fp8",
                84,
            ),
            ("gfx942_rowwise_plain_fp8", "gfx942", "ROWWISE_TILES", False, "fp8", 48),
            (
                "gfx942_unscaled_plain_bf16",
                "gfx942",
                "UNSCALED_TILES",
                False,
                "bf16",
                72,
            ),
            (
                "gfx950_rowwise_preshuffled_fp8",
                "gfx950",
                "ROWWISE_TILES",
                True,
                "fp8",
                108,
            ),
            (
                "gfx950_unscaled_plain_bf16",
                "gfx950",
                "UNSCALED_TILES",
                False,
                "bf16",
                156,
            ),
        ]
    )
    def test_kept_count_per_arch(
        self, _name, arch, tile_set, b_preshuffled, in_dtype, expected_kept
    ) -> None:
        from mslk.gemm.flydsl import grouped_dispatch

        candidates = _candidates(getattr(grouped_dispatch, tile_set))
        named_args = _named_args(b_preshuffled=b_preshuffled, in_dtype=in_dtype)
        with mock.patch(_ARCH_QUERY, return_value=arch):
            kept = grouped_dispatch._prune_tiles(candidates, named_args)

        self.assertEqual(len(kept), expected_kept)

    @parameterized.expand(
        [
            ("preshuffled_fp8", True, "fp8"),
            ("plain_fp8", False, "fp8"),
            ("preshuffled_bf16", True, "bf16"),
            ("plain_bf16", False, "bf16"),
        ]
    )
    def test_kept_and_dropped_agree_with_factory_validator(
        self, _name, b_preshuffled, in_dtype
    ) -> None:
        from mslk.gemm.flydsl import grouped_dispatch

        arch = "gfx942"
        candidates = _candidates(grouped_dispatch.UNSCALED_TILES)
        named_args = _named_args(b_preshuffled=b_preshuffled, in_dtype=in_dtype)
        with mock.patch(_ARCH_QUERY, return_value=arch):
            kept = grouped_dispatch._prune_tiles(candidates, named_args)

        kept_ids = {id(config) for config in kept}
        for config in candidates:
            with self.subTest(tile=config.kwargs):
                self.assertEqual(
                    id(config) in kept_ids,
                    _fits_factory_budget(
                        config,
                        b_preshuffled=b_preshuffled,
                        in_dtype=in_dtype,
                        arch=arch,
                    ),
                )

    def test_full_list_when_nothing_fits(self) -> None:
        from mslk.gemm.flydsl import grouped_dispatch

        candidates = _candidates(grouped_dispatch.ROWWISE_TILES)
        named_args = _named_args(b_preshuffled=True, in_dtype="fp8")
        no_lds = mock.patch(
            "mslk.flydsl.kernels.gemm.fp8_grouped_gemm_common.lds_capacity_bytes",
            return_value=0,
        )
        with mock.patch(_ARCH_QUERY, return_value="gfx942"), no_lds:
            kept = grouped_dispatch._prune_tiles(candidates, named_args)

        self.assertEqual(kept, candidates)

    def test_divisibility_still_applies_after_lds_filter(self) -> None:
        from mslk.gemm.flydsl import grouped_dispatch

        arch = "gfx942"
        candidates = _candidates(grouped_dispatch.ROWWISE_TILES)
        # 192 is divisible by tile_n=64 only, and 128 by tile_k=128 only.
        named_args = _named_args(b_preshuffled=False, in_dtype="fp8", n=192, k=128)
        with mock.patch(_ARCH_QUERY, return_value=arch):
            kept = grouped_dispatch._prune_tiles(candidates, named_args)

        self.assertTrue(kept)
        for config in kept:
            with self.subTest(tile=config.kwargs):
                self.assertEqual(config.kwargs["tile_n"], 64)
                self.assertEqual(config.kwargs["tile_k"], 128)
                self.assertTrue(
                    _fits_factory_budget(
                        config, b_preshuffled=False, in_dtype="fp8", arch=arch
                    )
                )


if __name__ == "__main__":
    unittest.main()
