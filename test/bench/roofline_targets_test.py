# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from mslk.bench.roofline.regime import CeilingType, classify_regime, Regime
from mslk.bench.roofline.targets import (
    classify_with_target,
    KernelTarget,
    TargetMatrix,
)


def matrix_with_row(**row_fields: object) -> TargetMatrix:
    row = KernelTarget(
        kernel_name="BF16Int4RowwiseFlyDSL",
        M=128,
        N=13312,
        K=6656,
        regime="R2",
        ceiling_type="mfma_peak",
        target_pct=42.0,
    )
    for name, value in row_fields.items():
        setattr(row, name, value)
    return TargetMatrix(gpu_arch="gfx950", targets=[row])


class ClassifyWithTargetTest(unittest.TestCase):
    def test_row_overrides_shape_rule(self) -> None:
        # classify_regime puts M <= 128 in R3 against HBM; the row says MFMA peak.
        self.assertEqual(
            classify_regime(128, 13312, 6656).ceiling_type, CeilingType.HBM_BW
        )
        classification, target = classify_with_target(
            matrix_with_row(), "BF16Int4RowwiseFlyDSL", 128, 13312, 6656
        )
        self.assertIsNotNone(target)
        self.assertEqual(classification.regime, Regime.R2_MEDIUM)
        self.assertEqual(classification.ceiling_type, CeilingType.MFMA_PEAK)
        self.assertEqual(classification.target_low, 42.0)

    def test_no_row_keeps_shape_rule(self) -> None:
        for kernel, m in (("OtherKernel", 128), ("BF16Int4RowwiseFlyDSL", 64)):
            classification, target = classify_with_target(
                matrix_with_row(), kernel, m, 13312, 6656
            )
            self.assertIsNone(target)
            self.assertEqual(classification, classify_regime(m, 13312, 6656))

    def test_no_matrix_keeps_shape_rule(self) -> None:
        classification, target = classify_with_target(
            None, "BF16Int4RowwiseFlyDSL", 128, 13312, 6656
        )
        self.assertIsNone(target)
        self.assertEqual(classification, classify_regime(128, 13312, 6656))


if __name__ == "__main__":
    unittest.main()
