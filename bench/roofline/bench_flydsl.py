#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlyDSL GEMM benchmark for roofline profiling.

Times one registered gemm_ops class (--kernel) per shape against empirical
roofline ceilings and the target matrix. Needs the full MSLK build, because
gemm_ops loads the C++ ops library.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import click
import numpy as np
import pandas as pd
import torch
import triton
from mslk.bench.gemm.gemm_bench import _COMPUTE_DTYPE_TO_ROOFLINE_KEY
from mslk.bench.gemm.gemm_ops import ComputeDtype, GemmOpBase, get_gemm_ops
from mslk.bench.roofline.capture import EmpiricalRoofline, load_empirical_roofline
from mslk.bench.roofline.regime import CeilingType
from mslk.bench.roofline.targets import (
    classify_with_target,
    load_target_matrix,
    lookup_target,
    save_target_matrix,
    TargetMatrix,
)

# MFMA peaks used when the roofline JSON has none for the op's compute dtype.
# Dtypes absent here require --roofline-json.
DATASHEET_MFMA_PEAK_TFLOPS = {ComputeDtype.FP8: 4600.0}


def find_gemm_op(class_name: str) -> GemmOpBase:
    ops_by_class_name = {type(op).__name__: op for op in get_gemm_ops()}
    if class_name not in ops_by_class_name:
        raise click.BadParameter(f"no registered gemm_ops class {class_name}")
    return ops_by_class_name[class_name]


def mfma_peak_tflops(
    compute_dtype: ComputeDtype, roofline: Optional[EmpiricalRoofline]
) -> float:
    roofline_key = _COMPUTE_DTYPE_TO_ROOFLINE_KEY[compute_dtype]
    if roofline and roofline_key in roofline.mfma_peak_tflops:
        return roofline.mfma_peak_tflops[roofline_key]
    if compute_dtype in DATASHEET_MFMA_PEAK_TFLOPS:
        return DATASHEET_MFMA_PEAK_TFLOPS[compute_dtype]
    raise ValueError(f"roofline JSON has no '{roofline_key}' MFMA peak")


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def moved_bytes(gemm_op: GemmOpBase, quantized: tuple, output: torch.Tensor) -> int:
    """Activation, weight and output bytes, as gemm_bench counts them. FP8 also
    counts its two row-scale vectors, keeping the existing FP8 baselines
    comparable."""
    activation, weight = quantized[0], quantized[1]
    total = tensor_bytes(activation) + tensor_bytes(weight) + tensor_bytes(output)
    if gemm_op.compute_dtype == ComputeDtype.FP8:
        activation_scale, weight_scale = quantized[2], quantized[3]
        total += tensor_bytes(activation_scale) + tensor_bytes(weight_scale)
    return total


@dataclass
class BenchResult:
    kernel: str
    M: int
    N: int
    K: int
    ms: float
    median_ms: float
    tflops: float
    gbps: float
    regime: str
    ceiling_type: str
    ceiling_value: float
    achieved_pct: float
    target_pct: float
    target_pass: Optional[bool]
    cov_pct: float
    n_iters: int


def bench_gemm_op(
    gemm_op: GemmOpBase,
    M: int,
    N: int,
    K: int,
    roofline: Optional[EmpiricalRoofline],
    target_matrix: Optional[TargetMatrix],
    n_iterations: int = 5,
    rep_ms: int = 200,
    device: int = 0,
) -> BenchResult:
    """Benchmark one gemm_ops class for a single shape."""
    torch.cuda.set_device(device)
    kernel_name = type(gemm_op).__name__

    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    # Same preparation order as gemm_bench.benchmark.
    quantized = gemm_op.quantize(*gemm_op.preprocess(A, B))

    # Warmup
    for _ in range(3):
        output = gemm_op.compute(*quantized)
    torch.cuda.synchronize()

    # Benchmark N iterations
    timings = []
    for _ in range(n_iterations):
        ms = triton.testing.do_bench(lambda: gemm_op.compute(*quantized), rep=rep_ms)
        timings.append(ms)

    arr = np.array(timings)
    median_ms = float(np.median(arr))
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr))
    cov_pct = (std_ms / mean_ms * 100) if mean_ms > 0 else 0.0

    tflops = 2 * M * N * K / (median_ms / 1e3) / 1e12
    gbps = moved_bytes(gemm_op, quantized, output) / (median_ms / 1e3) / 1e9

    # Regime, ceiling and pass mark: the target row's if the matrix has one.
    rc, _target = classify_with_target(target_matrix, kernel_name, M, N, K)

    # Ceiling and achieved %
    ceiling_value = 0.0
    achieved_pct = 0.0
    if rc.ceiling_type == CeilingType.MFMA_PEAK:
        ceiling_value = mfma_peak_tflops(gemm_op.compute_dtype, roofline)
        achieved_pct = (tflops / ceiling_value) * 100 if ceiling_value > 0 else 0
    elif rc.ceiling_type == CeilingType.HBM_BW:
        if roofline and roofline.hbm_bw_gbps > 0:
            ceiling_value = roofline.hbm_bw_gbps
        else:
            ceiling_value = triton.testing.get_dram_gbps()
        achieved_pct = (gbps / ceiling_value) * 100 if ceiling_value > 0 else 0

    target_pct = rc.target_low
    target_pass = None
    if achieved_pct > 0:
        target_pass = achieved_pct >= target_pct

    return BenchResult(
        kernel=kernel_name,
        M=M,
        N=N,
        K=K,
        ms=mean_ms,
        median_ms=median_ms,
        tflops=tflops,
        gbps=gbps,
        regime=rc.regime.value,
        ceiling_type=rc.ceiling_type.value,
        ceiling_value=ceiling_value,
        achieved_pct=achieved_pct,
        target_pct=target_pct,
        target_pass=target_pass,
        cov_pct=cov_pct,
        n_iters=n_iterations,
    )


@click.command()
@click.option(
    "--kernel",
    "kernel_class_name",
    default="FP8RowwisePreshuffleFlyDSL",
    help="gemm_ops class to benchmark, e.g. BF16Int4RowwiseFlyDSL",
)
@click.option("--roofline-json", default=None, type=click.Path(exists=True))
@click.option("--target-template", default=None, type=click.Path(exists=True))
@click.option("--output-dir", default="/tmp/mslk_flydsl_bench")
@click.option("--M", "m_vals", default="1,16,128,1024,4096,8192")
@click.option("--N", "n_vals", default="8192")
@click.option("--K", "k_vals", default="8192")
@click.option("--stat-iterations", default=5, type=int)
@click.option("--device", default=0, type=int)
@click.option(
    "--shapes",
    default=None,
    help="Preset: llama3_70b, llama3_405b, roofline_validation",
)
def main(
    kernel_class_name,
    roofline_json,
    target_template,
    output_dir,
    m_vals,
    n_vals,
    k_vals,
    stat_iterations,
    device,
    shapes,
):
    gemm_op = find_gemm_op(kernel_class_name)
    if not gemm_op.supported:
        raise click.UsageError(f"{kernel_class_name} is not supported on this device")
    if (
        roofline_json is None
        and gemm_op.compute_dtype not in DATASHEET_MFMA_PEAK_TFLOPS
    ):
        raise click.UsageError(
            f"{kernel_class_name} has no datasheet MFMA peak: pass --roofline-json"
        )
    os.makedirs(output_dir, exist_ok=True)

    roofline = load_empirical_roofline(roofline_json) if roofline_json else None
    target_matrix = None
    if target_template:
        target_matrix = load_target_matrix(target_template)

    # Build shape list
    if shapes == "llama3_70b":
        shape_list = []
        for m in [1, 16, 32, 64, 128]:
            shape_list += [(m, 1280, 8192), (m, 7168, 8192), (m, 8192, 3584)]
    elif shapes == "llama3_405b":
        shape_list = []
        for m in [1, 128]:
            shape_list += [(m, 13312, 6656), (m, 16384, 16384)]
    elif shapes == "roofline_validation":
        shape_list = [
            (8192, 8192, 8192),
            (4096, 4096, 4096),
            (1024, 8192, 8192),
            (128, 8192, 8192),
            (16, 8192, 8192),
            (1, 8192, 8192),
        ]
    else:
        Ms = [int(x) for x in m_vals.split(",")]
        Ns = [int(x) for x in n_vals.split(",")]
        Ks = [int(x) for x in k_vals.split(",")]
        shape_list = [(m, n, k) for m in Ms for n in Ns for k in Ks]

    if roofline:
        roofline_key = _COMPUTE_DTYPE_TO_ROOFLINE_KEY[gemm_op.compute_dtype]
        peak = roofline.mfma_peak_tflops.get(roofline_key, 0)
        print(
            f"Empirical roofline: {roofline_key.upper()} MFMA={peak:.1f} TFLOPS, "
            f"HBM={roofline.hbm_bw_gbps:.1f} GB/s"
        )
    shape_count = len(shape_list)
    print(f"Benchmarking {shape_count} shapes with {stat_iterations} iterations each\n")

    # Header
    print(
        f"{'Shape':<25} {'Regime':<6} {'Ms':<10} {'TFLOPS':<10} {'GB/s':<10} "
        f"{'Ceiling':<15} {'Achieved%':<10} {'Target%':<9} {'Pass':<5} {'CoV%':<7}"
    )
    print("-" * 120)

    results = []
    for M, N, K in shape_list:
        try:
            r = bench_gemm_op(
                gemm_op,
                M,
                N,
                K,
                roofline,
                target_matrix,
                n_iterations=stat_iterations,
                device=device,
            )
            pass_str = (
                "PASS" if r.target_pass else ("FAIL" if r.target_pass is False else "")
            )
            ceiling_str = f"{r.ceiling_type}={r.ceiling_value:.0f}"
            shape_str = f"({r.M},{r.N},{r.K})"
            print(
                f"{shape_str:<25} "
                f"{r.regime:<6} {r.median_ms:<10.3f} {r.tflops:<10.2f} {r.gbps:<10.2f} "
                f"{ceiling_str:<15} {r.achieved_pct:<10.2f} {r.target_pct:<9.1f} "
                f"{pass_str:<5} {r.cov_pct:<7.2f}"
            )
            results.append(r)
        except (RuntimeError, ValueError) as error:
            print(f"({M},{N},{K}) FAILED: {error}")

    # Export CSV
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"flydsl_bench_{timestamp}.csv")
        rows = [
            {
                "kernel": r.kernel,
                "M": r.M,
                "N": r.N,
                "K": r.K,
                "median_ms": r.median_ms,
                "tflops": r.tflops,
                "gbps": r.gbps,
                "regime": r.regime,
                "ceiling_type": r.ceiling_type,
                "ceiling_value": r.ceiling_value,
                "achieved_pct": r.achieved_pct,
                "target_pct": r.target_pct,
                "target_pass": r.target_pass,
                "cov_pct": r.cov_pct,
                "n_iters": r.n_iters,
            }
            for r in results
        ]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\nCSV saved to {csv_path}")

        # Summary
        passed = sum(1 for r in results if r.target_pass is True)
        failed = sum(1 for r in results if r.target_pass is False)
        total = passed + failed
        print(f"\nSummary: {passed}/{total} targets met")

        # Update target matrix if provided
        if target_matrix:
            for r in results:
                t = lookup_target(target_matrix, r.kernel, r.M, r.N, r.K)
                if t:
                    t.baseline_achieved_pct = r.achieved_pct
                    t.ceiling_value = r.ceiling_value
            matrix_path = os.path.join(output_dir, f"target_matrix_{timestamp}.yaml")
            save_target_matrix(target_matrix, matrix_path)
            print(f"Updated target matrix saved to {matrix_path}")


if __name__ == "__main__":
    main()
