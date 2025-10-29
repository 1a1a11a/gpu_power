#!/usr/bin/env python3

"""Benchmark INT8 / FP4 / FP8 / BF16 throughput using PyTorch + Transformer Engine."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

try:
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch import fp8 as te_fp8
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Transformer Engine is required for FP4/FP8 benchmarking. "
        "Install via `pip install transformer-engine`."
    ) from exc


@dataclass
class BenchmarkResult:
    dtype: str
    avg_ms: Optional[float]
    tflops: Optional[float]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, float | str | None]:
        payload: Dict[str, float | str | None] = {
            "avg_ms": self.avg_ms,
            "tflops": self.tflops,
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload


def _measure(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # milliseconds


def _calc_tflops(m: int, n: int, k: int, ms: float) -> float:
    return (2.0 * m * n * k) / (ms * 1.0e9)  # convert ms to s -> TFLOPs


def _bench_bf16(m: int, n: int, k: int, warmup: int, iters: int) -> BenchmarkResult:
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)

    def run() -> torch.Tensor:
        return torch.matmul(a, b)

    avg_ms = _measure(run, warmup, iters)
    return BenchmarkResult("bf16", avg_ms, _calc_tflops(m, n, k, avg_ms))


def _bench_int8(m: int, n: int, k: int, warmup: int, iters: int) -> BenchmarkResult:
    # Create FP32 model and quantize to INT8 using torchao (weight-only quantization)
    linear_fp32 = torch.nn.Linear(k, n, bias=False, device="cuda", dtype=torch.float32)
    linear_fp32.weight.data = torch.randn((n, k), device="cuda", dtype=torch.float32)

    # Use torchao for INT8 weight-only quantization (modifies in-place)
    # Weight-only quantization is much faster than dynamic quantization
    from torchao.quantization import quantize_, int8_weight_only
    quantize_(linear_fp32, int8_weight_only())
    # linear_fp32 is now quantized in-place

    # Use BF16 inputs for optimal INT8 performance (better kernel fusion with quantized weights)
    inp = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)

    def run() -> torch.Tensor:
        with torch.inference_mode():
            return linear_fp32(inp)

    avg_ms = _measure(run, warmup, iters)
    return BenchmarkResult("int8", avg_ms, _calc_tflops(m, n, k, avg_ms))


def _make_fp8_recipe(dtype: str):
    if dtype == "fp8":
        te_fp8.check_fp8_support()
        return te_fp8.get_default_fp8_recipe()

    if dtype == "fp4":
        try:
            te_fp8.check_nvfp4_support()
            return te_fp8.NVFP4BlockScaling()
        except Exception as e:
            raise RuntimeError(f"FP4 not supported on this GPU: {e}")

    raise ValueError(f"Unsupported FP8 recipe for dtype '{dtype}'")


def _bench_te_linear(
    dtype: str,
    m: int,
    n: int,
    k: int,
    warmup: int,
    iters: int,
) -> BenchmarkResult:
    recipe = _make_fp8_recipe(dtype)

    linear = te.Linear(
        k,
        n,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
    )

    inp = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)

    def run() -> torch.Tensor:
        with torch.inference_mode():
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                return linear(inp)

    avg_ms = _measure(run, warmup, iters)
    return BenchmarkResult(dtype, avg_ms, _calc_tflops(m, n, k, avg_ms))


def benchmark(
    dtypes: List[str],
    m: int,
    n: int,
    k: int,
    warmup: int,
    iters: int,
) -> Dict[str, BenchmarkResult]:
    results: Dict[str, BenchmarkResult] = {}

    for dtype in dtypes:
        dtype_norm = dtype.lower()
        try:
            if dtype_norm == "bf16":
                results[dtype_norm] = _bench_bf16(m, n, k, warmup, iters)
            elif dtype_norm == "int8":
                results[dtype_norm] = _bench_int8(m, n, k, warmup, iters)
            elif dtype_norm in {"fp8", "fp4"}:
                results[dtype_norm] = _bench_te_linear(dtype_norm, m, n, k, warmup, iters)
            else:
                raise SystemExit(f"Unsupported dtype '{dtype}'. Choose from bf16, int8, fp8, fp4.")
        except Exception as exc:
            results[dtype_norm] = BenchmarkResult(dtype_norm, None, None, str(exc))

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtypes",
        nargs="*",
        default=["bf16", "int8", "fp8"],
        help="Data types to benchmark (subset of bf16, int8, fp8, fp4)",
    )
    parser.add_argument("--m", type=int, default=8192, help="M dimension (batch size)")
    parser.add_argument("--n", type=int, default=8192, help="N dimension (output features)")
    parser.add_argument("--k", type=int, default=8192, help="K dimension (input features)")
    parser.add_argument("--warmup", type=int, default=20, help="Warm-up iterations")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save JSON results",
    )
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device is required for benchmarking.")

    args = _parse_args()
    dtypes = args.dtypes

    with torch.cuda.device(0):
        results = benchmark(dtypes, args.m, args.n, args.k, args.warmup, args.iters)

    payload = {
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": {name: result.to_dict() for name, result in results.items()},
    }

    print(json.dumps(payload, indent=2))

    if args.output is not None:
        path = args.output
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Saved results to {path}")


if __name__ == "__main__":
    main()

