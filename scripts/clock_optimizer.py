#!/usr/bin/env python3

"""
GPU clock frequency optimization for power sweeps.
Finds optimal SM and memory clock combinations for each power limit.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import threading

from . import power_control
from .benchmark_tflops import BenchmarkResult, benchmark


class PowerSampler:
    """Background sampler that polls ``nvidia-smi`` for power draw."""

    def __init__(self, *, interval_s: float, gpu_index: int) -> None:
        self.interval_s = max(interval_s, 0.05)
        self.gpu_index = gpu_index
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: List[float] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("PowerSampler already started")

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return

        self._stop.set()
        self._thread.join()
        self._thread = None

    def stats(self) -> Dict[str, float]:
        with self._lock:
            samples = list(self._samples)

        if not samples:
            raise RuntimeError("No power samples collected")

        avg = sum(samples) / len(samples)
        return {
            "power_samples": len(samples),
            "avg_power_w": avg,
            "min_power_w": min(samples),
            "max_power_w": max(samples),
        }

    def _loop(self) -> None:
        next_deadline = time.perf_counter()
        while not self._stop.is_set():
            try:
                telemetry = power_control.query_telemetry(gpu_index=self.gpu_index)
            except RuntimeError as exc:
                # Log once then keep trying.
                print(f"warning: telemetry query failed ({exc})", file=sys.stderr)
                telemetry = None

            if telemetry is not None and telemetry.power_w is not None:
                with self._lock:
                    self._samples.append(telemetry.power_w)

            next_deadline += self.interval_s
            delay = max(0.0, next_deadline - time.perf_counter())
            self._stop.wait(delay)


def query_supported_clocks(gpu_index: int = 0) -> Tuple[List[int], List[int]]:
    """Query the GPU for supported clock combinations using nvidia-smi.

    Args:
        gpu_index: GPU index to query

    Returns:
        Tuple of (sm_clocks, mem_clocks) lists in MHz
    """
    try:
        # Run nvidia-smi to get supported clocks
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu_index),
                "--query-supported-clocks=memory,graphics",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        sm_clocks = set()
        mem_clocks = set()

        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = line.split(",")
                if len(parts) == 2:
                    try:
                        mem_clock = int(parts[0].strip())
                        sm_clock = int(parts[1].strip())
                        mem_clocks.add(mem_clock)
                        sm_clocks.add(sm_clock)
                    except ValueError:
                        continue

        return sorted(list(sm_clocks)), sorted(list(mem_clocks))

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to reasonable defaults if nvidia-smi fails
        print("Warning: Could not query supported clocks, using defaults")
        return (
            list(range(2100, 3100, 100)),  # SM clocks: 2100-3000 in 100MHz steps
            [810, 7001, 13365, 14001],  # Common memory clocks
        )


def generate_clock_combinations(
    sm_range: Optional[List[int]] = None,
    mem_range: Optional[List[int]] = None,
    gpu_index: int = 0,
    max_combinations: int = 0,
) -> List[Tuple[int, int]]:
    """Generate combinations of SM and memory clocks.

    Args:
        sm_range: Optional [min, max, step] for SM clocks (MHz). If None, query from GPU.
        mem_range: Optional [min, max, step] for memory clocks (MHz). If None, query from GPU.
        gpu_index: GPU index to query for supported clocks
        max_combinations: Maximum number of combinations to return (0 = all)

    Returns:
        List of (sm_clock, mem_clock) tuples
    """
    if sm_range is None or mem_range is None:
        # Query actual supported clocks from the GPU
        supported_sm, supported_mem = query_supported_clocks(gpu_index)

        # Use supported clocks if available, otherwise fall back to reasonable ranges
        if supported_sm:
            sm_clocks = supported_sm
        else:
            raise ValueError("No supported SM clocks found")

        if supported_mem:
            mem_clocks = supported_mem
        else:
            raise ValueError("No supported memory clocks found")

    else:
        # Use provided ranges
        sm_min, sm_max, sm_step = sm_range
        mem_min, mem_max, mem_step = mem_range

        sm_clocks = list(range(sm_min, sm_max + 1, sm_step))
        mem_clocks = list(range(mem_min, mem_max + 1, mem_step))

    # Generate all combinations
    combinations = []
    for sm_clock in sm_clocks:
        for mem_clock in mem_clocks:
            combinations.append((sm_clock, mem_clock))

    # Limit combinations if requested
    if max_combinations > 0 and len(combinations) > max_combinations:
        # Use statistical percentiles instead of random sampling
        combinations = select_representative_combinations(
            sm_clocks, mem_clocks, max_combinations
        )
        combinations.sort()  # Sort for consistent ordering

    return combinations


def select_representative_combinations(
    sm_clocks: List[int], mem_clocks: List[int], num_combinations: int
) -> List[Tuple[int, int]]:
    """Select representative clock combinations using statistical percentiles.

    Args:
        sm_clocks: List of available SM clock frequencies
        mem_clocks: List of available memory clock frequencies
        num_combinations: Number of combinations to select

    Returns:
        List of representative (sm_clock, mem_clock) tuples
    """

    options_per_var = int(np.ceil(np.sqrt(num_combinations)))
    num_points = options_per_var + 1
    percentile_steps = np.linspace(0, 100, num_points)

    # Calculate percentiles for SM clocks
    sm_percentiles = np.percentile(sm_clocks, percentile_steps)
    sm_representative = [int(round(p)) for p in sm_percentiles]

    # Calculate percentiles for memory clocks
    mem_percentiles = np.percentile(mem_clocks, percentile_steps)
    mem_representative = [int(round(p)) for p in mem_percentiles]

    # Remove duplicates while preserving order
    sm_representative = list(dict.fromkeys(sm_representative))
    mem_representative = list(dict.fromkeys(mem_representative))
    combinations = [(sm, mem) for sm in sm_representative for mem in mem_representative]

    if len(combinations) > num_combinations:
        step = len(combinations) / num_combinations
        indices = [int(i * step) for i in range(num_combinations)]
        combinations = [combinations[i] for i in indices]

    return combinations


def warmup_gpu(power_limit: Optional[float], args: argparse.Namespace) -> None:
    """Warm up the GPU to stable operating temperature.

    Args:
        power_limit: Power limit to set for warm-up
        args: Command line arguments
    """
    print(f"Warming up GPU for power limit {power_limit}W...")

    # Set power limit for warm-up
    if power_limit is not None:
        power_control.set_power_limit(power_limit, gpu_index=args.gpu)

    # Use default clocks for warm-up (should be stable)
    # Don't override clocks here - let GPU use its defaults for warm-up

    # Run warm-up benchmark iterations
    warmup_iters = getattr(
        args, "warmup_iters", 20
    )  # Use warmup_iters if available, else default

    try:
        with torch.cuda.device(args.gpu):
            benchmark(
                ["bf16"],  # Use just BF16 for warm-up
                args.m,
                args.n,
                args.k,
                0,  # No additional warmup in benchmark function
                warmup_iters,
            )
    except Exception as e:
        print(f"Warning: Warm-up failed ({e}), continuing anyway...")

    print("GPU warm-up complete.")


def optimize_clocks_for_power_limit(
    power_limit: float,
    clock_combinations: List[Tuple[int, int]],
    args: argparse.Namespace,
) -> Tuple[int, int, Dict[str, BenchmarkResult]]:
    """Find the best clock combination for a given power limit.

    Args:
        power_limit: Power limit in watts
        clock_combinations: List of (sm_clock, mem_clock) tuples to test
        args: Command line arguments

    Returns:
        Tuple of (best_sm_clock, best_mem_clock, best_results)
    """
    best_tflops = 0.0
    best_sm_clock = 0
    best_mem_clock = 0
    best_results = {}

    print(f"Optimizing clocks for power limit {power_limit}W...")

    for sm_clock, mem_clock in clock_combinations:
        print(f"  Testing SM={sm_clock:<8d}MHz, MEM={mem_clock:<8d}MHz: ", end="")

        # Set power limit
        if power_limit is not None:
            power_control.set_power_limit(power_limit, gpu_index=args.gpu)

        # Set clocks
        power_control.set_sm_clock(sm_clock, gpu_index=args.gpu)
        power_control.set_mem_clock(mem_clock, gpu_index=args.gpu)

        # Settle
        if args.settle_seconds > 0:
            time.sleep(args.settle_seconds)

        # Run benchmark
        sampler = PowerSampler(interval_s=args.sample_interval, gpu_index=args.gpu)
        sampler.start()

        try:
            with torch.cuda.device(args.gpu):
                results = benchmark(
                    args.dtypes,
                    args.m,
                    args.n,
                    args.k,
                    args.warmup_iters,
                    args.iters,
                )
        finally:
            sampler.stop()

        # Check if this is better
        bf16_result = results.get("bf16")
        bf16_tflops = bf16_result.tflops if bf16_result else None
        fp8_result = results.get("fp8")
        fp8_tflops = fp8_result.tflops if fp8_result else None
        total_tflops = (bf16_tflops or 0) + (fp8_tflops or 0)
        if (
            bf16_tflops is not None
            and fp8_tflops is not None
            and total_tflops > best_tflops
        ):
            best_tflops = total_tflops
            best_sm_clock = sm_clock
            best_mem_clock = mem_clock
            best_results = results

        print(f"BF16={bf16_tflops or 0:8.1f} + FP8={fp8_tflops or 0:8.1f} = {total_tflops:8.1f}")

    best_results_bf16 = (
        best_results.get("bf16").tflops if best_results.get("bf16") else 0.0
    )
    best_results_fp8 = (
        best_results.get("fp8").tflops if best_results.get("fp8") else 0.0
    )
    print(
        f"Best clocks for {power_limit}W: SM={best_sm_clock:<8d}MHz, MEM={best_mem_clock:<8d}MHz, BF16 TFLOPs={best_results_bf16:.1f}, FP8 TFLOPs={best_results_fp8:.1f}"
    )
    return best_sm_clock, best_mem_clock, best_results


def optimize_all_power_limits(
    power_limits: List[float],
    sm_range: List[int],
    mem_range: List[int],
    args: argparse.Namespace,
) -> List[Dict]:
    """Optimize clock frequencies for all power limits.

    Args:
        power_limits: List of power limits to optimize
        sm_range: SM clock range [min, max, step]
        mem_range: Memory clock range [min, max, step]
        args: Command line arguments

    Returns:
        List of optimization results dictionaries
    """
    # Generate clock combinations for optimization
    clock_combinations = generate_clock_combinations(
        sm_range, mem_range, args.gpu, getattr(args, "max_combinations", 0)
    )
    print(f"Testing {len(clock_combinations)} clock combinations per power limit")

    results = []

    for power_limit in power_limits:
        # Warm up GPU for this power limit (if enabled)
        if getattr(args, "gpu_warmup", True):
            warmup_gpu(power_limit, args)

        # Find best clocks for this power limit
        best_sm_clock, best_mem_clock, benchmark_results = (
            optimize_clocks_for_power_limit(power_limit, clock_combinations, args)
        )

        # Helper function to safely get benchmark results
        def get_result(dtype: str, field: str) -> Optional[float]:
            result = benchmark_results.get(dtype)
            return getattr(result, field) if result else None

        # Store results
        result = {
            "power_limit_w": power_limit,
            "optimal_sm_clock_mhz": best_sm_clock,
            "optimal_mem_clock_mhz": best_mem_clock,
            "bf16_tflops": get_result("bf16", "tflops") or 0.0,
            "fp8_tflops": get_result("fp8", "tflops") or 0.0,
            "bf16_avg_ms": get_result("bf16", "avg_ms") or 0.0,
            "fp8_avg_ms": get_result("fp8", "avg_ms") or 0.0,
            "total_tflops": (get_result("bf16", "tflops") or 0.0) + (get_result("fp8", "tflops") or 0.0),
            "combinations_tested": len(clock_combinations),
        }

        results.append(result)

        print(
            f"Results: SM={best_sm_clock}MHz, MEM={best_mem_clock}MHz, Total={result['total_tflops']:.1f} TFLOPs\n"
        )

    return results


def save_optimization_results(results: List[Dict], output_path: Path, tag: str = ""):
    """Save optimization results to CSV file.

    Args:
        results: List of optimization result dictionaries
        output_path: Path to output CSV file
        tag: Optional tag to include in results
    """
    fieldnames = [
        "power_limit_w",
        "optimal_sm_clock_mhz",
        "optimal_mem_clock_mhz",
        "bf16_tflops",
        "fp8_tflops",
        "bf16_avg_ms",
        "fp8_avg_ms",
        "total_tflops",
        "combinations_tested",
    ]

    # Write CSV header
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Optimization results saved to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for clock optimization."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--gpu", type=int, default=0, help="GPU index to control")
    parser.add_argument(
        "--power-limits",
        type=float,
        nargs="*",
        default=[600, 480, 400, 320, 280, 240],
        help="Power limit targets in watts",
    )
    parser.add_argument(
        "--sm-clock-range",
        type=int,
        nargs=3,
        metavar=("MIN", "MAX", "STEP"),
        help="SM clock range for optimization: min max step. If not specified, queries GPU for supported clocks.",
    )
    parser.add_argument(
        "--mem-clock-range",
        type=int,
        nargs=3,
        metavar=("MIN", "MAX", "STEP"),
        help="Memory clock range for optimization: min max step. If not specified, queries GPU for supported clocks.",
    )
    parser.add_argument(
        "--settle-seconds", type=float, default=3.0, help="Delay after applying limits"
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.2,
        help="Telemetry sampling interval during benchmark (seconds)",
    )
    parser.add_argument(
        "--dtypes",
        nargs="*",
        default=["bf16", "fp8"],
        help="Numeric dtypes to benchmark",
    )
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    # parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--gpu-warmup",
        action="store_true",
        default=True,
        help="Enable GPU warm-up before optimization (default: enabled)",
    )
    parser.add_argument(
        "--no-gpu-warmup",
        action="store_false",
        dest="gpu_warmup",
        help="Disable GPU warm-up before optimization",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=100,
        help="Number of iterations for GPU warm-up (default: 100)",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=25,
        help="Maximum number of clock combinations to test using statistical percentiles (0 = test all, default: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("clock_optimization_results.csv"),
        help="Output CSV file",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag recorded with each measurement row",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for standalone clock optimization."""
    args = parse_args()

    # Ensure CUDA is available
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for benchmarking")

    # Run optimization
    results = optimize_all_power_limits(
        args.power_limits, args.sm_clock_range, args.mem_clock_range, args
    )

    # Save results
    save_optimization_results(results, args.output, args.tag)

    # Print summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)
    print(
        f"{'Power (W)':<10} {'SM (MHz)':<10} {'Mem (MHz)':<12} {'BF16 TFLOPs':<12} {'FP8 TFLOPs':<12}"
    )
    print("-" * 80)

    for result in results:
        print(
            f"{result['power_limit_w']:<10.0f} {result['optimal_sm_clock_mhz']:<10} "
            f"{result['optimal_mem_clock_mhz']:<12} {result['bf16_tflops']:<12.1f} {result['fp8_tflops']:<12.1f}"
        )

    print("=" * 80)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
