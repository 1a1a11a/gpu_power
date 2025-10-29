#!/usr/bin/env python3

"""Coordinate GPU power/clock sweeps and TFLOPs benchmarking."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch

import scripts.power_control as power_control
from scripts.benchmark_tflops import BenchmarkResult, benchmark


@dataclass
class SweepConfig:
    power_limit: Optional[float]
    sm_clock: Optional[int]
    mem_clock: Optional[int]


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


def _parse_list(values: Sequence[float | int]) -> List[float | int]:
    return list(values)


def _resolve_value(values: Optional[List[float | int]], index: int) -> Optional[float | int]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    if index < len(values):
        return values[index]
    raise ValueError("List length mismatch between power limits and clocks")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--gpu", type=int, default=0, help="GPU index to control")
    parser.add_argument(
        "--power-limits",
        type=float,
        nargs="*",
        default=[],
        help="Power limit targets in watts (omit to skip power changes)",
    )
    parser.add_argument(
        "--sm-clocks",
        type=int,
        nargs="*",
        default=[],
        help="SM clock targets (MHz); len must be 1 or match power limits",
    )
    parser.add_argument(
        "--mem-clocks",
        type=int,
        nargs="*",
        default=[],
        help="Memory clock targets (MHz); len must be 1 or match power limits",
    )
    parser.add_argument("--settle-seconds", type=float, default=3.0, help="Delay after applying limits")
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
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("power_sweep.csv"),
        help="CSV file to append results to",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag recorded with each measurement row",
    )

    return parser.parse_args()


def _ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for benchmarking")


def _write_csv_header(path: Path, fieldnames: Sequence[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def _append_csv(path: Path, fieldnames: Sequence[str], row: Dict[str, object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)


def _format_result(result: BenchmarkResult | None) -> Dict[str, Optional[float]]:
    if result is None:
        return {"avg_ms": None, "tflops": None, "error": "not run"}
    payload: Dict[str, Optional[float | str]] = {
        "avg_ms": result.avg_ms,
        "tflops": result.tflops,
    }
    if result.error is not None:
        payload["error"] = result.error
    return payload  # type: ignore[return-value]


def run_sweep(args: argparse.Namespace) -> None:
    _ensure_cuda()

    power_limits = _parse_list(args.power_limits)
    if not power_limits:
        # Guarantee at least one pass even when no power adjustments requested.
        power_limits = [None]  # type: ignore[list-item]

    sm_clocks = _parse_list(args.sm_clocks)
    mem_clocks = _parse_list(args.mem_clocks)

    fieldnames = [
        "timestamp",
        "tag",
        "gpu_index",
        "power_limit_w",
        "sm_clock_target_mhz",
        "mem_clock_target_mhz",
        "power_samples",
        "avg_power_w",
        "min_power_w",
        "max_power_w",
        "measured_sm_clock_mhz",
        "measured_mem_clock_mhz",
        "measured_temperature_c",
        "bf16_tflops",
        "bf16_avg_ms",
        "bf16_error",
        "fp8_tflops",
        "fp8_avg_ms",
        "fp8_error",
        "fp4_tflops",
        "fp4_avg_ms",
        "fp4_error",
        "raw_json",
    ]

    _write_csv_header(args.output, fieldnames)

    for idx, power_limit in enumerate(power_limits):
        sm_clock = _resolve_value(sm_clocks, idx)
        mem_clock = _resolve_value(mem_clocks, idx)

        if power_limit is not None:
            power_control.set_power_limit(power_limit, gpu_index=args.gpu)
        if sm_clock is not None:
            power_control.set_sm_clock(sm_clock, gpu_index=args.gpu)
        if mem_clock is not None:
            power_control.set_mem_clock(mem_clock, gpu_index=args.gpu)

        if args.settle_seconds > 0:
            time.sleep(args.settle_seconds)

        sampler = PowerSampler(interval_s=args.sample_interval, gpu_index=args.gpu)
        sampler.start()

        try:
            with torch.cuda.device(args.gpu):
                results = benchmark(
                    args.dtypes,
                    args.m,
                    args.n,
                    args.k,
                    args.warmup,
                    args.iters,
                )
        finally:
            sampler.stop()

        stats = sampler.stats()
        telemetry = power_control.query_telemetry(gpu_index=args.gpu)

        # Build CSV row.
        payload: Dict[str, object] = {
            "timestamp": time.time(),
            "tag": args.tag,
            "gpu_index": args.gpu,
            "power_limit_w": float(power_limit) if power_limit is not None else None,
            "sm_clock_target_mhz": int(sm_clock) if sm_clock is not None else None,
            "mem_clock_target_mhz": int(mem_clock) if mem_clock is not None else None,
            "measured_sm_clock_mhz": telemetry.sm_clock_mhz,
            "measured_mem_clock_mhz": telemetry.mem_clock_mhz,
            "measured_temperature_c": telemetry.temperature_c,
            **stats,
        }

        # Insert benchmark results per dtype.
        for dtype in ("bf16", "fp8", "fp4"):
            result = results.get(dtype)
            payload[f"{dtype}_tflops"] = result.tflops if result else None
            payload[f"{dtype}_avg_ms"] = result.avg_ms if result else None
            payload[f"{dtype}_error"] = result.error if result and result.error else None

        payload["raw_json"] = json.dumps(
            {
                "config": {
                    "power_limit": power_limit,
                    "sm_clock": sm_clock,
                    "mem_clock": mem_clock,
                },
                "results": {
                    dtype: res.to_dict() for dtype, res in results.items()
                },
                "telemetry": {
                    "avg_power": stats["avg_power_w"],
                    "samples": stats["power_samples"],
                    "snapshot": telemetry.__dict__,
                },
            }
        )

        _append_csv(args.output, fieldnames, payload)
        print(json.dumps(payload, indent=2))


def main() -> int:
    args = _parse_args()

    try:
        run_sweep(args)
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())


