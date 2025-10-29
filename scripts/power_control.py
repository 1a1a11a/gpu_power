#!/usr/bin/env python3

"""Helpers for manipulating NVIDIA GPU power / clock limits via ``nvidia-smi``.

The module exposes a small Python API geared towards automated benchmarking
workflows as well as a CLI wrapper for manual experimentation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass
class Telemetry:
    """Single-sample telemetry snapshot returned by :func:`query_telemetry`."""

    power_w: Optional[float]
    sm_clock_mhz: Optional[int]
    mem_clock_mhz: Optional[int]
    temperature_c: Optional[int]


def _run_nvidia_smi(
    args: Sequence[str],
    *,
    gpu_index: int = 0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    cmd = ["nvidia-smi", "-i", str(gpu_index), *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result


def set_power_limit(power_watts: float, *, gpu_index: int = 0) -> None:
    """Clamp the GPU's power limit to ``power_watts``."""

    _run_nvidia_smi(["-pl", str(int(power_watts))], gpu_index=gpu_index)


def set_sm_clock(min_mhz: int, max_mhz: Optional[int] = None, *, gpu_index: int = 0) -> None:
    """Limit SM/graphics clock to the provided range via ``-lgc``."""

    high = min_mhz if max_mhz is None else max_mhz
    _run_nvidia_smi(["-lgc", f"{int(min_mhz)},{int(high)}"], gpu_index=gpu_index)


def set_mem_clock(min_mhz: int, max_mhz: Optional[int] = None, *, gpu_index: int = 0) -> None:
    """Limit memory clock to the provided range via ``-lmc``."""

    high = min_mhz if max_mhz is None else max_mhz
    _run_nvidia_smi(["-lmc", f"{int(min_mhz)},{int(high)}"], gpu_index=gpu_index)


def set_application_clocks(mem_mhz: int, sm_mhz: int, *, gpu_index: int = 0) -> None:
    """Set application clocks using ``-ac`` (mem, graphics)."""

    _run_nvidia_smi(["-ac", f"{int(mem_mhz)},{int(sm_mhz)}"], gpu_index=gpu_index)


def reset_clocks(*, gpu_index: int = 0) -> None:
    """Reset all application clock limits to defaults."""

    _run_nvidia_smi(["-rac"], gpu_index=gpu_index)


def reset_power_limit(*, gpu_index: int = 0) -> None:
    """Reset power limit to the factory default."""

    _run_nvidia_smi(["-rgc"], gpu_index=gpu_index)
    _run_nvidia_smi(["-pl", "DEFAULT"], gpu_index=gpu_index)


def query_telemetry(
    *, gpu_index: int = 0, extra_fields: Optional[Iterable[str]] = None
) -> Telemetry:
    """Query a single telemetry sample via ``nvidia-smi --query-gpu``."""

    base_fields = [
        "power.draw",
        "clocks.sm",
        "clocks.mem",
        "temperature.gpu",
    ]

    if extra_fields is not None:
        fields = [*base_fields, *extra_fields]
    else:
        fields = base_fields

    result = _run_nvidia_smi(
        [
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ],
        gpu_index=gpu_index,
    )

    tokens = [token.strip() for token in result.stdout.strip().split(",")]
    if len(tokens) < 4:
        raise RuntimeError("Unexpected telemetry response: " + result.stdout)

    def _maybe_float(value: str) -> Optional[float]:
        return float(value) if value and value.lower() != "nan" else None

    def _maybe_int(value: str) -> Optional[int]:
        return int(float(value)) if value and value.lower() != "nan" else None

    telemetry = Telemetry(
        power_w=_maybe_float(tokens[0]),
        sm_clock_mhz=_maybe_int(tokens[1]),
        mem_clock_mhz=_maybe_int(tokens[2]),
        temperature_c=_maybe_int(tokens[3]),
    )

    return telemetry


def sample_power(
    duration_s: float,
    *,
    interval_s: float = 0.5,
    gpu_index: int = 0,
) -> Dict[str, float]:
    """Sample power draw periodically and return aggregates."""

    samples: List[float] = []
    deadline = time.time() + max(duration_s, 0.0)

    while True:
        telemetry = query_telemetry(gpu_index=gpu_index)
        if telemetry.power_w is not None:
            samples.append(telemetry.power_w)

        now = time.time()
        if now >= deadline:
            break

        sleep_for = max(interval_s, 0.0)
        time.sleep(min(sleep_for, max(0.0, deadline - now)))

    if not samples:
        raise RuntimeError("No power samples collected")

    avg = sum(samples) / len(samples)
    return {
        "samples": len(samples),
        "avg_power_w": avg,
        "min_power_w": min(samples),
        "max_power_w": max(samples),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser.add_argument("--gpu", type=int, default=0, help="GPU index")

    set_power = subparsers.add_parser("set-power", help="Set power limit (watts)")
    set_power.add_argument("power", type=float)

    set_sm = subparsers.add_parser("set-sm", help="Set SM clock min/max")
    set_sm.add_argument("min", type=int)
    set_sm.add_argument("max", type=int, nargs="?")

    set_mem = subparsers.add_parser("set-mem", help="Set memory clock min/max")
    set_mem.add_argument("min", type=int)
    set_mem.add_argument("max", type=int, nargs="?")

    set_ac = subparsers.add_parser("set-ac", help="Set application clocks (mem, sm)")
    set_ac.add_argument("mem", type=int)
    set_ac.add_argument("sm", type=int)

    subparsers.add_parser("reset-clocks", help="Reset application clocks")
    subparsers.add_parser("reset-power", help="Reset power limit")

    query = subparsers.add_parser("query", help="Query telemetry once")
    query.add_argument(
        "--fields",
        nargs="*",
        default=(),
        help="Additional fields to query via nvidia-smi",
    )

    sample = subparsers.add_parser("sample-power", help="Sample power draw over time")
    sample.add_argument("duration", type=float)
    sample.add_argument("--interval", type=float, default=0.5)

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    gpu_index = args.gpu

    try:
        if args.command == "set-power":
            set_power_limit(args.power, gpu_index=gpu_index)
            print(f"Power limit set to {args.power} W")
        elif args.command == "set-sm":
            set_sm_clock(args.min, args.max, gpu_index=gpu_index)
            hi = args.min if args.max is None else args.max
            print(f"SM clock limited to {args.min}-{hi} MHz")
        elif args.command == "set-mem":
            set_mem_clock(args.min, args.max, gpu_index=gpu_index)
            hi = args.min if args.max is None else args.max
            print(f"Memory clock limited to {args.min}-{hi} MHz")
        elif args.command == "set-ac":
            set_application_clocks(args.mem, args.sm, gpu_index=gpu_index)
            print(f"Application clocks set to mem={args.mem} MHz, sm={args.sm} MHz")
        elif args.command == "reset-clocks":
            reset_clocks(gpu_index=gpu_index)
            print("Application clocks reset")
        elif args.command == "reset-power":
            reset_power_limit(gpu_index=gpu_index)
            print("Power limit reset to default")
        elif args.command == "query":
            telemetry = query_telemetry(gpu_index=gpu_index, extra_fields=args.fields)
            print(telemetry)
        elif args.command == "sample-power":
            stats = sample_power(
                args.duration,
                interval_s=args.interval,
                gpu_index=gpu_index,
            )
            print(stats)
        else:  # pragma: no cover - defensive fallback
            raise ValueError(f"Unhandled command: {args.command}")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())


