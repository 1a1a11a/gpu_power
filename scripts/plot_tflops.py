#!/usr/bin/env python3

"""Plot TFLOPs versus power limit from ``run_power_sweep.py`` results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

# The sweep script may run on headless servers.  Use a non-interactive backend by default.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _load_curves(csv_path: Path, dtypes: Iterable[str]) -> Dict[str, List[Tuple[float, float]]]:
    curves: Dict[str, List[Tuple[float, float]]] = {dtype: [] for dtype in dtypes}

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            # Try different power column names (support both original and optimized formats)
            power = (_parse_float(row.get("power_limit_w")) or
                    _parse_float(row.get("avg_power_w")))
            if power is None:
                continue

            for dtype in dtypes:
                tflops_key = f"{dtype}_tflops"
                tflops = _parse_float(row.get(tflops_key))
                if tflops is None:
                    continue
                curves[dtype].append((power, tflops))

    # Sort each curve by power in descending order (high power first)
    for dtype in dtypes:
        curves[dtype].sort(key=lambda item: item[0], reverse=True)

    return curves


def _plot_curves(curves: Dict[str, List[Tuple[float, float]]], *, title: str, output: Path) -> None:
    plt.figure(figsize=(7.5, 5.5))

    for dtype, points in curves.items():
        if not points:
            continue
        xs, ys = zip(*points)
        plt.plot(xs, ys, marker="o", label=dtype.upper())

    plt.xlabel("Power Limit (W)", fontsize=14)
    plt.ylabel("Throughput (TFLOPs)", fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    plt.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        type=Path,
        help="CSV file produced by run_power_sweep.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tflops_vs_power.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--dtypes",
        nargs="*",
        default=["bf16", "fp8"],
        help="Data types to plot (columns named <dtype>_tflops)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="TFLOPs vs Power Limit",
        help="Plot title",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV file not found: {args.csv}")

    dtypes = [dtype.lower() for dtype in args.dtypes]
    curves = _load_curves(args.csv, dtypes)

    if not any(curves.values()):
        raise SystemExit("No usable data points found in CSV")

    _plot_curves(curves, title=args.title, output=args.output)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()


