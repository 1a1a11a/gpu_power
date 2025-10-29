#!/usr/bin/env python3

"""Utility script to report GPU benchmarking prerequisites."""

from __future__ import annotations

import importlib
import sys
from textwrap import indent


def _check_module(name: str, symbol: str | None = None) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:  # pragma: no cover - diagnostic path
        return False, f"missing ({exc})"

    if symbol is None:
        return True, getattr(module, "__version__", "unknown version")

    if not hasattr(module, symbol):  # pragma: no cover - diagnostic path
        return False, f"present but missing attribute '{symbol}'"

    return True, getattr(module, symbol)


def main() -> int:
    ok = True

    success, torch_info = _check_module("torch", "__version__")
    print("PyTorch:")
    print(indent(str(torch_info), "  "))
    if not success:
        ok = False
    else:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            print(indent("CUDA not available", "  "))
            ok = False
        else:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            print(indent(f"CUDA device: {props.name}", "  "))
            print(indent(f"Compute capability: {props.major}.{props.minor}", "  "))

    success, te_info = _check_module("transformer_engine")
    print("Transformer Engine:")
    print(indent(str(te_info), "  "))
    if not success:
        ok = False

    if ok:
        print("Environment ready ✅")
        return 0

    print("Environment incomplete ❌")
    return 1


if __name__ == "__main__":
    sys.exit(main())

