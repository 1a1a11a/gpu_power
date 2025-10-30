# GPU Power Benchmarking & FP8 Performance

Benchmark RTX PRO 6000 Blackwell GPU with **native FP8 tensor core performance testing** and power/clock optimization.

## 🏆 FP8 Performance Results

**Native FP8 tensor cores achieve 608 TFLOPs!**

| Precision | TFLOPs | Speedup vs FP32 | Notes |
|-----------|--------|-----------------|-------|
| **FP8 (E4M3/E5M2)** | **608** | **8.0x** | ✅ Native tensor cores |
| BF16 | 386 | 5.1x | 16-bit baseline |
| FP32 | 76 | 1.0x | 32-bit reference |

### Key Findings
- **FP8 is 1.57x faster than BF16** (608 vs 386 TFLOPs)
- **FP8 uses 2x less memory than BF16**, 4x less than FP32
- **Production ready** via NVIDIA Transformer Engine
- Tested on RTX PRO 6000 Blackwell (CC 12.0)

## Quick Start - FP8 Benchmark

```bash
# Install dependencies
pip install torch transformer-engine

# Run FP8 benchmark
python3 fp8_benchmark_te.py

# Custom matrix size (M N K)
python3 fp8_benchmark_te.py 8192 8192 8192 --warmup 5 --iters 10
```

## Requirements

### FP8 Benchmarking
- Python 3.8+
- PyTorch with CUDA support: `pip install torch`
- Transformer Engine: `pip install transformer-engine`
- NVIDIA RTX PRO 6000 Blackwell GPU
- CUDA 12.8+ drivers

### Power Sweeps (Optional)
- TorchAO (for INT8): `pip install torchao`
- Matplotlib: `pip install matplotlib`
- Admin privileges for `nvidia-smi` power/clock controls

## Power Sweep Benchmarks (Optional)

```bash
# Check environment
python -m scripts.check_env

# Run power sweep across different power limits
python -m scripts.run_power_sweep --power-limits 600 480 400 320 280 240

# Optimize clock frequencies for each power limit
python -m scripts.run_power_sweep --power-limits 600 480 400 320 280 240 --optimize-clocks

# Plot results
python -m scripts.plot_tflops power_sweep.csv --output tflops_vs_power.png
```

## Files

### FP8 Benchmarking
- **`fp8_benchmark_te.py`** - Native FP8 benchmark using Transformer Engine (608 TFLOPs)
- **`FP8_RESULTS.md`** - Detailed FP8 performance results and configuration
- **`Makefile`** - Quick install/test commands

### Power Sweep Tools (Optional)
- **`scripts/benchmark_tflops.py`** - Throughput benchmarking for INT8/FP4/FP8/BF16
- **`scripts/run_power_sweep.py`** - Power/clock optimization sweeps
- **`scripts/plot_tflops.py`** - Visualization of performance vs power
- **`scripts/power_control.py`** - GPU power/clock control utilities

## FP8 Technical Details

### FP8 Formats
- **E4M3**: 1 sign bit, 4 exponent bits, 3 mantissa bits (forward passes)
- **E5M2**: 1 sign bit, 5 exponent bits, 2 mantissa bits (backward passes)
- **Hybrid**: Automatically uses E4M3 for forward, E5M2 for gradients

### Why FP8?
- **8x speedup vs FP32** (608 vs 76 TFLOPs)
- **1.57x speedup vs BF16** (608 vs 386 TFLOPs)
- **Memory savings**: 4x less than FP32, 2x less than BF16
- **Same accuracy**: Minimal precision loss for AI training/inference

### Hardware
- RTX PRO 6000 Blackwell (Compute Capability 12.0)
- 188 streaming multiprocessors
- 1.79 TB/s memory bandwidth (512-bit @ 14 GHz)
- Native FP8 tensor cores
