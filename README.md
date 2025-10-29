# GPU Power Benchmarking

Benchmark RTX PRO 6000 GPU compute capability (INT8/FP4/FP8/BF16 TFLOPs) across varying power limits and clock frequencies, with automated plotting of performance curves.

## Requirements

- Linux with NVIDIA RTX PRO 6000 GPU
- Python 3.8+
- PyTorch with CUDA support
- Transformer Engine: `pip install transformer-engine`
- TorchAO (for INT8): `pip install torchao`
- Matplotlib: `pip install matplotlib`
- Admin privileges for `nvidia-smi` power/clock controls

## Quick Start

```bash
# Check environment
python check_env.py

# Run power sweep (adjust values for your GPU)
python run_power_sweep.py --power-limits 600 480 400 320 280 240 --sm-clocks 2400 --mem-clocks 9000
python run_power_sweep.py --power-limits 600 480 400 320 280 240

# Plot results
python plot_tflops.py power_sweep.csv --output tflops_vs_power.png
```

## Scripts

- **`check_env.py`** - Verify PyTorch + Transformer Engine installation and CUDA availability
- **`benchmark_tflops.py`** - Standalone throughput benchmarking for INT8/FP4/FP8/BF16 matrix multiplication
- **`power_control.py`** - CLI/API for NVIDIA GPU power limits, SM clocks, and memory clocks
- **`run_power_sweep.py`** - Orchestrator that combines power control + benchmarking + logging
- **`plot_tflops.py`** - Generate TFLOPs vs power consumption plots from CSV results

## Example Output

The sweep generates a CSV with columns like:
- Power limits and measured draw
- Clock frequencies (target vs measured)
- TFLOPs for each data type (BF16, INT8, FP8, FP4)
- Timing data and error messages

Use the plot script to visualize how compute performance scales with power consumption.

## Notes

- Requires root/admin access for GPU power/clock modifications
- Transformer Engine handles FP4/FP8 acceleration, TorchAO handles INT8 quantization
- Default benchmark uses 8192×8192×8192 matrix multiplication
- Adjust matrix sizes with `--m`, `--n`, `--k` flags for different workloads
