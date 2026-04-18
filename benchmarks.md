---
layout: default
title: Benchmarks
---

<div class="page-header">
  <h1>Benchmarks</h1>
  <p>Measure algorithm performance across CPU, CUDA, TPU, and Metal backends</p>
</div>

<section>
<div class="container" markdown="1">

## Quick Start

```bash
# Quick smoke test (TRG, small size, 1 trial)
python -m benchmarks.run --backend cpu --algorithm trg --size small --trials 1

# Full CPU baseline
python -m benchmarks.run --backend cpu -o benchmarks/results/cpu_baseline.json

# GPU comparison
python -m benchmarks.run --backend cuda -o benchmarks/results/cuda.json

# Show available backends
python -m benchmarks.run --list-backends
```

</div>
</section>

<section>
<div class="container" markdown="1">

## CLI Reference

```bash
python -m benchmarks.run [OPTIONS]
```

| Flag | Description | Values |
|------|-------------|--------|
| `-b`, `--backend` | Hardware backend | `cpu`, `cuda`, `tpu`, `metal` |
| `-a`, `--algorithm` | Algorithm(s) to benchmark | `dmrg`, `idmrg`, `trg`, `hotrg`, `ipeps`, `all` |
| `-s`, `--size` | Problem size(s) | `small`, `medium`, `large`, `all` |
| `-n`, `--trials` | Number of trials per config | Integer (default varies) |
| `-o`, `--output` | Save results to JSON | File path |
| `--csv` | Save results to CSV | File path |
| `--list-backends` | Show available backends | — |

</div>
</section>

<section>
<div class="container" markdown="1">

## Algorithm Sizes

Each algorithm defines three problem sizes. Larger sizes stress-test bond dimension scaling and hardware throughput.

| Algorithm | Size | Parameters |
|-----------|------|------------|
| DMRG | small | L=10, chi=20, 5 sweeps |
| DMRG | medium | L=20, chi=50, 10 sweeps |
| DMRG | large | L=40, chi=100, 10 sweeps |
| iDMRG | small | chi=16, 50 iterations |
| iDMRG | medium | chi=32, 100 iterations |
| iDMRG | large | chi=64, 200 iterations |
| TRG | small | chi=8, 16 steps |
| TRG | medium | chi=16, 20 steps |
| TRG | large | chi=32, 24 steps |
| HOTRG | small | chi=8, 12 steps |
| HOTRG | medium | chi=16, 16 steps |
| HOTRG | large | chi=32, 20 steps |
| iPEPS | small | D=2, chi=8, 100 SU steps |
| iPEPS | medium | D=3, chi=16, 200 SU steps |
| iPEPS | large | D=4, chi=24, 300 SU steps |

</div>
</section>

<section>
<div class="container" markdown="1">

## Supported Backends

| Backend | Flag | Requirements |
|---------|------|-------------|
| CPU | `cpu` | Default — works everywhere |
| NVIDIA GPU | `cuda` | `pip install tenax-tn[cuda13]` or `tenax-tn[cuda12]` |
| Google Cloud TPU | `tpu` | `pip install tenax-tn[tpu]` |
| Apple Silicon GPU | `metal` | `pip install tenax-tn[metal]` (macOS, experimental) |

Check what's available on your machine:

```bash
python -m benchmarks.run --list-backends
```

</div>
</section>

<section>
<div class="container" markdown="1">

## Output Formats

### JSON

Full structured output with timings, parameters, and device info:

```bash
python -m benchmarks.run -b cpu -a dmrg -s medium -n 3 -o results.json
```

### CSV

Flat table for analysis in pandas, Excel, or plotting tools:

```bash
python -m benchmarks.run -b cpu -a all -s all --csv results.csv
```

</div>
</section>

<section>
<div class="container" markdown="1">

## Example Experiments

### Is my GPU helping?

```bash
python -m benchmarks.run -b cpu -a dmrg -s medium -n 3 -o cpu.json
python -m benchmarks.run -b cuda -a dmrg -s medium -n 3 -o gpu.json
```

GPU typically helps when chi >= 64. For smaller bond dimensions, CPU may be faster due to transfer overhead.

### Algorithm scaling

Fix the backend and vary size to study computational scaling:

```bash
python -m benchmarks.run -b cpu -a dmrg -s small medium large -n 5 --csv dmrg_scaling.csv
```

### Tips

- **Report median time**, not mean — avoids JIT warmup skew from the first trial.
- **GPU memory** — large sizes may OOM on GPUs with limited VRAM.
- **First trial is slow** — JAX recompiles on first call. Use `--trials 5` or more for stable numbers.

### Chi ramping

`CTMConfig.chi_ramp` runs CTM convergence in stages at increasing chi.
Benchmarks show 1.2--2.1x speedup on GPU with identical energies:

| D | chi | Speedup |
|---|-----|---------|
| 2 | 16 | 1.33x |
| 2 | 48 | 1.41x |
| 3 | 36 | 1.51x |
| 4 | 16 | 2.13x |

</div>
</section>
