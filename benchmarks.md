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
| `-b`, `--backend` | Hardware backend | `cpu`, `cuda`, `tpu`, `metal`, `auto` (default) |
| `-a`, `--algorithm` | Algorithm(s) to benchmark | `dmrg`, `idmrg`, `trg`, `hotrg`, `ipeps`, `ipeps_ad`, `dmrg_sharded`, `all` |
| `-s`, `--size` | Problem size(s) | `small`, `medium`, `large`, `all` |
| `-n`, `--trials` | Number of trials per config | Integer (default 3) |
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
| DMRG | small | L=20, chi=32, 5 sweeps |
| DMRG | medium | L=40, chi=64, 5 sweeps |
| DMRG | large | L=80, chi=128, 3 sweeps |
| iDMRG | small | chi=32, 50 iterations |
| iDMRG | medium | chi=64, 100 iterations |
| iDMRG | large | chi=128, 50 iterations |
| TRG | small | chi=8, 15 steps |
| TRG | medium | chi=24, 20 steps |
| TRG | large | chi=48, 20 steps |
| HOTRG | small | chi=8, 10 steps |
| HOTRG | medium | chi=20, 15 steps |
| HOTRG | large | chi=40, 15 steps |
| iPEPS | small | D=2, chi=8, 50 SU steps |
| iPEPS | medium | D=3, chi=16, 50 SU steps |
| iPEPS | large | D=4, chi=24, 30 SU steps |
| iPEPS AD | small | D=2, chi=8, 20 AD steps |
| iPEPS AD | medium | D=2, chi=16, 30 AD steps |
| iPEPS AD | large | D=3, chi=16, 20 AD steps |

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
It is deprecated (#512) in favor of in-CTM chi growth (`ctmrg_heuristic_increase_chi=True` with `chi_max`).
Benchmarks of `chi_ramp` showed 1.2--2.1x speedup on GPU with identical energies:

| D | chi | Speedup |
|---|-----|---------|
| 2 | 16 | 1.33x |
| 2 | 48 | 1.41x |
| 3 | 36 | 1.51x |
| 4 | 16 | 2.13x |

</div>
</section>
