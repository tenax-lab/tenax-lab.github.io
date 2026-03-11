---
layout: default
---

<div class="hero">
  <h1>Tenax</h1>
  <p class="tagline">JAX-based tensor network library with symmetry-aware block-sparse tensors and label-based contraction</p>
  <div class="install-cmd"><span class="prompt">$</span> pip install tenax-tn</div>
  <div class="hero-links">
    <a href="https://github.com/tenax-lab/tenax">GitHub</a>
    <a href="https://tenax.readthedocs.io">Docs</a>
  </div>
</div>

<div class="experimental-banner">
  <div class="container">
    <strong>Experimental project</strong> — This library is under active development and largely written with the assistance of Claude Code (AI). While we test extensively, AI-generated code can contain subtle bugs. Please verify results against known benchmarks before using them in research.
  </div>
</div>

<section>
<div class="container" markdown="1">

## Key Features

<div class="features">
  <div class="feature">
    <h3>Symmetry-aware tensors</h3>
    <p>Block-sparse storage for U(1), Z<sub>n</sub>, and fermionic symmetries. Only allowed charge sectors are stored.</p>
  </div>
  <div class="feature">
    <h3>Label-based contraction</h3>
    <p>Legs identified by string labels; shared labels across tensors are automatically contracted.</p>
  </div>
  <div class="feature">
    <h3>Optimal contraction paths</h3>
    <p>Integrated with opt_einsum for finding optimal contraction orders in multi-tensor networks.</p>
  </div>
  <div class="feature">
    <h3>Pure JAX</h3>
    <p><code>jit</code>, <code>grad</code>, <code>vmap</code> work out of the box. GPU, TPU, and Metal acceleration.</p>
  </div>
  <div class="feature">
    <h3>AI integration</h3>
    <p>MCP server for running calculations from Claude. Built-in skills for ground states, debugging, benchmarking, and migration from ITensor, TeNPy, Cytnx, and quimb.</p>
  </div>
</div>

</div>
</section>

<section>
<div class="container" markdown="1">

## Algorithms

<div class="algorithms">
  <div class="algo-card">
    <h3>DMRG</h3>
    <p>Finite 1D chains and 2D cylinder ground states</p>
  </div>
  <div class="algo-card">
    <h3>iDMRG</h3>
    <p>Infinite chains and infinite cylinders</p>
  </div>
  <div class="algo-card">
    <h3>TRG / HOTRG</h3>
    <p>2D classical partition functions via tensor coarse-graining</p>
  </div>
  <div class="algo-card">
    <h3>iPEPS</h3>
    <p>2D ground states with simple update, AD optimization, QR projectors, 2-site unit cells, and split-CTMRG</p>
  </div>
  <div class="algo-card">
    <h3>Fermionic iPEPS</h3>
    <p>Simple-update fPEPS for spinless fermions with FermionParity symmetry</p>
  </div>
  <div class="algo-card">
    <h3>Excitations</h3>
    <p>Quasiparticle spectra via iPEPS at arbitrary momenta</p>
  </div>
  <div class="algo-card">
    <h3>AutoMPO</h3>
    <p>Build Hamiltonians from symbolic operator descriptions</p>
  </div>
</div>

</div>
</section>

<section>
<div class="container" markdown="1">

## AI-Powered Workflow

<div class="why-grid">
  <div class="why-card">
    <h3>MCP Server</h3>
    <p>Run DMRG, TRG, HOTRG, and more directly from Claude Code. Ask questions in natural language and get tensor network calculations.</p>
  </div>
  <div class="why-card">
    <h3>Claude Code Skills</h3>
    <p>Built-in prompt templates for ground states, benchmarking, debugging, teaching, and migrating from other libraries.</p>
  </div>
</div>

<a class="cta" href="/mcp/">Learn more about MCP & AI workflow</a>

</div>
</section>

<section>
<div class="container" markdown="1">

## Code Example

```python
from tenax import AutoMPO, DMRGConfig, build_random_mps, dmrg

L = 20
auto = AutoMPO(L)
for i in range(L - 1):
    auto += (1.0, "Sz", i, "Sz", i + 1)
    auto += (0.5, "Sp", i, "Sm", i + 1)
    auto += (0.5, "Sm", i, "Sp", i + 1)

mpo = auto.to_mpo()
mps = build_random_mps(L, bond_dim=4)
result = dmrg(mpo, mps, DMRGConfig(max_bond_dim=100, num_sweeps=10))
print(f"Ground state energy: {result.energy:.8f}")
```

<a class="cta" href="/examples/">See all examples</a>

</div>
</section>

<section>
<div class="container" markdown="1">

## Why Tenax?

<div class="why-grid">
  <div class="why-card">
    <h3>Fully differentiable</h3>
    <p>Every algorithm works with JAX's <code>grad</code> and <code>jit</code>. AD-based iPEPS optimization uses implicit differentiation through the CTM fixed point for stable gradients.</p>
  </div>
  <div class="why-card">
    <h3>Run anywhere</h3>
    <p>Same code runs on CPU, NVIDIA GPU (CUDA 12/13), Google Cloud TPU, and Apple Silicon (Metal). No code changes — just install the right backend.</p>
  </div>
  <div class="why-card">
    <h3>Benchmark suite</h3>
    <p>CLI-driven performance benchmarks for every algorithm across all backends. Compare wall-clock timings with a single command and export to JSON or CSV.</p>
  </div>
  <div class="why-card">
    <h3>From 1D to 2D</h3>
    <p>Covers the full range: finite DMRG, iDMRG (chains and infinite cylinders), 2D cylinder DMRG, iPEPS ground states, and quasiparticle excitation spectra.</p>
  </div>
</div>

</div>
</section>

<section>
<div class="container" markdown="1">

## Coming From Another Library?

Tenax shares core ideas with ITensor, TeNPy, Cytnx, quimb, and TensorKit.jl. Our migration guides map concepts and code patterns so you can translate your existing work.

<a class="cta" href="/migration/">View migration guides</a>

</div>
</section>

<section>
<div class="container" markdown="1">

## Installation

<div class="install-grid">
  <div class="install-option">
    <h3>CPU (default)</h3>
    <pre>pip install tenax-tn</pre>
  </div>
  <div class="install-option">
    <h3>NVIDIA GPU (CUDA 13)</h3>
    <pre>pip install tenax-tn[cuda13]</pre>
  </div>
  <div class="install-option">
    <h3>NVIDIA GPU (CUDA 12)</h3>
    <pre>pip install tenax-tn[cuda12]</pre>
  </div>
  <div class="install-option">
    <h3>Google Cloud TPU</h3>
    <pre>pip install tenax-tn[tpu]</pre>
  </div>
  <div class="install-option">
    <h3>Apple Silicon GPU</h3>
    <pre>pip install tenax-tn[metal]</pre>
  </div>
  <div class="install-option">
    <h3>From source</h3>
    <pre>git clone https://github.com/tenax-lab/tenax.git
cd tenax && pip install -e ".[dev]"</pre>
  </div>
</div>

</div>
</section>
