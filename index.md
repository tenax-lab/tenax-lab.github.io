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
    <a href="https://pypi.org/project/tenax-tn/">PyPI</a>
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
    <p>2D ground states with simple update and AD optimization</p>
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

<div class="features">
  <div class="feature highlight-feature">
    <h3>MCP Server</h3>
    <p>Tenax ships an <a href="https://modelcontextprotocol.io">MCP</a> server that gives any AI assistant direct access to tensor network tools — build Hamiltonians, run DMRG and TRG, optimize contraction paths, validate networks, and generate code, all through natural language.</p>
  </div>
  <div class="feature highlight-feature">
    <h3>Claude Code Skills</h3>
    <p>18+ specialized skills turn Claude Code into a tensor network tutor: DMRG workflows, iPEPS optimization, TRG analysis, symmetry systems, debugging, benchmarking, and migration guides from ITensor, TeNPy, Cytnx, and quimb.</p>
  </div>
</div>

<div class="mcp-tools">
  <h3>MCP Tools</h3>
  <div class="tool-grid">
    <div class="tool-card">
      <code>run_dmrg</code>
      <span>Ground-state search for 1D quantum Hamiltonians</span>
    </div>
    <div class="tool-card">
      <code>run_trg</code>
      <span>2D classical partition functions (Ising model)</span>
    </div>
    <div class="tool-card">
      <code>build_hamiltonian</code>
      <span>Build MPO from operator terms</span>
    </div>
    <div class="tool-card">
      <code>optimize_contraction</code>
      <span>Find optimal path and FLOP cost</span>
    </div>
    <div class="tool-card">
      <code>validate_network</code>
      <span>Check dimensions, charges, and flow consistency</span>
    </div>
    <div class="tool-card">
      <code>generate_code</code>
      <span>Turn descriptions into runnable Python</span>
    </div>
    <div class="tool-card">
      <code>list_operators</code>
      <span>Built-in spin-½ and spin-1 operators</span>
    </div>
    <div class="tool-card">
      <code>export_netfile</code>
      <span>Convert networks to .net format</span>
    </div>
  </div>
</div>

</div>
</section>

<section>
<div class="container" markdown="1">

## Code Examples

<div class="code-example" markdown="1">

### DMRG Ground State

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

</div>

<div class="code-example" markdown="1">

### TRG — 2D Ising Model

```python
from tenax import TRGConfig, trg, compute_ising_tensor, ising_free_energy_exact

beta = 0.44  # near critical temperature
tensor = compute_ising_tensor(beta)
log_Z = trg(tensor, TRGConfig(max_bond_dim=16, num_steps=20))

f_trg = float(-log_Z / beta)
f_exact = ising_free_energy_exact(beta)
print(f"TRG:   {f_trg:.8f}")
print(f"Exact: {f_exact:.8f}")
```

</div>

<div class="code-example" markdown="1">

### iPEPS AD Optimization

```python
import jax.numpy as jnp
from tenax import iPEPSConfig, CTMConfig, optimize_gs_ad, spin_half_ops

ops = spin_half_ops()
gate = (jnp.einsum("ij,kl->ikjl", ops["Sz"], ops["Sz"])
        + 0.5 * (jnp.einsum("ij,kl->ikjl", ops["Sp"], ops["Sm"])
                 + jnp.einsum("ij,kl->ikjl", ops["Sm"], ops["Sp"])))

config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=16),
    gs_num_steps=200,
)
A_opt, env, E_gs = optimize_gs_ad(gate, None, config)
print(f"Energy per site: {E_gs:.6f}")
```

</div>
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

Tenax provides migration guides with side-by-side concept mapping for:

<div class="migration-grid">
  <div class="migration-card">
    <h3>ITensor</h3>
    <p>Julia/C++ &rarr; Tenax. Maps Index, ITensor, MPS, AutoMPO to Tenax equivalents.</p>
  </div>
  <div class="migration-card">
    <h3>TeNPy</h3>
    <p>Maps Site, MPS, MPO, Model, and Engine classes to Tenax patterns.</p>
  </div>
  <div class="migration-card">
    <h3>Cytnx</h3>
    <p>UniTensor, Bond, Network &rarr; DenseTensor, SymmetricTensor, NetworkBlueprint.</p>
  </div>
  <div class="migration-card">
    <h3>quimb</h3>
    <p>Maps Tensor, TensorNetwork, DMRG, and TEBD to Tenax equivalents.</p>
  </div>
</div>

</div>
</section>

<section>
<div class="container" markdown="1">

## Installation

<div class="install-grid">
  <div class="install-option">
    <h3>CPU</h3>
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
    <h3>Development</h3>
    <pre>git clone https://github.com/tenax-lab/tenax.git
cd tenax && uv sync --all-extras --dev</pre>
  </div>
</div>

</div>
</section>

<footer>
  <div class="footer-links">
    <a href="https://github.com/tenax-lab/tenax">GitHub</a>
    <a href="https://pypi.org/project/tenax-tn/">PyPI</a>
    <a href="https://tenax.readthedocs.io">Docs</a>
  </div>
  <p>MIT License</p>
</footer>
