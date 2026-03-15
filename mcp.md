---
layout: default
title: MCP & AI Workflow
---

<div class="page-header">
  <h1>MCP & AI Workflow</h1>
  <p>Run tensor network calculations directly from Claude Code using MCP tools and the Tenax Toolkit plugin</p>
</div>

<section>
<div class="container" markdown="1">

## What is MCP?

The [Model Context Protocol](https://modelcontextprotocol.io) lets AI assistants call external tools. Tenax provides an MCP server that gives Claude Code direct access to tensor network algorithms — no boilerplate, no copy-paste.

Ask Claude to "find the ground state energy of a 20-site Heisenberg chain" and it will call `run_dmrg` with the right parameters, interpret the results, and explain the physics.

### Setup

Add the Tenax MCP server to your Claude Code configuration (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "tenax": {
      "command": "uv",
      "args": [
        "--directory", "/path/to/tenax-mcp",
        "run", "tenax-mcp"
      ]
    }
  }
}
```

</div>
</section>

<section>
<div class="container" markdown="1">

## MCP Tools

Tenax exposes 9 tools through the MCP server. Claude Code can call any of them directly during a conversation.

<div class="skills-grid">
  <div class="tool-card">
    <h3>run_dmrg</h3>
    <p>Run DMRG ground state search for a 1D quantum Hamiltonian. Specify the chain length, Hamiltonian terms, bond dimension, number of sweeps, and convergence tolerance. Returns the ground state energy, energy per sweep, and convergence status.</p>
  </div>
  <div class="tool-card">
    <h3>run_trg</h3>
    <p>Run the Tensor Renormalization Group on a 2D classical model. Currently supports the 2D Ising model. Provide inverse temperature or temperature, coupling constant, and TRG bond dimension. Returns free energy per site, exact solution, and relative error.</p>
  </div>
  <div class="tool-card">
    <h3>run_hotrg</h3>
    <p>Run Higher-Order TRG on a 2D classical model. Uses higher-order SVD for more accurate coarse-graining than standard TRG. Supports configurable direction order (alternating, horizontal, vertical). Returns free energy per site, exact solution, and relative error.</p>
  </div>
  <div class="tool-card">
    <h3>build_hamiltonian</h3>
    <p>Build an MPO Hamiltonian from operator terms and return diagnostic info (bond dimensions, number of terms, total Hilbert space dimension) without running a calculation.</p>
  </div>
  <div class="tool-card">
    <h3>optimize_contraction</h3>
    <p>Find the optimal contraction path and FLOP cost for a tensor network. Provide tensor labels and dimensions; returns the optimal contraction order, FLOP count, and speedup over naive contraction.</p>
  </div>
  <div class="tool-card">
    <h3>validate_network</h3>
    <p>Check tensor network validity — dimension matching, charge consistency, and flow directions. Returns a validation result with a list of any issues found.</p>
  </div>
  <div class="tool-card">
    <h3>generate_code</h3>
    <p>Generate complete, runnable Tenax Python code from a high-level description. Supports DMRG, TRG, HOTRG, iDMRG, iPEPS (1-site, 2-site, split-CTM), fermionic iPEPS, standard CTM with Tensor protocol, and quasiparticle excitations. Returns a ready-to-run Python script.</p>
  </div>
  <div class="tool-card">
    <h3>list_operators</h3>
    <p>List available built-in spin operators and their matrix representations. Supports spin-1/2 (Sz, Sp, Sm, Id) and spin-1 (Sz, Sp, Sm, Id in the |+1⟩, |0⟩, |-1⟩ basis).</p>
  </div>
  <div class="tool-card">
    <h3>export_netfile</h3>
    <p>Convert a tensor network description to <code>.net</code> file format (Cytnx-compatible). Provide tensor names and leg labels; returns a ready-to-use network file.</p>
  </div>
</div>

</div>
</section>

<section>
<div class="container" markdown="1">

## Tenax Toolkit (Claude Code Plugin)

The [Tenax Toolkit](https://github.com/tenax-lab/tenax-toolkit) is a Claude Code plugin that gives Claude 17 domain-specific skills for tensor network simulations. Install it once and Claude automatically knows how to guide you through DMRG, iPEPS, TRG, symmetry-aware tensors, and more.

### Install

In Claude Code, run:

```
/install-plugin tenax-lab/tenax-toolkit
```

That's it — all 17 skills are immediately available. Claude will automatically invoke them when your questions match (e.g., asking about DMRG triggers `tenax-dmrg-workflow`, asking about symmetries triggers `tenax-symmetry`).

### Skills

#### Algorithm Workflows

<div class="skills-grid">
  <div class="skill-card">
    <h4>tenax-dmrg-workflow</h4>
    <p>Complete DMRG ground-state calculation: finite DMRG, iDMRG, and 2D cylinder DMRG with AutoMPO Hamiltonians.</p>
  </div>
  <div class="skill-card">
    <h4>tenax-ipeps-workflow</h4>
    <p>iPEPS pipeline: simple update, AD-based optimization with CTM environments, and quasiparticle excitation spectra.</p>
  </div>
  <div class="skill-card">
    <h4>tenax-trg-workflow</h4>
    <p>TRG and HOTRG for 2D classical stat mech: partition functions, free energy, and phase transitions.</p>
  </div>
</div>

#### Building Blocks

<div class="skills-grid">
  <div class="skill-card">
    <h4>tenax-tensor-ops</h4>
    <p>Core tensor operations: DenseTensor, SymmetricTensor, label-based contraction, SVD, QR, and eigendecomposition.</p>
  </div>
  <div class="skill-card">
    <h4>tenax-symmetry</h4>
    <p>Symmetry system: U(1) and Z_n, TensorIndex with charges and FlowDirection, block-sparse operations.</p>
  </div>
  <div class="skill-card">
    <h4>tenax-autompo</h4>
    <p>Build Hamiltonian MPOs from natural-language model descriptions using AutoMPO.</p>
  </div>
  <div class="skill-card">
    <h4>tenax-blueprint</h4>
    <p>Design tensor network contractions using NetworkBlueprint and .net topology files.</p>
  </div>
  <div class="skill-card">
    <h4>tenax-observables</h4>
    <p>Compute expectation values, correlation functions, entanglement entropy, and order parameters.</p>
  </div>
</div>

#### Validation, Benchmarking & Teaching

<div class="skills-grid">
  <div class="skill-card">
    <h4>tenax-debugger</h4>
    <p>Diagnose shape mismatches, JAX tracing issues, gradient problems, and convergence failures.</p>
  </div>
  <div class="skill-card">
    <h4>tenax-benchmark</h4>
    <p>Design and run performance benchmarks across CPU, CUDA, TPU, and Metal backends.</p>
  </div>
  <div class="skill-card">
    <h4>tenax-ed-comparator</h4>
    <p>Run exact diagonalization and DMRG side-by-side to validate results and study truncation error.</p>
  </div>
  <div class="skill-card">
    <h4>tenax-homework</h4>
    <p>Generate scaffolded tensor network homework problems for graduate courses.</p>
  </div>
  <div class="skill-card">
    <h4>tenax-getting-started</h4>
    <p>Install Tenax, configure JAX backends, and run your first calculation.</p>
  </div>
</div>

#### Migration from Other Libraries

<div class="skills-grid">
  <div class="skill-card">
    <h4>tenax-migration-itensor</h4>
    <p>Translate ITensor (Julia/C++) code and concepts to Tenax.</p>
  </div>
  <div class="skill-card">
    <h4>tenax-migration-tenpy</h4>
    <p>Translate TeNPy code and concepts to Tenax.</p>
  </div>
  <div class="skill-card">
    <h4>tenax-migration-cytnx</h4>
    <p>Translate Cytnx code and concepts to Tenax.</p>
  </div>
  <div class="skill-card">
    <h4>tenax-migration-quimb</h4>
    <p>Translate quimb code and concepts to Tenax.</p>
  </div>
</div>

</div>
</section>
