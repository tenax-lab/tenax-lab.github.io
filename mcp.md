---
layout: default
title: MCP & AI Workflow
---

<div class="page-header">
  <h1>MCP & AI Workflow</h1>
  <p>Run tensor network calculations directly from Claude Code using the Model Context Protocol</p>
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

Tenax exposes 8 tools through the MCP server. Claude Code can call any of them directly during a conversation.

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
    <p>Generate complete, runnable Tenax Python code from a high-level description. Supports DMRG, TRG, iDMRG, and iPEPS algorithms. Returns a ready-to-run Python script.</p>
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

## Claude Code Skills

Beyond MCP tools, Tenax ships [Claude Code skills](https://github.com/tenax-lab/tenax/tree/main/.claude/skills) — prompt templates that teach Claude domain-specific workflows.

### Workflows

<div class="skills-grid">
  <div class="skill-card">
    <h4>tenax-workflow-ground-state</h4>
    <p>End-to-end ground state calculation: choose algorithm, build Hamiltonian, run DMRG/iDMRG/iPEPS, analyze results.</p>
  </div>
  <div class="skill-card">
    <h4>tenax-workflow-classical-stat-mech</h4>
    <p>Classical statistical mechanics: build tensor from partition function, run TRG/HOTRG, compare to exact solutions.</p>
  </div>
</div>

### Building Blocks

<div class="skills-grid">
  <div class="skill-card">
    <h4>tenax-symmetric-tensor</h4>
    <p>Create and manipulate symmetry-aware block-sparse tensors with U(1) and Z_n symmetries.</p>
  </div>
  <div class="skill-card">
    <h4>tenax-contraction</h4>
    <p>Label-based contraction, SVD, QR decomposition, and NetworkBlueprint usage.</p>
  </div>
  <div class="skill-card">
    <h4>tenax-auto-mpo</h4>
    <p>Build Hamiltonian MPOs from symbolic operator descriptions using AutoMPO.</p>
  </div>
  <div class="skill-card">
    <h4>tenax-observables</h4>
    <p>Compute expectation values, correlation functions, and entanglement entropy from MPS.</p>
  </div>
</div>

### Validation & Testing

<div class="skills-grid">
  <div class="skill-card">
    <h4>tenax-validate</h4>
    <p>Diagnose tensor shape mismatches, charge violations, contraction errors, and numerical issues.</p>
  </div>
  <div class="skill-card">
    <h4>tenax-benchmark</h4>
    <p>Design and run performance benchmarks across CPU, CUDA, TPU, and Metal backends.</p>
  </div>
</div>

### Teaching

<div class="skills-grid">
  <div class="skill-card">
    <h4>tenax-teach</h4>
    <p>Explain tensor network concepts at the student's level with Tenax code examples.</p>
  </div>
</div>

### Migration

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
