---
layout: default
title: Code Examples
---

<div class="page-header">
  <h1>Code Examples</h1>
  <p>Complete, runnable examples for every algorithm in Tenax</p>
</div>

<section>
<div class="container" markdown="1">

<div class="toc" markdown="1">

**Contents**

- [DMRG Ground State](#dmrg-ground-state)
- [2D Cylinder DMRG](#2d-cylinder-dmrg)
- [iDMRG](#idmrg)
- [Infinite Cylinder iDMRG](#infinite-cylinder-idmrg)
- [TRG — 2D Ising Model](#trg--2d-ising-model)
- [iPEPS Simple Update](#ipeps-simple-update)
- [iPEPS AD Optimization & Excitations](#ipeps-ad-optimization--excitations)
- [iPEPS with QR Projectors](#ipeps-with-qr-projectors)
- [iPEPS 2-Site AD Optimization](#ipeps-2-site-ad-optimization)
- [Split-CTMRG](#split-ctmrg)
- [AutoMPO](#autompo)
- [Tensor Display & Mermaid Export](#tensor-display--mermaid-export)
- [Runnable Scripts](#runnable-scripts)
{: style="list-style: none; padding: 0;" }

</div>

---

### DMRG Ground State

Finite DMRG for the spin-1/2 Heisenberg chain using AutoMPO:

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

---

### 2D Cylinder DMRG

Heisenberg model on a 6x3 cylinder mapped to a 1D chain via AutoMPO:

```python
from tenax import AutoMPO, DMRGConfig, build_random_mps, dmrg

Lx, Ly, N = 6, 3, 18
auto = AutoMPO(L=N, d=2)
for x in range(Lx):
    for y in range(Ly):
        # Within-ring bond (periodic y)
        i, j = x * Ly + y, x * Ly + (y + 1) % Ly
        auto += (1.0, "Sz", min(i,j), "Sz", max(i,j))
        auto += (0.5, "Sp", min(i,j), "Sm", max(i,j))
        auto += (0.5, "Sm", min(i,j), "Sp", max(i,j))
        # Between-ring bond (open x)
        if x < Lx - 1:
            i, j = x * Ly + y, (x + 1) * Ly + y
            auto += (1.0, "Sz", i, "Sz", j)
            auto += (0.5, "Sp", i, "Sm", j)
            auto += (0.5, "Sm", i, "Sp", j)

mpo = auto.to_mpo(compress=True)
mps = build_random_mps(N, physical_dim=2, bond_dim=16)
config = DMRGConfig(max_bond_dim=100, num_sweeps=10, verbose=True)
result = dmrg(mpo, mps, config)
print(f"E/N = {result.energy / N:.8f}")
```

---

### iDMRG

Infinite DMRG for the Heisenberg chain in the thermodynamic limit:

```python
from tenax import idmrg, build_bulk_mpo_heisenberg, iDMRGConfig

W = build_bulk_mpo_heisenberg(Jz=1.0, Jxy=1.0)
config = iDMRGConfig(max_bond_dim=32, max_iterations=100, convergence_tol=1e-8)
result = idmrg(W, config)
print(f"Energy per site: {result.energy_per_site:.6f}")  # ~ -0.4431
print(f"Converged: {result.converged}")
```

---

### Infinite Cylinder iDMRG

Heisenberg model on an infinite cylinder with circumference Ly:

```python
from tenax import build_bulk_mpo_heisenberg_cylinder, iDMRGConfig, idmrg

# Ly=4 cylinder: each super-site is a ring of 4 spins (d=16, D_w=14)
W = build_bulk_mpo_heisenberg_cylinder(Ly=4)
config = iDMRGConfig(max_bond_dim=200, max_iterations=200, convergence_tol=1e-4)
result = idmrg(W, config, d=16)
e_per_spin = result.energy_per_site / 4
print(f"Energy per spin: {e_per_spin:.6f}")
```

---

### TRG — 2D Ising Model

Tensor Renormalization Group for the 2D classical Ising model:

```python
from tenax import TRGConfig, trg, compute_ising_tensor, ising_free_energy_exact

beta = 0.44  # near critical temperature
T = compute_ising_tensor(beta)

config = TRGConfig(max_bond_dim=16, num_steps=20)
log_z_per_n = trg(T, config)
f_trg = float(-log_z_per_n / beta)
f_exact = ising_free_energy_exact(beta)
print(f"TRG:   {f_trg:.8f}")
print(f"Exact: {f_exact:.8f}")
```

---

### iPEPS Simple Update

2-site checkerboard iPEPS for the Heisenberg model:

```python
import jax.numpy as jnp
from tenax import iPEPSConfig, CTMConfig, ipeps

Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
gate = jnp.einsum("ij,kl->ikjl", Sz, Sz) \
     + 0.5 * (jnp.einsum("ij,kl->ikjl", Sp, Sm)
             + jnp.einsum("ij,kl->ikjl", Sm, Sp))

config = iPEPSConfig(
    max_bond_dim=2,
    num_imaginary_steps=200,
    dt=0.3,
    ctm=CTMConfig(chi=10, max_iter=40),
    unit_cell="2site",
)
energy, peps, (env_A, env_B) = ipeps(gate, None, config)
print(f"Energy per site: {energy:.6f}")  # ~ -0.65
```

---

### iPEPS AD Optimization & Excitations

Gradient-based ground state optimization via implicit differentiation through the CTM fixed point, followed by quasiparticle excitation spectra:

```python
import jax.numpy as jnp
from tenax import (
    iPEPSConfig, CTMConfig, optimize_gs_ad,
    ExcitationConfig, compute_excitations, make_momentum_path,
)

Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
gate = jnp.einsum("ij,kl->ikjl", Sz, Sz) \
     + 0.5 * (jnp.einsum("ij,kl->ikjl", Sp, Sm)
             + jnp.einsum("ij,kl->ikjl", Sm, Sp))

# AD ground-state optimization
config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=16, max_iter=50),
    gs_num_steps=200,
    gs_learning_rate=1e-3,
    su_init=True,
)
A_opt, env, E_gs = optimize_gs_ad(gate, None, config)
print(f"Ground-state energy: {E_gs:.6f}")

# Quasiparticle excitations
momenta = make_momentum_path("brillouin", num_points=20)
exc_config = ExcitationConfig(num_excitations=3)
result = compute_excitations(A_opt, env, gate, E_gs, momenta, exc_config)
print(result.energies.shape)  # (20, 3)
```

---

### iPEPS with QR Projectors

QR-based projectors replace the expensive eigendecomposition in CTM with a QR factorization, giving significant speedups for large bond dimensions:

```python
import jax.numpy as jnp
from tenax import iPEPSConfig, CTMConfig, optimize_gs_ad

Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
gate = jnp.einsum("ij,kl->ikjl", Sz, Sz) \
     + 0.5 * (jnp.einsum("ij,kl->ikjl", Sp, Sm)
             + jnp.einsum("ij,kl->ikjl", Sm, Sp))

config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(
        chi=16,
        max_iter=50,
        projector_method="qr",   # QR instead of eigh
        qr_warmup_steps=5,       # eigh warm-up before QR kicks in
    ),
    gs_num_steps=200,
    gs_learning_rate=1e-3,
    su_init=True,
)
A_opt, env, E_gs = optimize_gs_ad(gate, None, config)
print(f"Ground-state energy: {E_gs:.6f}")
```

---

### iPEPS 2-Site AD Optimization

For antiferromagnets (Neel order), use a 2-site checkerboard unit cell with independent tensors A and B. The backward pass uses implicit differentiation through the 2-site CTM fixed point:

```python
import jax.numpy as jnp
from tenax import iPEPSConfig, CTMConfig, optimize_gs_ad

Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
gate = jnp.einsum("ij,kl->ikjl", Sz, Sz) \
     + 0.5 * (jnp.einsum("ij,kl->ikjl", Sp, Sm)
             + jnp.einsum("ij,kl->ikjl", Sm, Sp))

config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=16, max_iter=50),
    gs_num_steps=200,
    gs_learning_rate=1e-3,
    unit_cell="2site",    # checkerboard A/B
    su_init=True,         # simple update initialization
    num_imaginary_steps=200,
    dt=0.01,
)
(A_opt, B_opt), (env_A, env_B), E_gs = optimize_gs_ad(gate, None, config)
print(f"Ground-state energy: {E_gs:.6f}")
```

---

### Split-CTMRG

Split-CTMRG (Rader & Lauchli, [arXiv:2502.10298](https://arxiv.org/abs/2502.10298)) keeps ket and bra layers separate in CTM edge tensors, reducing projector cost from O(chi^3 D^6) to O(chi^3 D^3):

```python
import jax.numpy as jnp
from tenax import (
    iPEPSConfig, CTMConfig, optimize_gs_ad,
    ctm_split, compute_energy_split_ctm,
)

Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
gate = jnp.einsum("ij,kl->ikjl", Sz, Sz) \
     + 0.5 * (jnp.einsum("ij,kl->ikjl", Sp, Sm)
             + jnp.einsum("ij,kl->ikjl", Sm, Sp))

# First optimize with standard CTM
config = iPEPSConfig(
    max_bond_dim=2,
    ctm=CTMConfig(chi=16, max_iter=50),
    gs_num_steps=200,
    gs_learning_rate=1e-3,
    su_init=True,
)
A_opt, env, E_gs = optimize_gs_ad(gate, None, config)

# Evaluate with split-CTMRG
split_config = CTMConfig(chi=16, max_iter=100, chi_I=10)
split_env = ctm_split(A_opt, split_config)
E = compute_energy_split_ctm(A_opt, split_env, gate, d=2)
print(f"Split-CTM energy: {E:.6f}")
```

`chi_I` controls the interlayer bond dimension between ket and bra edge tensors. Setting `chi_I = chi * D` makes the SVD split lossless; smaller values trade accuracy for speed.

---

### AutoMPO

Build Hamiltonian MPOs from symbolic operator descriptions, with support for custom operators and U(1) symmetric MPOs:

```python
from tenax import AutoMPO, build_auto_mpo
import numpy as np

# Class-based interface: Heisenberg chain
L = 10
auto = AutoMPO(L)
for i in range(L - 1):
    auto += (1.0, "Sz", i, "Sz", i + 1)
    auto += (0.5, "Sp", i, "Sm", i + 1)
    auto += (0.5, "Sm", i, "Sp", i + 1)
mpo = auto.to_mpo()

# Functional interface with custom operators
custom_ops = {
    "X": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]]),
    "Id": np.eye(2),
}
terms = [(1.0, "Z", i, "Z", i + 1) for i in range(L - 1)]
terms += [(0.5, "X", i) for i in range(L)]
mpo = build_auto_mpo(terms, L=L, site_ops=custom_ops)

# Build a symmetric (U(1) block-sparse) MPO
mpo_sym = auto.to_mpo(symmetric=True)
```

---

### Tensor Display & Mermaid Export

Tensors print as ASCII box diagrams with IN legs on the left and OUT legs on the right:

```python
# DenseTensor with 2 IN legs + 1 OUT leg
print(tensor)
#             ┌─────────┐
# phys (2) ──>┤ Dense   ├<── right (3)
# left (4) ──>┤ float64 │
#             └─────────┘

# SymmetricTensor with 2 IN legs + 2 OUT legs
print(sym_tensor)
#             ┌──────────────┐
# left (3) ──>┤ Symmetric    ├<── right (3)
# phys (2) ──>┤ U(1) float64 ├<── top (4)
#             │ 14blk nnz=14 │
#             └──────────────┘
#  charges: left{-1:1, 0:1, 1:1} phys{-1:1, 1:1}
#           right{-1:1, 0:1, 1:1} top{-1:1, 0:1, 1:1, 2:1}
```

Export a `TensorNetwork` as a [Mermaid](https://mermaid.js.org/) diagram:

```python
tn = TensorNetwork(name="MPS")
tn.add_node("A", tensor_A)
tn.add_node("B", tensor_B)
tn.connect("A", "bond", "B", "bond")
print(tn.to_mermaid())
# graph LR
#   A["A (2,3)"]
#   B["B (3,4)"]
#   A ---|bond| B
#   A -.- A_phys(("phys"))
#   B -.- B_phys(("phys"))
```

Paste the Mermaid output into GitHub markdown, Mermaid Live Editor, or VS Code to render interactive diagrams.

---

### Runnable Scripts

Complete example scripts are in the [`examples/`](https://github.com/tenax-lab/tenax/tree/main/examples) directory:

| Script | Algorithm | Model |
|--------|-----------|-------|
| `heisenberg_cylinder.py` | DMRG | Heisenberg on 4x2, 6x3, 8x4 cylinders |
| `heisenberg_infinite_cylinder.py` | iDMRG | Heisenberg on infinite Ly=2, Ly=4 cylinders |
| `heisenberg_ipeps_su.py` | iPEPS simple update | Heisenberg (1x1 and 2-site unit cells) |
| `heisenberg_ipeps_ad.py` | iPEPS AD optimization | Heisenberg (random vs SU init) |
| `heisenberg_ipeps_excitations.py` | iPEPS excitations | Heisenberg dispersion along Gamma-X-M-Gamma |
| `ising_trg.py` | TRG | 2D Ising vs Onsager exact |
| `ising_hotrg.py` | HOTRG | 2D Ising vs Onsager exact |

Run any example:

```bash
uv run python examples/<script>.py
```

</div>
</section>
