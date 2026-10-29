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
- [Gilt-TNR](#gilt-tnr)
- [iPEPS Simple Update](#ipeps-simple-update)
- [iPEPS AD Optimization & Excitations](#ipeps-ad-optimization--excitations)
- [iPEPS with QR Projectors](#ipeps-with-qr-projectors)
- [iPEPS 2-Site Shared-Tensor C4v AD](#ipeps-2-site-shared-tensor-c4v-ad)
- [iPEPS with L-BFGS Optimizer](#ipeps-with-l-bfgs-optimizer)
- [Split-CTMRG](#split-ctmrg)
- [AutoMPO](#autompo)
- [Tensor Display & Mermaid Export](#tensor-display--mermaid-export)
- [QTT — Quantic Tensor Trains](#qtt--quantic-tensor-trains)
- [Runnable Scripts](#runnable-scripts)
{: style="list-style: none; padding: 0;" }

</div>

---

### DMRG Ground State

Finite DMRG for the spin-1/2 Heisenberg chain using AutoMPO:

```python
import jax
from tenax import AutoMPO, DMRGConfig, FiniteMPS, dmrg

L = 20
auto = AutoMPO(L)
for i in range(L - 1):
    auto += (1.0, "Sz", i, "Sz", i + 1)
    auto += (0.5, "Sp", i, "Sm", i + 1)
    auto += (0.5, "Sm", i, "Sp", i + 1)

mpo = auto.to_mpo()
mps = FiniteMPS.random(L=L, d=2, chi=4, key=jax.random.PRNGKey(0))
result = dmrg(mpo, mps, DMRGConfig(max_bond_dim=100, num_sweeps=10))
print(f"Ground state energy: {result.energy:.8f}")
```

---

### 2D Cylinder DMRG

Heisenberg model on a 6x3 cylinder mapped to a 1D chain via AutoMPO:

```python
import jax
from tenax import AutoMPO, DMRGConfig, FiniteMPS, dmrg

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
mps = FiniteMPS.random(L=N, d=2, chi=16, key=jax.random.PRNGKey(0))
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

Heisenberg model on an infinite cylinder with circumference $L_y$:

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

`ising_free_energy_exact` returns the free energy per site $f = -\ln Z / (N\beta)$, so compare it with `-log_z_per_n / beta`.

---

### Gilt-TNR

GILT (graph-independent local truncation) removes corner-double-line entanglement before every TRG step; `gilt_tnr` is a drop-in counterpart of `trg` (and `gilt_hotrg` of `hotrg`):

```python
from tenax import (
    GiltConfig, GiltTNRConfig, TRGConfig, gilt_tnr, trg,
    compute_ising_tensor, ising_free_energy_exact,
)

beta_c = 0.44068679350977147  # Onsager critical point
T = compute_ising_tensor(beta_c)

log_z_gilt = gilt_tnr(T, GiltTNRConfig(max_bond_dim=8, num_steps=20,
                                       gilt=GiltConfig(gilt_eps=1e-6)))
log_z_trg = trg(T, TRGConfig(max_bond_dim=8, num_steps=20))
f_exact = ising_free_energy_exact(beta_c)
print(f"Gilt-TNR error: {abs(-log_z_gilt / beta_c - f_exact):.1e}")  # ~1e-4
print(f"TRG error:      {abs(-log_z_trg / beta_c - f_exact):.1e}")   # ~5e-3
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
    dt=0.05,
    ctm=CTMConfig(chi=10, max_iter=40),
    unit_cell="2site",
)
energy, peps, (env_A, env_B) = ipeps(gate, None, config)
print(f"Energy per site: {energy:.6f}")  # ~ -0.63
```

---

### iPEPS AD Optimization & Excitations

Gradient-based ground state optimization via implicit differentiation through the CTM fixed point, followed by quasiparticle excitation spectra:

```python
from tenax import (
    iPEPSConfig, CTMConfig, optimize_gs_ad,
    heisenberg_gate, sublattice_rotate_gate,
    ExcitationConfig, compute_excitations, make_momentum_path,
)

# A 1-site (uniform) ansatz needs the sublattice-rotated Heisenberg gate
H_rot = sublattice_rotate_gate(heisenberg_gate())

# AD ground-state optimization (L-BFGS, C4v-symmetric tensor)
config = iPEPSConfig(
    max_bond_dim=2,
    num_imaginary_steps=200,
    dt=0.05,
    ctm=CTMConfig(chi=16, max_iter=80, conv_tol=1e-8, projector_method="qr"),
    gs_num_steps=50,
    gs_c4v=True,
    su_init=True,
)
A_opt, env, E_gs = optimize_gs_ad(H_rot, None, config)
print(f"Ground-state energy: {E_gs:.6f}")  # ~ -0.66

# Quasiparticle excitations
momenta = make_momentum_path("brillouin", num_points=20)
exc_config = ExcitationConfig(num_excitations=3)
result = compute_excitations(A_opt, env, H_rot, E_gs, momenta, exc_config)
print(result.energies.shape)  # (20, 3)
```

> **Backward method:** The implicit-differentiation backward solves the
> adjoint fixed-point equation with a fused Neumann-series loop and falls
> back to GMRES when that loop does not converge (`CTMConfig(adjoint_method="gmres")`
> goes straight to GMRES).

> **Defaults changed in v0.9.0:** `gs_conv_criterion="grad_norm"` (expect more
> steps, and `converged=False` where earlier versions reported `True`),
> `CTMConfig.projector_backward="auto"` (the gradient includes the projector
> response; `"standard"` restores the old behaviour), `gs_stall_recovery="reset"`
> on 1-site cells too, and `CTMConfig.adjoint_chain_rule="auto"` (the joint VJP:
> same gradient, shorter compile).

> **Block-sparse AD:** `CTMConfig(ctm_ad_mode="root_implicit_symmetric")` runs
> the root-implicit engine on `SymmetricTensor` site tensors (1x1 only; pass an
> explicit symmetric `A_init`). It is expensive (~180 s per gradient at D=2,
> χ=4) and its gradient accuracy is not yet established — see the
> [AD paths guide](https://github.com/tenax-lab/tenax/blob/v0.9.0/docs/guide/algorithms/ipeps_ad_paths.md).

#### Growing chi inside CTM

Start CTM at a small chi and let it grow up to `chi_max` while it converges (the variPEPS in-CTM chi bump), so the optimizer always sees an environment converged at the current chi:

```python
from tenax import CTMConfig, iPEPSConfig, optimize_gs_ad

config = iPEPSConfig(
    max_bond_dim=2,
    dt=0.05,
    ctm=CTMConfig(chi=8, chi_max=32, max_iter=80, ctmrg_heuristic_increase_chi=True),
    gs_num_steps=50,
    gs_c4v=True,
)
A, env, E = optimize_gs_ad(H_rot, None, config)
```

The older scheduled `CTMConfig.chi_ramp` still works but is deprecated (#512).

---

### iPEPS with QR Projectors

QR-based projectors replace the SVD in CTM with a QR factorization, giving significant speedups for large bond dimensions:

```python
from tenax import (
    iPEPSConfig, CTMConfig, optimize_gs_ad,
    heisenberg_gate, sublattice_rotate_gate,
)

H_rot = sublattice_rotate_gate(heisenberg_gate())

config = iPEPSConfig(
    max_bond_dim=2,
    dt=0.05,
    ctm=CTMConfig(
        chi=16,
        max_iter=80,
        projector_method="qr",   # QR instead of the default SVD
        qr_warmup_steps=5,       # SVD warm-up sweeps before QR kicks in
    ),
    gs_num_steps=50,
    gs_c4v=True,
    su_init=True,
)
A_opt, env, E_gs = optimize_gs_ad(H_rot, None, config)
print(f"Ground-state energy: {E_gs:.6f}")
```

---

### iPEPS 2-Site Shared-Tensor C4v AD

For antiferromagnets (Neel order) on the square lattice, prefer the **2-site shared-tensor C4v path**: Tenax optimizes a single C4v-parameterized tensor `A` and derives `B` from `A` by sublattice rotation (`B = e^{iπσ^y/2}` on the physical leg). This ties the two sublattices together and avoids the A/B drift that makes the unconstrained 2-site AD path unstable. Spin-1/2 (d=2) only.

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
    ctm=CTMConfig(chi=16, max_iter=100, min_iter=50),
    gs_optimizer="lbfgs",
    gs_explicit_ad=True,
    gs_explicit_ad_steps=10,
    gs_explicit_ad_warmup=2,
    gs_num_steps=50,
    gs_line_search=True,
    unit_cell="2site",    # checkerboard A/B
    gs_c4v=True,          # shared-tensor C4v parameterization
    su_init=True,
    num_imaginary_steps=100,
    dt=0.05,
)
(A_opt, B_opt), (env_A, env_B), E_gs = optimize_gs_ad(gate, None, config)
print(f"Ground-state energy: {E_gs:.6f}")
```

---

### iPEPS with L-BFGS Optimizer

L-BFGS with Armijo backtracking line search converges faster than Adam near the minimum. Each line search trial runs fresh CTM convergence:

```python
from tenax import (
    iPEPSConfig, CTMConfig, optimize_gs_ad,
    heisenberg_gate, sublattice_rotate_gate,
)

H_rot = sublattice_rotate_gate(heisenberg_gate())

config = iPEPSConfig(
    max_bond_dim=2,
    dt=0.05,
    ctm=CTMConfig(chi=16, max_iter=80),
    gs_optimizer="lbfgs",          # L-BFGS with line search (the default)
    gs_num_steps=30,
    gs_line_search_max_steps=8,
    gs_c4v=True,
    su_init=True,
)
A_opt, env, E_gs = optimize_gs_ad(H_rot, None, config)
print(f"Ground-state energy: {E_gs:.6f}")
```

Available optimizers: `"lbfgs"` (default), `"adam"` (cosine lr decay), `"cg"`.

---

### Split-CTMRG

Split-CTMRG (Naumann, Weerda, Eisert, Rizzi & Schmoll, [arXiv:2502.10298](https://arxiv.org/abs/2502.10298)) keeps ket and bra layers separate in CTM edge tensors, reducing projector cost from $O(\chi^3 D^6)$ to $O(\chi^3 D^3)$:

```python
from tenax import (
    iPEPSConfig, CTMConfig, optimize_gs_ad,
    heisenberg_gate, sublattice_rotate_gate,
    ctm_split, compute_energy_split_ctm,
)

H_rot = sublattice_rotate_gate(heisenberg_gate())

# First optimize with standard CTM
config = iPEPSConfig(
    max_bond_dim=2,
    dt=0.05,
    ctm=CTMConfig(chi=16, max_iter=80, projector_method="qr"),
    gs_num_steps=50,
    gs_c4v=True,
    su_init=True,
)
A_opt, env, E_gs = optimize_gs_ad(H_rot, None, config)

# Evaluate with split-CTMRG (takes raw arrays)
A = A_opt.todense()
split_config = CTMConfig(chi=16, max_iter=100, chi_I=10)
split_env = ctm_split(A, split_config)
E = compute_energy_split_ctm(A, split_env, H_rot.todense(), d=2)
print(f"Split-CTM energy: {E:.6f}")
```

`chi_I` controls the interlayer bond dimension between ket and bra edge tensors. Setting $\chi_I = \chi \cdot D$ makes the SVD split lossless; smaller values trade accuracy for speed.

Split-CTMRG is dense-only: `ctm_split_tensor` and `CTMConfig(fuse_virtual_legs=False)` raise `NotImplementedError` on `SymmetricTensor` input. Symmetric iPEPS uses the default fused CTM.

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

### QTT — Quantic Tensor Trains

Compress a function on an exponentially fine grid using tensor cross interpolation from [tenax-qtt](https://github.com/tenax-lab/tenax-qtt):

```python
import math
from tenax_qtt import UniformGrid, GridSpec, cross_interpolation

# Define a 1D grid: 2^10 = 1024 points on [0, 2pi]
grid = GridSpec(variables=(UniformGrid(0, 2 * math.pi, 10),), layout="grouped")

# Build a QTT from sin(x) via tensor cross interpolation
result = cross_interpolation(lambda x: math.sin(x[0]), grid, tol=1e-8)
qtt = result.qtt

# Evaluate at a point
val = qtt.evaluate((1.0,))
print(f"sin(1.0) ~ {val.real:.6f}  (exact: {math.sin(1.0):.6f})")

# Integrate sin(x) over [0, 2pi] (exact answer: 0)
integral = qtt.integrate()
print(f"integral ~ {integral.real:.6e}")

# Bond dimensions (measures compression)
print(f"bond dims: {qtt.bond_dims}")
```

`pip install tenax-qtt` (coming soon) | [More examples](https://github.com/tenax-lab/tenax-qtt/tree/main/examples)

---

### Runnable Scripts

Complete example scripts are in the [`examples/`](https://github.com/tenax-lab/tenax/tree/main/examples) directory:

| Script | Algorithm | Model |
|--------|-----------|-------|
| `heisenberg_cylinder.py` | DMRG | Heisenberg on 4x2, 6x3, 8x4 cylinders |
| `heisenberg_infinite_cylinder.py` | iDMRG | Heisenberg on infinite $L_y=2$, $L_y=4$ cylinders |
| `heisenberg_ipeps_su.py` | iPEPS simple update | Heisenberg (1x1 and 2-site unit cells) |
| `heisenberg_ipeps_ad.py` | iPEPS AD optimization | Heisenberg (random vs SU init) |
| `heisenberg_ipeps_excitations.py` | iPEPS excitations | Heisenberg dispersion along Gamma-X-M-Gamma |
| `ising_trg.py` | TRG | 2D Ising vs Onsager exact |
| `ising_hotrg.py` | HOTRG | 2D Ising vs Onsager exact |
| `gilt_hotrg_ising.py` | Gilt-HOTRG | 2D Ising critical point |

Run any example:

```bash
uv run python examples/<script>.py
```

</div>
</section>
