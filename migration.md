---
layout: default
title: Migration Guides
---

<div class="page-header">
  <h1>Migration Guides</h1>
  <p>Translate your existing tensor network code to Tenax</p>
</div>

<section>
<div class="container" markdown="1">

Tenax shares core ideas with other tensor network libraries — label-based contraction, symmetry-aware tensors, DMRG. These guides map concepts and code patterns from each library to their Tenax equivalents.

<div class="toc" markdown="1">

**Libraries**

- [ITensor (Julia/C++)](#itensor)
- [TeNPy](#tenpy)
- [Cytnx](#cytnx)
- [quimb](#quimb)
{: style="list-style: none; padding: 0;" }

</div>

</div>
</section>

<section id="itensor">
<div class="container" markdown="1">

## ITensor

ITensor and Tenax share label-based contraction and AutoMPO. The main differences are language (Julia/C++ vs Python/JAX) and Tenax's explicit symmetry and flow direction on every index.

### Concept Mapping

| ITensor (Julia) | Tenax (Python) | Notes |
|-----------------|----------------|-------|
| `Index(dim, "label")` | `TensorIndex(sym, charges, flow, label)` | Tenax carries symmetry + flow |
| `ITensor(idx1, idx2)` | `DenseTensor(data, indices)` | Tenax requires explicit data |
| `randomITensor(...)` | `DenseTensor.random_normal(indices, key)` | JAX needs explicit RNG key |
| `dag(idx)` | `idx.dual()` | Flip FlowDirection |
| `A * B` | `contract(A, B)` | Both label-based |
| `svd(T, i1, i2)` | `truncated_svd(T, left_labels, right_labels)` | By labels, not Index objects |
| `AutoMPO()` | `AutoMPO(L, d)` | Very similar API |
| `dmrg(H, psi0, sweeps)` | `dmrg(mpo, mps, config)` | Config replaces Sweeps object |
| `siteinds("S=1/2", N)` | `build_random_mps(L, physical_dim=2)` | No site-type system |

### Key Differences

- **No tag system** — Tenax uses a single string label per index instead of ITensor's tag-based index matching.
- **Explicit RNG** — JAX requires `jax.random.PRNGKey(seed)` for all random operations (no global state).
- **FlowDirection** — Every `TensorIndex` carries an explicit IN or OUT flow. Contracted legs must form IN/OUT pairs for `SymmetricTensor`.
- **Operator names** — Tenax uses `"Sp"` / `"Sm"` (not `"S+"` / `"S-"`) and 0-based site indexing.

### Side-by-Side: DMRG

**ITensor (Julia):**

```julia
using ITensors
N = 20
sites = siteinds("S=1/2", N)
ampo = AutoMPO()
for j in 1:N-1
    ampo += ("Sz", j, "Sz", j+1)
    ampo += (0.5, "S+", j, "S-", j+1)
    ampo += (0.5, "S-", j, "S+", j+1)
end
H = MPO(ampo, sites)
psi0 = randomMPS(sites, linkdims=10)
sweeps = Sweeps(10)
setmaxdim!(sweeps, 10, 20, 50, 100)
energy, psi = dmrg(H, psi0, sweeps)
```

**Tenax (Python):**

```python
from tenax import AutoMPO, DMRGConfig, build_random_mps, dmrg

L = 20
auto = AutoMPO(L=L, d=2)
for i in range(L - 1):
    auto += (1.0, "Sz", i, "Sz", i + 1)
    auto += (0.5, "Sp", i, "Sm", i + 1)
    auto += (0.5, "Sm", i, "Sp", i + 1)
mpo = auto.to_mpo()
mps = build_random_mps(L, physical_dim=2, bond_dim=10)
config = DMRGConfig(max_bond_dim=100, num_sweeps=10)
result = dmrg(mpo, mps, config)
```

### What You Gain

- Python ecosystem (NumPy, SciPy, matplotlib, Jupyter)
- Autodiff — `jax.grad` through any contraction
- GPU/TPU/Metal — same code on all backends
- iPEPS + excitations — built-in 2D algorithms beyond DMRG

### What You Lose

- Tag system — replaced by simple string labels
- Per-sweep bond dimension schedule — use multiple DMRG runs instead
- Built-in `expect()` / `correlation_matrix()` — compute observables manually
- TEBD / TDVP — not yet implemented
- Non-Abelian symmetry — Tenax currently supports Abelian only (U(1), Z_n)

</div>
</section>

<section id="tenpy">
<div class="container" markdown="1">

## TeNPy

TeNPy uses an object-oriented hierarchy (Site → Lattice → Model → Engine). Tenax replaces this with a functional API — build the Hamiltonian directly with AutoMPO, run algorithms as pure functions.

### Concept Mapping

| TeNPy | Tenax | Notes |
|-------|-------|-------|
| `SpinHalfSite()` | `spin_half_ops()` | Returns operator dict, no Site object |
| `MPS.from_lat_product_state(...)` | `build_random_mps(L, d, chi)` | No lattice/product-state builder |
| `MPOModel` / `CouplingMPOModel` | `AutoMPO(L, d)` | Functional, not class-based |
| `TwoSiteDMRGEngine(psi, model, params)` | `dmrg(mpo, mps, config)` | Functional API |
| `eng.run()` | `result = dmrg(mpo, mps, config)` | Returns result dataclass |
| `psi.entanglement_entropy()` | Manual from singular values | No built-in method |
| `npc.tensordot(A, B, axes)` | `contract(A, B)` | Tenax uses label matching |

### Key Differences

- **No Model/Site/Lattice classes** — Tenax is functional. Build the Hamiltonian with AutoMPO by explicitly adding each coupling term.
- **No Engine pattern** — Algorithms are pure functions (`dmrg(mpo, mps, config)`) returning a result dataclass, not mutable engine objects.
- **NumPy vs JAX** — TeNPy is pure NumPy/SciPy with no GPU or autodiff. Tenax is pure JAX with GPU/TPU support and automatic differentiation.
- **Observables** — TeNPy has rich built-in measurements (`expectation_value`, `correlation_function`, `entanglement_entropy`). Tenax requires manual contraction.

### Side-by-Side: DMRG

**TeNPy:**

```python
from tenpy.models.xxz_chain import XXZChain
from tenpy.networks.mps import MPS
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine

model = XXZChain({"L": 20, "Jxx": 1.0, "Jz": 1.0, "bc_MPS": "finite"})
psi = MPS.from_lat_product_state(model.lat, [["up"], ["down"]])
eng = TwoSiteDMRGEngine(psi, model, {"trunc_params": {"chi_max": 100}})
E, psi = eng.run()
```

**Tenax:**

```python
from tenax import AutoMPO, DMRGConfig, build_random_mps, dmrg

L = 20
auto = AutoMPO(L=L, d=2)
for i in range(L - 1):
    auto += (1.0, "Sz", i, "Sz", i + 1)
    auto += (0.5, "Sp", i, "Sm", i + 1)
    auto += (0.5, "Sm", i, "Sp", i + 1)
mpo = auto.to_mpo()
mps = build_random_mps(L, physical_dim=2, bond_dim=16)
result = dmrg(mpo, mps, DMRGConfig(max_bond_dim=100, num_sweeps=10))
```

### What You Gain

- GPU/TPU support, JIT compilation
- Autodiff — gradient-based iPEPS optimization
- iPEPS + excitations — built-in 2D algorithms
- NetworkBlueprint — reusable contraction templates

### What You Lose

- Rich model library — TeNPy has dozens of pre-built models
- Built-in observables and entanglement measures
- Product state initialization
- Per-sweep parameter schedules
- TEBD / TDVP

</div>
</section>

<section id="cytnx">
<div class="container" markdown="1">

## Cytnx

Cytnx and Tenax share a label-based contraction philosophy and `.net` file support. The biggest differences are the backend (C++ vs JAX), Tenax's removal of row/column rank, and Tenax's built-in algorithms.

### Concept Mapping

| Cytnx | Tenax | Notes |
|-------|-------|-------|
| `UniTensor` | `DenseTensor` / `SymmetricTensor` | No row/col rank in Tenax |
| `Bond` | `TensorIndex` | Carries symmetry, charges, flow, label |
| `Bond.BD_IN` / `BD_OUT` | `FlowDirection.IN` / `OUT` | Same concept |
| `Network` | `NetworkBlueprint` | Same `.net` file format |
| `Contract(A, B)` | `contract(A, B)` | Both label-based |
| `Svd(T)` | `truncated_svd(T, left_labels, right_labels)` | Explicit label partition |
| `T.set_labels(...)` | `T.relabel(old, new)` | Immutable in Tenax |

### Key Differences

- **No row/column rank** — Cytnx's `UniTensor` tracks which legs are "row" vs "column" for implicit SVD partition. Tenax requires explicit `left_labels` / `right_labels` instead.
- **Immutable tensors** — Tenax tensors are JAX pytrees. Operations return new tensors; no in-place `set_labels()`.
- **`.net` files are compatible** — The same `.net` file format works in both libraries. Tenax ignores the semicolon in `TOUT:` (no row/column rank).
- **Built-in algorithms** — Cytnx is a tensor library; DMRG, TRG, etc. must be implemented by the user. Tenax includes production-ready algorithms.

### Side-by-Side: Network Contraction

**Cytnx:**

```cpp
auto net = Network("dmrg_eff_ham.net");
net.PutUniTensor("L", L_env);
net.PutUniTensor("W", W);
net.PutUniTensor("R", R_env);
auto result = net.Launch();
```

**Tenax:**

```python
from tenax import NetworkBlueprint

bp = NetworkBlueprint("dmrg_eff_ham.net")  # Same .net file format
bp.put_tensor("L", L_env)
bp.put_tensor("W", W)
bp.put_tensor("R", R_env)
result = bp.launch()
```

### What You Gain

- Autodiff — differentiate through any contraction
- JIT compilation — `jax.jit` for automatic optimization
- Built-in algorithms — DMRG, iDMRG, TRG, HOTRG, iPEPS, excitations
- AutoMPO — symbolic Hamiltonian construction
- GPU/TPU/Metal — same code on all backends

### What You Lose

- C++ performance for small tensors — JAX has JIT overhead
- Row/column rank semantics
- Non-Abelian symmetry — Tenax supports Abelian only (U(1), Z_n)
- Richer linear algebra (`Eig`, `Inv`, `Det` on UniTensor)

</div>
</section>

<section id="quimb">
<div class="container" markdown="1">

## quimb

Both quimb and Tenax use graph-based tensor network containers with label-based contraction, but differ in backend (NumPy/autoray vs JAX), symmetry support, and algorithm scope.

### Concept Mapping

| quimb | Tenax | Notes |
|-------|-------|-------|
| `qtn.Tensor(data, inds, tags)` | `DenseTensor(data, indices)` | No tags; labels on TensorIndex |
| `qtn.TensorNetwork(...)` | `TensorNetwork()` | Similar graph container |
| `tn ^ all` | `tn.contract()` | Method, not operator |
| `A & B` | `contract(A, B)` | Pairwise contraction |
| `A.reindex({"old": "new"})` | `A.relabel("old", "new")` | Immutable in Tenax |
| `qtn.DMRG2(ham)` | `dmrg(mpo, mps, config)` | Functional API |
| `qtn.SpinHam1D(S=0.5)` | `AutoMPO(L, d=2)` | Explicit site indices |

### Key Differences

- **Tags vs labels** — quimb tensors carry both `inds` (for contraction) and `tags` (for selection/grouping). Tenax has labels only; select tensors by node ID in `TensorNetwork`.
- **Symmetry support** — quimb has no built-in symmetry-aware tensors. Tenax has first-class `SymmetricTensor` with U(1) and Z_n.
- **Backend** — quimb uses `autoray` for backend flexibility. Tenax is pure JAX with native `jit`, `grad`, and `vmap`.
- **Hamiltonian construction** — quimb's `SpinHam1D` adds terms by operator pattern (applied to all bonds). Tenax's `AutoMPO` adds terms by explicit site indices, giving full control over geometry.

### Side-by-Side: DMRG

**quimb:**

```python
import quimb.tensor as qtn

builder = qtn.SpinHam1D(S=0.5)
builder += 1.0, "Z", "Z"
builder += 0.5, "+", "-"
builder += 0.5, "-", "+"
H = builder.build_mpo(20)

dmrg = qtn.DMRG2(H, bond_dims=[10, 20, 50, 100])
dmrg.solve(tol=1e-9)
```

**Tenax:**

```python
from tenax import AutoMPO, DMRGConfig, build_random_mps, dmrg

L = 20
auto = AutoMPO(L=L, d=2)
for i in range(L - 1):
    auto += (1.0, "Sz", i, "Sz", i + 1)
    auto += (0.5, "Sp", i, "Sm", i + 1)
    auto += (0.5, "Sm", i, "Sp", i + 1)
mpo = auto.to_mpo()
mps = build_random_mps(L, physical_dim=2, bond_dim=10)
result = dmrg(mpo, mps, DMRGConfig(max_bond_dim=100, num_sweeps=10))
```

### What You Gain

- Symmetry-aware tensors — `SymmetricTensor` with U(1), Z_n
- JIT compilation — built-in, not opt-in
- Autodiff — gradient-based iPEPS optimization
- iDMRG, TRG/HOTRG, iPEPS — built-in algorithms beyond DMRG
- NetworkBlueprint — reusable `.net` file contraction templates

### What You Lose

- Tag system — flexible tensor selection/grouping
- `cotengra` integration — advanced contraction path optimization
- Visualization — `tn.draw()` for tensor network diagrams
- Backend flexibility — Tenax is JAX-only
- TEBD / TDVP

</div>
</section>
