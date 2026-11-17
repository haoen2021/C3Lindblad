# Matsubara-axis impurity solver (`c3l.matsubara`)

This subpackage turns C³L into an imaginary-axis impurity solver. Given a pseudomode (PM) model, it returns the impurity Green function $G(i\omega_n)$ and self-energy $\Sigma(i\omega_n)$ on any fermionic Matsubara grid, including the discrete Lehmann representation (DLR) grid for a given $\beta$ and accuracy $\epsilon$. It provides the interface an imaginary-time DMFT loop needs.

**How it works.** When the PM bath represents an equilibrium bath at inverse temperature $\beta$, the impurity retarded function $G^R(z)$ is analytic for $\operatorname{Im} z>0$, and

$$
G(i\omega_n)=G^R(i\omega_n)\ \ (\omega_n>0),\qquad G(i\omega_n)=\big[G^R(-i\omega_n)\big]^\dagger\ \ (\omega_n<0).
$$

Every Matsubara point is therefore one more shift $z=i|\omega_n|$ of the shift-invariant Krylov space already used for real-axis spectra. The self-energy is the PM Dyson self-energy on the impurity block $S$,

$$
\Sigma=G_0^{-1}-G^{-1},\qquad G_0(z)=\big[(z-E+i\Lambda)^{-1}\big]_{SS},\qquad \Lambda=\Gamma_{\rm loss}+\Gamma_{\rm gain},
$$

which you can re-insert into any target hybridization $\Delta(i\omega_n)$.

## Install

Explicit Matsubara grids need nothing beyond the standard installation. DLR grids use [`pydlr`](https://github.com/jasonkaye/libdlr):

```bash
python -m pip install "c3l[dlr]"
```

## Quick start

```python
import numpy as np
from c3l import FermionicLindbladModel
from c3l.matsubara import dlr_matsubara_grid, matsubara_grid, solve_matsubara

model = FermionicLindbladModel(
    E=np.array([[-0.6, 0.28, 0.28], [0.28, -1.2, 0.0], [0.28, 0.0, 1.2]]),
    gamma_loss=np.diag([0.0, 0.06, 0.48]),
    gamma_gain=np.diag([0.0, 0.48, 0.06]),
    U=1.2,
)
beta = 10.0
grid = dlr_matsubara_grid(beta, eps=1e-10, wmax=4.0)   # or matsubara_grid(beta, n_iw=128)

solution = solve_matsubara(model, grid)    # CCSD + S13 response on the grid
G_iw = solution.green                      # shape (len(grid), 1, 1): impurity site 0
Sigma_iw = solution.self_energy
```

This toy model only illustrates the calls. For physically meaningful Matsubara data, the bath must be fitted to an equilibrium bath at the same $\beta$. [`examples/basic/05_matsubara_dlr.py`](../../../examples/basic/05_matsubara_dlr.py) does the full workflow: it fits a semicircle bath with RealTimeBath, solves on the DLR grid, inserts $\Sigma$ into the exact semicircle $\Delta(i\omega_n)$ and transforms to $G(\tau)$. It runs in about 15 s on one core.

## Choosing a grid

| Constructor | Grid |
| --- | --- |
| `matsubara_grid(beta, n_iw=N)` | symmetric window $n=-N,\dots,N-1$ (the ordering of TRIQS `MeshImFreq(beta, 'Fermion', N)`) |
| `matsubara_grid(beta, indices=n)` | the integers $n$, in your order |
| `matsubara_grid(beta, omega_n=w)` | real frequencies $\omega_n$ or the points $i\omega_n$, e.g. the nodes of an external DLR mesh; they must lie on the fermionic lattice of `beta` |
| `dlr_matsubara_grid(beta, eps, wmax=...)` or `lamb=...` | fermionic DLR nodes for cutoff $\Lambda=\beta\,\omega_{\max}$ and accuracy $\epsilon$ |

**DLR cutoff.** Besides $\beta$ and $\epsilon$, the DLR needs a cutoff: $\omega_{\max}$ must bound the support of the impurity spectral function. For Hubbard-type impurities, half the bandwidth plus $U$ is a safe choice. The number of nodes grows only like $\log\Lambda\,\log(1/\epsilon)$, so err on the large side.

**Grid attributes.** Every `MatsubaraGrid` has:
- `indices`, `omega_n`, `iw` and `len(grid)`.
- `upper_shifts`: the distinct points $i|\omega_n|$ that a retarded solver must evaluate.
- `from_upper(values)`: maps those values back onto the grid, using the adjoint for negative frequencies.
- On a DLR grid: `dlr_tau`, `dlr_matsubara`, `dlr_coefficients` and `dlr_poles` (see below), and `dlr`, the underlying `pydlr.dlr` basis (`None` on explicit grids).

## Running the solver

`solve_matsubara(model, grid, **options)` compiles and solves CCSD, builds the EOM response and evaluates it on the grid.

| Option | Default | Meaning |
| --- | --- | --- |
| `model` | — | a `FermionicLindbladModel` (default full-HF reference), or a compiled `ComponentStaticProblem` with a custom reference |
| `sites` | `model.interaction_sites` | impurity block $S$: spatial sites in spin mode, physical spin-orbitals in charge mode; must contain every interacting site |
| `rank` | `"S13"` | response space, `"S13"` or `"S135"` |
| `response_spin` | `0` | spin block in spin mode (charge mode returns full spin-orbital matrices) |
| `amplitudes` | `None` | converged CCSD amplitudes; skips the static solve |
| `policy` | `None` | `NewtonKrylovPolicy` for the static solve |
| `platform` | `None` | `None` for the NumPy kernels; `"cpu"` or `"gpu"` for the JAX static solver and response action (set `JAX_ENABLE_X64=true`) |
| `production` | `False` | passed to the JAX operators; `True` enforces that the requested device is the one used |
| `maximum_krylov_dimension`, `residual_tolerance`, `minimum_krylov_dimension`, `check_interval` | `320`, `1e-10`, `16`, `8` | Arnoldi controls, as in `component_retarded_matrix_green_arnoldi` |

The result (`MatsubaraImpuritySolution`) holds:

| Field | Content |
| --- | --- |
| `green` | PM impurity $G(i\omega_n)$, axes `(n, a, b)` = (frequency, sink, source) over `sites` |
| `self_energy` | $\Sigma(i\omega_n)=G_0^{-1}-G^{-1}$, including the static (Hartree–Fock) part |
| `noninteracting`, `hybridization` | $G_0(i\omega_n)$ and $\Delta_{\rm PM}(i\omega_n)=i\omega_n-h-G_0^{-1}$ |
| `one_body` | $h=E_{SS}$, including the chemical potential |
| `amplitudes`, `static_residual` | CCSD amplitudes and their residual norm |
| `response` | the `MatsubaraGreen` result, with `max_relative_residual` and the raw Arnoldi data at `upper_shifts` |
| `dyson_terms` | the `MatsubaraSelfEnergy` record, with condition numbers of $G_0$ and $G$ |

In spin mode, compute the other spin by reusing the amplitudes:

```python
down = solve_matsubara(model, grid, response_spin=1, amplitudes=solution.amplitudes)
```

## Inserting Σ into another bath

`solution.dyson(Delta_iw)` returns $[i\omega_n-h-\Delta(i\omega_n)-\Sigma(i\omega_n)]^{-1}$ for any hybridization on the same grid. Here $\Delta(i\omega_n)=\int d\omega\, J(\omega)/(i\omega_n-\omega)$, or $\sum_k V_kV_k^\dagger/(i\omega_n-\epsilon_k)$, with shape `(len(grid), m, m)`. Pass the continuous bath the pseudomodes were fitted to (or the current DMFT hybridization) to obtain the reconstructed impurity Green function. `solution.dyson(solution.hybridization)` returns `solution.green` exactly. The standalone form is `matsubara_dyson_green(grid, h, Delta_iw, Sigma_iw)`.

## Imaginary time with the DLR

A DLR grid expands values on its nodes as $G(i\omega_n)=\sum_l g_l/(i\omega_n-\omega_l)$, with real poles $\omega_l$ (`grid.dlr_poles`):

```python
G_tau = grid.dlr_tau(G_iw, tau)              # G(tau) for any tau in [0, beta]
G_dense = grid.dlr_matsubara(G_iw, dense.iw)  # G at any other Matsubara points
g = grid.dlr_coefficients(G_iw)              # the coefficients g_l themselves
```

These work for scalar `(n,)` or matrix `(n, m, m)` data and need only NumPy.

- **pydlr's own transforms:** `grid.dlr` is the `pydlr.dlr` basis, but its transforms are best avoided for matrix data. In pydlr 1.0.1, `dlr_from_matsubara` fails on `(n, m, m)` arrays with SciPy ≥ 1.18, `matsubara_from_dlr` returns the orbital transpose, and `eval_dlr_freq` assumes Hermitian coefficients.
- **Accuracy limit:** the PM Green function has Lorentzian (finite-lifetime) tails, so DLR interpolation of the raw PM $G$ is limited to roughly $10^{-8}$ even at $\epsilon=10^{-10}$. Check it against `solve_matsubara(model, matsubara_grid(beta, n_iw=...), amplitudes=solution.amplitudes)` when it matters.

## Bringing your own response

The grid and Dyson layers do not depend on how $G^R$ was computed. For example, you might use GPU-resident Arnoldi for a large S135 space, or combine real-axis and Matsubara shifts in one run. Then:

```python
from c3l.matsubara import matsubara_self_energy

G_upper = ...                       # G^R at grid.upper_shifts, shape (len(grid.upper_shifts), m, m)
G = grid.from_upper(G_upper)
terms = matsubara_self_energy(model, grid, G, sites=sites)   # terms.self_energy, terms.hybridization, ...
```

At a lower level:
- `component_matsubara_green(operator, sources, grid)` evaluates any exact response operator: `ComponentSigmaOperator`, `FullS135ResponseOperator`, or their JAX versions.
- `c3l.cc.component_response.component_retarded_matrix_green_shifts(operator, sources, shifts)` accepts arbitrary upper-half-plane shifts, e.g. `np.r_[omega + 1j*eta, grid.upper_shifts]`.

## Sketch of a DMFT loop

```python
grid = dlr_matsubara_grid(beta, eps, wmax=wmax)
Delta_iw = initial_hybridization(grid.iw)
for iteration in range(n_iterations):
    model = pseudomode_model_from(Delta_iw, beta)   # bath fit, see below
    solution = solve_matsubara(model, grid, sites=impurity_sites)
    Sigma_iw = solution.self_energy
    G_loc = lattice_local_green(grid.iw, Sigma_iw)  # k-sum, user supplied
    Delta_iw = grid.iw[:, None, None] * I - h_imp - Sigma_iw - np.linalg.inv(G_loc)
```

**Not provided yet:** `pseudomode_model_from`. C³L's bath fitting (`c3l.fitting`, RealTimeBath) fits the bath's lesser and greater real-time correlation functions. Going from a Matsubara $\Delta(i\omega_n)$ to those correlations requires the bath spectral function, for example via an analytic continuation of $\Delta$ or a pole/DLR fit with positive weights. Once a PM model exists, `matrix_pseudomode_model` embeds it and this package does the rest.

## Limitations

- **Equilibrium only.** The Matsubara identification assumes the PM bath represents an equilibrium bath at the same $\beta$; nonequilibrium (biased) baths have no Matsubara function.
- **`sites` must contain every interacting site;** otherwise $G_0^{-1}-G^{-1}$ is not the impurity self-energy, and the call raises `ValueError`.
- **Spin mode returns one spin block per call.**
- **The default full-HF reference can fail at strong coupling.** Given a model, `solve_matsubara` builds the default general full-HF reference. This may not converge, e.g. for the particle-hole-symmetric SIAM at $U=6\Gamma$, where it raises `RuntimeError: the default general full-HF reference did not converge`. For half-filled single-orbital models, use the reference of the SIAM examples instead and pass the compiled problem:

  ```python
  from c3l.cc.component_tensor import ComponentStaticProblem
  from c3l.cc.factorized import factorized_cc_problem

  static = ComponentStaticProblem.compile(factorized_cc_problem(
      model, "CCSD", m_up=0.5, reference_mode="legacy_half_filled"))
  solution = solve_matsubara(static, grid, rank="S135")
  ```
- **The NumPy/JAX Arnoldi used here keeps the Krylov basis in host memory.** For very large S135 spaces, use the GPU-resident path and pass its output through `grid.from_upper` (see above).

## Layout and tests

| File | Content |
| --- | --- |
| `grid.py` | `MatsubaraGrid`, `matsubara_grid`, `dlr_matsubara_grid` |
| `dyson.py` | `impurity_sites`, PM $G_0$ and $\Delta_{\rm PM}$, `matsubara_self_energy`, `matsubara_dyson_green` |
| `solver.py` | `component_matsubara_green`, `solve_matsubara` |

[`tests/test_matsubara.py`](../../../tests/test_matsubara.py) covers:
- **Exact oracle:** against exact Lindblad $G^R(i|\omega_n|)$ for S13 and S135, including negative frequencies.
- **U = 0:** $\Sigma=0$, with the analytic PM hybridization.
- **Symmetry:** spin mode against charge mode.
- **Backends:** JAX against NumPy.
- **DLR:** nodes against `pydlr`, and τ and Matsubara interpolation of a non-Hermitian matrix pole model.
