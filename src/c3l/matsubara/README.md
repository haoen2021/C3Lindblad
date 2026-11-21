# Matsubara Green functions

`c3l.matsubara` computes the impurity Green function $`G(\mathrm{i}\omega_n)`$ and self-energy $`\Sigma(\mathrm{i}\omega_n)`$ on a fermionic Matsubara grid, for use as an impurity solver in imaginary-time DMFT. The pseudomode bath must be fitted to an equilibrium bath at the same inverse temperature $`\beta`$. Then

$$
G(\mathrm{i}\omega_n)=G^R(\mathrm{i}\omega_n)\quad(\omega_n>0),\qquad G(\mathrm{i}\omega_n)=G^R(-\mathrm{i}\omega_n)^\dagger\quad(\omega_n<0),
$$

so each Matsubara point is one more shift $`z=\mathrm{i}\lvert\omega_n\rvert`$ of the Krylov space used for real-frequency spectra. The self-energy is $`\Sigma=G_0^{-1}-G^{-1}`$ on the impurity sites $`S`$, with the pseudomode noninteracting Green function

$$
G_0(z)=\big[(z-E+\mathrm{i}\Lambda)^{-1}\big]_{SS},\qquad \Lambda=\Gamma_{\mathrm{loss}}+\Gamma_{\mathrm{gain}}.
$$

## Install

Explicit grids need only the standard [installation](../../../README.md#install). DLR grids require [pydlr](https://github.com/jasonkaye/libdlr):

```bash
python -m pip install "c3l[dlr]"
```

## Quick start

This CPU example computes $`G(\mathrm{i}\omega_n)`$ and $`\Sigma(\mathrm{i}\omega_n)`$ on a DLR grid for the model of the [quick start](../../../README.md#quick-start). That bath is not fitted to an equilibrium bath, so the example only shows the calls.

<details>
<summary>Show the Matsubara example</summary>

```python
import numpy as np
from c3l import FermionicLindbladModel
from c3l.matsubara import dlr_matsubara_grid, solve_matsubara

model = FermionicLindbladModel(
    E=np.array([[-0.6, 0.28, 0.28], [0.28, -1.2, 0.0], [0.28, 0.0, 1.2]]),
    gamma_loss=np.diag([0.0, 0.06, 0.48]),
    gamma_gain=np.diag([0.0, 0.48, 0.06]),
    U=1.2,
)
grid = dlr_matsubara_grid(beta=10.0, eps=1e-10, wmax=4.0)
solution = solve_matsubara(model, grid)
print(f"Arnoldi residual: {solution.response.max_relative_residual:.2e}")
np.savez("matsubara.npz", iw=grid.iw, G=solution.green, Sigma=solution.self_energy)
```

`matsubara.npz` contains $`G`$ and $`\Sigma`$ with axes (frequency, sink, source) over the impurity sites.

</details>

## Grids

| Grid | Matsubara points |
| --- | --- |
| `matsubara_grid(beta, n_iw=N)` | $`n=-N,\dots,N-1`$, as in TRIQS `MeshImFreq` |
| `matsubara_grid(beta, indices=n)` | the integers $`n`$, in the given order |
| `matsubara_grid(beta, omega_n=w)` | frequencies $`\omega_n`$ or points $`\mathrm{i}\omega_n`$ on the fermionic lattice of `beta` |
| `dlr_matsubara_grid(beta, eps, wmax=w)` | DLR nodes for accuracy `eps` and cutoff $`\Lambda=\beta\omega_{\max}`$; or give `lamb` |

Here $`\omega_n=(2n+1)\pi/\beta`$. For the DLR, $`\omega_{\max}`$ must bound the impurity spectrum: use the half bandwidth plus $`U`$ for Hubbard-type impurities. The number of DLR nodes grows only as $`\log\Lambda\,\log(1/\epsilon)`$, so choose $`\omega_{\max}`$ generously. Only distinct $`\lvert\omega_n\rvert`$ are computed (`grid.upper_shifts`); `grid.from_upper` fills negative frequencies with the adjoint.

## Solver options

`solve_matsubara(model, grid, ...)` solves CCSD, builds the EOM response and evaluates it on the grid.

+ `rank`: `"S13"` (default) or `"S135"`.
+ `platform`: `None` (default) for the NumPy kernels, or `"cpu"`/`"gpu"` for the JAX solvers with `JAX_ENABLE_X64=true`. `production=True` requires the requested device.
+ `sites`: impurity sites, spatial sites in spin mode and spin-orbitals in charge mode. The default is `model.interaction_sites`; every interacting site must be included.
+ `response_spin`: spin block in spin mode (default 0). For the other spin, pass `amplitudes=solution.amplitudes` to skip the static solve.
+ `policy`: `NewtonKrylovPolicy` for the static solve.
+ `maximum_krylov_dimension`, `residual_tolerance`, `minimum_krylov_dimension`, `check_interval` (defaults 320, 1e-10, 16, 8): Arnoldi controls. The Arnoldi stops at `maximum_krylov_dimension` without an error; check `solution.response.max_relative_residual`.

`model` is a `FermionicLindbladModel` with the default full-HF reference, or a compiled `ComponentStaticProblem` with another reference.

## Output

`solution.green` and `solution.self_energy` have axes (frequency, sink, source) over `solution.sites`; $`\Sigma`$ includes the static Hartree–Fock part. `solution.noninteracting` and `solution.hybridization` are $`G_0`$ and $`\Delta_{\mathrm{PM}}=\mathrm{i}\omega_n-h-G_0^{-1}`$, with $`h=E_{SS}`$ in `solution.one_body` (including the chemical potential). `solution.amplitudes` and `solution.static_residual` hold the CCSD solution.

To insert $`\Sigma`$ into another hybridization on the same grid, e.g. the continuous bath the pseudomodes were fitted to, use

```python
G = solution.dyson(Delta_iw)  # [i w_n - h - Delta(i w_n) - Sigma(i w_n)]^{-1}; Delta_iw has shape (len(grid), m, m)
```

## Imaginary time

On a DLR grid, `grid.dlr_tau(G_iw, tau)` returns $`G(\tau)`$ for $`\tau\in[0,\beta]`$, and `grid.dlr_matsubara(G_iw, iw)` returns $`G`$ at other Matsubara points. `grid.dlr_coefficients` and `grid.dlr_poles` give the expansion $`G(\mathrm{i}\omega_n)=\sum_l g_l/(\mathrm{i}\omega_n-\omega_l)`$. Use these methods rather than pydlr's transforms for matrix data. The Lorentzian tails of the pseudomode $`G`$ limit DLR interpolation of `solution.green` to about $`10^{-8}`$; check against an explicit grid when this matters.

## Other response solvers

$`G^R`$ at `grid.upper_shifts` from any solver, e.g. GPU-resident Arnoldi, can be mapped with `grid.from_upper` and passed to `matsubara_self_energy(model, grid, G, sites=...)`. `component_matsubara_green(operator, sources, grid)` evaluates any exact response operator. `c3l.cc.component_response.component_retarded_matrix_green_shifts` accepts any shifts with $`\operatorname{Im} z>0`$, so real-frequency and Matsubara points can share one Krylov space.

## Example and tests

[05_matsubara_dlr.py](../../../examples/basic/05_matsubara_dlr.py) fits a semicircle bath with RealTimeBath, solves on the DLR grid, inserts $`\Sigma`$ into the exact semicircle hybridization and transforms to $`G(\tau)`$. It takes about 15 s on one CPU core; `--modes` sets the pseudomodes per branch.

```bash
python /path/to/C3L/examples/basic/05_matsubara_dlr.py --output /path/to/work/matsubara_dlr.npz
```

[test_matsubara.py](../../../tests/test_matsubara.py) compares with exact Lindblad Green functions for S13 and S135; run it with `python -m pytest tests/test_matsubara.py`.
