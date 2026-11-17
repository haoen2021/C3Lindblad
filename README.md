# C³L (Coupled-Cluster Coupled Lindblad)

[![CI](https://github.com/haoen2021/C3Lindblad/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/haoen2021/C3Lindblad/actions/workflows/ci.yml)

<p align="center">
  <img src="C3L/c3l.svg" alt="C³L — Coupled-Cluster Coupled Lindblad" width="480">
</p>

C³L (Coupled-Cluster Coupled Lindblad, "C cubed L") applies coupled-cluster theory to non-Markovian impurity dynamics. By representing bath memory with coupled Lindblad pseudomodes, it provides a scalable route to correlated steady states, dynamical Green functions, spectra, and nonequilibrium transport in multiorbital systems with general two-body interactions.

## Install

C³L requires Python 3.11 or newer.

```bash
python -m pip install c3l
```

The installation includes JAX and [RealTimeBath](https://github.com/Hertz4/RealTimeBath) for bath fitting. In Python, use `import c3l`.

CPU and GPU calculations use the same installation. On Linux x86-64 and ARM64, CUDA 13 dependencies are installed automatically. GPU execution requires a compatible NVIDIA GPU and driver; see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

For some examples using additional utilities, install the optional benchmarking tools:

```bash
python -m pip install "c3l[benchmark]"
```

### From source

From the root of a downloaded or cloned source tree:

```bash
python -m pip install .
```

<details>
<summary>For developers</summary>

For development, use `python -m pip install -e ".[test]"`; run the tests with `python -m pytest`.

</details>

## Quick start

This CPU example computes a CCSD steady state and its $\mathcal S_{1,3}$ spectral function for an impurity coupled to two bath orbitals.

<details>
<summary>Show the CCSD and response example</summary>

```python
import numpy as np
from c3l import FermionicLindbladModel
from c3l.cc.factorized import factorized_cc_problem
from c3l.cc.component_tensor import (
    ComponentStaticProblem, ComponentEOMProblem, component_eom_context, component_eom_sources,
)
from c3l.cc.component_solver import solve_component_ccsd
from c3l.cc.eom import eom_ccsd_problem
from c3l.cc.component_response import ComponentSigmaOperator, component_retarded_green

model = FermionicLindbladModel(
    E=np.array([[-0.6, 0.28, 0.28], [0.28, -1.2, 0.0], [0.28, 0.0, 1.2]]),
    gamma_loss=np.diag([0.0, 0.06, 0.48]),
    gamma_gain=np.diag([0.0, 0.48, 0.06]),
    U=1.2,
)
problem = ComponentStaticProblem.compile(
    factorized_cc_problem(model, "CCSD")
)
solution = solve_component_ccsd(problem)
print(f"CCSD residual: {solution.packed_residual_norm:.2e}")

eom = eom_ccsd_problem(problem.oracle)
context = component_eom_context(
    ComponentEOMProblem.compile(eom, problem, charge=1), solution.amplitudes
)
omega = np.linspace(-4.0, 4.0, 81)
response = component_retarded_green(
    ComponentSigmaOperator(context), component_eom_sources(context), omega, eta=0.0
)
np.savez("spectrum.npz", omega=omega, G_R=response.retarded, A=-response.retarded.imag / np.pi)
```

`spectrum.npz` contains the impurity Green function and spectral density $A(\omega)=-\Im G^R(\omega)/\pi$ per spin.

</details>

### Matsubara Green functions and self-energies

For an equilibrium bath fitted at inverse temperature $\beta$, the impurity Green function on the Matsubara axis is the retarded response at $z=i|\omega_n|$ (with $G(-i\omega_n)=G(i\omega_n)^\dagger$), so all Matsubara points share one Krylov space. `c3l.matsubara` returns $G(i\omega_n)$ and the self-energy $\Sigma=G_{0,\mathrm{PM}}^{-1}-G^{-1}$ on a given grid, or on the DLR grid for given $\beta$, accuracy $\epsilon$ and cutoff $\Lambda=\beta\omega_{\max}$ (requires `pip install "c3l[dlr]"`). This is the interface for using C³L as an impurity solver in imaginary-time DMFT.

**See the [Matsubara solver guide](src/c3l/matsubara/README.md)** for grid choices, solver options and outputs, inserting $\Sigma$ into another bath, DLR imaginary-time transforms, using your own (e.g. GPU-resident) response, a DMFT-loop sketch and current limitations.

<details>
<summary>Show the Matsubara example</summary>

```python
from c3l.matsubara import dlr_matsubara_grid, matsubara_grid, solve_matsubara

grid = dlr_matsubara_grid(beta=10.0, eps=1e-10, wmax=6.0)   # or matsubara_grid(10.0, n_iw=256)
solution = solve_matsubara(model, grid, rank="S13")        # CCSD + EOM; platform="gpu" for JAX
G_iw, Sigma_iw = solution.green, solution.self_energy        # axes (n, a, b) over impurity sites
G_target = solution.dyson(Delta_iw)                          # [iw - h - Delta - Sigma]^{-1}
```

On a DLR grid, `grid.dlr_tau(G_iw, tau)` and `grid.dlr_matsubara(G_iw, iw)` give $G(\tau)$ and $G$ at other Matsubara points. Responses computed elsewhere (e.g. GPU-resident Arnoldi) can be evaluated at `grid.upper_shifts`, mapped with `grid.from_upper`, and passed to `matsubara_self_energy`. See [examples/basic/05_matsubara_dlr.py](examples/basic/05_matsubara_dlr.py) for a fitted SIAM bath.

</details>

## Examples

See the [examples guide](examples/README.md) for more examples. These optional scripts are provided in the source repository. For the Matsubara-axis solver, see [examples/basic/05_matsubara_dlr.py](examples/basic/05_matsubara_dlr.py) and the [Matsubara solver guide](src/c3l/matsubara/README.md).

## Citing C³L

If you use C³L in your research, please cite our preprint:

Hao-En Li, Zhen Huang, and Lin Lin, [Efficient simulations of spectra and transport for multiorbital quantum impurities](https://arxiv.org/abs/2611.05185), 2026.

```bibtex
@misc{LiHuangLin2026,
      title={Efficient simulations of spectra and transport for multiorbital quantum impurities}, 
      author={Hao-En Li and Zhen Huang and Lin Lin},
      year={2026},
      eprint={2611.05185},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2611.05185}, 
}
```

If your calculation employed the bath fitting based on [RealTimeBath](https://github.com/Hertz4/RealTimeBath), cite

Zhen Huang, Gunhee Park, Garnet Kin-Lic Chan, and Lin Lin, [Coupled Lindblad Pseudomode Theory for Simulating Open Quantum Systems](https://journals.aps.org/prl/abstract/10.1103/qfkp-jc7s), **Phys. Rev. Lett. 2026**, 136, 090403

## License

[MIT License](LICENSE). Copyright (c) 2026 Hao-En Li.
