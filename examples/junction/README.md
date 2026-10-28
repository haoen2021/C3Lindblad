# Minimal model for the resonance-resolved transport in molecular junctions

[Back to examples](../README.md)

This example is a four-orbital molecule coupled to metallic tip and substrate reservoirs, motivated by the single-molecule experiments of [Kimura et al., Nature 570, 210–213 (2019)](https://doi.org/10.1038/s41586-019-1284-2) and [Sellies et al., Nat. Nanotechnol. 20, 27–35 (2025)](https://doi.org/10.1038/s41565-024-01791-2). Coulomb repulsion, spin exchange, and pair hopping produce competing molecular states and distinct electron-removal resonances. At $\beta=10$, the calculations benchmark the equilibrium LUMO Green function and follow the current, energy-resolved transport, and neutral spin-response ratio $R$ as the extraction bias increases.

`cc.py` handles equilibrium/bias CCSD and EOM; `mbpt.py` handles continuum GW/2ndB; `equilibrium.py` handles CT-HYB/NCA/OCA; `observables.py` dumps current densities and isolated-molecule spectral functions. `model.py` contains their shared model.

The standard [C3L installation](../../README.md#install) includes NumPy, SciPy, Numba, JAX and RealTimeBath, with CUDA 13 GPU dependencies on supported Linux systems. The impurity comparisons additionally require [TRIQS](https://github.com/TRIQS/triqs) + [triqs_cthyb](https://github.com/TRIQS/cthyb) + [mpi4py](https://mpi4py.readthedocs.io/en/stable/) for CT-HYB, or TRIQS + [triqs_xca](https://github.com/TRIQS/xca) for NCA/OCA.

Use the parent launcher with an external working directory containing `junction/inputs/`; the model and solver settings come from the parent `parameters.json`:

```bash
python /path/to/scripts/run.py --work-dir /path/to/work junction.cc static --phi=-0.6
python /path/to/scripts/run.py --work-dir /path/to/work junction.cc eom --phi=-0.6 --rank S135
python /path/to/scripts/run.py --work-dir /path/to/work junction.mbpt --method GW --phi=-0.6
python /path/to/scripts/run.py --work-dir /path/to/work junction.equilibrium cthyb --group 0
python /path/to/scripts/run.py --work-dir /path/to/work junction.equilibrium oca
# Use nca for the NCA comparison; --phi=0 for equilibrium GW/2ndB.
python /path/to/scripts/run.py --work-dir /path/to/work junction.observables currents
python /path/to/scripts/run.py --work-dir /path/to/work junction.observables atomic
```

CC defaults to equilibrium (`--phi=0`). Outputs under `work/junction/results/` contain the LUMO imaginary-time Green function, $R$, extraction currents, and electron-removal spectrum for isolated molecule, with solver checkpoints and convergence checks needed to reproduce them. The scripts store orbitals as `(h1, h2, l1, l2)`.

EOM uses GPU-resident Arnoldi, including reconstruction for the full-action residual check. If the requested cap does not fit in VRAM, the run fails rather than falling back to streaming.

[Back to examples](../README.md)
