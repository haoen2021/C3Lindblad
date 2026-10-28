# Scripts

Install C3L following the [installation instructions](../README.md#install). Set model and solver parameters in [parameters.json](parameters.json). Run calculations in a compute allocation, with inputs and outputs in an external working directory. For example:

```bash
python /path/to/C3L/examples/run.py --work-dir /path/to/work siam_modes.example04 --modes 1 2 3 4 5 6
python /path/to/C3L/examples/run.py --work-dir /path/to/work exchange_model
```
(Example 04 generates its bath fits and static checkpoints; `exchange_model` needs no input files. Other calculations generally require the archived bath fits and checkpoints in the matching case directory under `--work-dir`)

For cold/warm GPU timings of static CCSD followed by the joint Matsubara and real-frequency response:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python /path/to/C3L/examples/run.py \
  --work-dir /path/to/work inchmoss.gpu_timing --rank S135 \
  --bath /path/to/bath_N5.npz --reference /path/to/static_reference.npz \
  --output results/inchmoss_S135
```

Use `junction.gpu_timing` with the junction N4 inputs, or `--rank S13` for the smaller response space. Add `--device cpu` to run the same JAX solver on CPU; GPU is the default.

| Calculation | Scripts |
| --- | --- |
| SIAM | `siam_ed/`, `siam_modes/`, `siam_tdvp/` |
| Hubbard dimer | `dimer/` |
| Kanamori model | `kanamori/` |
| Inchworm QMC benchmark | `inchmoss/` |
| Molecular junction | [Commands](junction/README.md) |
| TDVP comparisons | `tdvp/` |

For SIAM steady states and S13/S135 spectra at $U=2,4,6,8,10$, run `siam_ed.gpu --output /path/to/results`, reusing the archived bath. Use `--U` to override the interaction list.

## CPU and GPU solvers

The JAX drivers accept `--device cpu|gpu`, defaulting to `gpu`. Static residuals/JVPs, GMRES, EOM actions and resident Arnoldi use the selected device with the same equations and tolerances. The standard C3L installation includes JAX and, on supported Linux systems, its CUDA 13 GPU dependencies. GPU execution requires a compatible NVIDIA GPU and driver.

The large basis stays in GPU memory or CPU RAM. Insufficient memory raises an error: reduce the Krylov cap or request more memory.

## Optional dependencies

RealTimeBath fitting and JAX solvers are included in the standard C3L installation. The comparison calculations below require additional packages; install only those needed for your calculation:

+ [TRIQS](https://github.com/TRIQS/triqs) with [triqs_cthyb](https://github.com/TRIQS/cthyb)/[mpi4py](https://mpi4py.readthedocs.io/en/stable/) or [triqs_xca](https://github.com/TRIQS/xca) for impurity comparisons
+ [Julia](https://julialang.org/) with [ITensors](https://github.com/ITensor/ITensors.jl), [ITensorMPS](https://github.com/ITensor/ITensorMPS.jl) and [JSON](https://github.com/JuliaIO/JSON.jl) for TDVP
