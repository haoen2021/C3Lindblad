r"""Matsubara-axis impurity Green functions and self-energies.

For a pseudomode (PM) model whose bath represents an equilibrium bath at
inverse temperature ``beta``, the impurity retarded function :math:`G^R(z)`
is analytic for :math:`\operatorname{Im} z>0`.  Its values at the fermionic
Matsubara frequencies :math:`\omega_n=(2n+1)\pi/\beta` give the imaginary-axis
Green function,

.. math::

   G(i\omega_n)=G^R(i\omega_n)\quad(\omega_n>0),\qquad
   G(i\omega_n)=\big[G^R(-i\omega_n)\big]^\dagger\quad(\omega_n<0),

so a Matsubara grid costs one shift :math:`z=i|\omega_n|` per distinct
:math:`|\omega_n|` in the same shift-invariant Krylov space used for
real-axis spectra.  The self-energy is the auxiliary PM Dyson self-energy on
the impurity block ``S``,

.. math::

   \Sigma(i\omega_n)=G_{0}^{-1}(i\omega_n)-G^{-1}(i\omega_n),\qquad
   G_{0}^R(z)=\big[(z-E+i\Lambda)^{-1}\big]_{SS},\qquad
   \Lambda=\Gamma_{\rm loss}+\Gamma_{\rm gain},

which is exact block elimination of the bath whenever every interacting site
lies in ``S``.  This is the quantity an imaginary-time DMFT loop consumes;
:func:`matsubara_dyson_green` inserts it into any target hybridization, for
example the continuous bath that the pseudomodes were fitted to.

Grids are either explicit (indices, frequencies, or a symmetric ``n_iw``
window) or the fermionic discrete Lehmann representation (DLR) grid for a
given ``beta``, accuracy ``eps`` and cutoff :math:`\Lambda=\beta\omega_{\max}`.
DLR grids use the optional ``pydlr`` package (``pip install "c3l[dlr]"``);
the ``pydlr.dlr`` object is kept on the grid for imaginary-time transforms.
"""

from .dyson import (
    MatsubaraSelfEnergy,
    impurity_sites,
    matsubara_dyson_green,
    matsubara_noninteracting_green,
    matsubara_self_energy,
    noninteracting_retarded_green,
)
from .grid import MatsubaraGrid, dlr_matsubara_grid, matsubara_grid
from .solver import (
    MatsubaraGreen,
    MatsubaraImpuritySolution,
    component_matsubara_green,
    solve_matsubara,
)

__all__ = [
    "MatsubaraGreen",
    "MatsubaraGrid",
    "MatsubaraImpuritySolution",
    "MatsubaraSelfEnergy",
    "component_matsubara_green",
    "dlr_matsubara_grid",
    "impurity_sites",
    "matsubara_dyson_green",
    "matsubara_grid",
    "matsubara_noninteracting_green",
    "matsubara_self_energy",
    "noninteracting_retarded_green",
    "solve_matsubara",
]
