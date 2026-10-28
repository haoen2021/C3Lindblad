"""Stationary self-consistent charge-RPA GW, spin-degenerate density interaction.

P</>(t)=-2i G</>(t) [G>/<(-t)]^T, W_R=(1-v P_R)^-1 v,
W</>=W_R P</> W_A, Sigma_c</>(t)=i G</>(t) W</>(t).
Products in P and Sigma are elementwise; RPA products are matrix products.
Bare instantaneous v belongs to HF exchange and is excluded from W_c^R.
Primary conventions: Myohanen et al. arXiv:0906.2136 Sec III B,D/Eq39;
arXiv:1006.2912 Fig2. No spin-dependent self-interaction correction is applied.
"""

import numpy as np
from .second_born import Solver, adjoint


class GWSolver(Solver):
    def causal(self, lesser_t, greater_t):
        xp = self.xp
        m = self.config.n * self.config.padding
        theta = xp.where(xp.arange(m) < m // 2, 1.0, 0.0)
        theta = xp.where((xp.arange(m) == 0) | (xp.arange(m) == m // 2), 0.5, theta)
        return self.to_frequency((greater_t - lesser_t) * theta[None, :, None, None])

    def screening(self, gl, gg):
        xp = self.xp
        lt, gt = self.to_time(gl), self.to_time(gg)
        reverse = (-xp.arange(self.config.n * self.config.padding)) % (
            self.config.n * self.config.padding
        )
        plt = -2j * lt * xp.swapaxes(gt[:, reverse], -1, -2)
        pgt = -2j * gt * xp.swapaxes(lt[:, reverse], -1, -2)
        pr = self.causal(plt, pgt)
        pl, pg = self.to_frequency(plt), self.to_frequency(pgt)
        pl, pg = 0.5 * (pl - adjoint(pl, xp)), 0.5 * (pg - adjoint(pg, xp))
        wr = xp.linalg.solve(self.eye - self.w @ pr, xp.broadcast_to(self.w, pr.shape))
        wa = adjoint(wr, xp)
        wl, wg = wr @ pl @ wa, wr @ pg @ wa
        wl, wg = 0.5 * (wl - adjoint(wl, xp)), 0.5 * (wg - adjoint(wg, xp))
        return (pr, pl, pg, wr, wl, wg), (lt, gt)

    def correlation(self, gl, gg):
        xp = self.xp
        screened, (lt, gt) = self.screening(gl, gg)
        pr, pl, pg, wr, wl, wg = screened
        slt = 1j * lt * self.to_time(wl)
        sgt = 1j * gt * self.to_time(wg)
        sr = self.causal(slt, sgt)
        sl, sg = self.to_frequency(slt), self.to_frequency(sgt)
        return (sr, 0.5 * (sl - adjoint(sl, xp)), 0.5 * (sg - adjoint(sg, xp)))

    def _step(self, state):
        xp = self.xp
        gr, gl, gg = self.dyson(state)
        rho = -1j * xp.sum(gl, axis=1) * self.dw / (2 * np.pi)
        rho = 0.5 * (rho + adjoint(rho, xp))
        hf = (
            2
            * xp.einsum("ik,bk->bi", self.w, xp.diagonal(rho, axis1=-2, axis2=-1))[
                :, :, None
            ]
            * self.eye
            - self.w[None] * rho
        )
        proposed = (
            (hf, *self.correlation(gl, gg))
            if self.config.correlation
            else (hf, *(xp.zeros_like(gr),) * 3)
        )
        defects = xp.stack(
            [
                xp.max(xp.abs(a - b), axis=tuple(range(1, a.ndim)))
                for a, b in zip(proposed, state)
            ],
            axis=1,
        )
        return proposed, defects
