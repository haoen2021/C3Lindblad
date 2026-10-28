"""Continuum GF2/GW cumulants used in the molecular-junction comparison."""

import numpy as np


def adjoint(a):
    return a.conj().swapaxes(-1, -2)


def hermitian_basis(d):
    basis = []
    for i in range(d):
        a = np.zeros((d, d), complex)
        a[i, i] = 1
        basis.append(a)
    for i in range(d):
        for j in range(i + 1, d):
            a = np.zeros((d, d), complex)
            a[i, j] = a[j, i] = 1 / np.sqrt(2)
            basis.append(a)
            a = np.zeros((d, d), complex)
            a[i, j] = -1j / np.sqrt(2)
            a[j, i] = 1j / np.sqrt(2)
            basis.append(a)
    return np.array(basis)


def cumulant(w, gl, gg, u, method, padding=3):
    n = len(w)
    dw = w[1] - w[0]
    d = gl.shape[1]
    m = padding * n
    pad = (m - n) // 2
    basis = hermitian_basis(d)
    # H_int = 1/2 V_ab :O_a O_b:, O_a = c^dag T_a c.
    v = np.einsum("pqrs,arp,bsq->ab", u, basis, basis, optimize=True)
    np.testing.assert_allclose(v.imag, 0, atol=1e-12)
    v = v.real

    def tt(a):
        return (
            np.fft.fft(
                np.fft.ifftshift(np.pad(a, ((pad, pad), (0, 0), (0, 0))), axes=0),
                axis=0,
            )
            * dw
            / (2 * np.pi)
        )

    def ff(a):
        return (np.fft.fftshift(np.fft.ifft(a, axis=0), axes=0) * 2 * np.pi / dw)[
            pad : pad + n
        ].copy()

    lt, gt = tt(gl), tt(gg)
    reverse = (-np.arange(m)) % m

    def polar(a, b):
        return -1j * np.einsum(
            "aij,tjk,bkl,tli->tab", basis, a, basis, b, optimize="optimal"
        )

    pl_t = polar(lt, gt[reverse])
    pg_t = polar(gt, lt[reverse])
    theta = np.zeros(m)
    theta[: m // 2] = 1
    theta[0] = theta[m // 2] = 0.5
    pr = ff((pg_t - pl_t) * theta[:, None, None])
    pl = ff(pl_t)
    del pl_t, pg_t, lt, gt
    pa = adjoint(pr)
    corr = np.zeros((d * d, d * d), complex)
    for start in range(0, n, 256):
        sl = slice(start, start + 256)
        r, l, a = pr[sl], pl[sl], pa[sl]
        if method == "2B":
            xless = r @ v @ l + l @ v @ a
        elif method == "GW":
            wr = np.linalg.solve(np.eye(d * d) - 2 * v @ r, np.broadcast_to(v, r.shape))
            wa = adjoint(wr)
            wl = wr @ (2 * l) @ wa
            xless = r @ wr @ l + r @ wl @ a + l @ wa @ a
        else:
            raise ValueError(method)
        corr += 1j * xless.sum(axis=0) * dw / (2 * np.pi)
    direct = np.einsum("arp,bsq,ab->pqrs", basis, basis, corr, optimize=True)
    checks = dict(
        covariance_hermiticity=float(np.max(abs(corr - adjoint(corr)))),
        covariance_imaginary=float(np.max(abs(corr.imag))),
        particle_exchange=float(np.max(abs(direct - direct.transpose(1, 0, 3, 2)))),
        cumulant_hermiticity=float(
            np.max(abs(direct - direct.transpose(2, 3, 0, 1).conj()))
        ),
    )
    return direct, checks


def spin_components(rho, direct, method, i=0, j=2):
    same = direct - direct.swapaxes(2, 3) if method == "2B" else direct
    den = direct[i, j, i, j]
    exchange = direct[i, j, j, i]
    # Spin-conserving normal order, identical to reference Eqs. 91-102.
    zz_c = -0.5 * (same[i, j, j, i] + den)
    trans_c = -exchange
    wick = -1.5 * rho[i, j] * rho[j, i]
    # Alternative use of anticommutation in the same-spin zz term.
    zz_density_c = 0.5 * (same[i, j, i, j] - den)
    return dict(
        dot=wick + zz_c + trans_c,
        zz=wick / 3 + zz_c,
        transverse=2 * wick / 3 + trans_c,
        wick=wick,
        cumulant=zz_c + trans_c,
        zz_density_order=wick / 3 + zz_density_c,
        dot_density_order=wick + zz_density_c + trans_c,
        opposite_spin_density_cumulant=den,
        opposite_spin_exchange_cumulant=exchange,
        spin_isotropy_defect=zz_c - trans_c / 2,
    )
