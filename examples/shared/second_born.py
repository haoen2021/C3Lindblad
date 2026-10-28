"""Stationary spin-symmetric continuum second Born; all energies relative to mu.

FFT convention: G(t)=integral dw/(2pi) exp(-i*w*t)G(w). Frequency
zero-padding (default 3x) prevents cubic wraparound on the retained window.
The time grid is periodic: negative-time index is (-arange(M)) % M.
The Nyquist-time point is assigned theta=1/2, and its decay is diagnosed.
No physical time propagation, pseudomodes, or display broadening is used.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    beta: float
    mu: float
    n: int = 32768
    window: float = 16.0
    padding: int = 3
    correlation: bool = True
    mixing: float = 0.15
    tolerance: float = 1.0e-8
    max_iterations: int = 3000
    backend: str = "gpu"

    def validate(self):
        if self.n < 32 or self.n % 2 or self.padding < 2:
            raise ValueError("Need even n >= 32 and padding >= 2 (3 recommended).")
        if not (0 < self.mixing <= 1) or self.window <= 0:
            raise ValueError("Invalid mixing or frequency window.")


def contraction(a, b_reverse, w, xp=np):
    """a=G^{</>}(t), b_reverse=G^{>/<}(-t); arbitrary batch/time axes."""
    bubble = a * xp.swapaxes(b_reverse, -1, -2)
    direct = 2 * a * (w @ bubble @ w.T)
    # Sum over one orbital explicitly to avoid a (time,i,j,k,l) tensor.
    exchange = xp.stack(
        [
            xp.sum(
                a[..., i, :, None] * w.T * (b_reverse @ (w[i, :, None] * a)), axis=-2
            )
            for i in range(w.shape[0])
        ],
        axis=-2,
    )
    return direct - exchange


def adjoint(a, xp=np):
    return xp.swapaxes(a.conj(), -1, -2)


class Solver:
    def __init__(self, config, biases, h, w):
        config.validate()
        self.config = config
        self.biases = np.asarray(biases, float)
        self.xp = np
        self.jax = None
        if config.backend in ("cpu", "gpu"):
            try:
                import jax

                jax.config.update("jax_enable_x64", True)
                jax.config.update("jax_platform_name", config.backend)
                import jax.numpy as jnp

                devices = jax.devices(config.backend)
                self.xp, self.jax = jnp, jax
                self.device = str(devices[0])
            except ImportError:
                if config.backend == "gpu":
                    raise
                self.device = "NumPy CPU fallback (JAX unavailable)"
        else:
            self.device = "NumPy CPU reference"
        xp = self.xp
        self.omega = np.linspace(
            -config.window, config.window, config.n, endpoint=False
        )
        self.dw = 2 * config.window / config.n
        self.w = xp.asarray(w)
        self.h = xp.asarray(h)
        self.eye = xp.eye(len(h), dtype=xp.complex128)
        self.frequency = xp.asarray(self.omega)
        self.delta_r, self.delta_l, self.delta_g, self.gammas, self.fermis = (
            self.embedding()
        )
        self.step = self.jax.jit(self._step) if self.jax else self._step

    def to_time(self, a):
        xp, c = self.xp, self.config
        pad = (c.padding - 1) * c.n // 2
        padded = xp.pad(a, ((0, 0), (pad, pad), (0, 0), (0, 0)))
        return (
            xp.fft.fft(xp.fft.ifftshift(padded, axes=1), axis=1) * self.dw / (2 * np.pi)
        )

    def to_frequency(self, a):
        xp, c = self.xp, self.config
        padded = xp.fft.fftshift(xp.fft.ifft(a, axis=1), axes=1) * (2 * np.pi / self.dw)
        pad = (c.padding - 1) * c.n // 2
        return padded[:, pad : pad + c.n]

    def dyson(self, state):
        xp = self.xp
        hf, sr, sl, sg = state
        gr = xp.linalg.inv(
            self.frequency[None, :, None, None] * self.eye
            - self.h[None, None]
            - self.delta_r
            - hf[:, None]
            - sr
        )
        ga = adjoint(gr, xp)
        gl, gg = gr @ (self.delta_l + sl) @ ga, gr @ (self.delta_g + sg) @ ga
        # Enforce exact algebraic symmetries to avoid roundoff growth in
        # unphysical Hermitian lesser/greater directions during long iterations.
        return gr, 0.5 * (gl - adjoint(gl, xp)), 0.5 * (gg - adjoint(gg, xp))

    def _step(self, state):
        xp, c = self.xp, self.config
        gr, gl, gg = self.dyson(state)
        rho = -1j * xp.sum(gl, axis=1) * self.dw / (2 * np.pi)
        rho = 0.5 * (rho + adjoint(rho, xp))
        hf = (
            2
            * xp.einsum("ik,bk->bi", self.w, xp.diagonal(rho, axis1=-2, axis2=-1))[
                :, :, None
            ]
            * self.eye
        )
        hf = hf - self.w[None] * rho
        if c.correlation:
            lt, gt = self.to_time(gl), self.to_time(gg)
            m = c.padding * c.n
            reverse = (-xp.arange(m)) % m
            slt = contraction(lt, gt[:, reverse], self.w, xp)
            sgt = contraction(gt, lt[:, reverse], self.w, xp)
            theta = xp.where(xp.arange(m) < m // 2, 1.0, 0.0)
            # Half weights preserve SigmaR-SigmaA=Sigma>-Sigma< discretely.
            theta = xp.where((xp.arange(m) == 0) | (xp.arange(m) == m // 2), 0.5, theta)
            sr = self.to_frequency((sgt - slt) * theta[None, :, None, None])
            sl, sg = self.to_frequency(slt), self.to_frequency(sgt)
            sl, sg = 0.5 * (sl - adjoint(sl, xp)), 0.5 * (sg - adjoint(sg, xp))
        else:
            sr, sl, sg = (xp.zeros_like(gr),) * 3
        proposed = (hf, sr, sl, sg)
        defects = xp.stack(
            [
                xp.max(xp.abs(a - b), axis=tuple(range(1, a.ndim)))
                for a, b in zip(proposed, state)
            ],
            axis=1,
        )
        return proposed, defects

    def initial_state(self):
        xp, c = self.xp, self.config
        b, n = len(self.biases), len(self.h)
        # Half-filled HF start; final occupations are solved, not imposed.
        hf = 2 * xp.sum(self.w, axis=1) * 0.5 * self.eye - self.w * 0.5 * self.eye
        return (
            xp.broadcast_to(hf, (b, n, n)).astype(xp.complex128),
            *(xp.zeros((b, c.n, n, n), dtype=xp.complex128) for _ in range(3)),
        )

    def solve(self, state=None, callback=None):
        xp, c = self.xp, self.config
        state = (
            self.initial_state()
            if state is None
            else tuple(xp.asarray(x) for x in state)
        )
        history = []
        for iteration in range(1, c.max_iterations + 1):
            proposed, defects = self.step(state)
            host_defects = np.asarray(defects)
            if not np.isfinite(host_defects).all():
                raise FloatingPointError(
                    "Nonfinite Dyson/fixed-point defect; inspect bound poles/grid."
                )
            history.append(host_defects)
            if np.max(host_defects) < c.tolerance:
                break
            active = xp.asarray(np.max(host_defects, axis=1) >= c.tolerance)
            state = tuple(
                xp.where(
                    active.reshape((-1,) + (1,) * (a.ndim - 1)),
                    (1 - c.mixing) * a + c.mixing * b,
                    a,
                )
                for a, b in zip(state, proposed)
            )
            if callback and iteration % 50 == 0:
                callback(iteration, state, host_defects)
        # Defects must describe exactly the returned state (including max-iter case).
        _, defects = self.step(state)
        greens = self.dyson(state)
        return state, greens, np.asarray(defects), np.asarray(history)
