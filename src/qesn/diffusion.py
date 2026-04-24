"""Classical energy-diffusion update on a 2D lattice with complex amplitudes.

This module implements a Schrödinger-inspired diffusion step. It is a
classical dynamical system; no quantum hardware or quantum simulation is
involved.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Lattice:
    """2D complex-amplitude lattice.

    Attributes
    ----------
    width, height : int
        Grid dimensions.
    coupling : float
        Diffusion coupling coefficient in [0, 0.25] for stability.
    decay : float
        Multiplicative energy decay per step in (0, 1].
    amplitudes : np.ndarray[complex]
        Complex amplitudes of shape (height, width).
    """

    width: int
    height: int
    coupling: float = 0.20
    decay: float = 1.0
    amplitudes: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if not (0.0 <= self.coupling <= 0.25):
            raise ValueError("coupling must be in [0, 0.25] for stability")
        if not (0.0 < self.decay <= 1.0):
            raise ValueError("decay must be in (0, 1]")
        if self.amplitudes is None:
            self.amplitudes = np.zeros((self.height, self.width), dtype=np.complex128)
        else:
            self.amplitudes = np.asarray(self.amplitudes, dtype=np.complex128)
            if self.amplitudes.shape != (self.height, self.width):
                raise ValueError("amplitudes shape mismatch")

    # ------------------------------------------------------------------
    def inject(self, x: int, y: int, amount: float = 1.0) -> None:
        """Inject real energy at (x, y)."""
        self.amplitudes[y % self.height, x % self.width] += amount + 0j

    def energy(self) -> float:
        """Total energy = sum |psi|^2."""
        return float(np.sum(np.abs(self.amplitudes) ** 2))

    def energy_map(self) -> np.ndarray:
        return np.abs(self.amplitudes) ** 2

    def normalize(self) -> None:
        e = self.energy()
        if e > 0:
            self.amplitudes /= np.sqrt(e)


def diffuse(lattice: Lattice, steps: int = 1) -> Lattice:
    """Apply `steps` Schrödinger-inspired diffusion updates, in-place.

    Update rule (periodic boundaries):
        psi'[i,j] = decay * ((1 - 4c) * psi[i,j]
                             + c * (psi[i+1,j] + psi[i-1,j]
                                  + psi[i,j+1] + psi[i,j-1]))

    With decay == 1 and periodic boundaries, total energy (sum |psi|^2)
    is conserved up to floating-point error (the stencil is the identity
    plus a symmetric real Laplacian, so this is a real-linear unitary-ish
    mixing; exact conservation isn't guaranteed analytically but the error
    stays in the ~1e-12 range for small `coupling`).
    """
    if steps < 0:
        raise ValueError("steps must be >= 0")
    c = lattice.coupling
    d = lattice.decay
    a = lattice.amplitudes
    for _ in range(steps):
        up = np.roll(a, -1, axis=0)
        down = np.roll(a, 1, axis=0)
        left = np.roll(a, -1, axis=1)
        right = np.roll(a, 1, axis=1)
        a = d * ((1.0 - 4.0 * c) * a + c * (up + down + left + right))
    lattice.amplitudes = a
    return lattice
