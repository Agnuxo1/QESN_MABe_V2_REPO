"""Quantum foam simulation utilities shared by QESN demos.

This module mirrors the behaviour of the original C++ implementation so that
Python demos and inference stay numerically aligned with exported checkpoints.
"""

from __future__ import annotations

import numpy as np


class QuantumFoam2D:
    """Discrete 2D quantum foam grid with simple diffusion dynamics."""

    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self.grid_size = self.width * self.height

        self.energy_grid = np.zeros(self.grid_size, dtype=np.float64)
        self.coupling_strength = 0.10
        self.diffusion_rate = 0.05
        self.decay_rate = 0.01
        self.quantum_noise = 0.0005

    def reset(self) -> None:
        self.energy_grid.fill(0.0)

    def set_coupling_strength(self, strength: float) -> None:
        self.coupling_strength = float(strength)

    def set_diffusion_rate(self, rate: float) -> None:
        self.diffusion_rate = float(rate)

    def set_decay_rate(self, rate: float) -> None:
        self.decay_rate = float(rate)

    def set_quantum_noise(self, noise: float) -> None:
        self.quantum_noise = float(noise)

    def inject_energy(self, x: int, y: int, energy: float) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = y * self.width + x
            self.energy_grid[idx] += float(energy)

    def time_step(self, dt: float = 0.002) -> None:
        new_grid = self.energy_grid.copy()
        width = self.width
        height = self.height
        coupling = self.coupling_strength
        diffusion_rate = self.diffusion_rate
        decay_rate = self.decay_rate
        noise_scale = self.quantum_noise

        for y in range(height):
            base = y * width
            for x in range(width):
                idx = base + x
                energy = self.energy_grid[idx]

                neighbours = []
                if x > 0:
                    neighbours.append(idx - 1)
                if x < width - 1:
                    neighbours.append(idx + 1)
                if y > 0:
                    neighbours.append(idx - width)
                if y < height - 1:
                    neighbours.append(idx + width)

                coupling_energy = 0.0
                for nidx in neighbours:
                    coupling_energy += coupling * (self.energy_grid[nidx] - energy)

                diffusion = diffusion_rate * coupling_energy
                decay = -decay_rate * energy
                noise = noise_scale * np.random.normal(0.0, 1.0)

                new_grid[idx] += dt * (diffusion + decay) + noise

        np.maximum(new_grid, 0.0, out=new_grid)
        self.energy_grid = new_grid

    def observe_gaussian(self, sigma: int = 1) -> np.ndarray:
        if sigma != 1:
            raise ValueError("observe_gaussian currently only supports sigma=1 to match C++ export")

        kernel = np.array(
            [[0.075, 0.124, 0.075],
             [0.124, 0.204, 0.124],
             [0.075, 0.124, 0.075]],
            dtype=np.float64,
        )

        observed = np.zeros_like(self.energy_grid)
        width = self.width
        height = self.height

        for y in range(height):
            for x in range(width):
                total_weight = 0.0
                total_energy = 0.0
                for ky in range(-1, 2):
                    ny = y + ky
                    if ny < 0 or ny >= height:
                        continue
                    for kx in range(-1, 2):
                        nx = x + kx
                        if nx < 0 or nx >= width:
                            continue
                        weight = kernel[ky + 1, kx + 1]
                        idx = ny * width + nx
                        total_energy += weight * self.energy_grid[idx]
                        total_weight += weight
                if total_weight > 0.0:
                    observed[y * width + x] = total_energy / total_weight

        return observed
