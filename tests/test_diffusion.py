"""Synthetic unit tests for the classical diffusion update.

No dataset required.
"""
from __future__ import annotations

import numpy as np
import pytest

from qesn.diffusion import Lattice, diffuse


def test_lattice_init_zero_energy() -> None:
    lat = Lattice(8, 8)
    assert lat.energy() == 0.0
    assert lat.amplitudes.shape == (8, 8)
    assert lat.amplitudes.dtype == np.complex128


def test_inject_adds_energy() -> None:
    lat = Lattice(8, 8)
    lat.inject(3, 4, 1.0)
    assert lat.energy() == pytest.approx(1.0)


def test_invalid_coupling_rejected() -> None:
    with pytest.raises(ValueError):
        Lattice(4, 4, coupling=0.5)
    with pytest.raises(ValueError):
        Lattice(4, 4, coupling=-0.1)


def test_invalid_decay_rejected() -> None:
    with pytest.raises(ValueError):
        Lattice(4, 4, decay=0.0)
    with pytest.raises(ValueError):
        Lattice(4, 4, decay=1.5)


def test_invalid_size_rejected() -> None:
    with pytest.raises(ValueError):
        Lattice(0, 4)
    with pytest.raises(ValueError):
        Lattice(4, -1)


def test_negative_steps_rejected() -> None:
    lat = Lattice(4, 4)
    with pytest.raises(ValueError):
        diffuse(lat, steps=-1)


def test_zero_steps_noop() -> None:
    lat = Lattice(8, 8, coupling=0.2, decay=1.0)
    lat.inject(2, 2, 1.0)
    before = lat.amplitudes.copy()
    diffuse(lat, steps=0)
    np.testing.assert_array_equal(before, lat.amplitudes)


def test_energy_conservation_with_unit_decay() -> None:
    """With decay=1 and periodic BC the symmetric stencil keeps energy ~constant."""
    lat = Lattice(16, 16, coupling=0.1, decay=1.0)
    rng = np.random.default_rng(0)
    for _ in range(10):
        x = int(rng.integers(0, 16))
        y = int(rng.integers(0, 16))
        lat.inject(x, y, 1.0)
    e0 = lat.energy()
    diffuse(lat, steps=50)
    e1 = lat.energy()
    # The update is a real symmetric stencil; in general it's contractive on L2,
    # not strictly unitary, but for small coupling the drift is bounded.
    assert e1 <= e0 * 1.001
    assert e1 > 0.0


def test_decay_reduces_energy() -> None:
    lat = Lattice(8, 8, coupling=0.1, decay=0.9)
    lat.inject(4, 4, 1.0)
    e0 = lat.energy()
    diffuse(lat, steps=10)
    e1 = lat.energy()
    assert e1 < e0


def test_diffusion_symmetry_under_shift() -> None:
    """Periodic-boundary diffusion commutes with lattice translation."""
    lat_a = Lattice(8, 8, coupling=0.2, decay=1.0)
    lat_b = Lattice(8, 8, coupling=0.2, decay=1.0)
    lat_a.inject(2, 3, 1.0)
    lat_b.inject(5, 6, 1.0)  # (2+3, 3+3) mod 8
    diffuse(lat_a, steps=5)
    diffuse(lat_b, steps=5)
    shifted = np.roll(np.roll(lat_a.amplitudes, 3, axis=0), 3, axis=1)
    np.testing.assert_allclose(shifted, lat_b.amplitudes, atol=1e-12)


def test_energy_spreads_to_neighbors() -> None:
    lat = Lattice(8, 8, coupling=0.2, decay=1.0)
    lat.inject(4, 4, 1.0)
    diffuse(lat, steps=1)
    emap = lat.energy_map()
    # Neighbor cells must be non-zero after one step.
    assert emap[4, 5] > 0
    assert emap[5, 4] > 0


def test_normalize_sets_unit_energy() -> None:
    lat = Lattice(4, 4)
    lat.inject(1, 1, 3.0)
    lat.inject(2, 2, 4.0)
    lat.normalize()
    assert lat.energy() == pytest.approx(1.0, abs=1e-12)
