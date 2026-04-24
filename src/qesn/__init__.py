"""QESN — Quantum-inspired Echo State Network.

A classical energy-diffusion reservoir on a 2D lattice with complex
amplitudes. Inspired by Schrödinger dynamics; NOT a quantum computer
simulation. See BENCHMARK_DISCLAIMER.md at the repo root.
"""

from .diffusion import Lattice, diffuse
from .config import QESNConfig, load_config

__version__ = "1.0.0"
__all__ = ["Lattice", "diffuse", "QESNConfig", "load_config", "__version__"]
