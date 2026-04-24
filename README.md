# QESN-MABe V2

**Quantum-inspired Echo State Network on a 2D lattice — classical, NOT a quantum computer simulation.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python ≥3.10](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](pyproject.toml)
[![PyPI: qesn-mabe](https://img.shields.io/badge/PyPI-qesn--mabe-informational)](https://pypi.org/project/qesn-mabe/)

QESN is a **classical energy-diffusion reservoir** that uses complex-valued
amplitudes and a **Schrödinger-inspired diffusion update**. It runs on ordinary
CPUs/GPUs — no quantum hardware is involved.

> Earlier versions of this repo described the model as "genuine quantum
> mechanical evolution" or said it "runs the Schrödinger equation". That
> framing is retracted. The update rule is a symmetric, complex-valued
> Laplacian stencil with optional multiplicative decay. It is inspired by the
> form of the Schrödinger equation, not a rigorous simulation of it.

---

## What this is

- A 2D lattice of complex amplitudes.
- A diffusion update (5-point stencil with periodic boundaries) applied for N steps.
- A small Python reference implementation (`src/qesn/`) + a C++20 core (`src/`, `include/`).
- A CLI (`qesn-mabe`) for synthetic simulations.
- Pure-Python synthetic unit tests under `tests/`.

## What this is NOT

- Not a quantum computer simulation.
- Not a variational quantum circuit.
- Not running on quantum hardware.
- Not a physically rigorous Schrödinger solver.
- Not shipping the MABe 2022 dataset, weights, or a benchmark reproduction script.

---

## Install

### Python package (recommended)

```bash
pip install qesn-mabe
```

With optional extras:

```bash
pip install "qesn-mabe[arrow]"   # adds pyarrow for Parquet I/O
pip install "qesn-mabe[dev]"     # pytest + build + twine
```

From source:

```bash
git clone https://github.com/Agnuxo1/QESN_MABe_V2_REPO
cd QESN_MABe_V2_REPO
pip install -e ".[dev]"
pytest
```

### C++ core (optional)

The C++ core is optional and only needed if you want to explore the original
training binary.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Options:

| Option | Default | Effect |
| --- | --- | --- |
| `QESN_WITH_CUDA`  | `OFF` | Enable CUDA GPU acceleration (requires `nvcc`). |
| `QESN_WITH_ARROW` | `OFF` | Link Apache Arrow + Parquet. |
| `QESN_WITH_OPENMP`| `ON`  | Enable OpenMP parallelism if available. |

If Eigen3 is not found via `find_package`, CMake falls back to
`FetchContent` from the upstream Eigen repo — no manual path setup required.

---

## CLI usage

```bash
qesn-mabe info
qesn-mabe status
qesn-mabe simulate --lattice-size 64 --steps 100 --output run.json
```

The `simulate` subcommand runs a **synthetic** diffusion simulation (random
energy injections on a fresh lattice) and writes metrics as JSON. It does not
load the MABe dataset and does not produce classification labels.

---

## Library usage

```python
from qesn import Lattice, diffuse

lat = Lattice(width=64, height=64, coupling=0.20, decay=1.0)
lat.inject(32, 32, amount=1.0)
diffuse(lat, steps=100)
print(lat.energy(), lat.energy_map().shape)
```

---

## Benchmarks

Historical README versions compared QESN against ResNet-50 + LSTM, Transformer,
GCN, SlowFast, etc. with F1 numbers such as `QESN F1 ≈ 0.48`. Those numbers
are **unverified** and **not reproducible from this repo alone**.

**Full details:** [BENCHMARK_DISCLAIMER.md](BENCHMARK_DISCLAIMER.md).

No benchmark numbers are claimed by this release.

---

## Tests

Synthetic only — no dataset needed.

```bash
pytest
```

The suite covers the diffusion update (energy conservation, translation
symmetry, validation of parameters), config round-trips, and the CLI.

---

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Author

Francisco Angulo de Lafuente — <agnuxo1@gmail.com>
