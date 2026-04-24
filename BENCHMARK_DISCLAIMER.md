# Benchmark Disclaimer

The F1 numbers in prior versions of the README (and in comparison tables pitting
QESN against ResNet-50 + LSTM, Transformer, GCN, SlowFast, etc.) were obtained
on an **internal preprocessing of MABe 2022 that is not included in this
repository**. They are **unverified** and **not reproducible from this repo
alone**.

This release ships:

- The C++20 reservoir source code (`src/`, `include/`).
- A Python reference implementation of the diffusion dynamics (`src/qesn/`).
- Synthetic unit tests (`tests/`) that do **not** require the MABe dataset.
- A CLI (`qesn-mabe`) that runs synthetic simulations only.

No pretrained weights, no preprocessed dataset splits, and no benchmark
reproduction script are included. Any table of the form "QESN F1 = 0.48 vs
ResNet F1 = 0.52" should be read as a historical claim by the author that has
**not** been independently verified in this repository.

If you want to reproduce benchmark numbers you will need to:

1. Obtain the MABe 2022 dataset yourself (Kaggle / MABe website).
2. Write your own preprocessing pipeline (the original one is not shipped).
3. Train the model end-to-end.
4. Evaluate on a held-out split of your choosing.

The maintainers make no accuracy guarantees for this release.

## Nature of the model

QESN is a **classical energy-diffusion reservoir** on a 2D lattice that uses
complex amplitudes and a Schrödinger-**inspired** update rule. It is **not**:

- A quantum computer simulation.
- A variational quantum circuit.
- Running on quantum hardware.
- Solving the Schrödinger equation in any physically rigorous sense.

It is a classical dynamical system whose update has been decorated with
complex-valued state for expressiveness. Earlier documents described it as
"genuine quantum mechanical evolution" — that framing is retracted.
