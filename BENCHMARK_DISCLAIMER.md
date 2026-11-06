<!-- SPDX-License-Identifier: Apache-2.0 | Author: Francisco Angulo de Lafuente | ORCID: https://orcid.org/0009-0001-1634-7063 | Attribution recorded: 2026-11-06 -->

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

The repository contains preprocessing and training code, but not an MABe 2022 training split or a verifiable benchmark evaluation record. It also tracks demo artifacts under `data/exposure_dataset/` and a binary at `kaggle_model/model_weights.bin` (64×64). `create_demo_data.py` describes synthetic demo-data generation, but this review did not independently reproduce the exact tracked data files. The current `create_demo_weights.py` generates 32×32 weights and does not match the tracked binary. The adjacent `best_accuracy: 0.925906` value has no verifiable training record. Treat the binary origin, training data, and reported accuracy as unverified. Any historical F1 table remains unverified and is **not** reproducible from this repository alone.

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
