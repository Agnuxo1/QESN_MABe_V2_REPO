# License scope record — QESN-MABe V2

## Current package evidence

The root [`LICENSE`](LICENSE), `pyproject.toml`, `CITATION.cff`, current `README.md`, and package command output identify Apache-2.0. Commit `d8c3164288ce0add0a142765c7dda02bc7e31a00` introduced the Apache license and package metadata together with `src/qesn/`, `tests/`, `README.md`, and `BENCHMARK_DISCLAIMER.md`. New SPDX file identifiers in this attribution pass are limited to `src/qesn/*.py`, `tests/*.py`, `README.md`, and `BENCHMARK_DISCLAIMER.md`.

`CMakeLists.txt` already contains an Apache-2.0 header; it is preserved. No global scope conclusion is made for unrelated legacy content.

## Conflicting legacy MIT statements

Older source files, deployment scripts, and documents contain explicit MIT wording while the repository root and current package metadata say Apache-2.0. These statements were preserved; no new MIT notice was added. The scanned path/line inventory is:

- `CHANGELOG.md` (lines 162, 178)
- `COMPLETION_REPORT.md` (lines 347, 431, 451)
- `CONTRIBUTING.md` (lines 217)
- `DEPLOYMENT_GUIDE.md` (lines 40)
- `INSTALLATION.md` (lines 6)
- `PROJECT_STATUS.md` (lines 326, 338)
- `QUICK_START.md` (lines 448)
- `README_OLD.md` (lines 3)
- `README_PROFESSIONAL.md` (lines 883)
- `create_optimized_notebooks.py` (lines 7)
- `demo_espectacular.py` (lines 13)
- `demo_espectacular_no_emoji.py` (lines 12)
- `demo_espectacular_optimized.py` (lines 7)
- `demo_profesional.py` (lines 12)
- `demo_simple.py` (lines 12)
- `examples/kaggle_submission.py` (lines 7)
- `huggingface_space/README.md` (lines 10, 85)
- `include/io/dataset_loader.h` (lines 5)
- `include/training/trainer.h` (lines 5)
- `install_dependencies.py` (lines 7)
- `install_dependencies_no_emoji.py` (lines 7)
- `notebooks/qesn_demo_2.py` (lines 7)
- `python/qesn_inference.py` (lines 7)
- `python/qesn_inference_optimized.py` (lines 7)
- `scripts/deploy_huggingface.py` (lines 7, 347)
- `scripts/deploy_kaggle.py` (lines 7)
- `scripts/train.bat` (lines 6)
- `src/io/dataset_loader.cpp` (lines 5)
- `src/main.cpp` (lines 136)
- `src/training/trainer.cpp` (lines 5)
- `update_all_demos.py` (lines 7)
- `validation_pipeline.py` (lines 7)

The available history does not determine whether each legacy MIT notice is an intentional separate grant or stale metadata. This record does not overwrite those notices or extend either license to data, notebooks, figures, results, or model weights. License scope remains **partially unresolved**.

## Data, model, and contest boundary

`create_demo_data.py` describes synthetic demo-data generation for `data/exposure_dataset/`; this review did not independently reproduce every tracked data file. `kaggle_model/model_weights.bin` is present, encodes 64×64 weights, and is not reproduced by the current 32×32 `create_demo_weights.py`; its training lineage and adjacent `best_accuracy` value remain unverified. Both artifacts were preserved byte-for-byte.

The official [MABe Challenge 2022 rules](https://www.aicrowd.com/challenges/mabe-2022-mouse-triplets/challenge_rules) require a team's own original entry and grant organizers use rights when an entry is submitted. No linked AIcrowd submission was found in the reviewed repository history. The current [Kaggle MABe data page](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/data) reports CC BY 4.0 for that page's data; that does not establish rights for unrelated or historical artifacts in this repository.
