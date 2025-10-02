"""Helpers to load the exported QESN checkpoint for demos and scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .qesn_inference import QESNInference
from .qesn_inference_optimized import QESNInferenceOptimized


def resolve_model_dir(candidate: Optional[str]) -> Path:
    if candidate:
        model_dir = Path(candidate)
    else:
        model_dir = Path(os.environ.get("QESN_MODEL_DIR", Path(__file__).resolve().parents[1] / "kaggle_model"))
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory '{model_dir}' not found. Provide a path or set QESN_MODEL_DIR."
        )
    if not (model_dir / "model_weights.bin").exists() or not (model_dir / "model_config.json").exists():
        raise FileNotFoundError(
            f"Directory '{model_dir}' does not contain model_weights.bin and model_config.json."
        )
    return model_dir


def load_inference(model_dir: Optional[str] = None, optimized: bool = True) -> QESNInference:
    resolved = resolve_model_dir(model_dir)
    weights_path = str(resolved / "model_weights.bin")
    config_path = str(resolved / "model_config.json")
    
    if optimized:
        return QESNInferenceOptimized(weights_path, config_path)
    else:
        return QESNInference(weights_path, config_path)
