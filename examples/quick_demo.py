#!/usr/bin/env python3
"""Quick demonstration that loads the exported QESN checkpoint and runs inference."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from python.model_loader import load_inference, resolve_model_dir  # noqa: E402


def simulate_keypoints(behavior: str, frames: int, mice: int = 4) -> np.ndarray:
    keypoints = np.zeros((frames, mice, 18, 3), dtype=np.float64)

    if behavior == "aggressive":
        for frame in range(frames):
            for mouse in range(mice):
                angle = frame * 0.2 + mouse * np.pi / 2.0
                base_x = 512.0 + 20.0 * np.cos(angle)
                base_y = 285.0 + 20.0 * np.sin(angle)
                offsets = np.random.normal(0.0, 10.0, size=(18, 2))
                confidences = np.random.uniform(0.7, 1.0, size=(18, 1))
                keypoints[frame, mouse, :, 0:2] = np.array([base_x, base_y]) + offsets
                keypoints[frame, mouse, :, 2:3] = confidences

    elif behavior == "social":
        for frame in range(frames):
            progress = frame / max(frames - 1, 1)
            for mouse in range(mice):
                start_x = 200.0 + mouse * 200.0
                start_y = 200.0 + mouse * 100.0
                target_x = 400.0 + np.sin(progress * np.pi) * 100.0
                target_y = 300.0 + np.cos(progress * np.pi) * 50.0
                start = np.array([start_x, start_y])
                target = np.array([target_x, target_y])
                current = start + (target - start) * progress
                offsets = np.random.normal(0.0, 8.0, size=(18, 2))
                confidences = np.random.uniform(0.8, 1.0, size=(18, 1))
                keypoints[frame, mouse, :, 0:2] = current + offsets
                keypoints[frame, mouse, :, 2:3] = confidences

    elif behavior == "exploration":
        for frame in range(frames):
            for mouse in range(mice):
                centers = np.random.uniform([100.0, 100.0], [900.0, 500.0])
                offsets = np.random.normal(0.0, 15.0, size=(18, 2))
                confidences = np.random.uniform(0.6, 1.0, size=(18, 1))
                keypoints[frame, mouse, :, 0:2] = centers + offsets
                keypoints[frame, mouse, :, 2:3] = confidences

    else:
        raise ValueError(f"Unknown behavior '{behavior}'")

    return keypoints


def optionally_validate_against_reference(
    inference,
    keypoints: np.ndarray,
    video_dims: Tuple[int, int],
    tolerance: float,
) -> None:
    reference_repo = os.environ.get("QESN_GPU_REPO")
    if not reference_repo:
        return

    sys.path.insert(0, str(Path(reference_repo) / "python_interface"))
    try:
        from qesn_mabe_inference import QESNInference as LegacyInference  # type: ignore
    except ImportError:
        print("WARN  Could not import qesn_mabe_inference from QESN_GPU_REPO; validation skipped.")
        sys.path.pop(0)
        return

    try:
        legacy_model_dir = Path(reference_repo) / "kaggle_model"
        legacy_model = LegacyInference(str(legacy_model_dir))
        pred_idx_new, probs_new, _ = inference.predict(keypoints, *video_dims)
        pred_idx_old, probs_old, _ = legacy_model.predict(keypoints, window_size=inference.window_size)

        max_diff = float(np.max(np.abs(probs_new - probs_old)))
        if pred_idx_new != pred_idx_old or max_diff > tolerance:
            print("FAIL  Inference does not match the exported reference model.")
            print(f"       max difference: {max_diff:.6f}")
        else:
            print("OK    Probabilities match the exported reference (max delta = {:.6f}).".format(max_diff))
    finally:
        sys.path.pop(0)


def demo_predictions(inference, behaviors: Iterable[str]) -> None:
    video_width = 1024
    video_height = 570
    for behavior in behaviors:
        print(f"\n[DEMO] Running synthetic behavior: {behavior}")
        keypoints = simulate_keypoints(behavior, inference.window_size)
        pred_idx, probs, pred_name = inference.predict(keypoints, video_width, video_height)
        print(f"   predicted class : {pred_name} (index {pred_idx})")
        print(f"   confidence      : {probs[pred_idx]:.3f}")
        top5 = np.argsort(probs)[-5:][::-1]
        print("   top-5 classes")
        for rank, idx in enumerate(top5, start=1):
            print(f"     {rank}. {inference.class_names[idx]} -> {probs[idx]:.3f}")

        optionally_validate_against_reference(inference, keypoints, (video_width, video_height), tolerance=1e-6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QESN quick demo using the exported checkpoint")
    parser.add_argument("--model-dir", help="Directory with model_weights.bin and model_config.json")
    parser.add_argument(
        "--behaviors",
        nargs="*",
        default=["aggressive", "social", "exploration"],
        help="Synthetic behaviors to simulate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = resolve_model_dir(args.model_dir)
    inference = load_inference(str(model_dir))
    demo_predictions(inference, args.behaviors)


if __name__ == "__main__":
    main()
