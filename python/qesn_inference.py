#!/usr/bin/env python3
"""
QESN-MABe V2: Quantum Energy State Network for Mouse Behavior Classification
Author: Francisco Angulo de Lafuente
License: MIT
GitHub: https://github.com/Agnuxo1
ResearchGate: https://www.researchgate.net/profile/Francisco-Angulo-Lafuente-3
Kaggle: https://www.kaggle.com/franciscoangulo
HuggingFace: https://huggingface.co/Agnuxo
Wikipedia: https://es.wikipedia.org/wiki/Francisco_Angulo_de_Lafuente
"""

from __future__ import annotations

import json
import struct
import sys
from typing import List, Tuple

import numpy as np

try:
    from .quantum_foam import QuantumFoam2D
except ImportError:  # pragma: no cover - script execution without package context
    from quantum_foam import QuantumFoam2D


class QESNInference:
    """Physics-faithful inference engine that mirrors the C++ export."""

    CLASS_NAMES = [
        "allogroom", "approach", "attack", "attemptmount", "avoid",
        "biteobject", "chase", "chaseattack", "climb", "defend",
        "dig", "disengage", "dominance", "dominancegroom", "dominancemount",
        "ejaculate", "escape", "exploreobject", "flinch", "follow",
        "freeze", "genitalgroom", "huddle", "intromit", "mount",
        "rear", "reciprocalsniff", "rest", "run", "selfgroom",
        "shepherd", "sniff", "sniffbody", "sniffface", "sniffgenital",
        "submit", "tussle"
    ]

    def __init__(self, weights_path: str, config_path: str):
        self.config_path = config_path
        self.weights_path = weights_path

        self.load_config(config_path)
        self.num_classes = int(self.config.get("num_classes", len(self.CLASS_NAMES)))
        self.class_names = self.config.get("class_names", self.CLASS_NAMES[: self.num_classes])
        self.window_size = int(self.config.get("window_size", 30))
        self.stride = int(self.config.get("stride", 15))
        self.dt = float(self.config.get("time_step", 0.002))
        self.energy_injection = float(self.config.get("energy_injection", 0.05))
        self.confidence_threshold = float(self.config.get("confidence_threshold", 0.5))

        self.load_weights(weights_path)
        self._initialise_foam()

    def load_config(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        print(f"Loaded config from {path}: {self.config}")

    def load_weights(self, path: str) -> None:
        with open(path, "rb") as f:
            grid_w, = struct.unpack("Q", f.read(8))
            grid_h, = struct.unpack("Q", f.read(8))
            weight_count, = struct.unpack("Q", f.read(8))
            bias_count, = struct.unpack("Q", f.read(8))

            self.grid_width = int(grid_w)
            self.grid_height = int(grid_h)
            grid_size = self.grid_width * self.grid_height

            expected_weights = self.num_classes * grid_size
            if weight_count != expected_weights:
                raise ValueError(
                    f"Weight count mismatch: file has {weight_count}, expected {expected_weights}"
                )
            if bias_count != self.num_classes:
                raise ValueError(
                    f"Bias count mismatch: file has {bias_count}, expected {self.num_classes}"
                )

            weights_flat = struct.unpack(f"{weight_count}d", f.read(weight_count * 8))
            biases = struct.unpack(f"{bias_count}d", f.read(bias_count * 8))

        self.weights = np.array(weights_flat, dtype=np.float64).reshape(self.num_classes, grid_size)
        self.biases = np.array(biases, dtype=np.float64)
        print(
            f"Model loaded from {path}: grid {self.grid_width}x{self.grid_height}, "
            f"weights {self.weights.shape}, biases {self.biases.shape}"
        )

    def _initialise_foam(self) -> None:
        self.foam = QuantumFoam2D(self.grid_width, self.grid_height)
        self.foam.set_coupling_strength(self.config.get("coupling_strength", 0.1))
        self.foam.set_diffusion_rate(self.config.get("diffusion_rate", 0.05))
        self.foam.set_decay_rate(self.config.get("decay_rate", 0.01))
        self.foam.set_quantum_noise(self.config.get("quantum_noise", 0.0005))

    def predict(
        self,
        keypoints: np.ndarray,
        video_width: int,
        video_height: int,
        window_size: int | None = None,
    ) -> Tuple[int, np.ndarray, str]:
        energy_map = self.encode_window(keypoints, video_width, video_height, window_size)
        logits = self.weights @ energy_map + self.biases
        probs = self.softmax(logits)

        pred_idx = int(np.argmax(probs))
        pred_name = self.class_names[pred_idx]
        return pred_idx, probs, pred_name

    def encode_window(
        self,
        keypoints: np.ndarray,
        video_width: int,
        video_height: int,
        window_size: int | None = None,
    ) -> np.ndarray:
        if window_size is None:
            window = self.window_size
        else:
            window = int(window_size)

        frames = min(keypoints.shape[0], window)
        if frames <= 0:
            raise ValueError("encode_window received an empty keypoint sequence")

        self.foam.reset()
        energy = self.energy_injection
        threshold = self.confidence_threshold
        width = float(video_width)
        height = float(video_height)

        for frame_idx in range(frames):
            frame = keypoints[frame_idx]

            if frame.ndim == 3:
                for mouse in range(frame.shape[0]):
                    self._inject_frame(frame[mouse], width, height, energy, threshold)
            elif frame.ndim == 2:
                self._inject_frame(frame, width, height, energy, threshold)
            else:
                raise ValueError(f"Unsupported frame shape: {frame.shape}")

            self.foam.time_step(self.dt)

        return self.foam.observe_gaussian(1)

    def _inject_frame(
        self,
        keypoints: np.ndarray,
        video_width: float,
        video_height: float,
        energy: float,
        threshold: float,
    ) -> None:
        grid_w = self.grid_width
        grid_h = self.grid_height

        for kp in keypoints:
            if kp.size < 2:
                continue
            x, y = kp[0], kp[1]
            conf = kp[2] if kp.size >= 3 else 1.0
            if np.isnan(x) or np.isnan(y) or conf < threshold:
                continue

            nx = min(max(x / video_width, 0.0), 0.999)
            ny = min(max(y / video_height, 0.0), 0.999)
            gx = int(nx * grid_w)
            gy = int(ny * grid_h)
            self.foam.inject_energy(gx, gy, energy)

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp_logits = np.exp(shifted)
        return exp_logits / exp_logits.sum()

    def batch_predict(
        self,
        keypoints_list: List[np.ndarray],
        video_widths: List[int],
        video_heights: List[int],
        window_size: int | None = None,
    ) -> List[Tuple[int, np.ndarray, str]]:
        results: List[Tuple[int, np.ndarray, str]] = []
        for keypoints, width, height in zip(keypoints_list, video_widths, video_heights):
            results.append(self.predict(keypoints, width, height, window_size))
        return results


def example_usage() -> None:
    print("Loading QESN model...")
    model = QESNInference("model_weights.bin", "model_config.json")

    video_width = 1024
    video_height = 570
    window_size = model.window_size

    keypoints = np.random.rand(window_size, 4, 18, 3)
    keypoints[:, :, :, 0] *= video_width
    keypoints[:, :, :, 1] *= video_height
    keypoints[:, :, :, 2] = 0.5 + 0.5 * np.random.rand(window_size, 4, 18)

    pred_idx, probs, pred_name = model.predict(keypoints, video_width, video_height)

    print(f"\nPrediction: {pred_name} (index {pred_idx})")
    print(f"Confidence: {probs[pred_idx]:.4f}")
    print("\nTop 5 predictions:")
    top5_indices = np.argsort(probs)[-5:][::-1]
    for idx in top5_indices:
        print(f"  {model.class_names[idx]}: {probs[idx]:.4f}")


def kaggle_submission_template() -> None:
    print("""
# Kaggle Submission Template

import pandas as pd
import pyarrow.parquet as pq

model = QESNInference('/kaggle/input/your-dataset/model_weights.bin',
                      '/kaggle/input/your-dataset/model_config.json')

test_metadata = pd.read_csv('/kaggle/input/mabe-2022/test.csv')
submission_rows = []

for _, row in test_metadata.iterrows():
    video_id = row['video_id']
    video_width = row['width']
    video_height = row['height']

    tracking_path = f"/kaggle/input/mabe-2022/test_tracking/{video_id}.parquet"
    tracking_df = pq.read_table(tracking_path).to_pandas()
    keypoints = convert_tracking_to_keypoints(tracking_df)

    for start in range(0, len(keypoints) - model.window_size + 1, model.stride):
        window = keypoints[start:start + model.window_size]
        pred_idx, probs, pred_name = model.predict(window, video_width, video_height)
        submission_rows.append({
            'video_id': video_id,
            'frame_start': start,
            'frame_end': start + model.window_size,
            'prediction': pred_name,
            'confidence': float(probs[pred_idx])
        })

submission = pd.DataFrame(submission_rows)
submission.to_csv('submission.csv', index=False)
""")


if __name__ == "__main__":
    print("=" * 60)
    print("QESN-MABe V2 Inference")
    print("=" * 60)
    print()

    if len(sys.argv) > 1 and sys.argv[1] == "--template":
        kaggle_submission_template()
    else:
        example_usage()

    print()
    print("=" * 60)
    print("For Kaggle submission template, run:")
    print("  python qesn_inference.py --template")
    print("=" * 60)
