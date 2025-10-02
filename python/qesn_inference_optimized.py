#!/usr/bin/env python3
"""
QESN-MABe V2: Quantum Energy State Network OPTIMIZADO para Máxima Precisión
Author: Francisco Angulo de Lafuente
License: MIT

Versión optimizada con todas las mejoras del plan de precisión:
- Limpieza de datos y balanceo temporal
- Física cuántica adaptativa (dt dinámico, acoplamiento adaptativo, energía adaptativa)
- Clasificador mejorado (regularización L2, temperatura softmax)
- Validación cruzada y métricas avanzadas
"""

from __future__ import annotations

import json
import struct
import sys
from typing import List, Tuple, Optional
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

try:
    from .quantum_foam import QuantumFoam2D
except ImportError:  # pragma: no cover - script execution without package context
    from quantum_foam import QuantumFoam2D


class QESNInferenceOptimized:
    """Motor de inferencia optimizado que implementa todas las mejoras del plan de precisión"""

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
        self.window_size = int(self.config.get("window_size", 60))
        self.stride = int(self.config.get("stride", 30))
        self.dt = float(self.config.get("time_step", 0.002))
        self.energy_injection = float(self.config.get("energy_injection", 0.05))
        self.confidence_threshold = float(self.config.get("confidence_threshold", 0.3))

        # Parámetros optimizados
        self.weight_decay = float(self.config.get("weight_decay", 2e-5))
        self.softmax_temperature = float(self.config.get("softmax_temperature", 0.95))
        self.adaptive_dt = bool(self.config.get("adaptive_dt", True))
        self.adaptive_coupling = bool(self.config.get("adaptive_coupling", True))
        self.adaptive_energy = bool(self.config.get("adaptive_energy", True))
        self.data_cleaning = bool(self.config.get("data_cleaning", True))
        self.temporal_balancing = bool(self.config.get("temporal_balancing", True))

        self.load_weights(weights_path)
        self._initialise_foam()
        self._setup_calibration()

    def load_config(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        print(f"Configuración optimizada cargada desde {path}")

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

        # Aplicar regularización L2
        self.weights = np.array(weights_flat, dtype=np.float64).reshape(self.num_classes, grid_size)
        self.biases = np.array(biases, dtype=np.float64)
        
        # Regularización L2 en pesos
        self.weights = self.weights * (1 - self.weight_decay)
        
        print(f"Modelo optimizado cargado desde {path}: grid {self.grid_width}x{self.grid_height}")
        print(f"Pesos: {self.weights.shape}, Sesgos: {self.biases.shape}")
        print(f"Regularización L2: {self.weight_decay}")

    def _initialise_foam(self) -> None:
        self.foam = QuantumFoam2D(self.grid_width, self.grid_height)
        self.foam.set_coupling_strength(self.config.get("coupling_strength", 0.5))
        self.foam.set_diffusion_rate(self.config.get("diffusion_rate", 0.5))
        self.foam.set_decay_rate(self.config.get("decay_rate", 0.001))
        self.foam.set_quantum_noise(self.config.get("quantum_noise", 0.0005))

    def _setup_calibration(self) -> None:
        """Configurar calibración post-hoc para mejorar probabilidades"""
        self.calibrator = None
        self.is_calibrated = False

    def clean_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Limpiar keypoints eliminando frames con tracking defectuoso"""
        if not self.data_cleaning:
            return keypoints
            
        cleaned_keypoints = []
        
        for frame_idx in range(keypoints.shape[0]):
            frame = keypoints[frame_idx]
            
            # Calcular confianza media del frame
            if frame.ndim == 3:  # [mice, keypoints, coords]
                confidences = frame[:, :, 2] if frame.shape[2] >= 3 else np.ones_like(frame[:, :, 0])
                mean_confidence = np.mean(confidences)
            elif frame.ndim == 2:  # [keypoints, coords]
                confidences = frame[:, 2] if frame.shape[1] >= 3 else np.ones_like(frame[:, 0])
                mean_confidence = np.mean(confidences)
            else:
                mean_confidence = 1.0
            
            # Filtrar frames con confianza baja
            if mean_confidence >= self.confidence_threshold:
                cleaned_keypoints.append(frame)
            else:
                # Interpolación suave para frames problemáticos
                if len(cleaned_keypoints) > 0:
                    prev_frame = cleaned_keypoints[-1]
                    interpolated = 0.7 * prev_frame + 0.3 * frame
                    cleaned_keypoints.append(interpolated)
                else:
                    # Si es el primer frame, usar valores por defecto
                    default_frame = np.zeros_like(frame)
                    default_frame[:, :, 2] = self.confidence_threshold  # confianza mínima
                    cleaned_keypoints.append(default_frame)
        
        return np.array(cleaned_keypoints)

    def calculate_window_entropy(self, keypoints: np.ndarray) -> float:
        """Calcular entropía de la ventana para ajustes adaptativos"""
        if keypoints.size == 0:
            return 0.0
            
        # Calcular variación espacial
        if keypoints.ndim >= 3:
            positions = keypoints[:, :, :2]  # x, y coordinates
            spatial_variance = np.var(positions)
        else:
            spatial_variance = np.var(keypoints)
            
        # Convertir a entropía (log de la varianza + 1 para evitar log(0))
        entropy = np.log(spatial_variance + 1.0)
        return min(entropy, 10.0)  # Limitar entropía máxima

    def adaptive_physics_parameters(self, keypoints: np.ndarray) -> Tuple[float, float, float]:
        """Calcular parámetros físicos adaptativos basados en el contenido de la ventana"""
        
        if not (self.adaptive_dt or self.adaptive_coupling or self.adaptive_energy):
            return self.dt, self.foam.coupling_strength, self.energy_injection
        
        entropy = self.calculate_window_entropy(keypoints)
        
        # DT dinámico: reducir para movimientos rápidos
        if self.adaptive_dt:
            if entropy > 5.0:  # Movimientos rápidos
                dt_adaptive = 0.0015
            elif entropy > 2.0:  # Movimientos moderados
                dt_adaptive = 0.0018
            else:  # Movimientos lentos
                dt_adaptive = 0.002
        else:
            dt_adaptive = self.dt
            
        # Acoplamiento adaptativo basado en entropía
        if self.adaptive_coupling:
            coupling_adaptive = 0.45 + (entropy / 10.0) * 0.07  # Oscila entre 0.45 y 0.52
        else:
            coupling_adaptive = self.foam.coupling_strength
            
        # Energía adaptativa basada en conteo de keypoints válidos
        if self.adaptive_energy:
            valid_keypoints = np.sum(keypoints[:, :, 2] >= self.confidence_threshold) if keypoints.ndim == 3 else np.sum(keypoints[:, 2] >= self.confidence_threshold)
            total_keypoints = keypoints.size // 3 if keypoints.ndim == 3 else keypoints.shape[0]
            energy_ratio = valid_keypoints / max(total_keypoints, 1)
            energy_adaptive = 0.04 + energy_ratio * 0.02  # Oscila entre 0.04 y 0.06
        else:
            energy_adaptive = self.energy_injection
            
        return dt_adaptive, coupling_adaptive, energy_adaptive

    def predict(
        self,
        keypoints: np.ndarray,
        video_width: int,
        video_height: int,
        window_size: int | None = None,
        return_confidence: bool = True,
    ) -> Tuple[int, np.ndarray, str]:
        """Predicción optimizada con todas las mejoras implementadas"""
        
        # Limpiar keypoints si está habilitado
        if self.data_cleaning:
            keypoints = self.clean_keypoints(keypoints)
        
        energy_map = self.encode_window_optimized(keypoints, video_width, video_height, window_size)
        
        # Clasificación con regularización
        logits = self.weights @ energy_map + self.biases
        
        # Aplicar temperatura softmax
        probs = self.softmax_tempered(logits, self.softmax_temperature)
        
        # Calibración post-hoc si está disponible
        if self.is_calibrated and hasattr(self, 'calibrator'):
            probs = self.calibrator.predict_proba(energy_map.reshape(1, -1))[0]

        pred_idx = int(np.argmax(probs))
        pred_name = self.class_names[pred_idx]
        
        if return_confidence:
            return pred_idx, probs, pred_name
        else:
            return pred_idx, pred_name

    def encode_window_optimized(
        self,
        keypoints: np.ndarray,
        video_width: int,
        video_height: int,
        window_size: int | None = None,
    ) -> np.ndarray:
        """Codificación de ventana optimizada con física adaptativa"""
        
        if window_size is None:
            window = self.window_size
        else:
            window = int(window_size)

        frames = min(keypoints.shape[0], window)
        if frames <= 0:
            raise ValueError("encode_window received an empty keypoint sequence")

        # Calcular parámetros físicos adaptativos
        dt_adaptive, coupling_adaptive, energy_adaptive = self.adaptive_physics_parameters(keypoints)
        
        # Aplicar parámetros adaptativos al foam
        self.foam.reset()
        self.foam.set_coupling_strength(coupling_adaptive)
        
        threshold = self.confidence_threshold
        width = float(video_width)
        height = float(video_height)

        for frame_idx in range(frames):
            frame = keypoints[frame_idx]

            if frame.ndim == 3:
                for mouse in range(frame.shape[0]):
                    self._inject_frame_optimized(frame[mouse], width, height, energy_adaptive, threshold)
            elif frame.ndim == 2:
                self._inject_frame_optimized(frame, width, height, energy_adaptive, threshold)
            else:
                raise ValueError(f"Unsupported frame shape: {frame.shape}")

            # Usar dt adaptativo
            self.foam.time_step(dt_adaptive)

        return self.foam.observe_gaussian(1)

    def _inject_frame_optimized(
        self,
        keypoints: np.ndarray,
        video_width: float,
        video_height: float,
        energy: float,
        threshold: float,
    ) -> None:
        """Inyección de frame optimizada con normalización por frame"""
        
        grid_w = self.grid_width
        grid_h = self.grid_height
        
        valid_keypoints = 0
        total_energy = 0.0

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
            
            # Normalizar energía por frame para evitar saturaciones
            frame_energy = energy * conf
            self.foam.inject_energy(gx, gy, frame_energy)
            
            valid_keypoints += 1
            total_energy += frame_energy
        
        # Balanceo temporal: ajustar energía total si hay pocos keypoints válidos
        if self.temporal_balancing and valid_keypoints > 0:
            expected_energy = len(keypoints) * energy
            if total_energy < expected_energy * 0.5:  # Menos del 50% de energía esperada
                # Redistribuir energía adicional
                additional_energy = (expected_energy - total_energy) / valid_keypoints
                for kp in keypoints:
                    if kp.size >= 3 and kp[2] >= threshold:
                        nx = min(max(kp[0] / video_width, 0.0), 0.999)
                        ny = min(max(kp[1] / video_height, 0.0), 0.999)
                        gx = int(nx * grid_w)
                        gy = int(ny * grid_h)
                        self.foam.inject_energy(gx, gy, additional_energy)

    @staticmethod
    def softmax_tempered(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Softmax con temperatura para suavizar distribuciones"""
        scaled_logits = logits / temperature
        shifted = scaled_logits - np.max(scaled_logits)
        exp_logits = np.exp(shifted)
        return exp_logits / exp_logits.sum()

    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Calibrar el modelo usando validación cruzada"""
        try:
            # Extraer features de validación
            features_val = []
            for i in range(len(X_val)):
                keypoints, width, height = X_val[i]
                energy_map = self.encode_window_optimized(keypoints, width, height)
                features_val.append(energy_map)
            
            features_val = np.array(features_val)
            
            # Entrenar calibrador
            self.calibrator = CalibratedClassifierCV(method='isotonic', cv=3)
            self.calibrator.fit(features_val, y_val)
            self.is_calibrated = True
            
            print("Modelo calibrado exitosamente")
        except Exception as e:
            print(f"Error en calibración: {e}")
            self.is_calibrated = False

    def evaluate_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluar métricas avanzadas del modelo"""
        
        predictions = []
        probabilities = []
        
        for i in range(len(X_test)):
            keypoints, width, height = X_test[i]
            pred_idx, probs, _ = self.predict(keypoints, width, height)
            predictions.append(pred_idx)
            probabilities.append(probs)
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Métricas principales
        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(y_test, predictions, average='macro')
        weighted_f1 = f1_score(y_test, predictions, average='weighted')
        
        # Análisis por clase
        class_report = classification_report(y_test, predictions, target_names=self.class_names, output_dict=True)
        
        # Calcular F1 mínimo para clases minoritarias
        f1_scores = [class_report[cls]['f1-score'] for cls in self.class_names if cls in class_report]
        min_f1 = min(f1_scores) if f1_scores else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'min_class_f1': min_f1,
            'classification_report': class_report,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        print(f"Precisión: {accuracy:.4f}")
        print(f"F1 Macro: {macro_f1:.4f}")
        print(f"F1 Ponderado: {weighted_f1:.4f}")
        print(f"F1 Mínimo (clases minoritarias): {min_f1:.4f}")
        
        return metrics

    def batch_predict(
        self,
        keypoints_list: List[np.ndarray],
        video_widths: List[int],
        video_heights: List[int],
        window_size: int | None = None,
    ) -> List[Tuple[int, np.ndarray, str]]:
        """Predicción por lotes optimizada"""
        results: List[Tuple[int, np.ndarray, str]] = []
        for keypoints, width, height in zip(keypoints_list, video_widths, video_heights):
            results.append(self.predict(keypoints, width, height, window_size))
        return results


def example_usage() -> None:
    print("Cargando modelo QESN optimizado...")
    model = QESNInferenceOptimized("kaggle_model/model_weights.bin", "kaggle_model/model_config.json")

    video_width = 1024
    video_height = 570
    window_size = model.window_size

    # Generar datos sintéticos para prueba
    keypoints = np.random.rand(window_size, 4, 18, 3)
    keypoints[:, :, :, 0] *= video_width
    keypoints[:, :, :, 1] *= video_height
    keypoints[:, :, :, 2] = 0.5 + 0.5 * np.random.rand(window_size, 4, 18)

    pred_idx, probs, pred_name = model.predict(keypoints, video_width, video_height)

    print(f"\nPredicción: {pred_name} (índice {pred_idx})")
    print(f"Confianza: {probs[pred_idx]:.4f}")
    print("\nTop 5 predicciones:")
    top5_indices = np.argsort(probs)[-5:][::-1]
    for idx in top5_indices:
        print(f"  {model.class_names[idx]}: {probs[idx]:.4f}")


if __name__ == "__main__":
    print("=" * 80)
    print("QESN-MABe V2: Motor de Inferencia OPTIMIZADO")
    print("=" * 80)
    print()
    
    example_usage()
    
    print()
    print("=" * 80)
    print("Motor optimizado listo para máxima precisión!")
    print("=" * 80)
