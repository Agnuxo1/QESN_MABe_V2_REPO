#!/usr/bin/env python3
"""
Script para crear datos de demostración para el modelo QESN.
Genera datos sintéticos de keypoints y etiquetas para probar el pipeline completo.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Agregar el directorio raíz del proyecto al PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from python.model_loader import load_inference

def create_demo_data():
    """Crea datos de demostración sintéticos para probar el modelo QESN."""
    
    # Cargar el modelo para obtener las clases
    model = load_inference()
    behaviors = model.class_names
    print(f"Clases disponibles: {len(behaviors)}")
    print(f"Primeras 10 clases: {behaviors[:10]}")
    
    # Crear directorio de datos si no existe
    data_dir = Path("data/exposure_dataset")
    tracking_dir = data_dir / "tracking"
    data_dir.mkdir(parents=True, exist_ok=True)
    tracking_dir.mkdir(parents=True, exist_ok=True)
    
    # Parámetros del dataset sintético
    num_videos = 10  # Más videos para mejor diversidad
    frames_per_video = 600  # Videos más largos
    mice_per_frame = 4
    keypoints_per_mouse = 18
    window_size = 60
    stride = 30
    
    # Definir patrones de movimiento específicos para diferentes comportamientos
    behavior_patterns = {
        'mount': {'speed': 'high', 'proximity': 'close', 'movement': 'circular'},
        'sniff': {'speed': 'low', 'proximity': 'close', 'movement': 'linear'},
        'chase': {'speed': 'very_high', 'proximity': 'far', 'movement': 'linear'},
        'attack': {'speed': 'high', 'proximity': 'close', 'movement': 'erratic'},
        'escape': {'speed': 'very_high', 'proximity': 'far', 'movement': 'linear'},
        'selfgroom': {'speed': 'very_low', 'proximity': 'self', 'movement': 'stationary'},
        'allogroom': {'speed': 'low', 'proximity': 'close', 'movement': 'stationary'},
        'rear': {'speed': 'low', 'proximity': 'self', 'movement': 'vertical'},
        'dig': {'speed': 'low', 'proximity': 'ground', 'movement': 'stationary'},
        'freeze': {'speed': 'none', 'proximity': 'self', 'movement': 'stationary'},
        'follow': {'speed': 'medium', 'proximity': 'close', 'movement': 'linear'},
        'approach': {'speed': 'medium', 'proximity': 'close', 'movement': 'linear'},
        'avoid': {'speed': 'high', 'proximity': 'far', 'movement': 'linear'},
        'defend': {'speed': 'low', 'proximity': 'close', 'movement': 'stationary'},
        'tussle': {'speed': 'high', 'proximity': 'close', 'movement': 'erratic'},
        'huddle': {'speed': 'very_low', 'proximity': 'close', 'movement': 'stationary'},
        'exploreobject': {'speed': 'low', 'proximity': 'object', 'movement': 'circular'},
        'biteobject': {'speed': 'low', 'proximity': 'object', 'movement': 'stationary'},
        'flinch': {'speed': 'high', 'proximity': 'self', 'movement': 'erratic'},
        'disengage': {'speed': 'medium', 'proximity': 'far', 'movement': 'linear'},
        'dominance': {'speed': 'medium', 'proximity': 'close', 'movement': 'stationary'},
        'submit': {'speed': 'low', 'proximity': 'close', 'movement': 'stationary'},
        'intromit': {'speed': 'low', 'proximity': 'close', 'movement': 'stationary'},
        'ejaculate': {'speed': 'very_low', 'proximity': 'close', 'movement': 'stationary'},
        'attemptmount': {'speed': 'medium', 'proximity': 'close', 'movement': 'circular'},
        'dominancemount': {'speed': 'medium', 'proximity': 'close', 'movement': 'circular'},
        'dominancegroom': {'speed': 'low', 'proximity': 'close', 'movement': 'stationary'},
        'genitalgroom': {'speed': 'low', 'proximity': 'self', 'movement': 'stationary'},
        'sniffbody': {'speed': 'low', 'proximity': 'close', 'movement': 'linear'},
        'sniffface': {'speed': 'low', 'proximity': 'close', 'movement': 'linear'},
        'sniffgenital': {'speed': 'low', 'proximity': 'close', 'movement': 'linear'},
        'reciprocalsniff': {'speed': 'low', 'proximity': 'close', 'movement': 'linear'},
        'climb': {'speed': 'medium', 'proximity': 'object', 'movement': 'vertical'},
        'run': {'speed': 'very_high', 'proximity': 'far', 'movement': 'linear'},
        'rest': {'speed': 'none', 'proximity': 'self', 'movement': 'stationary'},
        'chaseattack': {'speed': 'very_high', 'proximity': 'close', 'movement': 'erratic'}
    }
    
    # Crear datos de keypoints sintéticos
    labels_data = []
    
    for video_id in range(num_videos):
        video_name = f"video_{video_id:03d}"
        print(f"Creando datos para {video_name}...")
        
        # Generar keypoints sintéticos con patrones específicos por comportamiento
        keypoints_data = []
        
        # Dividir el video en segmentos con diferentes comportamientos
        segments = 8  # 8 segmentos por video
        segment_frames = frames_per_video // segments
        
        for segment in range(segments):
            # Seleccionar comportamiento para este segmento
            behavior = np.random.choice(behaviors)
            pattern = behavior_patterns.get(behavior, {'speed': 'medium', 'proximity': 'close', 'movement': 'linear'})
            
            start_frame = segment * segment_frames
            end_frame = min((segment + 1) * segment_frames, frames_per_video)
            
            for frame in range(start_frame, end_frame):
                t = (frame - start_frame) / (end_frame - start_frame)
                
                for mouse_id in range(mice_per_frame):
                    for kp_id in range(keypoints_per_mouse):
                        # Generar movimiento basado en el patrón del comportamiento
                        if pattern['movement'] == 'stationary':
                            base_x = 200 + mouse_id * 50 + kp_id * 2
                            base_y = 200 + mouse_id * 50 + kp_id * 2
                        elif pattern['movement'] == 'linear':
                            base_x = 200 + mouse_id * 50 + kp_id * 2 + 50 * t
                            base_y = 200 + mouse_id * 50 + kp_id * 2
                        elif pattern['movement'] == 'circular':
                            angle = 2 * np.pi * t + mouse_id * np.pi/2
                            radius = 30 + kp_id * 2
                            base_x = 200 + mouse_id * 50 + radius * np.cos(angle)
                            base_y = 200 + mouse_id * 50 + radius * np.sin(angle)
                        elif pattern['movement'] == 'erratic':
                            base_x = 200 + mouse_id * 50 + kp_id * 2 + 30 * np.sin(10 * t)
                            base_y = 200 + mouse_id * 50 + kp_id * 2 + 30 * np.cos(10 * t)
                        elif pattern['movement'] == 'vertical':
                            base_x = 200 + mouse_id * 50 + kp_id * 2
                            base_y = 200 + mouse_id * 50 + kp_id * 2 + 50 * t
                        else:
                            base_x = 200 + mouse_id * 50 + kp_id * 2
                            base_y = 200 + mouse_id * 50 + kp_id * 2
                        
                        # Agregar ruido basado en la velocidad del comportamiento
                        noise_scale = {'none': 1, 'very_low': 2, 'low': 3, 'medium': 5, 'high': 8, 'very_high': 12}
                        noise = noise_scale.get(pattern['speed'], 5)
                        
                        x = base_x + np.random.normal(0, noise)
                        y = base_y + np.random.normal(0, noise)
                        
                        # Confianza basada en el comportamiento
                        if pattern['speed'] == 'none':
                            confidence = np.random.uniform(0.9, 1.0)
                        elif pattern['speed'] == 'very_high':
                            confidence = np.random.uniform(0.6, 0.8)
                        else:
                            confidence = np.random.uniform(0.7, 0.9)
                        
                        keypoints_data.append({
                            'frame': frame,
                            'track_id': mouse_id,
                            'keypoint': kp_id,
                            'x': x,
                            'y': y,
                            'confidence': confidence
                        })
                
                # Agregar etiquetas para algunos frames del segmento
                if frame % 20 == 0:  # Etiqueta cada 20 frames
                    labels_data.append({
                        'video_id': video_name,
                        'frame': frame,
                        'behavior': behavior
                    })
        
        # Guardar keypoints como parquet
        kp_df = pd.DataFrame(keypoints_data)
        kp_df.to_parquet(tracking_dir / f"{video_name}.parquet", index=False)
    
    # Guardar etiquetas
    labels_df = pd.DataFrame(labels_data)
    labels_df.to_csv(data_dir / "labels.csv", index=False)
    
    print(f"\nDatos de demostración creados:")
    print(f"  Videos: {num_videos}")
    print(f"  Frames por video: {frames_per_video}")
    print(f"  Total de etiquetas: {len(labels_data)}")
    print(f"  Archivos guardados en: {data_dir}")
    print(f"  Archivos de tracking: {len(list(tracking_dir.glob('*.parquet')))}")
    
    return data_dir

if __name__ == "__main__":
    create_demo_data()
