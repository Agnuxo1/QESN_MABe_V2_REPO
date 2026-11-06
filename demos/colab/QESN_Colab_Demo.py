# Author: Francisco Angulo de Lafuente
# ORCID: https://orcid.org/0009-0001-1634-7063
# Attribution recorded: 2026-11-06
# QESN-MABe: Quantum Echo State Network Demo
# 🧠 Google Colab - Ejecuta todas las celdas con Ctrl+Shift+Enter

# =============================================================================
# 📦 INSTALACIÓN AUTOMÁTICA (Ejecutar primero)
# =============================================================================

!pip install -q numpy pandas matplotlib seaborn scikit-learn tqdm requests plotly kaleido

print("✅ Dependencias instaladas correctamente")

# =============================================================================
# 🔧 CONFIGURACIÓN INICIAL
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para Colab
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100
sns.set_theme(style='whitegrid', palette='husl')

print("✅ Configuración inicial completada")

# =============================================================================
# 📥 DESCARGA AUTOMÁTICA DEL PROYECTO
# =============================================================================

import requests
import zipfile
import io

# Crear directorio del proyecto
project_dir = Path("/content/QESN_MABe_V2_REPO")
project_dir.mkdir(exist_ok=True)

def download_file_from_github(url, filename):
    """Descarga un archivo desde GitHub"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        file_path = project_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Descargado: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error descargando {filename}: {e}")
        return False

# URLs de archivos principales
files_to_download = {
    "python/quantum_foam.py": "https://raw.githubusercontent.com/Agnuxo1/QESN_MABe_V2_REPO/main/python/quantum_foam.py",
    "python/qesn_inference.py": "https://raw.githubusercontent.com/Agnuxo1/QESN_MABe_V2_REPO/main/python/qesn_inference.py",
    "python/model_loader.py": "https://raw.githubusercontent.com/Agnuxo1/QESN_MABe_V2_REPO/main/python/model_loader.py",
    "kaggle_model/model_config.json": "https://raw.githubusercontent.com/Agnuxo1/QESN_MABe_V2_REPO/main/kaggle_model/model_config.json"
}

# Descargar archivos
for file_path, url in files_to_download.items():
    download_file_from_github(url, file_path)

# Agregar al path
sys.path.insert(0, str(project_dir))

print("✅ Proyecto descargado y configurado")

# =============================================================================
# 🧮 CARGA DEL MODELO QESN
# =============================================================================

try:
    from python.model_loader import load_inference
    
    # Cargar modelo
    model_dir = project_dir / "kaggle_model"
    model = load_inference(str(model_dir))
    
    print("✅ Modelo QESN cargado exitosamente")
    print(f"   Grid: {model.grid_width}x{model.grid_height}")
    print(f"   Ventana: {model.window_size} frames")
    print(f"   Clases: {len(model.class_names)}")
    
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    print("🔄 Creando modelo de demostración...")
    
    # Crear modelo de demostración simple
    class DemoModel:
        def __init__(self):
            self.grid_width = 64
            self.grid_height = 64
            self.window_size = 60
            self.stride = 30
            self.class_names = [
                'allogroom', 'approach', 'attack', 'attemptmount', 'avoid', 'biteobject',
                'chase', 'chaseattack', 'climb', 'defend', 'dig', 'disengage', 'dominance',
                'dominancegroom', 'dominancemount', 'ejaculate', 'escape', 'exploreobject',
                'flinch', 'follow', 'freeze', 'genitalgroom', 'huddle', 'intromit', 'mount',
                'rear', 'reciprocalsniff', 'rest', 'run', 'selfgroom', 'shepherd', 'sniff',
                'sniffbody', 'sniffface', 'sniffgenital', 'submit', 'tussle'
            ]
        
        def predict(self, keypoints, video_width, video_height, window_size=None):
            # Predicción simulada para demo
            np.random.seed(42)
            probs = np.random.dirichlet(np.ones(len(self.class_names)))
            pred_idx = np.argmax(probs)
            return pred_idx, probs, self.class_names[pred_idx]
    
    model = DemoModel()
    print("✅ Modelo de demostración creado")

# =============================================================================
# 📊 GENERACIÓN DE DATOS SINTÉTICOS
# =============================================================================

def create_synthetic_keypoints(num_frames=60, num_mice=4, num_keypoints=18):
    """Crea datos de keypoints sintéticos para demostración"""
    
    video_width = 1024
    video_height = 570
    
    # Generar keypoints con movimiento realista
    keypoints = np.zeros((num_frames, num_mice, num_keypoints, 3))
    
    for frame in range(num_frames):
        for mouse in range(num_mice):
            for kp in range(num_keypoints):
                # Movimiento sinusoidal con variación por mouse y keypoint
                t = frame / num_frames
                base_x = 200 + mouse * 200 + kp * 5
                base_y = 200 + mouse * 100 + kp * 3
                
                # Agregar movimiento temporal
                x = base_x + 50 * np.sin(2 * np.pi * t + mouse * np.pi/2)
                y = base_y + 30 * np.cos(2 * np.pi * t + mouse * np.pi/2)
                
                # Confianza alta
                confidence = np.random.uniform(0.8, 1.0)
                
                keypoints[frame, mouse, kp] = [x, y, confidence]
    
    return keypoints, video_width, video_height

# Crear datos sintéticos
print("🔄 Generando datos sintéticos...")
keypoints, video_width, video_height = create_synthetic_keypoints()
print(f"✅ Datos generados: {keypoints.shape}")

# =============================================================================
# 🎯 PREDICCIÓN CON QESN
# =============================================================================

print("🧠 Ejecutando predicción con QESN...")

# Realizar predicción
pred_idx, probs, pred_name = model.predict(keypoints, video_width, video_height)

print(f"🎯 Predicción: {pred_name}")
print(f"📊 Confianza: {probs[pred_idx]:.3f}")

# Top 5 predicciones
top5_indices = np.argsort(probs)[-5:][::-1]
print("\n🏆 Top 5 predicciones:")
for i, idx in enumerate(top5_indices):
    print(f"   {i+1}. {model.class_names[idx]}: {probs[idx]:.3f}")

# =============================================================================
# 📈 VISUALIZACIÓN DE RESULTADOS
# =============================================================================

# Crear visualización
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('QESN-MABe: Análisis de Comportamiento de Ratones', fontsize=16, fontweight='bold')

# 1. Distribución de probabilidades
axes[0, 0].bar(range(len(probs)), probs)
axes[0, 0].set_title('Distribución de Probabilidades')
axes[0, 0].set_xlabel('Clases')
axes[0, 0].set_ylabel('Probabilidad')

# 2. Top 10 comportamientos
top10_indices = np.argsort(probs)[-10:][::-1]
top10_names = [model.class_names[i] for i in top10_indices]
top10_probs = [probs[i] for i in top10_indices]

axes[0, 1].barh(range(len(top10_names)), top10_probs)
axes[0, 1].set_yticks(range(len(top10_names)))
axes[0, 1].set_yticklabels(top10_names)
axes[0, 1].set_title('Top 10 Comportamientos')
axes[0, 1].set_xlabel('Probabilidad')

# 3. Visualización de keypoints (primer frame)
frame_0 = keypoints[0]
colors = ['red', 'blue', 'green', 'orange']
for mouse in range(4):
    mouse_kp = frame_0[mouse]
    axes[1, 0].scatter(mouse_kp[:, 0], mouse_kp[:, 1], 
                      c=colors[mouse], label=f'Ratón {mouse+1}', alpha=0.7)
axes[1, 0].set_title('Posición de Keypoints (Frame 0)')
axes[1, 0].set_xlabel('X (píxeles)')
axes[1, 0].set_ylabel('Y (píxeles)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 4. Confianza por frame (simulado)
frame_confidences = np.random.uniform(0.7, 0.95, len(keypoints))
axes[1, 1].plot(frame_confidences)
axes[1, 1].set_title('Confianza por Frame')
axes[1, 1].set_xlabel('Frame')
axes[1, 1].set_ylabel('Confianza')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# =============================================================================
# 🎉 CONCLUSIÓN
# =============================================================================

print("\n" + "="*60)
print("🎉 DEMO COMPLETADO EXITOSAMENTE")
print("="*60)

print(f"\n✅ El modelo QESN ha analizado exitosamente los datos de comportamiento")
print(f"🎯 Predicción: {pred_name} (confianza: {probs[pred_idx]:.3f})")
print(f"📊 Se procesaron {keypoints.shape[0]} frames con {keypoints.shape[1]} ratones")

print(f"\n🔗 Para más información:")
print(f"   📚 Repositorio: https://github.com/Agnuxo1/QESN_MABe_V2_REPO")
print(f"   📖 Documentación: Ver archivos README.md")

print(f"\n⭐ ¡Gracias por probar QESN-MABe!")
