# QESN-MABe: Quantum Echo State Network Demo
# 🏆 Kaggle Notebooks - Ejecuta todas las celdas con Shift+Enter

# =============================================================================
# 📦 INSTALACIÓN AUTOMÁTICA (Ejecutar primero)
# =============================================================================

# Kaggle ya tiene la mayoría de dependencias, solo instalamos las que faltan
import subprocess
import sys

def install_if_missing(package):
    try:
        __import__(package)
        print(f"✅ {package} ya está disponible")
    except ImportError:
        print(f"🔄 Instalando {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} instalado")

# Instalar dependencias adicionales si es necesario
install_if_missing("tqdm")
install_if_missing("requests")

print("✅ Todas las dependencias están listas")

# =============================================================================
# 🔧 CONFIGURACIÓN INICIAL
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para Kaggle
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100
sns.set_theme(style='whitegrid', palette='husl')

print("✅ Configuración inicial completada")

# =============================================================================
# 📥 CONFIGURACIÓN DEL PROYECTO PARA KAGGLE
# =============================================================================

import os
from pathlib import Path

# En Kaggle, trabajamos en el directorio de trabajo
work_dir = Path("/kaggle/working")
project_dir = work_dir / "QESN_MABe_V2_REPO"
project_dir.mkdir(exist_ok=True)

print(f"📁 Directorio de trabajo: {work_dir}")
print(f"📁 Directorio del proyecto: {project_dir}")

# =============================================================================
# 🧮 MODELO QESN SIMPLIFICADO PARA KAGGLE
# =============================================================================

class QESNDemoModel:
    """Modelo QESN simplificado para demostración en Kaggle"""
    
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
        
        # Simular parámetros cuánticos
        self.coupling_strength = 0.5
        self.diffusion_rate = 0.05
        self.decay_rate = 0.001
        self.quantum_noise = 0.0005
        
    def simulate_quantum_foam(self, keypoints):
        """Simula la evolución de la espuma cuántica"""
        frames, mice, keypoints_per_mouse, _ = keypoints.shape
        
        # Crear grid cuántico
        quantum_grid = np.zeros((self.grid_width, self.grid_height))
        
        # Inyectar energía desde keypoints
        for frame in range(frames):
            for mouse in range(mice):
                for kp in range(keypoints_per_mouse):
                    x, y, conf = keypoints[frame, mouse, kp]
                    
                    # Mapear coordenadas al grid
                    grid_x = int((x / 1024) * self.grid_width)
                    grid_y = int((y / 570) * self.grid_height)
                    
                    # Asegurar que esté dentro del grid
                    grid_x = max(0, min(self.grid_width-1, grid_x))
                    grid_y = max(0, min(self.grid_height-1, grid_y))
                    
                    # Inyectar energía con confianza
                    quantum_grid[grid_y, grid_x] += conf * 0.1
        
        # Simular difusión cuántica
        for _ in range(30):  # 30 pasos de evolución
            new_grid = quantum_grid.copy()
            for y in range(1, self.grid_height-1):
                for x in range(1, self.grid_width-1):
                    # Difusión con vecinos
                    neighbors = (
                        quantum_grid[y-1, x] + quantum_grid[y+1, x] +
                        quantum_grid[y, x-1] + quantum_grid[y, x+1]
                    )
                    new_grid[y, x] = (
                        0.6 * quantum_grid[y, x] + 
                        0.1 * neighbors + 
                        np.random.normal(0, self.quantum_noise)
                    )
            quantum_grid = np.maximum(new_grid, 0)  # No energía negativa
        
        return quantum_grid
    
    def predict(self, keypoints, video_width=1024, video_height=570, window_size=None):
        """Predicción principal del modelo"""
        
        # Simular evolución cuántica
        quantum_state = self.simulate_quantum_foam(keypoints)
        
        # Extraer características del estado cuántico
        features = quantum_state.flatten()
        
        # Simular clasificador lineal (pesos aleatorios pero consistentes)
        np.random.seed(42)
        weights = np.random.randn(len(self.class_names), len(features)) * 0.01
        biases = np.random.randn(len(self.class_names)) * 0.1
        
        # Calcular logits
        logits = np.dot(weights, features) + biases
        
        # Softmax para probabilidades
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Predicción
        pred_idx = np.argmax(probs)
        
        return pred_idx, probs, self.class_names[pred_idx]

# Crear modelo
model = QESNDemoModel()
print("✅ Modelo QESN creado exitosamente")
print(f"   Grid: {model.grid_width}x{model.grid_height}")
print(f"   Ventana: {model.window_size} frames")
print(f"   Clases: {len(model.class_names)}")

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
# 📊 ANÁLISIS DETALLADO
# =============================================================================

print("\n" + "="*60)
print("📊 ANÁLISIS DETALLADO DEL MODELO QESN")
print("="*60)

print(f"\n🎯 Predicción Principal:")
print(f"   Comportamiento: {pred_name}")
print(f"   Índice: {pred_idx}")
print(f"   Confianza: {probs[pred_idx]:.3f}")

print(f"\n📈 Estadísticas del Modelo:")
print(f"   Grid Cuántico: {model.grid_width}x{model.grid_height}")
print(f"   Ventana Temporal: {model.window_size} frames")
print(f"   Total de Clases: {len(model.class_names)}")
print(f"   Datos de Entrada: {keypoints.shape}")

print(f"\n🔬 Análisis de Keypoints:")
print(f"   Frames: {keypoints.shape[0]}")
print(f"   Ratones: {keypoints.shape[1]}")
print(f"   Keypoints por ratón: {keypoints.shape[2]}")
print(f"   Dimensiones: {keypoints.shape[3]} (x, y, confianza)")

print(f"\n⚛️ Parámetros Cuánticos:")
print(f"   Coupling Strength: {model.coupling_strength}")
print(f"   Diffusion Rate: {model.diffusion_rate}")
print(f"   Decay Rate: {model.decay_rate}")
print(f"   Quantum Noise: {model.quantum_noise}")

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
print(f"   🏆 Kaggle: https://www.kaggle.com/franciscoangulo")

print(f"\n💡 Próximos pasos:")
print(f"   1. Experimenta con diferentes parámetros")
print(f"   2. Prueba con datos reales de MABe")
print(f"   3. Contribuye al proyecto en GitHub")
print(f"   4. Comparte tus resultados en Kaggle")

print(f"\n⭐ ¡Gracias por probar QESN-MABe en Kaggle!")

# =============================================================================
# 🏆 INSTRUCCIONES PARA KAGGLE
# =============================================================================

print(f"\n" + "="*60)
print("🏆 INSTRUCCIONES PARA KAGGLE NOTEBOOKS")
print("="*60)

print(f"\n📋 Para usar este demo en Kaggle:")
print(f"   1. Crea un nuevo notebook en Kaggle")
print(f"   2. Copia y pega este código")
print(f"   3. Ejecuta todas las celdas (Shift+Enter)")
print(f"   4. ¡Disfruta del análisis!")

print(f"\n🔧 Características de Kaggle:")
print(f"   • GPU/TPU disponible")
print(f"   • Datasets públicos")
print(f"   • Colaboración en tiempo real")
print(f"   • Competencias de ML")

print(f"\n💻 Recursos de Kaggle:")
print(f"   • Notebooks gratuitos")
print(f"   • Datasets de comportamiento animal")
print(f"   • Comunidad activa")
print(f"   • Competencias MABe")
