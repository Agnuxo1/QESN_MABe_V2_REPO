#!/usr/bin/env python3
"""
Script para crear pesos de demostración para QESN-MABe V2
"""

import struct
import numpy as np
import os

def create_demo_weights():
    """Crear archivo de pesos de demostración"""
    
    # Parámetros del modelo optimizados según checkpoint base
    grid_width = 32
    grid_height = 32
    num_classes = 37
    grid_size = grid_width * grid_height
    
    # Crear directorio si no existe
    os.makedirs("kaggle_model", exist_ok=True)
    
    # Generar pesos aleatorios pero determinísticos
    np.random.seed(42)  # Para reproducibilidad
    
    weights_flat = np.random.normal(0, 0.1, num_classes * grid_size).astype(np.float64)
    biases = np.random.normal(0, 0.1, num_classes).astype(np.float64)
    
    # Escribir archivo binario
    with open("kaggle_model/model_weights.bin", "wb") as f:
        # Escribir dimensiones
        f.write(struct.pack("Q", grid_width))
        f.write(struct.pack("Q", grid_height))
        f.write(struct.pack("Q", len(weights_flat)))
        f.write(struct.pack("Q", len(biases)))
        
        # Escribir pesos y sesgos
        f.write(struct.pack(f"{len(weights_flat)}d", *weights_flat))
        f.write(struct.pack(f"{len(biases)}d", *biases))
    
    print(f"Modelo de demostración creado:")
    print(f"  - Grid: {grid_width}x{grid_height}")
    print(f"  - Clases: {num_classes}")
    print(f"  - Pesos: {len(weights_flat)} valores")
    print(f"  - Sesgos: {len(biases)} valores")
    print(f"  - Archivo: kaggle_model/model_weights.bin")

if __name__ == "__main__":
    create_demo_weights()
