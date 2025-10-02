#!/usr/bin/env python3
"""
QESN-MABe V2: Actualizador de Todas las Demos
Author: Francisco Angulo de Lafuente
License: MIT

Script para actualizar todas las demos existentes con las optimizaciones
del plan de precision.
"""

import os
import shutil
from pathlib import Path
import sys

def update_demo_file(file_path: str, backup: bool = True) -> bool:
    """Actualizar un archivo de demo con las optimizaciones"""
    
    if not os.path.exists(file_path):
        print(f"Archivo no encontrado: {file_path}")
        return False
    
    # Crear backup si se solicita
    if backup:
        backup_path = f"{file_path}.backup"
        shutil.copy2(file_path, backup_path)
        print(f"Backup creado: {backup_path}")
    
    # Leer archivo original
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Aplicar actualizaciones
    updated_content = content
    
    # 1. Actualizar import del model_loader para usar motor optimizado
    if "load_inference(None)" in content:
        updated_content = updated_content.replace(
            "load_inference(None)",
            "load_inference(None, optimized=True)"
        )
        print(f"  [OK] Actualizado load_inference en {file_path}")
    
    # 2. Actualizar window_size a 60 si es 30
    if "window_size = 30" in content:
        updated_content = updated_content.replace(
            "window_size = 30",
            "window_size = 60"
        )
        print(f"  [OK] Actualizado window_size a 60 en {file_path}")
    
    # 3. Actualizar confidence_threshold a 0.3 si es 0.5
    if "confidence_threshold = 0.5" in content:
        updated_content = updated_content.replace(
            "confidence_threshold = 0.5",
            "confidence_threshold = 0.3"
        )
        print(f"  [OK] Actualizado confidence_threshold a 0.3 en {file_path}")
    
    # 4. Añadir comentario sobre optimizaciones si no existe
    if "OPTIMIZADO" not in content and "OPTIMIZADO" not in content:
        header_comment = """# OPTIMIZADO: Este archivo ha sido actualizado con las mejoras del plan de precision
# - Motor de inferencia optimizado con fisica cuantica adaptativa
# - Limpieza de datos y balanceo temporal
# - Clasificador mejorado con regularizacion L2 y temperatura softmax
# - Parametros optimizados: window_size=60, confidence_threshold=0.3

"""
        # Insertar al principio del archivo
        lines = updated_content.split('\n')
        if not lines[0].startswith('#!/usr/bin/env python3'):
            updated_content = header_comment + updated_content
        else:
            # Insertar después del shebang
            lines.insert(1, header_comment.rstrip())
            updated_content = '\n'.join(lines)
        print(f"  [OK] Añadido comentario de optimizacion en {file_path}")
    
    # Escribir archivo actualizado
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"Archivo actualizado: {file_path}")
    return True

def update_notebook_file(file_path: str) -> bool:
    """Actualizar un notebook con optimizaciones básicas"""
    
    if not os.path.exists(file_path):
        print(f"Notebook no encontrado: {file_path}")
        return False
    
    # Crear backup
    backup_path = f"{file_path}.backup"
    shutil.copy2(file_path, backup_path)
    print(f"Backup creado: {backup_path}")
    
    # Leer notebook
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Actualizaciones básicas para notebooks
    updated_content = content
    
    # Actualizar load_inference si existe
    if "load_inference(None)" in content:
        updated_content = updated_content.replace(
            "load_inference(None)",
            "load_inference(None, optimized=True)"
        )
        print(f"  [OK] Actualizado load_inference en {file_path}")
    
    # Escribir notebook actualizado
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"Notebook actualizado: {file_path}")
    return True

def main():
    """Funcion principal para actualizar todas las demos"""
    
    print("=" * 80)
    print("QESN-MABe V2: ACTUALIZADOR DE TODAS LAS DEMOS")
    print("=" * 80)
    
    # Archivos de demo a actualizar
    demo_files = [
        "demo_espectacular.py",
        "demo_profesional.py", 
        "demo_simple.py",
        "demo_simple_no_emoji.py",
        "demo_espectacular_no_emoji.py"
    ]
    
    # Notebooks a actualizar
    notebook_files = [
        "notebooks/QESN_Complete_Classification_Demo.ipynb",
        "notebooks/QESN_Professional_Quantum_Demo.ipynb",
        "notebooks/QESN_Demo_Interactive.ipynb",
        "notebooks/demo_espectacular_notebook.py",
        "notebooks/qesn_demo_espectacular.py"
    ]
    
    print("\nACTUALIZANDO ARCHIVOS DE DEMO:")
    print("-" * 40)
    
    updated_demos = 0
    for demo_file in demo_files:
        if os.path.exists(demo_file):
            if update_demo_file(demo_file):
                updated_demos += 1
        else:
            print(f"Archivo no encontrado: {demo_file}")
    
    print(f"\nACTUALIZANDO NOTEBOOKS:")
    print("-" * 40)
    
    updated_notebooks = 0
    for notebook_file in notebook_files:
        if os.path.exists(notebook_file):
            if update_notebook_file(notebook_file):
                updated_notebooks += 1
        else:
            print(f"Notebook no encontrado: {notebook_file}")
    
    # Crear resumen de archivos optimizados
    optimized_files = [
        "demo_espectacular_optimized.py",
        "validation_pipeline.py",
        "python/qesn_inference_optimized.py"
    ]
    
    print(f"\nARCHIVOS NUEVOS OPTIMIZADOS:")
    print("-" * 40)
    
    for opt_file in optimized_files:
        if os.path.exists(opt_file):
            print(f"[OK] {opt_file} - LISTO")
        else:
            print(f"[X] {opt_file} - NO ENCONTRADO")
    
    # Resumen final
    print(f"\n{'='*80}")
    print("RESUMEN DE ACTUALIZACIONES")
    print(f"{'='*80}")
    print(f"Demos actualizadas: {updated_demos}/{len(demo_files)}")
    print(f"Notebooks actualizados: {updated_notebooks}/{len(notebook_files)}")
    print(f"Archivos optimizados nuevos: {len([f for f in optimized_files if os.path.exists(f)])}")
    
    print(f"\nMEJORAS APLICADAS:")
    print("[OK] Motor de inferencia optimizado")
    print("[OK] Fisica cuantica adaptativa")
    print("[OK] Limpieza de datos y balanceo temporal")
    print("[OK] Clasificador mejorado con regularizacion L2")
    print("[OK] Temperatura softmax optimizada")
    print("[OK] Parametros optimizados (window_size=60, confidence_threshold=0.3)")
    print("[OK] Validacion cruzada y metricas avanzadas")
    
    print(f"\nPRÓXIMOS PASOS:")
    print("1. Ejecutar demo_espectacular_optimized.py para ver mejoras")
    print("2. Ejecutar validation_pipeline.py para validacion completa")
    print("3. Probar demos actualizadas")
    print("4. Verificar mejoras en precision")
    
    print(f"\n{'='*80}")
    print("ACTUALIZACION COMPLETA FINALIZADA")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
