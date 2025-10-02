#!/usr/bin/env python3
"""
Script para crear y subir QESN-MABe Demo a HuggingFace Spaces
Usando la API de HuggingFace Hub
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
import requests

# Configuracion
HF_TOKEN = "[TU_TOKEN_AQUI]"
SPACE_NAME = "QESN-MABe-Demo"
REPO_ID = f"Agnuxo/{SPACE_NAME}"
LOCAL_FOLDER = "huggingface_space"

def create_huggingface_space():
    """Crea el Space en HuggingFace"""
    
    print("Creando HuggingFace Space...")
    
    # Inicializar API
    api = HfApi(token=HF_TOKEN)
    
    try:
        # Crear el Space usando la API directamente
        response = requests.post(
            "https://huggingface.co/api/repos/create",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={
                "name": SPACE_NAME,
                "organization": "Agnuxo",
                "type": "space",
                "sdk": "streamlit",
                "private": False
            }
        )
        
        if response.status_code == 201:
            print(f"Space creado: https://huggingface.co/spaces/{REPO_ID}")
            return True
        elif response.status_code == 409:
            print(f"Space ya existe: https://huggingface.co/spaces/{REPO_ID}")
            return True
        else:
            print(f"Error creando Space: {response.status_code} - {response.text}")
            return False
        
    except Exception as e:
        print(f"Error creando Space: {str(e)}")
        return False

def upload_files_to_space():
    """Sube los archivos al Space"""
    
    print("Subiendo archivos al Space...")
    
    # Verificar que existe la carpeta
    if not Path(LOCAL_FOLDER).exists():
        print(f"No se encontro la carpeta: {LOCAL_FOLDER}")
        return False
    
    try:
        # Subir archivos
        upload_folder(
            folder_path=LOCAL_FOLDER,
            repo_id=REPO_ID,
            repo_type="space",
            token=HF_TOKEN,
            commit_message="Initial upload: QESN-MABe Demo with interactive interface"
        )
        print("Archivos subidos exitosamente")
        
    except Exception as e:
        print(f"Error subiendo archivos: {str(e)}")
        return False
    
    return True

def verify_space():
    """Verifica que el Space este funcionando"""
    
    print("Verificando Space...")
    
    space_url = f"https://huggingface.co/spaces/{REPO_ID}"
    
    try:
        response = requests.get(space_url, timeout=10)
        if response.status_code == 200:
            print(f"Space verificado: {space_url}")
            return True
        else:
            print(f"Space creado pero puede estar inicializando: {space_url}")
            return True
            
    except Exception as e:
        print(f"Error verificando Space: {str(e)}")
        return False

def main():
    """Funcion principal"""
    
    print("Iniciando creacion de HuggingFace Space...")
    print(f"Carpeta local: {LOCAL_FOLDER}")
    print(f"Space: {REPO_ID}")
    
    # Verificar archivos locales
    required_files = ["app.py", "requirements.txt", "README.md"]
    for file in required_files:
        file_path = Path(LOCAL_FOLDER) / file
        if not file_path.exists():
            print(f"Archivo requerido no encontrado: {file_path}")
            return False
        else:
            print(f"Archivo encontrado: {file}")
    
    # Crear Space
    if not create_huggingface_space():
        return False
    
    # Subir archivos
    if not upload_files_to_space():
        return False
    
    # Verificar Space
    if not verify_space():
        return False
    
    print("\n" + "="*60)
    print("HuggingFace Space creado exitosamente!")
    print("="*60)
    print(f"URL: https://huggingface.co/spaces/{REPO_ID}")
    print(f"GitHub: https://github.com/Agnuxo1/QESN_MABe_V2_REPO")
    print(f"Kaggle: https://www.kaggle.com/franciscoangulo")
    print("\nEl Space puede tardar unos minutos en inicializarse.")
    print("Una vez listo, podras usar la interfaz interactiva.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)