#!/usr/bin/env python3
"""
Script para crear HuggingFace Space - Version simple
"""

import requests
import json
import os
from pathlib import Path

# Configuracion
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Set via environment variable
SPACE_NAME = "QESN-MABe-Demo"
REPO_ID = f"Agnuxo/{SPACE_NAME}"

def create_space():
    """Crea el Space usando requests"""
    
    url = "https://huggingface.co/api/repos/create"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    data = {
        "name": SPACE_NAME,
        "organization": "Agnuxo", 
        "type": "space",
        "sdk": "streamlit",
        "private": False
    }
    
    print("Creando Space...")
    response = requests.post(url, headers=headers, json=data)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 201:
        print("Space creado exitosamente!")
        print(f"URL: https://huggingface.co/spaces/{REPO_ID}")
        return True
    elif response.status_code == 409:
        print("Space ya existe!")
        print(f"URL: https://huggingface.co/spaces/{REPO_ID}")
        return True
    else:
        print("Error en la respuesta")
        try:
            error_text = response.text
            print(f"Error: {error_text}")
        except:
            print("No se pudo leer el error")
        return False

def upload_file(file_path, repo_path):
    """Sube un archivo al Space"""
    
    url = f"https://huggingface.co/api/spaces/{REPO_ID}/upload"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (repo_path, f)}
            data = {'path': repo_path}
            
            response = requests.post(url, headers=headers, files=files, data=data)
            
            print(f"Subiendo {repo_path}: Status {response.status_code}")
            
            if response.status_code == 200:
                print(f"Archivo subido: {repo_path}")
                return True
            else:
                print(f"Error subiendo {repo_path}")
                return False
    except Exception as e:
        print(f"Error con archivo {file_path}: {str(e)}")
        return False

def main():
    """Funcion principal"""
    
    print("=== Creando HuggingFace Space ===")
    
    # Crear Space
    if not create_space():
        print("No se pudo crear el Space")
        return False
    
    # Subir archivos
    files_to_upload = [
        ("huggingface_space/app.py", "app.py"),
        ("huggingface_space/requirements.txt", "requirements.txt"),
        ("huggingface_space/README.md", "README.md")
    ]
    
    print("\nSubiendo archivos...")
    success_count = 0
    
    for local_path, repo_path in files_to_upload:
        if Path(local_path).exists():
            if upload_file(local_path, repo_path):
                success_count += 1
        else:
            print(f"Archivo no encontrado: {local_path}")
    
    print(f"\nArchivos subidos: {success_count}/{len(files_to_upload)}")
    print(f"Space URL: https://huggingface.co/spaces/{REPO_ID}")
    
    return success_count > 0

if __name__ == "__main__":
    success = main()
    if success:
        print("\nSpace creado y configurado exitosamente!")
    else:
        print("\nHubo problemas en la creacion del Space")
