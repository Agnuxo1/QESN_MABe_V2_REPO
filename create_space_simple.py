#!/usr/bin/env python3
"""
Script simple para crear HuggingFace Space usando requests
"""

import requests
import json
import os
from pathlib import Path

# Configuracion
HF_TOKEN = "[TU_TOKEN_AQUI]"
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
    
    if response.status_code == 201:
        print(f"Space creado exitosamente!")
        print(f"URL: https://huggingface.co/spaces/{REPO_ID}")
        return True
    elif response.status_code == 409:
        print(f"Space ya existe: https://huggingface.co/spaces/{REPO_ID}")
        return True
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def upload_file(file_path, repo_path):
    """Sube un archivo al Space"""
    
    url = f"https://huggingface.co/api/spaces/{REPO_ID}/upload"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }
    
    with open(file_path, 'rb') as f:
        files = {'file': (repo_path, f)}
        data = {'path': repo_path}
        
        response = requests.post(url, headers=headers, files=files, data=data)
        
        if response.status_code == 200:
            print(f"Archivo subido: {repo_path}")
            return True
        else:
            print(f"Error subiendo {repo_path}: {response.status_code}")
            return False

def main():
    """Funcion principal"""
    
    print("=== Creando HuggingFace Space ===")
    
    # Crear Space
    if not create_space():
        return False
    
    # Subir archivos
    files_to_upload = [
        ("huggingface_space/app.py", "app.py"),
        ("huggingface_space/requirements.txt", "requirements.txt"),
        ("huggingface_space/README.md", "README.md")
    ]
    
    print("\nSubiendo archivos...")
    for local_path, repo_path in files_to_upload:
        if Path(local_path).exists():
            upload_file(local_path, repo_path)
        else:
            print(f"Archivo no encontrado: {local_path}")
    
    print(f"\nSpace listo: https://huggingface.co/spaces/{REPO_ID}")
    return True

if __name__ == "__main__":
    main()
