#!/usr/bin/env python3
"""
Script para descargar y descomprimir el dataset MABe desde GitHub.
Este script maneja la descarga automática del dataset comprimido.
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
from urllib.parse import urlparse
import subprocess

def download_file(url: str, output_path: Path) -> bool:
    """Descarga un archivo desde una URL."""
    try:
        print(f"Descargando {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgreso: {progress:.1f}%", end='', flush=True)
        
        print(f"\nDescarga completada: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error descargando {url}: {e}")
        return False

def extract_rar(file_path: Path, extract_to: Path) -> bool:
    """Extrae un archivo RAR usando unzip o 7zip."""
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        
        # Intentar con unrar primero
        try:
            result = subprocess.run(['unrar', 'x', str(file_path), str(extract_to)], 
                                 capture_output=True, text=True, check=True)
            print(f"Archivo RAR extraído exitosamente con unrar")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Intentar con 7zip
        try:
            result = subprocess.run(['7z', 'x', str(file_path), f'-o{extract_to}'], 
                                 capture_output=True, text=True, check=True)
            print(f"Archivo RAR extraído exitosamente con 7zip")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Intentar con WinRAR en Windows
        try:
            result = subprocess.run(['winrar', 'x', str(file_path), str(extract_to)], 
                                 capture_output=True, text=True, check=True)
            print(f"Archivo RAR extraído exitosamente con WinRAR")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        print("Error: No se encontró un extractor de RAR instalado.")
        print("Por favor instala uno de los siguientes:")
        print("- unrar (Linux/Mac): sudo apt install unrar o brew install unrar")
        print("- 7zip (Windows/Linux/Mac): https://www.7-zip.org/")
        print("- WinRAR (Windows): https://www.win-rar.com/")
        return False
        
    except Exception as e:
        print(f"Error extrayendo {file_path}: {e}")
        return False

def extract_zip(file_path: Path, extract_to: Path) -> bool:
    """Extrae un archivo ZIP."""
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Archivo ZIP extraído exitosamente")
        return True
    except Exception as e:
        print(f"Error extrayendo ZIP {file_path}: {e}")
        return False

def extract_tar(file_path: Path, extract_to: Path) -> bool:
    """Extrae un archivo TAR/TAR.GZ."""
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        with tarfile.open(file_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
        print(f"Archivo TAR extraído exitosamente")
        return True
    except Exception as e:
        print(f"Error extrayendo TAR {file_path}: {e}")
        return False

def setup_dataset(github_url: str, dataset_name: str = "mabe_dataset") -> bool:
    """
    Descarga y configura el dataset desde GitHub.
    
    Args:
        github_url: URL del archivo comprimido en GitHub
        dataset_name: Nombre del directorio donde extraer el dataset
    """
    
    # Crear directorio de datos
    data_dir = Path("data")
    dataset_dir = data_dir / dataset_name
    downloads_dir = data_dir / "downloads"
    
    data_dir.mkdir(exist_ok=True)
    downloads_dir.mkdir(exist_ok=True)
    
    # Obtener nombre del archivo desde la URL
    parsed_url = urlparse(github_url)
    filename = Path(parsed_url.path).name
    file_path = downloads_dir / filename
    
    print(f"Configurando dataset desde: {github_url}")
    print(f"Directorio de destino: {dataset_dir}")
    print(f"Archivo temporal: {file_path}")
    
    # Descargar archivo si no existe
    if not file_path.exists():
        if not download_file(github_url, file_path):
            return False
    else:
        print(f"Archivo ya existe: {file_path}")
    
    # Extraer archivo según su extensión
    file_extension = file_path.suffix.lower()
    
    if file_extension == '.rar':
        success = extract_rar(file_path, dataset_dir)
    elif file_extension == '.zip':
        success = extract_zip(file_path, dataset_dir)
    elif file_extension in ['.tar', '.gz', '.bz2', '.xz']:
        success = extract_tar(file_path, dataset_dir)
    else:
        print(f"Formato de archivo no soportado: {file_extension}")
        return False
    
    if success:
        print(f"\n✅ Dataset configurado exitosamente en: {dataset_dir}")
        
        # Mostrar contenido del dataset
        if dataset_dir.exists():
            print(f"\nContenido del dataset:")
            for item in dataset_dir.rglob("*"):
                if item.is_file():
                    print(f"  📄 {item.relative_to(dataset_dir)}")
                elif item.is_dir() and item != dataset_dir:
                    print(f"  📁 {item.relative_to(dataset_dir)}/")
        
        # Limpiar archivo temporal (opcional)
        try:
            file_path.unlink()
            print(f"\n🗑️ Archivo temporal eliminado: {file_path}")
        except:
            print(f"\n⚠️ No se pudo eliminar el archivo temporal: {file_path}")
        
        return True
    else:
        print(f"\n❌ Error configurando el dataset")
        return False

def main():
    """Función principal para configurar el dataset."""
    
    # URLs de ejemplo - reemplaza con tu URL real
    github_urls = {
        "mabe_dataset": "https://github.com/tu-usuario/tu-repo/releases/download/v1.0/mabe_dataset.rar",
        "exposure_dataset": "https://github.com/tu-usuario/tu-repo/releases/download/v1.0/exposure_dataset.rar"
    }
    
    print("🔧 Configurador de Dataset QESN-MABe")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        # URL proporcionada como argumento
        github_url = sys.argv[1]
        dataset_name = "mabe_dataset"
    else:
        # Seleccionar dataset interactivamente
        print("Selecciona el dataset a descargar:")
        for i, (name, url) in enumerate(github_urls.items(), 1):
            print(f"{i}. {name}")
        
        try:
            choice = int(input("\nIngresa el número (1-2): ")) - 1
            dataset_names = list(github_urls.keys())
            if 0 <= choice < len(dataset_names):
                dataset_name = dataset_names[choice]
                github_url = github_urls[dataset_name]
            else:
                print("Opción inválida")
                return False
        except (ValueError, KeyboardInterrupt):
            print("Operación cancelada")
            return False
    
    return setup_dataset(github_url, dataset_name)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
