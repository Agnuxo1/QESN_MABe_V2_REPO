# Author: Francisco Angulo de Lafuente
# ORCID: https://orcid.org/0009-0001-1634-7063
# Attribution recorded: 2026-11-06
# Dataset Configuration
# Configuración para descargar y configurar el dataset MABe

# URLs de descarga (reemplaza con tus URLs reales)
DATASET_URLS = {
    "mabe_dataset": "https://github.com/tu-usuario/tu-repo/releases/download/v1.0/mabe_dataset.rar",
    "exposure_dataset": "https://github.com/tu-usuario/tu-repo/releases/download/v1.0/exposure_dataset.rar"
}

# Configuración del dataset
DATASET_CONFIG = {
    "data_dir": "data",
    "downloads_dir": "data/downloads",
    "extract_to": "data/mabe_dataset",
    "cleanup_after_extract": True
}

# Estructura esperada del dataset
EXPECTED_STRUCTURE = {
    "labels.csv": "Archivo con etiquetas de comportamiento",
    "tracking/": "Directorio con archivos .parquet de keypoints",
    "results/": "Directorio para resultados (se crea automáticamente)"
}

# Instrucciones de instalación de extractores RAR
RAR_EXTRACTORS = {
    "windows": [
        "7-Zip: https://www.7-zip.org/",
        "WinRAR: https://www.win-rar.com/"
    ],
    "linux": [
        "unrar: sudo apt install unrar",
        "7zip: sudo apt install p7zip-full"
    ],
    "macos": [
        "unrar: brew install unrar",
        "7zip: brew install p7zip"
    ]
}
