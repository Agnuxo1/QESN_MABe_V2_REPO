<!-- Author: Francisco Angulo de Lafuente | ORCID: https://orcid.org/0009-0001-1634-7063 | Attribution recorded: 2026-11-06 -->

# 📊 Configuración del Dataset MABe

## Descarga Automática del Dataset

El proyecto incluye un script automático para descargar y configurar el dataset MABe desde GitHub.

### 🚀 Uso Rápido

```bash
# Descargar y configurar el dataset automáticamente
python setup_dataset.py

# O especificar URL directamente
python setup_dataset.py "https://github.com/tu-usuario/tu-repo/releases/download/v1.0/mabe_dataset.rar"
```

### 📋 Requisitos Previos

Para extraer archivos RAR, necesitas instalar uno de estos extractores:

#### Windows
- **7-Zip** (Recomendado): https://www.7-zip.org/
- **WinRAR**: https://www.win-rar.com/

#### Linux
```bash
# Ubuntu/Debian
sudo apt install unrar
# o
sudo apt install p7zip-full

# CentOS/RHEL
sudo yum install unrar
```

#### macOS
```bash
# Con Homebrew
brew install unrar
# o
brew install p7zip
```

### 🔧 Configuración Manual

Si prefieres configurar manualmente:

1. **Descarga el archivo RAR** desde GitHub Releases
2. **Extrae el contenido** en el directorio `data/mabe_dataset/`
3. **Verifica la estructura**:
   ```
   data/
   └── mabe_dataset/
       ├── labels.csv
       └── tracking/
           ├── video_001.parquet
           ├── video_002.parquet
           └── ...
   ```

### 🎯 Estructura del Dataset

El dataset debe contener:

- **`labels.csv`**: Archivo con etiquetas de comportamiento
  - Columnas: `video_id`, `frame`, `behavior`
- **`tracking/`**: Directorio con archivos de keypoints
  - Archivos `.parquet` con columnas: `frame`, `track_id`, `keypoint`, `x`, `y`, `confidence`

### 🔄 Datos Sintéticos (Fallback)

Si no tienes el dataset real, el sistema automáticamente creará datos sintéticos:

```bash
python create_demo_data.py
```

Esto generará datos de demostración para probar el modelo.

### 🐛 Solución de Problemas

#### Error: "No se encontró un extractor de RAR"
- Instala uno de los extractores mencionados arriba
- En Windows, asegúrate de que 7-Zip esté en el PATH

#### Error: "Dataset no encontrado"
- Verifica que el archivo se extrajo correctamente
- Comprueba que `data/mabe_dataset/labels.csv` existe
- Ejecuta `python setup_dataset.py` para descarga automática

#### Error de permisos
- En Linux/Mac: `chmod +x setup_dataset.py`
- En Windows: Ejecuta como administrador si es necesario

### 📝 Variables de Entorno

Puedes personalizar las rutas usando variables de entorno:

```bash
# Windows
set QESN_DATASET_ROOT=data/mi_dataset
set QESN_MODEL_DIR=kaggle_model

# Linux/Mac
export QESN_DATASET_ROOT=data/mi_dataset
export QESN_MODEL_DIR=kaggle_model
```

### 🔗 Enlaces Útiles

- [GitHub Releases](https://github.com/tu-usuario/tu-repo/releases)
- [Documentación MABe](https://mabe-dataset.github.io/)
- [7-Zip Download](https://www.7-zip.org/)
