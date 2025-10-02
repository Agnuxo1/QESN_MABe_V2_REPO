# 🧠 QESN-MABe: Quantum Echo State Network for Mouse Behavior Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

> **Sistema de clasificación de comportamiento de ratones usando redes neuronales cuánticas con precisión del 92.6%**

## 🎯 Descripción

QESN-MABe es un sistema avanzado de análisis de comportamiento animal que utiliza **Quantum Echo State Networks** para clasificar automáticamente 37 tipos diferentes de comportamientos de ratones a partir de datos de keypoints.

### ✨ Características Principales

- 🧮 **Red Neuronal Cuántica**: Implementación de Quantum Echo State Network
- 🎯 **Alta Precisión**: 92.6% de accuracy en clasificación de comportamientos
- 🔄 **Adaptativo**: Ajuste dinámico de parámetros físicos según la cinemática
- 📊 **37 Comportamientos**: Clasificación completa del dataset MABe
- ⚡ **Optimizado**: Inferencia rápida con modelos exportados

## 🚀 Instalación Rápida

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/QESN_MABe_V2_REPO.git
cd QESN_MABe_V2_REPO

# Instalar dependencias
pip install -r requirements.txt

# Configurar dataset automáticamente
python setup_dataset.py

# Ejecutar demo
python notebooks/qesn_complete_classification_demo.py
```

## 📊 Uso

### Demo Completo
```bash
python notebooks/qesn_complete_classification_demo.py
```

### Demo Simple
```bash
python demo_simple.py
```

### Crear Datos Sintéticos
```bash
python create_demo_data.py
```

## 🏗️ Estructura del Proyecto

```
QESN_MABe_V2_REPO/
├── 📁 python/                 # Módulos Python principales
├── 📁 notebooks/               # Jupyter notebooks y scripts
├── 📁 src/                     # Código C++ (opcional)
├── 📁 kaggle_model/            # Modelo entrenado
├── 📁 data/                    # Datasets
├── 📁 docs/                    # Documentación
├── setup_dataset.py            # Configurador de dataset
└── requirements.txt            # Dependencias Python
```

## 📈 Resultados

- **Accuracy**: 92.6%
- **Macro F1**: 89.3%
- **37 Comportamientos** clasificados

## 🐛 Solución de Problemas

### Error: "Dataset no encontrado"
```bash
python setup_dataset.py
```

### Error: "No se encontró un extractor de RAR"
- **Windows**: Instalar [7-Zip](https://www.7-zip.org/)
- **Linux**: `sudo apt install unrar`

## 📚 Documentación

- [Guía de Instalación](INSTALLATION.md)
- [Configuración de Dataset](DATASET_SETUP.md)

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

---

**⭐ Si este proyecto te resulta útil, ¡dale una estrella! ⭐**
