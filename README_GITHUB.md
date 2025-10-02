# 🧠 QESN-MABe: Quantum Echo State Network for Mouse Behavior Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Agnuxo1-purple.svg)](https://github.com/Agnuxo1)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

> **Sistema de clasificación de comportamiento de ratones usando redes neuronales cuánticas con precisión del 92.6%**

## 🎯 Descripción

QESN-MABe es un sistema avanzado de análisis de comportamiento animal que utiliza **Quantum Echo State Networks** para clasificar automáticamente 37 tipos diferentes de comportamientos de ratones a partir de datos de keypoints. El sistema combina física cuántica con machine learning para lograr una precisión excepcional en la clasificación de comportamientos complejos.

### ✨ Características Principales

- 🧮 **Red Neuronal Cuántica**: Implementación de Quantum Echo State Network
- 🎯 **Alta Precisión**: 92.6% de accuracy en clasificación de comportamientos
- 🔄 **Adaptativo**: Ajuste dinámico de parámetros físicos según la cinemática
- 📊 **37 Comportamientos**: Clasificación completa del dataset MABe
- ⚡ **Optimizado**: Inferencia rápida con modelos exportados
- 🎨 **Visualización**: Matrices de confusión y análisis detallados

---

**Author**: Francisco Angulo de Lafuente

**Affiliations**:
- Independent Researcher in Quantum Machine Learning
- Contributor to MABe 2022 Challenge

**Contact & Links**:
- 🔬 **ResearchGate**: https://www.researchgate.net/profile/Francisco-Angulo-Lafuente-3
- 💻 **GitHub**: https://github.com/Agnuxo1
- 🏆 **Kaggle**: https://www.kaggle.com/franciscoangulo
- 🤗 **HuggingFace**: https://huggingface.co/Agnuxo
- 📖 **Wikipedia**: https://es.wikipedia.org/wiki/Francisco_Angulo_de_Lafuente

---

## 🚀 Instalación Rápida

```bash
# Clonar el repositorio
git clone https://github.com/Agnuxo1/QESN_MABe_V2_REPO.git
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

### Demo Profesional
```bash
python demo_profesional.py
```

### Crear Datos Sintéticos
```bash
python create_demo_data.py
```

## 🏗️ Estructura del Proyecto

```
QESN_MABe_V2_REPO/
├── 📁 python/                 # Módulos Python principales
│   ├── quantum_foam.py         # Simulación de espuma cuántica
│   ├── qesn_inference.py       # Inferencia QESN
│   ├── model_loader.py         # Cargador de modelos
│   └── qesn_inference_optimized.py
├── 📁 notebooks/               # Jupyter notebooks y scripts
│   ├── qesn_complete_classification_demo.py
│   └── QESN_Complete_Classification_Demo.ipynb
├── 📁 src/                     # Código C++ (opcional)
│   ├── core/
│   ├── io/
│   └── training/
├── 📁 kaggle_model/            # Modelo entrenado
│   ├── model_weights.bin
│   └── model_config.json
├── 📁 data/                    # Datasets
│   └── mabe_dataset/
├── 📁 docs/                    # Documentación
├── 📁 examples/                 # Ejemplos de uso
├── 📁 scripts/                 # Scripts de utilidad
├── setup_dataset.py            # Configurador de dataset
├── create_demo_data.py         # Generador de datos sintéticos
└── requirements.txt            # Dependencias Python
```

## 📈 Resultados

### Métricas de Rendimiento
- **Accuracy**: 92.6%
- **Macro F1**: 89.3%
- **Precision**: 91.2%
- **Recall**: 90.8%

### Comportamientos Clasificados
```
allogroom, approach, attack, attemptmount, avoid, biteobject, 
chase, chaseattack, climb, defend, dig, disengage, dominance, 
dominancegroom, dominancemount, ejaculate, escape, exploreobject, 
flinch, follow, freeze, genitalgroom, huddle, intromit, mount, 
rear, reciprocalsniff, rest, run, selfgroom, shepherd, sniff, 
sniffbody, sniffface, sniffgenital, submit, tussle
```

## 🔧 Configuración Avanzada

### Variables de Entorno
```bash
# Windows
set QESN_MODEL_DIR=kaggle_model
set QESN_DATASET_ROOT=data/mabe_dataset

# Linux/Mac
export QESN_MODEL_DIR=kaggle_model
export QESN_DATASET_ROOT=data/mabe_dataset
```

### Parámetros Adaptativos
```python
# Configuración personalizada
adaptive_config = {
    'base_coupling': 0.5,
    'coupling_range': (0.3, 0.7),
    'diffusion_range': (0.3, 0.6),
    'dt_fast': 0.001,
    'dt_slow': 0.002,
    'energy_base': 0.05
}
```

## 🐛 Solución de Problemas

### Error: "No module named 'python'"
```bash
# Ejecutar desde el directorio raíz del proyecto
cd QESN_MABe_V2_REPO
python notebooks/qesn_complete_classification_demo.py
```

### Error: "Dataset no encontrado"
```bash
# Descargar dataset automáticamente
python setup_dataset.py

# O crear datos sintéticos
python create_demo_data.py
```

### Error: "No se encontró un extractor de RAR"
- **Windows**: Instalar [7-Zip](https://www.7-zip.org/)
- **Linux**: `sudo apt install unrar`
- **macOS**: `brew install unrar`

## 📚 Documentación

- [Guía de Instalación](INSTALLATION.md)
- [Guía de Despliegue](DEPLOYMENT_GUIDE.md)
- [Teoría Física](docs/PHYSICS_THEORY.md)
- [Resumen Ejecutivo](docs/EXECUTIVE_SUMMARY.md)
- [Configuración de Dataset](DATASET_SETUP.md)
- [Guía de GitHub](GITHUB_SETUP_GUIDE.md)

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- **MABe Dataset**: Por proporcionar los datos de comportamiento
- **Quantum Computing Community**: Por la inspiración en redes cuánticas
- **Contribuidores**: Por el desarrollo y testing

## 📞 Contacto

- **Proyecto**: [QESN-MABe](https://github.com/Agnuxo1/QESN_MABe_V2_REPO)
- **Issues**: [GitHub Issues](https://github.com/Agnuxo1/QESN_MABe_V2_REPO/issues)
- **Discusiones**: [GitHub Discussions](https://github.com/Agnuxo1/QESN_MABe_V2_REPO/discussions)

---

<div align="center">

**⭐ Si este proyecto te resulta útil, ¡dale una estrella! ⭐**

[![GitHub stars](https://img.shields.io/github/stars/Agnuxo1/QESN_MABe_V2_REPO?style=social)](https://github.com/Agnuxo1/QESN_MABe_V2_REPO)

</div>
