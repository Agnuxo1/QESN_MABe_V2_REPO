# 🚀 QESN-MABe V2: Instalación Rápida

**Autor**: Francisco Angulo de Lafuente  
**Fecha**: Octubre 2025

---

## 🎯 **Instalación Automática (Recomendada)**

### **Opción 1: Script Automático (Windows)**
```bash
# Ejecutar el instalador automático
install_dependencies.bat
```

### **Opción 2: Script Python (Multiplataforma)**
```bash
# Ejecutar el instalador Python
python install_dependencies.py
```

### **Opción 3: Instalación Manual**
```bash
# Instalar dependencias principales
pip install numpy pandas matplotlib seaborn plotly ipywidgets scipy tqdm

# Instalar dependencias opcionales
pip install jupyter notebook jupyterlab pyarrow h5py
```

---

## 🧪 **Verificar Instalación**

### **Test Rápido**
```python
# Ejecutar este código para verificar
python -c "import numpy, pandas, matplotlib, seaborn, plotly; print('✅ Todas las librerías instaladas correctamente!')"
```

### **Test Completo**
```python
# Ejecutar el demo simple
python demo_simple_no_emoji.py
```

---

## 📓 **Ejecutar Demostraciones**

### **1. Demo Simplificado (Sin Dependencias Externas)**
```bash
python demo_simple_no_emoji.py
```
- ✅ Funciona solo con librerías estándar de Python
- ✅ No requiere instalaciones adicionales
- ✅ Visualización ASCII de la red cuántica

### **2. Demo Completo (Con Visualizaciones)**
```bash
python examples/quick_demo.py
```
- ✅ Requiere: numpy, pandas, matplotlib, seaborn, plotly
- ✅ Visualizaciones interactivas
- ✅ Gráficos profesionales

### **3. Notebook Interactivo**
```bash
# Abrir Jupyter Notebook
jupyter notebook notebooks/QESN_Demo_Interactive.ipynb

# O usar Jupyter Lab
jupyter lab notebooks/QESN_Demo_Interactive.ipynb
```
- ✅ Instalación automática de dependencias
- ✅ Visualizaciones interactivas con Plotly
- ✅ Widgets interactivos
- ✅ Demostración completa

---

## 🔧 **Solución de Problemas**

### **Error: ModuleNotFoundError**
```bash
# Si falta alguna librería específica
pip install [nombre_libreria]

# Ejemplo para plotly
pip install plotly
```

### **Error: Permission Denied**
```bash
# Usar --user para instalación local
pip install --user numpy pandas matplotlib seaborn plotly

# O usar conda si está disponible
conda install numpy pandas matplotlib seaborn plotly
```

### **Error: Unicode/Emoji**
- Usar `demo_simple_no_emoji.py` en lugar de versiones con emojis
- Configurar terminal para UTF-8

### **Error: Jupyter No Encontrado**
```bash
# Instalar Jupyter
pip install jupyter notebook

# O usar Google Colab
# Subir el notebook a: https://colab.research.google.com/
```

---

## 📊 **Requisitos del Sistema**

### **Mínimos**
- ✅ Python 3.8+
- ✅ 2 GB RAM
- ✅ 1 GB espacio en disco

### **Recomendados**
- ✅ Python 3.9+
- ✅ 4 GB RAM
- ✅ 2 GB espacio en disco
- ✅ Navegador web moderno (para Jupyter)

### **Opcionales**
- ✅ Anaconda/Miniconda
- ✅ Visual Studio Code con extensión Python
- ✅ Git (para clonar repositorio)

---

## 🎯 **Ejecución Paso a Paso**

### **Paso 1: Verificar Python**
```bash
python --version
# Debe mostrar Python 3.8 o superior
```

### **Paso 2: Instalar Dependencias**
```bash
# Opción más fácil
install_dependencies.bat

# O manualmente
pip install numpy pandas matplotlib seaborn plotly ipywidgets
```

### **Paso 3: Ejecutar Demo**
```bash
# Demo simple (sin dependencias externas)
python demo_simple_no_emoji.py

# Demo completo (con visualizaciones)
python examples/quick_demo.py
```

### **Paso 4: Abrir Notebook**
```bash
# Iniciar Jupyter
jupyter notebook

# Navegar a: notebooks/QESN_Demo_Interactive.ipynb
# Ejecutar todas las celdas
```

---

## 🚀 **Demostraciones Disponibles**

### **1. Simulación Cuántica Interactiva**
- Visualización de red cuántica 64×64
- Evolución temporal de energía
- Parámetros ajustables

### **2. Análisis de Comportamiento**
- 37 clases de comportamiento MABe 2022
- Distribución de frecuencias
- Análisis estadístico

### **3. Clasificación en Tiempo Real**
- Predicción de comportamientos
- Visualización de confianza
- Top 10 predicciones

### **4. Análisis de Rendimiento**
- Comparación con métodos clásicos
- Métricas detalladas
- Gráficos de eficiencia

---

## 📚 **Recursos Adicionales**

### **Documentación**
- **README.md**: Documentación principal
- **docs/PHYSICS_THEORY.md**: Teoría cuántica
- **CONTRIBUTING.md**: Guía de contribución

### **Ejemplos**
- **examples/quick_demo.py**: Demo completo
- **examples/kaggle_submission.py**: Script para Kaggle
- **notebooks/QESN_Demo_Interactive.ipynb**: Notebook interactivo

### **Scripts**
- **install_dependencies.py**: Instalador automático
- **install_dependencies.bat**: Instalador Windows
- **scripts/deploy_kaggle.py**: Deployment Kaggle
- **scripts/deploy_huggingface.py**: Deployment HuggingFace

---

## 🆘 **Soporte**

### **Problemas Comunes**
1. **Python no encontrado**: Instalar desde python.org
2. **Pip no funciona**: Usar `python -m pip` en lugar de `pip`
3. **Permisos denegados**: Usar `--user` flag
4. **Emojis no se muestran**: Usar versión sin emojis

### **Contacto**
- **GitHub Issues**: https://github.com/Agnuxo1/QESN-MABe-V2/issues
- **Email**: Contactar via GitHub
- **Documentación**: Ver docs/ directory

---

## 🎉 **¡Listo para Empezar!**

Una vez completada la instalación, puedes:

1. **Explorar la física cuántica** en acción
2. **Ver clasificación de comportamientos** en tiempo real
3. **Comparar con métodos clásicos**
4. **Experimentar con parámetros**
5. **Preparar para deployment** en plataformas

---

**¡Disfruta explorando QESN!** 🚀🧬✨

---

**Autor**: Francisco Angulo de Lafuente  
**GitHub**: https://github.com/Agnuxo1  
**Licencia**: MIT
