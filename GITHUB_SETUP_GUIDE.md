# 🚀 Guía Completa para Subir el Proyecto a GitHub

## 📋 Pasos para Subir Todo el Proyecto

### Opción 1: Automático (Recomendado)

```bash
# Ejecutar configuración automática
python setup_github.py

# O en Windows
setup_github.bat
```

### Opción 2: Manual

#### 1. Crear repositorio en GitHub
1. Ve a https://github.com/new
2. Nombre: `QESN_MABe_V2_REPO` (o el que prefieras)
3. Descripción: "Quantum Echo State Network for Mouse Behavior Analysis"
4. **NO** marques "Initialize with README"
5. Haz clic en "Create repository"

#### 2. Configurar Git local
```bash
# Agregar repositorio remoto (reemplaza USERNAME con tu usuario)
git remote add origin https://github.com/USERNAME/QESN_MABe_V2_REPO.git

# Verificar configuración
git remote -v
```

#### 3. Subir el proyecto
```bash
# Subir todo el proyecto
git push -u origin master
```

## 📁 Lo que se subirá

✅ **Archivos incluidos:**
- Todo el código Python (`python/`, `notebooks/`, `scripts/`)
- Documentación completa (`docs/`, `README.md`, etc.)
- Modelos entrenados (`kaggle_model/`)
- Scripts de configuración (`setup_dataset.py`, etc.)
- Datos sintéticos de ejemplo (`data/exposure_dataset/`)
- Configuración de proyecto (`.gitignore`, `requirements.txt`)

❌ **Archivos excluidos** (por `.gitignore`):
- Archivos temporales (`__pycache__/`, `*.pyc`)
- Datos grandes (`*.parquet`, `*.bin` - opcional)
- Archivos del sistema (`.DS_Store`, `Thumbs.db`)
- Logs y archivos temporales

## 🔧 Configuración del Dataset

### Después de subir el proyecto:

1. **Comprimir tu dataset real:**
   ```bash
   # Crear archivo RAR con tu dataset
   # Subirlo a GitHub Releases
   ```

2. **Actualizar URL en setup_dataset.py:**
   ```python
   DATASET_URLS = {
       "mabe_dataset": "https://github.com/USERNAME/REPO/releases/download/v1.0/mabe_dataset.rar"
   }
   ```

3. **Crear Release en GitHub:**
   - Ve a tu repositorio > Releases > Create a new release
   - Tag: `v1.0`
   - Título: "Dataset MABe v1.0"
   - Sube tu archivo `.rar` como asset

## 🎯 Verificación

Después de subir, verifica que:

1. **Repositorio visible:** https://github.com/USERNAME/QESN_MABe_V2_REPO
2. **README se muestra correctamente**
3. **Estructura de carpetas es correcta**
4. **Scripts funcionan:** `python setup_dataset.py`

## 🐛 Solución de Problemas

### Error: "Repository not found"
- Verifica que el repositorio existe en GitHub
- Verifica que tienes permisos de escritura
- Usa HTTPS en lugar de SSH si hay problemas

### Error: "Authentication failed"
```bash
# Configurar credenciales
git config --global user.name "Tu Nombre"
git config --global user.email "tu@email.com"

# O usar GitHub CLI
gh auth login
```

### Error: "Large files detected"
```bash
# Si tienes archivos grandes, usa Git LFS
git lfs install
git lfs track "*.bin"
git lfs track "*.parquet"
git add .gitattributes
```

## 📚 Próximos Pasos

1. **Documentación:**
   - Actualizar README.md con tu información
   - Agregar badges de estado
   - Crear CONTRIBUTING.md

2. **CI/CD:**
   - Configurar GitHub Actions
   - Tests automáticos
   - Despliegue automático

3. **Distribución:**
   - PyPI package
   - Docker container
   - HuggingFace Spaces

## 🔗 Enlaces Útiles

- [GitHub Docs](https://docs.github.com/)
- [Git Tutorial](https://git-scm.com/docs/gittutorial)
- [GitHub CLI](https://cli.github.com/)
- [Git LFS](https://git-lfs.github.com/)

---

**¡Tu proyecto QESN-MABe estará disponible públicamente en GitHub!** 🎉
