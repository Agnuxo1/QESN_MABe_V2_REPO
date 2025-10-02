# 🚀 QUICK START GUIDE - Instalaciones Existentes

## ✅ Tu Entorno (PERFECTO)

```
✅ Visual Studio 2022:  E:\VS2022
✅ OpenCV:              E:\OpenCV
✅ Eigen 3.4.0:         E:\eigen-3.4.0
✅ CUDA 12:             Instalado
✅ Python + Anaconda:   Instalado
```

**¡No necesitas vcpkg!** Todo está adaptado para tus instalaciones.

---

## 🎯 PASOS EXACTOS (30 minutos total)

### **Paso 1: Instalar PyArrow (2 minutos)**

```bash
# Abre Anaconda Prompt o CMD
pip install pyarrow pandas tqdm
```

**Por qué**: PyArrow en Python es MUY fácil de instalar vs C++ Arrow.

---

### **Paso 2: Preprocesar Datos Parquet (10-15 minutos)**

```bash
cd E:\QESN_MABe_V2
python scripts\preprocess_parquet.py
```

**Qué hace**: Convierte archivos `.parquet` a binarios `.bin` que C++ puede leer fácilmente.

**Entrada esperada**:
```
How many sequences to process? (Enter for all, or number):
```

**Recomendación**: Escribe `10` para empezar (testing rápido).

**Output esperado**:
```
Processing 10 sequences...

[1/10] 44566106 (1024x570)
  Frames: 18456, Mice: 2, Keypoints: 18
  Saved: E:\QESN_MABe_V2\data\preprocessed\AdaptableSnail\44566106.bin (23.45 MB)
...
Success: 10
Output: E:\QESN_MABe_V2\data\preprocessed
```

---

### **Paso 3: Compilar C++ (5-10 minutos)**

```bash
cd E:\QESN_MABe_V2
scripts\build.bat
```

**Output esperado**:
```
========================================
QESN-MABe V2 Build System
========================================

Using existing installations:
  - Visual Studio 2022: E:\VS2022
  - Eigen 3.4.0: E:\eigen-3.4.0
  - CUDA 12: Detected automatically

Configuring CMake...
----------------------------------------
-- The CXX compiler identification is MSVC 19.XX
-- Using local Eigen3 installation at E:/eigen-3.4.0
-- CUDA found: 12.0
-- OpenMP found

Building project (Release)...
----------------------------------------
[1/5] Building CXX object CMakeFiles/qesn_train.dir/src/core/quantum_neuron.cpp.obj
[2/5] Building CXX object CMakeFiles/qesn_train.dir/src/core/quantum_foam.cpp.obj
[3/5] Building CXX object CMakeFiles/qesn_train.dir/src/io/dataset_loader.cpp.obj
[4/5] Building CXX object CMakeFiles/qesn_train.dir/src/training/trainer.cpp.obj
[5/5] Building CXX object CMakeFiles/qesn_train.dir/src/main.cpp.obj
[6/6] Linking CXX executable Release\qesn_train.exe

========================================
Build successful!
========================================

Executable: build\Release\qesn_train.exe
```

---

### **Paso 4: Test Rápido (5 minutos)**

```bash
build\Release\qesn_train.exe ^
    --data E:\QESN_MABe_V2\data\preprocessed ^
    --max-sequences 5 ^
    --epochs 1 ^
    --batch 4
```

**Output esperado**:
```
QESN-MABe V2 Training
========================================
Loading dataset from: E:\QESN_MABe_V2\data\preprocessed
Loaded 5 sequences (123,456 frames total)

Initializing model...
Grid: 64x64, Classes: 37, Parameters: 151,552

Epoch 1/1:
  Batch 1/25: Loss 3.215, Acc 0.087
  Batch 2/25: Loss 3.102, Acc 0.112
  ...
  Epoch Loss: 3.045, Acc: 0.134

Checkpoint saved: checkpoints/epoch_1.bin
```

---

### **Paso 5: Entrenamiento Completo (12-15 horas)**

Después del test, entrena con todos los datos:

```bash
scripts\train.bat
```

**Monitorea que**:
- Loss baje de ~3.2 a <1.5
- Accuracy suba a 55-65%
- No haya NaN/Inf

---

## 🔧 MODIFICACIONES REALIZADAS

### 1. CMakeLists.txt
✅ Usa Eigen de `E:\eigen-3.4.0`
✅ Detecta CUDA 12 automáticamente
✅ NO requiere vcpkg

### 2. dataset_loader.cpp
✅ Lee archivos `.bin` (generados por Python)
✅ NO requiere Apache Arrow C++

### 3. build.bat
✅ Busca VS2022 en `E:\VS2022`
✅ Configuración automática

### 4. preprocess_parquet.py
✅ Convierte parquet → binarios
✅ Normaliza por dimensiones del video
✅ Mapea 37 acciones

---

## ⚠️ TROUBLESHOOTING

### Error: "CMake not found"
**Solución**:
```bash
# Agrega CMake de VS2022 al PATH
set PATH=%PATH%;E:\VS2022\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin
```

### Error: "Eigen3 not found"
**Solución**: Verifica que `E:\eigen-3.4.0` existe y contiene `Eigen/Core`

### Error: "Python pyarrow not found"
**Solución**:
```bash
pip install pyarrow pandas tqdm
```

### Error: Dataset paths incorrect
**Solución**: Edita `scripts\preprocess_parquet.py` líneas 15-17:
```python
TRAIN_CSV = r"E:\QESN-MABe\train.csv"
TRACKING_ROOT = r"E:\QESN-MABe\train_tracking"
ANNOTATION_ROOT = r"E:\QESN-MABe\train_annotation"
```

---

## 📊 MÉTRICAS ESPERADAS

### Test Rápido (5 videos, 1 época):
```
Time: ~5 minutos
Loss: ~3.0
Accuracy: ~12-15%
```

### Entrenamiento Completo (100 videos, 30 épocas):
```
Time: 12-15 horas
Final Loss: ~1.4-1.6
Final Accuracy: 55-65%
F1-Score: 0.40-0.50
```

---

## ✅ CHECKLIST RÁPIDO

- [ ] PyArrow instalado (`pip install pyarrow pandas tqdm`)
- [ ] Datos preprocesados (`python scripts\preprocess_parquet.py`)
- [ ] Proyecto compilado (`scripts\build.bat`)
- [ ] Test ejecutado (5 videos, 1 época)
- [ ] Entrenamiento completo lanzado

---

## 🎯 RESUMEN DE COMANDOS

```bash
# 1. Instalar PyArrow
pip install pyarrow pandas tqdm

# 2. Preprocesar (escribe 10 cuando pregunte)
cd E:\QESN_MABe_V2
python scripts\preprocess_parquet.py

# 3. Compilar
scripts\build.bat

# 4. Test rápido
build\Release\qesn_train.exe --data data\preprocessed --max-sequences 5 --epochs 1 --batch 4

# 5. Entrenamiento completo
scripts\train.bat
```

---

## 🚀 SIGUIENTE ACCIÓN

**AHORA MISMO**:
```bash
pip install pyarrow pandas tqdm
```

**Luego (2 minutos)**:
```bash
cd E:\QESN_MABe_V2
python scripts\preprocess_parquet.py
```

Escribe `10` cuando pregunte cuántos videos procesar.

---

**Tiempo total**: 30 minutos hasta primer entrenamiento
**Ventaja**: No necesitas vcpkg (ahorra 1-2 horas)
**Estado**: ✅ Listo para ejecutar

---

**Generado**: 2025-10-01
**Para**: Francisco Angulo de Lafuente
**Entorno**: VS2022 + Eigen + CUDA 12
