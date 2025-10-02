# 🚀 QESN-MABe V2 - Installation & Setup Guide

**Author:** Francisco Angulo de Lafuente
**License:** MIT
**GitHub:** https://github.com/Agnuxo1

---

## ⚠️ IMPORTANT: Installation takes 30-60 minutes

The vcpkg package manager will download and compile Apache Arrow, Parquet, and dependencies.
This is normal and only needs to be done once.

---

## 📋 Prerequisites

### Required Software
- **Windows 10/11** with Visual Studio 2022
- **Git** for Windows
- **CMake** 3.20 or higher
- **Visual Studio 2022** with C++ development tools

### System Requirements
- **RAM:** 8 GB minimum, 16 GB recommended
- **Disk Space:** 10 GB free (for vcpkg compilation)
- **CPU:** Multi-core recommended (compilation is parallelized)

---

## 🔧 Step-by-Step Installation

### Step 1: Install vcpkg (Package Manager)

Open PowerShell or Command Prompt as Administrator:

```bash
# Clone vcpkg
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg

# Bootstrap vcpkg
.\bootstrap-vcpkg.bat
```

**Expected output:**
```
vcpkg package management program version 2025-XX-XX
Telemetry: Data collected anonymously (opt-out with --disable-metrics)
```

---

### Step 2: Install Dependencies (30-60 minutes)

This is the LONGEST step. The libraries will be compiled from source.

```bash
cd C:\vcpkg

# Install all dependencies at once
.\vcpkg install arrow:x64-windows parquet:x64-windows eigen3:x64-windows
```

**What gets installed:**
- **Apache Arrow** (21.0.0) - 15-20 minutes
- **Parquet** - 5 minutes
- **Eigen3** - 2 minutes
- **Dependencies:** OpenSSL, Boost, Thrift, etc.

**Progress indicators:**
```
Installing 1/82 vcpkg-cmake...
Installing 2/82 zlib...
Installing 3/82 bzip2...
...
Installing 82/82 arrow...
```

**If compilation fails:**

Common issue: Out of memory
```bash
# Solution: Reduce parallel jobs
.\vcpkg install arrow:x64-windows --x-buildtrees-root=C:\temp\vcpkg-build
```

---

### Step 3: Verify Installation

```bash
# Check installed packages
C:\vcpkg\vcpkg list

# Expected output:
# arrow:x64-windows                     21.0.0
# parquet:x64-windows                   0
# eigen3:x64-windows                    3.4.0
```

---

### Step 4: Set Environment Variable

Add vcpkg to your system PATH (CRITICAL for CMake):

**Option A: Temporary (for current session)**
```bash
set CMAKE_PREFIX_PATH=C:\vcpkg\installed\x64-windows
```

**Option B: Permanent (recommended)**
1. Open System Properties → Environment Variables
2. Add new **System Variable**:
   - Name: `CMAKE_PREFIX_PATH`
   - Value: `C:\vcpkg\installed\x64-windows`

---

## 🏗️ Building QESN-MABe V2

### Step 5: Build the Project

```bash
cd E:\QESN_MABe_V2
scripts\build.bat
```

**Build process:**
1. Creates `build/` directory
2. Runs CMake configuration
3. Compiles C++ sources (5-10 minutes)
4. Generates `build\Release\qesn_train.exe`

**Expected output:**
```
========================================
QESN-MABe V2 Build System
========================================

Creating build directory...
Configuring CMake...
-- The CXX compiler identification is MSVC 19.44
-- Detecting CXX compiler ABI info - done
-- Found Arrow: C:/vcpkg/installed/x64-windows
-- Found Parquet: C:/vcpkg/installed/x64-windows
-- Found Eigen3: C:/vcpkg/installed/x64-windows

Building project (Release)...
[100%] Built target qesn_train

========================================
Build successful!
========================================

Executable: build\Release\qesn_train.exe
```

---

## 🧪 Testing the Installation

### Quick Test (no data required)

```bash
build\Release\qesn_train.exe --help
```

**Expected output:**
```
QESN-MABe V2: Quantum Energy State Network for Mouse Behavior Classification
Author: Francisco Angulo de Lafuente

Usage: qesn_train.exe [options]

Required options:
  --metadata <path>      Path to metadata CSV file
  --tracking <path>      Path to tracking directory (parquet files)
  --annotation <path>    Path to annotation directory (CSV files)
...
```

---

## 📊 Training Setup

### Step 6: Prepare MABe Dataset

Download the MABe 2022 dataset from Kaggle:
```
E:\QESN-MABe\
├── train.csv                    # Metadata
├── train_tracking\              # Parquet files
│   ├── video001.parquet
│   ├── video002.parquet
│   └── ...
└── train_annotation\            # Annotation CSVs
    ├── video001.csv
    ├── video002.csv
    └── ...
```

### Step 7: Update Paths in train.bat

Edit `scripts\train.bat` and update these lines:

```batch
set "METADATA=E:\QESN-MABe\train.csv"
set "TRACKING=E:\QESN-MABe\train_tracking"
set "ANNOTATION=E:\QESN-MABe\train_annotation"
```

### Step 8: Start Training

```bash
cd E:\QESN_MABe_V2
scripts\train.bat
```

**Training duration:** 12-15 hours for 100 videos, 30 epochs

**Expected progress:**
```
Epoch 1/30
  Processed 100/5420 windows
  Training   - Loss: 3.2145, Acc: 12.50%, F1: 0.0823
  Validation - Loss: 3.1872, Acc: 14.32%, F1: 0.0912

Epoch 10/30
  Training   - Loss: 2.1023, Acc: 41.23%, F1: 0.3245
  Validation - Loss: 2.2341, Acc: 38.76%, F1: 0.3012

Epoch 30/30
  Training   - Loss: 1.4382, Acc: 58.71%, F1: 0.4823
  Validation - Loss: 1.6234, Acc: 54.32%, F1: 0.4512
  *** New best model saved! ***
```

---

## 🐛 Troubleshooting

### Problem 1: CMake can't find Arrow/Parquet

**Error:**
```
CMake Error: Could not find a package configuration file provided by "Arrow"
```

**Solution:**
```bash
# Verify vcpkg installation
C:\vcpkg\vcpkg list | findstr arrow

# Reinstall if missing
C:\vcpkg\vcpkg remove arrow:x64-windows
C:\vcpkg\vcpkg install arrow:x64-windows --recurse

# Set CMAKE_PREFIX_PATH
set CMAKE_PREFIX_PATH=C:\vcpkg\installed\x64-windows
```

---

### Problem 2: Out of Memory during Training

**Error:**
```
std::bad_alloc
```

**Solution:**
```bash
# Reduce batch size in train.bat
--batch 16    # Instead of 32
```

---

### Problem 3: Accuracy Stuck at ~2.7%

**This indicates the model is predicting randomly (1/37 = 2.7%)**

**Diagnosis:**
1. Check data loading:
   ```cpp
   // In dataset_loader.cpp, add debug output
   std::cout << "Frame 0, Mouse 0, Keypoint 0: ("
             << frame.mice[0][0].x << ", "
             << frame.mice[0][0].y << ")" << std::endl;
   ```

2. Verify keypoints are NOT circular (not synthetic data)

3. Check normalization:
   ```cpp
   // Must divide by actual video dimensions
   x /= sequence.width;   // NOT hardcoded 1024
   y /= sequence.height;  // NOT hardcoded 570
   ```

---

### Problem 4: vcpkg Installation Fails

**Error:**
```
error: building arrow:x64-windows failed with: BUILD_FAILED
```

**Solutions:**

**A) Disk space:**
```bash
# Check free space (need 10+ GB)
dir C:\

# Clean vcpkg cache
C:\vcpkg\vcpkg remove --outdated
```

**B) Network issues:**
```bash
# Retry installation
C:\vcpkg\vcpkg install arrow:x64-windows --clean-after-build
```

**C) Use precompiled binaries (if available):**
```bash
# Enable binary caching
set VCPKG_BINARY_SOURCES=clear;nuget,https://vcpkg.github.io/vcpkg-ce-catalog/,read
C:\vcpkg\vcpkg install arrow:x64-windows
```

---

## 📈 Performance Expectations

### Training Metrics (37 Classes)

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| **Training Accuracy** | 55-65% | Much better than random (2.7%) |
| **Validation Accuracy** | 50-60% | Some overfitting expected |
| **F1-Score (Macro)** | 0.40-0.50 | Impacted by rare classes |
| **F1-Score (sniff)** | 0.65-0.75 | Most frequent class |
| **F1-Score (ejaculate)** | 0.01-0.10 | Only 3 training samples! |

### Prediction Distribution

**Healthy model:**
- Top class: ~35% (sniff)
- Top 5 classes: ~70%
- At least 10 classes predicted: ✓

**Broken model (V1):**
- One class: 99% ❌
- Circular synthetic data ❌

---

## 🎯 Validation Checklist

Before submitting to Kaggle:

- [ ] Training completes 30 epochs without crashes
- [ ] Validation accuracy > 50%
- [ ] Checkpoint file size ~1.2 MB (151,552 parameters)
- [ ] No NaN/Inf in `training_history.csv`
- [ ] Predictions span >10 classes (check with debug output)
- [ ] Confidence distribution: 0.20-0.80 (not all 0.99)
- [ ] Local inference with `qesn_inference.py` works

---

## 📞 Support

**Issues:** https://github.com/Agnuxo1/QESN-MABe-V2/issues
**Documentation:** See `README.md` and `docs/MASTER_PLAN.md`
**MABe Competition:** https://www.kaggle.com/competitions/MABe-mouse-behavior-detection

---

## 🏆 Next Steps

1. ✅ Installation complete
2. ✅ Build successful
3. → **Train the model** (`scripts\train.bat`)
4. → **Validate results** (accuracy > 50%)
5. → **Test inference** (`python/qesn_inference.py`)
6. → **Submit to Kaggle**

**Good luck!** 🚀

---

**Created by:** Francisco Angulo de Lafuente
**ResearchGate:** https://www.researchgate.net/profile/Francisco-Angulo-Lafuente-3
**Kaggle:** https://www.kaggle.com/franciscoangulo
**HuggingFace:** https://huggingface.co/Agnuxo
**Wikipedia:** https://es.wikipedia.org/wiki/Francisco_Angulo_de_Lafuente
