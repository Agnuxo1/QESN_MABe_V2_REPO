<!-- Author: Francisco Angulo de Lafuente | ORCID: https://orcid.org/0009-0001-1634-7063 | Attribution recorded: 2026-11-06 -->

# 🎯 QESN-MABe V2: Master Plan

## 📋 PROJECT OVERVIEW

**Objective**: Create clean, production-ready Quantum Energy State Network for MABe 2022 Challenge
**Version**: 2.0 (Complete Rewrite with Real Data)
**Target Accuracy**: 0.55-0.65 F1-Score on 37 behavioral classes

---

## 🔬 CORE PHILOSOPHY

**NO COMPROMISES ON QUANTUM PHYSICS**
- 100% real quantum simulation preserved
- No CNN, no Transformers, no traditional ML
- Pure physics-based computation using energy diffusion
- Future-ready for actual quantum hardware

---

## ⚠️ CRITICAL FIXES FROM V1

### 1. **DATA SOURCE** ✅ MANDATORY FIX
- ❌ **V1**: Synthetic circular trajectories, cycling labels
- ✅ **V2**: Real parquet files from MABe dataset
- **Impact**: Model never saw real mouse behavior in V1
- **Implementation**: New `ParquetDataLoader` using Apache Arrow C++

### 2. **COORDINATE NORMALIZATION** ✅ MANDATORY FIX
- ❌ **V1**: Hardcoded 1024×570 in Python
- ✅ **V2**: Dynamic normalization using actual video dimensions
- **Impact**: Grid mapping was wrong for different resolutions
- **Implementation**: Pass `video_width`, `video_height` from metadata

### 3. **WINDOW SIZE ALIGNMENT** ✅ MANDATORY FIX
- ❌ **V1**: Train=60 frames, Inference=30 frames
- ✅ **V2**: Both=30 frames
- **Impact**: Temporal context mismatch
- **Implementation**: Set `WINDOW_SIZE=30` everywhere

### 4. **ENERGY INJECTION CONSISTENCY** ✅ MANDATORY FIX
- ❌ **V1**: C++ variable, Python fixed
- ✅ **V2**: Both use fixed 0.05 energy per keypoint
- **Impact**: Activation magnitude discrepancy
- **Implementation**: `energy = 0.05` in trainer.cpp

---

## 📂 PROJECT STRUCTURE

```
E:\QESN_MABe_V2\
├── include/
│   ├── core/
│   │   ├── quantum_neuron.h     [PHYSICS] ← Copy from V1, NO changes
│   │   └── quantum_foam.h       [PHYSICS] ← Copy from V1, NO changes
│   ├── io/
│   │   ├── dataset_loader.h     [NEW] Real parquet loader
│   │   └── data_pipeline.h      [COPY] Batch pipeline
│   └── training/
│       └── trainer.h            [MODIFIED] 37 classes
│
├── src/
│   ├── core/
│   │   ├── quantum_neuron.cpp   [PHYSICS] ← Copy verbatim
│   │   └── quantum_foam.cpp     [PHYSICS] ← Copy verbatim
│   ├── io/
│   │   ├── dataset_loader.cpp   [NEW] Parquet loading implementation
│   │   └── data_pipeline.cpp    [COPY] Keep as-is
│   ├── training/
│   │   └── trainer.cpp          [MODIFIED] Real data + 37 classes
│   └── main.cpp                 [NEW] Clean entry point
│
├── python/
│   └── qesn_inference.py        [NEW] Fixed normalization
│
├── kaggle/
│   ├── QESN_MABe_Submission.py  [NEW] Correct configuration
│   └── model_weights.bin        [GENERATED] After training
│
├── scripts/
│   ├── build.bat                [NEW] Windows build script
│   ├── train.bat                [NEW] Training launcher
│   └── export_kaggle.bat        [NEW] Package for Kaggle
│
├── checkpoints/                 [OUTPUT] Training checkpoints
├── docs/
│   ├── MASTER_PLAN.md          [THIS FILE]
│   ├── PHYSICS_SPEC.md         [NEW] Quantum physics documentation
│   └── KAGGLE_GUIDE.md         [NEW] Submission instructions
│
├── CMakeLists.txt              [NEW] Build configuration
└── README.md                    [NEW] Project documentation
```

---

## 🔧 IMPLEMENTATION ROADMAP

### Phase 1: Foundation (QUANTUM PHYSICS) ✅ NO CHANGES
**Files**: `quantum_neuron.{h,cpp}`, `quantum_foam.{h,cpp}`
**Action**: Direct copy from V1 - these are PERFECT
**Duration**: 5 minutes
**Validation**: Compile standalone physics module

### Phase 2: Data Loading (CRITICAL REWRITE) ⚠️
**Files**: `dataset_loader.{h,cpp}`
**Dependencies**: Apache Arrow, Parquet C++ library
**Actions**:
1. Install Arrow: `vcpkg install arrow`
2. Implement `ParquetDataLoader::loadRealData()`
3. Parse CSV annotations with start_frame/stop_frame
4. Map 37 action names to indices
5. Normalize coordinates by video dimensions
**Duration**: 3-4 hours
**Validation**: Load 10 videos, verify keypoints non-synthetic

### Phase 3: Training System (37 CLASSES UPDATE) 🔨
**Files**: `trainer.{h,cpp}`
**Changes**:
- `NUM_BEHAVIOR_CLASSES = 37` (from 4)
- `weights_.resize(37 * grid_size)`
- `biases_.resize(37)`
- Implement `computeClassWeights()` for imbalance
- WINDOW_SIZE = 30
- Energy injection = 0.05 fixed
**Duration**: 2-3 hours
**Validation**: Train 1 epoch on 10 videos

### Phase 4: Inference System (PYTHON) 🐍
**Files**: `python/qesn_inference.py`, `kaggle/QESN_MABe_Submission.py`
**Changes**:
- Load 37 CLASS_NAMES
- Dynamic normalization: `video_width`, `video_height` from test.csv
- WINDOW_SIZE = 30
- Energy = 0.05
**Duration**: 2 hours
**Validation**: Local inference on 1 test video

### Phase 5: Build & Train (FULL SYSTEM) 🚀
**Actions**:
1. Compile C++ project
2. Train 30 epochs on full dataset
3. Export checkpoint
4. Validate on test data locally
**Duration**: 12-15 hours (mostly training)
**Validation**: Accuracy > 55%, diverse predictions

### Phase 6: Kaggle Deployment 📦
**Actions**:
1. Package kaggle/ directory
2. Upload to Kaggle datasets
3. Create inference notebook
4. Submit to competition
**Duration**: 1 hour
**Validation**: Submission accepted, LB score > 0.50

---

## 📊 DATASET SPECIFICATIONS

### Training Data Structure
```
E:\QESN-MABe\
├── train.csv                    [METADATA] 847 videos
├── train_tracking/              [PARQUET] Keypoints
│   ├── AdaptableSnail/
│   │   ├── 44566106.parquet    [18,456 frames × 4 mice × 18 keypoints]
│   │   └── ...
│   └── ...
└── train_annotation/            [CSV] Behavior annotations
    ├── AdaptableSnail/
    │   ├── 44566106.csv        [start_frame, stop_frame, action, ...]
    │   └── ...
    └── ...
```

### 37 Behavior Classes
```cpp
const std::vector<std::string> MABE_ACTION_NAMES = {
    "allogroom", "approach", "attack", "attemptmount", "avoid",
    "biteobject", "chase", "chaseattack", "climb", "defend",
    "dig", "disengage", "dominance", "dominancegroom", "dominancemount",
    "ejaculate", "escape", "exploreobject", "flinch", "follow",
    "freeze", "genitalgroom", "huddle", "intromit", "mount",
    "rear", "reciprocalsniff", "rest", "run", "selfgroom",
    "shepherd", "sniff", "sniffbody", "sniffface", "sniffgenital",
    "submit", "tussle"
};
```

### Class Imbalance (Real Distribution)
```
sniff:           37,837 (45.25%)
sniffgenital:     7,862  (9.40%)
attack:           7,462  (8.92%)
rear:             4,408  (5.27%)
...
ejaculate:            3  (0.00%) ← EXTREME IMBALANCE
```

**Solution**: Inverse frequency weighting in loss function

---

## 🔬 QUANTUM PHYSICS PARAMETERS (PRESERVED)

```cpp
HyperParams {
    // Quantum Simulation
    double coupling_strength = 0.10;    // Inter-neuron coupling
    double diffusion_rate = 0.05;        // Energy diffusion
    double decay_rate = 0.01;            // Energy decay
    double quantum_noise = 0.0005;       // Quantum fluctuations

    // Grid Topology
    std::size_t grid_width = 64;
    std::size_t grid_height = 64;

    // Temporal Evolution
    double time_step = 0.002;            // dt = 2ms

    // Training Config
    std::size_t window_size = 30;        // FIXED: 30 frames
    std::size_t stride = 15;             // 50% overlap
    std::size_t batch_size = 32;
    std::size_t epochs = 30;
    double learning_rate = 0.001;
    double weight_decay = 1e-5;
};
```

**Physics Equations (DO NOT MODIFY)**:
```
∂E/∂t = D∇²E - λE + κ∑(E_neighbor - E) + ξ(t)

Where:
- E: Energy field
- D: Diffusion coefficient (0.05)
- λ: Decay rate (0.01)
- κ: Coupling strength (0.10)
- ξ: Quantum noise ~N(0, 0.0005)
```

---

## 🎯 SUCCESS CRITERIA

### Training Metrics
- ✅ Loss converges below 1.5 by epoch 30
- ✅ Training accuracy > 60%
- ✅ Validation accuracy > 55%
- ✅ No NaN/Inf in checkpoints
- ✅ Per-class F1 > 0.10 for all classes

### Inference Quality
- ✅ Predictions span all 37 classes (not 99% one class)
- ✅ Confidence distribution: 0.20-0.80 (not all 0.12)
- ✅ 5,000-20,000 predictions per submission (not 902)
- ✅ No overlapping intervals for same agent/target pair

### Kaggle Submission
- ✅ Submission format valid (CSV accepted)
- ✅ Public LB score > 0.50
- ✅ Private LB score > 0.45
- ✅ No runtime errors in notebook

---

## 🛠️ DEPENDENCIES

### C++ Libraries
```bash
# Core
- C++20 compiler (MSVC 2022 or GCC 11+)
- CMake 3.20+

# Required
- Apache Arrow/Parquet (for reading parquet files)
- Eigen3 (linear algebra)
- OpenMP (parallelization)

# Optional
- CUDA 12.x + cuBLAS (GPU acceleration)
```

### Python Libraries
```bash
pip install numpy pandas pyarrow tqdm
```

### Installation (Windows)
```bash
# Install vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install dependencies
.\vcpkg install arrow:x64-windows eigen3:x64-windows
```

---

## 📝 DEVELOPMENT CHECKLIST

### Week 1: Foundation
- [x] Create project structure
- [ ] Copy quantum physics code (verbatim)
- [ ] Install Arrow/Parquet library
- [ ] Implement real parquet loading
- [ ] Test loading 10 videos
- [ ] Verify keypoints are real (not synthetic)

### Week 2: Training System
- [ ] Update trainer for 37 classes
- [ ] Implement class weighting
- [ ] Fix window size to 30
- [ ] Fix energy injection to 0.05
- [ ] Compile full project
- [ ] Train 1 epoch on 10 videos
- [ ] Validate loss decreases

### Week 3: Full Training
- [ ] Train 30 epochs on 100 videos
- [ ] Monitor accuracy/loss curves
- [ ] Export checkpoint
- [ ] Validate no NaN/Inf
- [ ] Test inference locally
- [ ] Verify diverse predictions

### Week 4: Kaggle Submission
- [ ] Create Python inference script
- [ ] Fix normalization to dynamic
- [ ] Test on local test data
- [ ] Package for Kaggle
- [ ] Upload dataset
- [ ] Create notebook
- [ ] Submit to competition
- [ ] Analyze LB score

---

## 🚨 RISK MITIGATION

### Risk 1: Parquet Loading Fails
**Mitigation**: Fallback to CSV parsing (slower but reliable)
**Backup Plan**: Pre-convert parquet to binary format

### Risk 2: Out of Memory During Training
**Mitigation**: Reduce batch size from 32 to 16
**Backup Plan**: Train on subset (50 videos) first

### Risk 3: Training Doesn't Converge
**Mitigation**: Tune learning rate (try 0.0001-0.01 range)
**Backup Plan**: Increase window size to 60 if 30 fails

### Risk 4: Kaggle Submission Timeout
**Mitigation**: Limit videos to 5000 frames max
**Backup Plan**: Use stride=30 instead of 15

---

## 📈 EXPECTED PERFORMANCE

### Training (30 epochs on 100 videos)
- **Time**: ~12-15 hours on CPU (OpenMP)
- **Memory**: ~8-12 GB RAM
- **Disk**: ~2 GB checkpoints

### Inference (Kaggle)
- **Time**: ~25-30 minutes (offline mode)
- **Memory**: ~4 GB RAM
- **Output**: 5,000-20,000 predictions

### F1-Score Targets
```
Class Tier 1 (Frequent): sniff, attack, approach
  → Target F1: 0.60-0.75

Class Tier 2 (Medium): rear, mount, chase
  → Target F1: 0.30-0.50

Class Tier 3 (Rare): ejaculate, biteobject
  → Target F1: 0.05-0.20 (acceptable)

Overall Macro F1: 0.40-0.50
```

---

## 🎓 LEARNING OUTCOMES

### Technical Skills
- Apache Arrow/Parquet C++ API
- Large-scale dataset management
- Class imbalance handling
- Physics-based ML architectures

### Scientific Contributions
- First published QESN for behavior recognition
- Validation of quantum simulation for real-world data
- Novel architecture without backpropagation

---

## 📞 SUPPORT & CONTACT

**Issues**: Check `docs/TROUBLESHOOTING.md`
**Questions**: Review `docs/FAQ.md`
**Updates**: Follow `docs/CHANGELOG.md`

---

**Status**: ⏳ IN PROGRESS
**Last Updated**: 2025-10-01
**Responsible**: Development Team
**Priority**: P0 - CRITICAL

---

**Next Action**: Phase 1 - Copy quantum physics code
**ETA to First Train**: 2-3 days
**ETA to Kaggle Submission**: 1-2 weeks
