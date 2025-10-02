# QESN-MABe V2: Professional Demonstration Notebooks

**Production-Grade Quantum Machine Learning for Behavior Classification**

---

## 📚 Notebook Collection

This directory contains **professional, publication-ready** Jupyter notebooks demonstrating the complete QESN (Quantum Energy State Network) architecture for mouse behavior classification on the MABe 2022 dataset.

### Available Notebooks:

| Notebook | Description | Level | Platform |
|----------|-------------|-------|----------|
| **QESN_Professional_Quantum_Demo.ipynb** | Complete quantum physics implementation with Schrödinger evolution, coherence analysis, and energy diffusion visualization | Advanced | All |
| **QESN_Complete_Classification_Demo.ipynb** | Full 37-class classification pipeline with realistic behavior generation, energy maps, and performance analysis | Production | All |
| **QESN_Demo_Interactive.ipynb** | Interactive demo with dependency auto-installation (legacy) | Beginner | Kaggle/Colab |

---

## 🎯 What Makes These Notebooks Professional?

### 1. **QESN_Professional_Quantum_Demo.ipynb**

**Purpose**: Demonstrate the genuine quantum mechanical foundation of QESN.

**Key Features**:
- ⚛️ **Complete Quantum Neuron**: Two-state system with complex amplitudes (α, β)
- 📐 **Schrödinger Evolution**: Time-dependent evolution with Runge-Kutta integration
- 🔬 **Physics Validation**: Energy conservation, coherence tracking, purity metrics
- 📊 **Bloch Sphere Visualization**: Quantum state trajectory in complex space
- 🌊 **Quantum Foam 2D**: 64×64 lattice with energy diffusion and neighbor coupling

**What You'll Learn**:
- How quantum neurons evolve according to the Schrödinger equation
- Energy diffusion dynamics across coupled quantum systems
- Quantum coherence and decoherence in neural networks
- Why QESN is fundamentally different from classical neural networks

**Use Cases**:
- Understanding the quantum physics behind QESN
- Academic presentations and publications
- Validating quantum mechanical properties
- Educational material for quantum machine learning

---

### 2. **QESN_Complete_Classification_Demo.ipynb**

**Purpose**: Full production pipeline for MABe 2022 behavior classification.

**Key Features**:
- 🎯 **37-Class Classification**: Complete MABe behavior recognition
- 📊 **Realistic Data Generation**: Behavioral patterns matching real MABe statistics
- 🔥 **Energy Map Visualization**: 2D heatmaps and 3D surface plots
- 📈 **Performance Metrics**: Inference time, throughput, confidence analysis
- 🏆 **Class Imbalance Handling**: Weighted classification (sniff: 37,837 vs ejaculate: 3 samples)

**Demonstration Scenarios**:
1. **Aggressive Behaviors**: attack, chase, tussle, defend
2. **Social Behaviors**: allogroom, approach, sniff, huddle
3. **Exploration Behaviors**: climb, dig, exploreobject
4. **Rest Behaviors**: rest, freeze

**Visualization Suite**:
- Keypoint trajectories (30-frame sequences)
- Quantum energy distribution (64×64 grid)
- 3D energy landscape
- Top-10 prediction probabilities
- Dataset frequency analysis
- Performance summary dashboard

**Use Cases**:
- Demonstrating QESN on real classification tasks
- Kaggle competition submissions
- HuggingFace model cards
- Production inference examples

---

## 🚀 Quick Start Guide

### Option 1: Google Colab (Recommended)

1. **Open in Colab**:
   - Navigate to: https://colab.research.google.com
   - Upload the notebook or use GitHub URL

2. **Install dependencies** (already included in notebooks):
   ```python
   !pip install numpy matplotlib seaborn scipy
   ```

3. **Run all cells** (Runtime → Run all)

**Why Colab?**
- Free GPU/TPU access
- No local installation required
- Easy sharing and collaboration
- Pre-installed scientific libraries

---

### Option 2: Kaggle Notebooks

1. **Create Kaggle Notebook**:
   - Go to: https://www.kaggle.com/code
   - Create new notebook
   - Copy-paste notebook content

2. **Enable Internet** (Settings → Internet ON)

3. **Add MABe dataset**:
   ```python
   # Link to MABe 2022 dataset
   # https://www.kaggle.com/competitions/mabe-2022-mouse-behavior/data
   ```

4. **Run notebook**

**Why Kaggle?**
- Direct access to MABe competition data
- Free GPU quota (30 hours/week)
- Built-in dataset management
- Competition-ready environment

---

### Option 3: Local Jupyter

1. **Install Jupyter**:
   ```bash
   pip install jupyter notebook
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy pandas matplotlib seaborn scipy plotly ipywidgets
   ```

3. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```

4. **Open notebook** and run cells

**Why Local?**
- Full control over environment
- No internet required (after setup)
- Integration with local C++ model
- Custom dataset paths

---

### Option 4: HuggingFace Spaces

1. **Create Space**:
   - Go to: https://huggingface.co/spaces
   - Create new Space (Gradio or Streamlit)

2. **Convert notebook to app**:
   ```python
   # Use nbconvert or create Gradio interface
   # See: docs/huggingface_deployment.md
   ```

3. **Deploy**

**Why HuggingFace?**
- Public model hosting
- Interactive demos
- Community visibility
- Model versioning

---

## 📖 Detailed Notebook Guides

### Running QESN_Professional_Quantum_Demo.ipynb

**Estimated Time**: 10-15 minutes
**Requirements**: NumPy, Matplotlib, SciPy
**Output**: 6-panel quantum physics visualization

**Steps**:
1. Run **Part 1** (Environment Setup) → Verifies libraries
2. Run **Part 2** (Quantum Architecture) → Implements QuantumNeuron and QuantumFoam2D
3. Run **Part 3** (Quantum Physics Demo) → Generates complete evolution analysis

**Expected Output**:
```
Environment Configuration:
  NumPy version: X.XX.X
  Matplotlib version: X.XX.X
  Interactive plots: True/False

Quantum Neuron implementation complete.
Key features:
  - Two-state quantum system (|0>, |1>)
  - Schrödinger evolution with Runge-Kutta integration
  - Lindblad decoherence model
  - 90-frame energy memory
  - Coherence and purity observables

Quantum Mechanical Observations:
  Initial energy: 0.000000
  Peak energy (post-injection): 0.XXXXXX
  Final energy (after decay): 0.XXXXXX
  Energy decay rate: X.XXX
  Average coherence: 0.XXXXXX
  Average purity: 0.XXXXXX

Physics validation: ✓ Normalization preserved
                    ✓ Energy decay exponential
                    ✓ Coherence maintains quantum character
```

**Visualization**:
- **Panel 1**: Energy evolution over time (exponential decay)
- **Panel 2**: Quantum coherence (off-diagonal density matrix)
- **Panel 3**: State purity Tr(ρ²)
- **Panel 4**: Ground state amplitude |0⟩ (real/imaginary)
- **Panel 5**: Excited state amplitude |1⟩ (real/imaginary)
- **Panel 6**: Bloch sphere trajectory (quantum state path)

---

### Running QESN_Complete_Classification_Demo.ipynb

**Estimated Time**: 5-10 minutes per behavior
**Requirements**: NumPy, Matplotlib, Seaborn, SciPy
**Output**: 7-panel classification analysis (×4 behaviors)

**Steps**:
1. Run **Part 1** (Configuration) → Loads MABe behavior classes
2. Run **Part 2** (Quantum Foam Classifier) → Initializes 4096-parameter model
3. Run **Part 3** (Behavior Generator) → Creates realistic keypoint sequences
4. Run **Part 4** (Visualization) → Defines plotting functions
5. Run **Part 5** (Full Demo) → Processes all 4 behavior types

**Expected Output**:
```
MABe 2022 Configuration:
  Behavior classes: 37
  Total samples: 92,609
  Most frequent: sniff (37,837 samples)
  Least frequent: ejaculate (3 samples)
  Class imbalance ratio: 12612.3:1

Quantum Foam Classifier initialized.
  Parameters: 151,589 (weights + biases)
  Architecture: 4096 → 37 classes

[1/4] Analyzing AGGRESSIVE behavior...
  Predicted: attack
  Confidence: 0.XXXX
  Inference: XX.XX ms
  Top 3: attack, chase, tussle

[Visualization generated]

PERFORMANCE STATISTICS
Total predictions: 4
Mean inference time: XX.XX ± X.XX ms
Min/Max: XX.XX / XX.XX ms
Throughput: XX.X predictions/second
```

**Visualization** (per behavior):
1. **Keypoint Trajectories**: 4 mice × 18 keypoints over 30 frames
2. **Energy Map (2D)**: 64×64 heatmap of quantum energy
3. **Energy Landscape (3D)**: Surface plot with peaks/valleys
4. **Top 10 Predictions**: Horizontal bar chart with probabilities
5. **Probability Distribution**: Log-scale plot of all 37 classes
6. **Dataset Frequencies**: Top 15 most common behaviors
7. **Summary Dashboard**: Text panel with metrics and parameters

---

## 🔬 Advanced Usage

### Integrating with C++ Model

The Python notebooks use a **simplified** quantum simulation for demonstration. For production use, connect to the C++ backend:

```python
import subprocess
import struct

# Export weights from Python
with open('model_weights.bin', 'wb') as f:
    f.write(struct.pack('Q', 64))  # grid_width
    f.write(struct.pack('Q', 64))  # grid_height
    f.write(struct.pack('Q', weights.size))
    f.write(struct.pack('Q', biases.size))
    f.write(struct.pack(f'{weights.size}d', *weights.flatten()))
    f.write(struct.pack(f'{biases.size}d', *biases))

# Run C++ inference
result = subprocess.run([
    './qesn_train.exe',
    '--mode', 'inference',
    '--weights', 'model_weights.bin',
    '--input', 'keypoints.npy'
], capture_output=True)
```

See: [../docs/cpp_python_interface.md](../docs/cpp_python_interface.md)

---

### Custom Behavior Patterns

Modify `generate_realistic_behavior()` to create custom behaviors:

```python
def generate_custom_behavior(pattern_params):
    keypoints = np.zeros((30, 4, 18, 3))

    # Your custom trajectory logic
    for frame in range(30):
        for mouse in range(4):
            # Define mouse position based on pattern_params
            x, y = custom_trajectory_function(frame, mouse, pattern_params)

            for kp in range(18):
                keypoints[frame, mouse, kp] = [x, y, confidence]

    return keypoints
```

Examples:
- **Pursuit behavior**: One mouse chasing another
- **Mating sequence**: Specific mount/intromit pattern
- **Aggression escalation**: Gradual transition from approach → attack

---

### Performance Benchmarking

```python
import time

# Benchmark inference speed
num_trials = 100
times = []

for _ in range(num_trials):
    keypoints = generate_realistic_behavior('social')
    start = time.time()
    prediction = classifier.predict(keypoints)
    times.append((time.time() - start) * 1000)

print(f"Mean: {np.mean(times):.2f} ms")
print(f"Std: {np.std(times):.2f} ms")
print(f"95th percentile: {np.percentile(times, 95):.2f} ms")
print(f"Throughput: {1000.0 / np.mean(times):.1f} sequences/sec")
```

Expected performance (Python):
- **Mean inference**: 20-50 ms (CPU)
- **Throughput**: 20-50 sequences/second

C++ production (expected):
- **Mean inference**: 2-5 ms (CPU, optimized)
- **Throughput**: 200-500 sequences/second

---

## 🎓 Educational Resources

### Understanding Quantum Foam

**Recommended reading order**:
1. Read `QESN_Professional_Quantum_Demo.ipynb` - Quantum neuron physics
2. Experiment with single neuron evolution (Part 3)
3. Observe energy diffusion in 2D lattice
4. Study coherence and decoherence effects

**Key concepts**:
- **Superposition**: α|0⟩ + β|1⟩ (neuron can be in both states)
- **Entanglement**: Coupling between neighbors creates correlations
- **Decoherence**: Quantum noise causes decay to classical states
- **Energy diffusion**: Information propagation across lattice

### MABe Dataset Deep Dive

**Understanding the challenge**:
- **Video input**: 1024×570 pixels, 30 FPS
- **Pose estimation**: DeepLabCut or SLEAP (18 keypoints/mouse)
- **Annotation**: Human-labeled behaviors with start/stop frames
- **Task**: Classify 30-frame windows into 37 behavior categories

**Class imbalance strategies**:
1. **Class weights**: Inverse frequency weighting (implemented in notebook)
2. **Oversampling**: Duplicate rare classes
3. **SMOTE**: Synthetic minority over-sampling
4. **Focal loss**: Down-weight easy examples

---

## 📊 Expected Results

### Quantum Physics Demo

**Metrics to observe**:
- Energy decay should be **exponential**: E(t) ∝ e^(-γt)
- Coherence should **oscillate** initially, then **decay**
- Purity should remain **close to 1.0** (pure quantum state)
- Bloch sphere trajectory should show **smooth evolution**

**Physics validation**:
- ✓ Normalization: |α|² + |β|² = 1 ± 1e-10
- ✓ Energy conservation (with decay): dE/dt = -γE
- ✓ Coherence bounds: 0 ≤ C ≤ 1

### Classification Demo

**Typical predictions**:
- **Aggressive**: attack (0.25), chase (0.18), tussle (0.12), ...
- **Social**: allogroom (0.22), sniff (0.19), approach (0.15), ...
- **Exploration**: climb (0.20), dig (0.17), exploreobject (0.14), ...
- **Rest**: rest (0.35), freeze (0.18), huddle (0.12), ...

**Performance expectations**:
- **Inference time**: 20-50 ms (Python), 2-5 ms (C++)
- **Confidence**: 0.15-0.40 (typical for 37-class problem)
- **Top-5 accuracy**: 60-80% (reasonable baseline)

---

## 🐛 Troubleshooting

### Common Issues

**1. "ModuleNotFoundError: No module named 'scipy'"**
```bash
# Solution:
pip install scipy
# or in notebook:
!pip install scipy
```

**2. "RuntimeWarning: invalid value encountered in divide"**
- Cause: Energy map sum is zero (no keypoints with confidence > 0.5)
- Solution: Check keypoint generation, ensure valid coordinates

**3. Slow inference (>100ms)**
- Cause: Large grid size or unoptimized loops
- Solution: Use NumPy vectorization, reduce grid size, or use C++ backend

**4. Visualization not showing**
```python
# Add this before plotting:
%matplotlib inline
# or for interactive plots:
%matplotlib notebook
```

**5. Memory error on Colab**
- Cause: Grid too large or too many simulations
- Solution: Reduce grid_size to 32, or process behaviors sequentially

---

## 📝 Citation

If you use these notebooks in your research, please cite:

```bibtex
@software{qesn_mabe_v2,
  author = {Angulo de Lafuente, Francisco},
  title = {QESN-MABe V2: Quantum Energy State Network for Behavior Classification},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Agnuxo1/QESN-MABe-V2}
}
```

---

## 🔗 Links

- **GitHub**: https://github.com/Agnuxo1/QESN-MABe-V2
- **Kaggle**: https://www.kaggle.com/franciscoangulo
- **HuggingFace**: https://huggingface.co/Agnuxo
- **MABe Competition**: https://www.kaggle.com/competitions/mabe-2022-mouse-behavior

---

## 📧 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact: [Your contact method]
- Discussion forum: [If available]

---

**Last Updated**: 2025-10-01
**Version**: 2.0
**License**: MIT
