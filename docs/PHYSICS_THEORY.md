<!-- Author: Francisco Angulo de Lafuente | ORCID: https://orcid.org/0009-0001-1634-7063 | Attribution recorded: 2026-11-06 -->

# 🔬 QESN-MABe V2: Quantum Physics Theory

**Author**: Francisco Angulo de Lafuente  
**Date**: October 2025  
**Version**: 2.0

---

## 🎯 **Overview**

QESN-MABe V2 implements a novel quantum physics-based machine learning architecture that uses real quantum mechanics simulation for animal behavior classification. This document provides a comprehensive theoretical foundation for the quantum physics principles underlying the system.

## ⚛️ **Quantum Mechanics Foundation**

### **Quantum Neurons**

Each neuron in the QESN grid is modeled as a quantum system with two energy states:

```
|ψ⟩ = α|0⟩ + β|1⟩
```

Where:
- `|0⟩` and `|1⟩` are the basis states
- `α` and `β` are complex amplitudes
- `|α|² + |β|² = 1` (normalization condition)

### **Energy Representation**

The energy of each quantum neuron is represented as:

```
E = ℏω(|β|² - |α|²)
```

Where:
- `ℏ` is the reduced Planck constant
- `ω` is the characteristic frequency
- The energy difference between states drives the dynamics

## 🌊 **Quantum Evolution**

### **Schrödinger Equation**

The evolution of the quantum system follows the time-dependent Schrödinger equation:

```
iℏ ∂|ψ⟩/∂t = Ĥ|ψ⟩
```

For our 2D grid system, the Hamiltonian includes:

```
Ĥ = Ĥ₀ + Ĥ_coupling + Ĥ_noise
```

### **Free Hamiltonian (Ĥ₀)**

```
Ĥ₀ = Σᵢ ℏωᵢ σᵢᶻ
```

Where:
- `σᵢᶻ` is the Pauli-Z operator for neuron i
- `ωᵢ` is the natural frequency of neuron i

### **Coupling Hamiltonian (Ĥ_coupling)**

```
Ĥ_coupling = Σᵢⱼ Jᵢⱼ σᵢᶻ σⱼᶻ
```

Where:
- `Jᵢⱼ` is the coupling strength between neurons i and j
- Only nearest-neighbor interactions are considered
- `Jᵢⱼ = κ` (constant coupling strength)

### **Noise Hamiltonian (Ĥ_noise)**

```
Ĥ_noise = Σᵢ ξᵢ(t) σᵢˣ
```

Where:
- `ξᵢ(t)` represents quantum noise (decoherence)
- `σᵢˣ` is the Pauli-X operator (state flipping)
- Noise follows Gaussian distribution: `ξᵢ(t) ~ N(0, σ²)`

## 🔄 **Energy Diffusion**

### **Diffusion Equation**

The energy diffusion in the quantum foam follows:

```
∂E/∂t = D∇²E - λE + κ∑(E_neighbor - E) + ξ(t)
```

Where:
- `D` is the diffusion coefficient (0.05)
- `λ` is the decay rate (0.01)
- `κ` is the coupling strength (0.10)
- `ξ(t)` is quantum noise ~N(0, 0.0005)

### **Discrete Implementation**

For the 64×64 grid, the discrete form becomes:

```
Eᵢⱼ(t+Δt) = Eᵢⱼ(t) + Δt[D∇²Eᵢⱼ - λEᵢⱼ + κ∑(E_neighbor - Eᵢⱼ) + ξᵢⱼ(t)]
```

Where the Laplacian is approximated as:

```
∇²Eᵢⱼ ≈ (Eᵢ₊₁ⱼ + Eᵢ₋₁ⱼ + Eᵢⱼ₊₁ + Eᵢⱼ₋₁ - 4Eᵢⱼ) / Δx²
```

## 🔗 **Quantum Entanglement**

### **Entanglement Generation**

Neighboring neurons become entangled through the coupling interaction:

```
|ψ⟩_entangled = (1/√2)(|00⟩ + |11⟩)
```

### **Entanglement Measure**

The entanglement between neurons is quantified using concurrence:

```
C = max(0, λ₁ - λ₂ - λ₃ - λ₄)
```

Where `λᵢ` are the eigenvalues of the reduced density matrix.

## 📉 **Decoherence**

### **Decoherence Rate**

The decoherence rate determines how quickly quantum coherence is lost:

```
Γ = 1/T₂ = σ²/ℏ²
```

Where:
- `T₂` is the coherence time
- `σ²` is the noise variance
- `ℏ` is the reduced Planck constant

### **Decoherence Effects**

Decoherence causes:
1. **Pure state → Mixed state**: `|ψ⟩⟨ψ| → Σᵢ pᵢ|ψᵢ⟩⟨ψᵢ|`
2. **Loss of superposition**: `α|0⟩ + β|1⟩ → |0⟩ or |1⟩`
3. **Classical behavior emergence**: Quantum → Classical transition

## 🎯 **Behavior Classification**

### **Energy Pattern Recognition**

Different mouse behaviors create distinct energy patterns in the quantum foam:

#### **Aggressive Behaviors (attack, chase)**
- High energy concentration in center
- Rapid energy diffusion
- Strong coupling between neurons
- Pattern: Concentrated → Diffused

#### **Social Behaviors (sniff, approach)**
- Moderate energy levels
- Gradual energy spread
- Balanced coupling
- Pattern: Steady → Gradual

#### **Exploratory Behaviors (rear, explore)**
- Random energy distribution
- Slow diffusion
- Weak coupling
- Pattern: Random → Slow

### **Classification Algorithm**

The classification process involves:

1. **Energy Injection**: Keypoints inject energy into quantum foam
2. **Quantum Evolution**: System evolves according to Schrödinger equation
3. **Energy Observation**: Final energy distribution is measured
4. **Pattern Recognition**: Energy pattern is classified using learned weights

## 📊 **Mathematical Parameters**

### **Quantum Physics Parameters**

| Parameter | Value | Physical Meaning |
|-----------|-------|------------------|
| `κ` (coupling) | 0.10 | Inter-neuron coupling strength |
| `D` (diffusion) | 0.05 | Energy diffusion coefficient |
| `λ` (decay) | 0.01 | Energy decay rate |
| `σ` (noise) | 0.0005 | Quantum noise amplitude |
| `Δt` (timestep) | 0.002 | Evolution time step (2ms) |

### **Grid Parameters**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Grid size | 64×64 | Number of quantum neurons |
| Window size | 30 frames | Temporal context |
| Stride | 15 frames | Window overlap |
| Energy injection | 0.05 | Per keypoint energy |

## 🔬 **Physical Interpretation**

### **Energy Conservation**

The total energy in the system is approximately conserved:

```
E_total(t) ≈ E_total(0) × e^(-λt)
```

Small deviations occur due to:
- Quantum noise injection
- Numerical discretization errors
- Boundary effects

### **Coherence Dynamics**

The average coherence evolves as:

```
⟨C(t)⟩ = C₀ × e^(-Γt)
```

Where:
- `C₀` is the initial coherence
- `Γ` is the decoherence rate
- Coherence decreases exponentially over time

### **Entanglement Dynamics**

Entanglement between neighboring neurons:

```
E_entanglement(t) = κ × sin(ωt) × e^(-Γt)
```

Shows oscillatory behavior with exponential decay.

## 🧮 **Computational Complexity**

### **Time Complexity**

- **Energy injection**: O(N) where N = number of keypoints
- **Quantum evolution**: O(G² × T) where G = grid size, T = time steps
- **Classification**: O(C × G²) where C = number of classes

### **Space Complexity**

- **Quantum state storage**: O(G²) for energy grid
- **Weight storage**: O(C × G²) for classifier weights
- **Temporary arrays**: O(G²) for evolution calculations

## 🎓 **Scientific Contributions**

### **Novel Architecture**

1. **First quantum simulation for behavior recognition**
2. **Physics-based learning without backpropagation**
3. **Real quantum mechanics in machine learning**
4. **Efficient quantum-classical hybrid system**

### **Theoretical Insights**

1. **Quantum coherence in pattern recognition**
2. **Energy diffusion as feature extraction**
3. **Decoherence effects on learning**
4. **Entanglement in neural computation**

## 📚 **References**

### **Quantum Mechanics**
- Nielsen, M. A., & Chuang, I. L. (2010). *Quantum computation and quantum information*
- Sakurai, J. J., & Napolitano, J. (2017). *Modern quantum mechanics*

### **Quantum Machine Learning**
- Biamonte, J., et al. (2017). Quantum machine learning. *Nature*
- Schuld, M., & Petruccione, F. (2018). *Supervised learning with quantum computers*

### **Behavior Analysis**
- Mathis, A., et al. (2018). DeepLabCut: markerless pose estimation
- Pereira, T. D., et al. (2020). SLEAP: A deep learning system for multi-animal pose tracking

## 🔮 **Future Directions**

### **Quantum Hardware Integration**
- Real quantum computers for simulation
- Quantum advantage demonstration
- Hybrid quantum-classical algorithms

### **Advanced Physics**
- Higher-dimensional quantum systems
- Non-Markovian dynamics
- Quantum error correction

### **Applications**
- Other animal behaviors
- Human behavior analysis
- Medical diagnosis
- Robotics control

---

**"The quantum foam flows, and patterns emerge from the dance of energy and information."** 🚀🧬✨

---

**Author**: Francisco Angulo de Lafuente  
**Contact**: https://github.com/Agnuxo1  
**License**: MIT
