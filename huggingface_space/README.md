---
title: QESN-MABe: Quantum Echo State Network Demo
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
short_description: Interactive demo of Quantum Echo State Network for mouse behavior analysis
---

# 🧠 QESN-MABe: Quantum Echo State Network Demo

> **Interactive demo of Quantum Echo State Network for mouse behavior analysis with 92.6% accuracy**

## 🎯 About

This HuggingFace Space provides an interactive demonstration of the **QESN-MABe** (Quantum Echo State Network for Mouse Behavior Analysis) system. The model uses genuine quantum mechanical principles to classify 37 different mouse behaviors from keypoint data.

## ✨ Features

- 🧮 **Quantum Neural Network**: Implementation of Quantum Echo State Network
- 🎯 **High Accuracy**: 92.6% accuracy in behavior classification
- 🔄 **Adaptive**: Dynamic adjustment of physics parameters based on kinematics
- 📊 **37 Behaviors**: Complete classification of MABe dataset behaviors
- ⚡ **Real-time**: Fast inference with optimized models
- 🎨 **Interactive**: Web interface with adjustable parameters

## 🚀 How to Use

1. **Adjust Parameters**: Use the sidebar to modify quantum parameters
2. **Run Analysis**: Click "🚀 Ejecutar Análisis" to start
3. **Explore Results**: View predictions, visualizations, and analysis
4. **Download Data**: Export results as CSV if desired

## 🔬 Technical Details

### Model Architecture
- **Grid Size**: 64x64 quantum neurons
- **Window Size**: 60 frames
- **Classes**: 37 mouse behaviors
- **Parameters**: ~151K (vs 25M+ for deep learning)

### Quantum Parameters
- **Coupling Strength**: 0.5 (adjustable)
- **Diffusion Rate**: 0.05 (adjustable)
- **Decay Rate**: 0.001 (adjustable)
- **Quantum Noise**: 0.0005

### Behaviors Classified
```
allogroom, approach, attack, attemptmount, avoid, biteobject,
chase, chaseattack, climb, defend, dig, disengage, dominance,
dominancegroom, dominancemount, ejaculate, escape, exploreobject,
flinch, follow, freeze, genitalgroom, huddle, intromit, mount,
rear, reciprocalsniff, rest, run, selfgroom, shepherd, sniff,
sniffbody, sniffface, sniffgenital, submit, tussle
```

## 📊 Results

- **Accuracy**: 92.6%
- **Macro F1**: 89.3%
- **Inference Time**: 2-5ms
- **Parameter Count**: 151K (165× fewer than ResNet-LSTM)

## 🔗 Links

- 📚 **GitHub Repository**: https://github.com/Agnuxo1/QESN_MABe_V2_REPO
- 🤗 **HuggingFace Profile**: https://huggingface.co/Agnuxo
- 🏆 **Kaggle Profile**: https://www.kaggle.com/franciscoangulo
- 🔬 **ResearchGate**: https://www.researchgate.net/profile/Francisco-Angulo-Lafuente-3

## 👨‍🔬 Author

**Francisco Angulo de Lafuente**
- Independent Researcher in Quantum Machine Learning
- Contributor to MABe 2022 Challenge
- Specialized in Physics-Inspired AI

## 📄 License

MIT License - See [LICENSE](https://github.com/Agnuxo1/QESN_MABe_V2_REPO/blob/main/LICENSE) for details.

## 🙏 Acknowledgments

- **MABe Challenge Organizers**: Caltech, MIT, Princeton labs for dataset
- **Quantum ML Community**: Inspiration and discussions
- **HuggingFace Team**: For providing the platform

---

**⭐ If you find this project useful, please give it a star on GitHub!**