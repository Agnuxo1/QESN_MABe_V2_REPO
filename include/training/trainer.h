// QESN-MABe V2: Quantum Energy State Network for Mouse Behavior Classification
// Author: Francisco Angulo de Lafuente
// License: MIT
// GitHub: https://github.com/Agnuxo1
// ResearchGate: https://www.researchgate.net/profile/Francisco-Angulo-Lafuente-3
// Kaggle: https://www.kaggle.com/franciscoangulo
// HuggingFace: https://huggingface.co/Agnuxo
// Wikipedia: https://es.wikipedia.org/wiki/Francisco_Angulo_de_Lafuente

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <random>

#include "../core/quantum_foam.h"
#include "../io/dataset_loader.h"

namespace QESN {

struct HyperParams {
    // Quantum Physics
    double coupling_strength = 0.10;
    double diffusion_rate = 0.05;
    double decay_rate = 0.01;
    double quantum_noise = 0.0005;

    // Temporal Processing
    std::size_t window_size = 30;  // FIXED: 30 frames (not 60!)
    std::size_t stride = 15;

    // Training
    std::size_t batch_size = 32;
    std::size_t epochs = 30;
    double learning_rate = 0.001;
    double weight_decay = 1e-5;

    // Grid
    std::size_t grid_width = 64;
    std::size_t grid_height = 64;
};

struct Metrics {
    double loss = 0.0;
    double accuracy = 0.0;
    double f1 = 0.0;

    std::array<double, NUM_BEHAVIOR_CLASSES> per_class_f1{};
    std::array<int, NUM_BEHAVIOR_CLASSES> true_positive{};
    std::array<int, NUM_BEHAVIOR_CLASSES> false_positive{};
    std::array<int, NUM_BEHAVIOR_CLASSES> false_negative{};

    void reset();
    void accumulate(int truth, int prediction);
    void finalize(std::size_t samples);
};

struct TrainingHistory {
    std::vector<double> losses;
    std::vector<double> validation_losses;
    std::vector<double> accuracies;
    std::vector<double> validation_accuracies;

    void save(const std::string& path) const;
};

class QESNTrainer {
public:
    QESNTrainer(HyperParams params, std::shared_ptr<MABeDatasetLoader> loader);

    void initialiseModel();
    void train(const std::string& checkpoint_dir,
               const std::string& best_checkpoint_path,
               const std::string& export_dir);

    Metrics evaluate(const std::vector<FrameWindow>& windows);

    void saveCheckpoint(const std::string& directory, std::size_t epoch, const Metrics& metrics);
    void saveBestModel(const std::string& path, const Metrics& metrics);
    void exportForInference(const std::string& directory) const;

private:
    HyperParams params_;
    std::shared_ptr<MABeDatasetLoader> loader_;
    std::unique_ptr<QuantumFoam2D> foam_;

    // 37-class classifier: W[37 × grid_size], b[37]
    std::vector<double> weights_;  // flattened: 37 × (grid_width × grid_height)
    std::vector<double> biases_;   // 37 biases
    std::vector<double> class_weights_;  // for imbalanced dataset

    TrainingHistory history_;
    double best_accuracy_;
    std::mt19937 rng_;

    std::vector<FrameWindow> training_windows_;
    std::vector<FrameWindow> validation_windows_;

    void splitDataset(double validation_ratio);
    void computeClassWeights();

    std::vector<double> encodeWindow(const FrameWindow& window);
    std::vector<double> forward(const std::vector<double>& energy_map) const;
    std::vector<double> softmax(const std::vector<double>& logits) const;
    double crossEntropy(const std::vector<double>& probs, int label) const;

    void updateWeights(const std::vector<double>& energy_map,
                       const std::vector<double>& probs,
                       int label,
                       double lr);

    int labelToIndex(const FrameLabel& label) const;
};

} // namespace QESN
