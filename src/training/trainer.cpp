// QESN-MABe V2: Quantum Energy State Network for Mouse Behavior Classification
// Author: Francisco Angulo de Lafuente
// License: MIT
// GitHub: https://github.com/Agnuxo1
// ResearchGate: https://www.researchgate.net/profile/Francisco-Angulo-Lafuente-3
// Kaggle: https://www.kaggle.com/franciscoangulo
// HuggingFace: https://huggingface.co/Agnuxo
// Wikipedia: https://es.wikipedia.org/wiki/Francisco_Angulo_de_Lafuente

#include "../../include/training/trainer.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <filesystem>

namespace QESN {

// Metrics implementation
void Metrics::reset() {
    loss = 0.0;
    accuracy = 0.0;
    f1 = 0.0;
    per_class_f1.fill(0.0);
    true_positive.fill(0);
    false_positive.fill(0);
    false_negative.fill(0);
}

void Metrics::accumulate(int truth, int prediction) {
    if (truth < 0 || truth >= static_cast<int>(NUM_BEHAVIOR_CLASSES)) return;
    if (prediction < 0 || prediction >= static_cast<int>(NUM_BEHAVIOR_CLASSES)) return;

    if (truth == prediction) {
        true_positive[truth]++;
    } else {
        false_negative[truth]++;
        false_positive[prediction]++;
    }
}

void Metrics::finalize(std::size_t samples) {
    if (samples == 0) return;

    // Compute accuracy
    int total_correct = 0;
    for (int tp : true_positive) {
        total_correct += tp;
    }
    accuracy = static_cast<double>(total_correct) / samples;

    // Compute per-class F1 and macro F1
    double f1_sum = 0.0;
    int valid_classes = 0;

    for (std::size_t i = 0; i < NUM_BEHAVIOR_CLASSES; ++i) {
        int tp = true_positive[i];
        int fp = false_positive[i];
        int fn = false_negative[i];

        double precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
        double recall = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;

        if (precision + recall > 0.0) {
            per_class_f1[i] = 2.0 * precision * recall / (precision + recall);
            f1_sum += per_class_f1[i];
            valid_classes++;
        } else {
            per_class_f1[i] = 0.0;
        }
    }

    f1 = (valid_classes > 0) ? f1_sum / valid_classes : 0.0;
}

void TrainingHistory::save(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Cannot save history to: " << path << std::endl;
        return;
    }

    file << "epoch,loss,val_loss,accuracy,val_accuracy\n";
    for (std::size_t i = 0; i < losses.size(); ++i) {
        file << i << ","
             << losses[i] << ","
             << (i < validation_losses.size() ? validation_losses[i] : 0.0) << ","
             << accuracies[i] << ","
             << (i < validation_accuracies.size() ? validation_accuracies[i] : 0.0) << "\n";
    }

    file.close();
}

// QESNTrainer implementation
QESNTrainer::QESNTrainer(HyperParams params, std::shared_ptr<MABeDatasetLoader> loader)
    : params_(params), loader_(loader), foam_(std::make_unique<QuantumFoam2D>()),
      best_accuracy_(0.0), rng_(std::random_device{}()) {
}

void QESNTrainer::initialiseModel() {
    std::cout << "Initializing QESN model..." << std::endl;

    // Initialize quantum foam
    foam_->initialise(params_.grid_width, params_.grid_height);
    foam_->setCouplingStrength(params_.coupling_strength);
    foam_->setDiffusionRate(params_.diffusion_rate);
    foam_->setDecayRate(params_.decay_rate);
    foam_->setQuantumNoise(params_.quantum_noise);

    std::cout << "Quantum foam: " << params_.grid_width << "x" << params_.grid_height
              << " grid, " << (params_.grid_width * params_.grid_height) << " neurons" << std::endl;

    // Xavier initialization for 37 classes
    std::size_t grid_size = params_.grid_width * params_.grid_height;
    double stddev = std::sqrt(2.0 / (grid_size + NUM_BEHAVIOR_CLASSES));
    std::normal_distribution<double> dist(0.0, stddev);

    weights_.resize(NUM_BEHAVIOR_CLASSES * grid_size);
    biases_.resize(NUM_BEHAVIOR_CLASSES, 0.0);

    for (auto& w : weights_) {
        w = dist(rng_);
    }

    std::cout << "Classifier: " << NUM_BEHAVIOR_CLASSES << " classes, "
              << weights_.size() << " weights, " << biases_.size() << " biases" << std::endl;
    std::cout << "Total parameters: " << (weights_.size() + biases_.size()) << std::endl;

    // Compute class weights for imbalanced dataset
    computeClassWeights();

    // Split dataset
    splitDataset(0.2);  // 20% validation

    std::cout << "Training windows: " << training_windows_.size() << std::endl;
    std::cout << "Validation windows: " << validation_windows_.size() << std::endl;
}

void QESNTrainer::computeClassWeights() {
    class_weights_.resize(NUM_BEHAVIOR_CLASSES);

    // Inverse frequency weighting
    double total_samples = 0.0;
    for (int freq : MABE_ACTION_FREQUENCIES) {
        total_samples += static_cast<double>(freq);
    }

    for (std::size_t i = 0; i < NUM_BEHAVIOR_CLASSES; ++i) {
        double freq = static_cast<double>(MABE_ACTION_FREQUENCIES[i]) / total_samples;
        class_weights_[i] = 1.0 / (freq + 1e-6);
    }

    // Normalize so mean = 1.0
    double sum = std::accumulate(class_weights_.begin(), class_weights_.end(), 0.0);
    double mean = sum / NUM_BEHAVIOR_CLASSES;
    for (auto& w : class_weights_) {
        w /= mean;
    }

    std::cout << "Class weights computed (range: "
              << *std::min_element(class_weights_.begin(), class_weights_.end()) << " - "
              << *std::max_element(class_weights_.begin(), class_weights_.end()) << ")" << std::endl;
}

void QESNTrainer::splitDataset(double validation_ratio) {
    // Generate all windows
    auto all_windows = loader_->windows(params_.window_size, params_.stride);

    // Shuffle
    std::shuffle(all_windows.begin(), all_windows.end(), rng_);

    // Split
    std::size_t val_count = static_cast<std::size_t>(all_windows.size() * validation_ratio);
    std::size_t train_count = all_windows.size() - val_count;

    training_windows_.assign(all_windows.begin(), all_windows.begin() + train_count);
    validation_windows_.assign(all_windows.begin() + train_count, all_windows.end());
}

std::vector<double> QESNTrainer::encodeWindow(const FrameWindow& window) {
    foam_->reset();
    const Sequence& sequence = *window.sequence;

    for (std::size_t offset = 0; offset < window.size; ++offset) {
        const Frame& frame = sequence.frames[window.start_index + offset];

        // Convert frame to energy map (normalized by video dimensions!)
        auto energy_map = loader_->frameToEnergyMap(
            sequence, frame,
            params_.grid_width, params_.grid_height
        );

        // Inject energy into quantum foam
        for (std::size_t idx = 0; idx < energy_map.size(); ++idx) {
            if (energy_map[idx] > 0.0) {
                std::size_t x = idx % params_.grid_width;
                std::size_t y = idx / params_.grid_width;
                foam_->injectEnergy(x, y, 0.05);  // FIXED energy injection
            }
        }

        // Evolve quantum system
        foam_->timeStep(0.002);  // dt = 2ms
    }

    // Observe energy field with Gaussian smoothing
    return foam_->observeGaussian(1);
}

std::vector<double> QESNTrainer::forward(const std::vector<double>& energy_map) const {
    std::vector<double> logits(NUM_BEHAVIOR_CLASSES, 0.0);
    std::size_t grid_size = params_.grid_width * params_.grid_height;

    // Matrix-vector multiply: logits = W * energy_map + b
    for (std::size_t c = 0; c < NUM_BEHAVIOR_CLASSES; ++c) {
        double sum = 0.0;
        const double* weight_row = &weights_[c * grid_size];
        for (std::size_t i = 0; i < grid_size; ++i) {
            sum += weight_row[i] * energy_map[i];
        }
        logits[c] = sum + biases_[c];
    }

    return logits;
}

std::vector<double> QESNTrainer::softmax(const std::vector<double>& logits) const {
    std::vector<double> probs(logits.size());

    // Numerical stability: subtract max
    double max_logit = *std::max_element(logits.begin(), logits.end());

    double sum = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }

    for (auto& p : probs) {
        p /= sum;
    }

    return probs;
}

double QESNTrainer::crossEntropy(const std::vector<double>& probs, int label) const {
    if (label < 0 || label >= static_cast<int>(NUM_BEHAVIOR_CLASSES)) {
        return 0.0;
    }

    // Weighted cross-entropy for class imbalance
    double weight = class_weights_[label];
    return -weight * std::log(std::max(probs[label], 1e-10));
}

void QESNTrainer::updateWeights(const std::vector<double>& energy_map,
                                 const std::vector<double>& probs,
                                 int label,
                                 double lr) {
    if (label < 0 || label >= static_cast<int>(NUM_BEHAVIOR_CLASSES)) {
        return;
    }

    std::size_t grid_size = params_.grid_width * params_.grid_height;

    // Gradient descent with weight decay
    for (std::size_t c = 0; c < NUM_BEHAVIOR_CLASSES; ++c) {
        double target = (c == static_cast<std::size_t>(label)) ? 1.0 : 0.0;
        double error = probs[c] - target;
        double weight_gradient_scale = error * class_weights_[label];

        // Update weights
        double* weight_row = &weights_[c * grid_size];
        for (std::size_t i = 0; i < grid_size; ++i) {
            double gradient = weight_gradient_scale * energy_map[i];
            double l2_penalty = params_.weight_decay * weight_row[i];
            weight_row[i] -= lr * (gradient + l2_penalty);
        }

        // Update bias
        biases_[c] -= lr * weight_gradient_scale;
    }
}

int QESNTrainer::labelToIndex(const FrameLabel& label) const {
    return label.action_index;
}

Metrics QESNTrainer::evaluate(const std::vector<FrameWindow>& windows) {
    Metrics metrics;
    metrics.reset();

    double total_loss = 0.0;

    for (const auto& window : windows) {
        // Get label
        FrameLabel label = loader_->aggregateLabels(window);
        int truth = labelToIndex(label);

        if (truth < 0) continue;

        // Encode and forward
        auto energy_map = encodeWindow(window);
        auto logits = forward(energy_map);
        auto probs = softmax(logits);

        // Loss
        total_loss += crossEntropy(probs, truth);

        // Prediction
        int prediction = static_cast<int>(
            std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()))
        );

        metrics.accumulate(truth, prediction);
    }

    metrics.loss = total_loss / windows.size();
    metrics.finalize(windows.size());

    return metrics;
}

void QESNTrainer::train(const std::string& checkpoint_dir,
                         const std::string& best_checkpoint_path,
                         const std::string& export_dir) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Starting QESN Training" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Create checkpoint directory
    std::filesystem::create_directories(checkpoint_dir);
    std::filesystem::create_directories(export_dir);

    for (std::size_t epoch = 0; epoch < params_.epochs; ++epoch) {
        std::cout << "Epoch " << (epoch + 1) << "/" << params_.epochs << std::endl;

        // Shuffle training data
        std::shuffle(training_windows_.begin(), training_windows_.end(), rng_);

        // Training
        Metrics train_metrics;
        train_metrics.reset();
        double total_loss = 0.0;

        std::size_t processed = 0;
        for (const auto& window : training_windows_) {
            // Get label
            FrameLabel label = loader_->aggregateLabels(window);
            int truth = labelToIndex(label);

            if (truth < 0) continue;

            // Encode and forward
            auto energy_map = encodeWindow(window);
            auto logits = forward(energy_map);
            auto probs = softmax(logits);

            // Loss
            double loss = crossEntropy(probs, truth);
            total_loss += loss;

            // Prediction
            int prediction = static_cast<int>(
                std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()))
            );

            train_metrics.accumulate(truth, prediction);

            // Update weights
            updateWeights(energy_map, probs, truth, params_.learning_rate);

            // Progress
            processed++;
            if (processed % 100 == 0) {
                std::cout << "  Processed " << processed << "/" << training_windows_.size()
                          << " windows\r" << std::flush;
            }
        }

        train_metrics.loss = total_loss / training_windows_.size();
        train_metrics.finalize(training_windows_.size());

        std::cout << "  Training   - Loss: " << std::fixed << std::setprecision(4)
                  << train_metrics.loss
                  << ", Acc: " << std::setprecision(2) << (train_metrics.accuracy * 100.0) << "%"
                  << ", F1: " << std::setprecision(4) << train_metrics.f1 << std::endl;

        // Validation
        Metrics val_metrics = evaluate(validation_windows_);

        std::cout << "  Validation - Loss: " << std::fixed << std::setprecision(4)
                  << val_metrics.loss
                  << ", Acc: " << std::setprecision(2) << (val_metrics.accuracy * 100.0) << "%"
                  << ", F1: " << std::setprecision(4) << val_metrics.f1 << std::endl;

        // Save history
        history_.losses.push_back(train_metrics.loss);
        history_.accuracies.push_back(train_metrics.accuracy);
        history_.validation_losses.push_back(val_metrics.loss);
        history_.validation_accuracies.push_back(val_metrics.accuracy);

        // Save checkpoint every 5 epochs
        if ((epoch + 1) % 5 == 0) {
            saveCheckpoint(checkpoint_dir, epoch + 1, val_metrics);
        }

        // Save best model
        if (val_metrics.accuracy > best_accuracy_) {
            best_accuracy_ = val_metrics.accuracy;
            saveBestModel(best_checkpoint_path, val_metrics);
            std::cout << "  *** New best model saved! ***" << std::endl;
        }

        std::cout << std::endl;
    }

    // Save final history
    history_.save(checkpoint_dir + "/training_history.csv");

    // Export for inference
    exportForInference(export_dir);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Training Complete!" << std::endl;
    std::cout << "Best validation accuracy: " << std::fixed << std::setprecision(2)
              << (best_accuracy_ * 100.0) << "%" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

void QESNTrainer::saveCheckpoint(const std::string& directory, std::size_t epoch, const Metrics& metrics) {
    std::string filename = directory + "/checkpoint_epoch_" + std::to_string(epoch) + ".bin";
    std::ofstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Cannot save checkpoint: " << filename << std::endl;
        return;
    }

    // Write header
    std::uint64_t grid_w = params_.grid_width;
    std::uint64_t grid_h = params_.grid_height;
    std::uint64_t weight_count = weights_.size();
    std::uint64_t bias_count = biases_.size();

    file.write(reinterpret_cast<const char*>(&grid_w), sizeof(grid_w));
    file.write(reinterpret_cast<const char*>(&grid_h), sizeof(grid_h));
    file.write(reinterpret_cast<const char*>(&weight_count), sizeof(weight_count));
    file.write(reinterpret_cast<const char*>(&bias_count), sizeof(bias_count));

    // Write weights and biases
    file.write(reinterpret_cast<const char*>(weights_.data()), weights_.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(biases_.data()), biases_.size() * sizeof(double));

    file.close();

    std::cout << "  Checkpoint saved: " << filename << std::endl;
}

void QESNTrainer::saveBestModel(const std::string& path, const Metrics& metrics) {
    std::ofstream file(path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Cannot save best model: " << path << std::endl;
        return;
    }

    // Write header
    std::uint64_t grid_w = params_.grid_width;
    std::uint64_t grid_h = params_.grid_height;
    std::uint64_t weight_count = weights_.size();
    std::uint64_t bias_count = biases_.size();

    file.write(reinterpret_cast<const char*>(&grid_w), sizeof(grid_w));
    file.write(reinterpret_cast<const char*>(&grid_h), sizeof(grid_h));
    file.write(reinterpret_cast<const char*>(&weight_count), sizeof(weight_count));
    file.write(reinterpret_cast<const char*>(&bias_count), sizeof(bias_count));

    // Write weights and biases
    file.write(reinterpret_cast<const char*>(weights_.data()), weights_.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(biases_.data()), biases_.size() * sizeof(double));

    file.close();
}

void QESNTrainer::exportForInference(const std::string& directory) const {
    std::filesystem::create_directories(directory);

    // Save weights binary
    std::string weights_path = directory + "/model_weights.bin";
    std::ofstream weights_file(weights_path, std::ios::binary);

    if (weights_file.is_open()) {
        std::uint64_t grid_w = params_.grid_width;
        std::uint64_t grid_h = params_.grid_height;
        std::uint64_t weight_count = weights_.size();
        std::uint64_t bias_count = biases_.size();

        weights_file.write(reinterpret_cast<const char*>(&grid_w), sizeof(grid_w));
        weights_file.write(reinterpret_cast<const char*>(&grid_h), sizeof(grid_h));
        weights_file.write(reinterpret_cast<const char*>(&weight_count), sizeof(weight_count));
        weights_file.write(reinterpret_cast<const char*>(&bias_count), sizeof(bias_count));

        weights_file.write(reinterpret_cast<const char*>(weights_.data()), weights_.size() * sizeof(double));
        weights_file.write(reinterpret_cast<const char*>(biases_.data()), biases_.size() * sizeof(double));

        weights_file.close();
        std::cout << "Exported weights: " << weights_path << std::endl;
    }

    // Save config JSON
    std::string config_path = directory + "/model_config.json";
    std::ofstream config_file(config_path);

    if (config_file.is_open()) {
        config_file << "{\n";
        config_file << "  \"grid_width\": " << params_.grid_width << ",\n";
        config_file << "  \"grid_height\": " << params_.grid_height << ",\n";
        config_file << "  \"num_classes\": " << NUM_BEHAVIOR_CLASSES << ",\n";
        config_file << "  \"window_size\": " << params_.window_size << ",\n";
        config_file << "  \"coupling_strength\": " << params_.coupling_strength << ",\n";
        config_file << "  \"diffusion_rate\": " << params_.diffusion_rate << ",\n";
        config_file << "  \"decay_rate\": " << params_.decay_rate << ",\n";
        config_file << "  \"quantum_noise\": " << params_.quantum_noise << "\n";
        config_file << "}\n";

        config_file.close();
        std::cout << "Exported config: " << config_path << std::endl;
    }
}

} // namespace QESN
