// QESN-MABe V2: Quantum Energy State Network for Mouse Behavior Classification
// Author: Francisco Angulo de Lafuente
// License: MIT
// GitHub: https://github.com/Agnuxo1
// ResearchGate: https://www.researchgate.net/profile/Francisco-Angulo-Lafuente-3
// Kaggle: https://www.kaggle.com/franciscoangulo
// HuggingFace: https://huggingface.co/Agnuxo
// Wikipedia: https://es.wikipedia.org/wiki/Francisco_Angulo_de_Lafuente

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace QESN {

// 37 MABe 2022 behavior classes
constexpr std::size_t NUM_BEHAVIOR_CLASSES = 37;

extern const std::vector<std::string> MABE_ACTION_NAMES;
extern const std::vector<int> MABE_ACTION_FREQUENCIES;

struct Keypoint {
    double x;
    double y;
    double confidence;
};

struct Frame {
    static constexpr int MAX_MICE = 4;
    static constexpr int MAX_KEYPOINTS = 18;

    int frame_index;
    double timestamp;
    Keypoint mice[MAX_MICE][MAX_KEYPOINTS];
};

struct FrameLabel {
    int action_index;  // 0-36 for 37 classes
    std::string action_name;
    int agent_id;
    int target_id;
};

struct Sequence {
    std::string video_id;
    std::string lab_id;
    int width;
    int height;
    double fps;
    double duration;

    std::vector<Frame> frames;
    std::unordered_map<int, FrameLabel> labels;  // frame_index -> label
};

struct FrameWindow {
    const Sequence* sequence;
    std::size_t start_index;
    std::size_t size;
};

class MABeDatasetLoader {
public:
    MABeDatasetLoader() = default;

    // Load preprocessed binary files
    void loadDataset(const std::string& preprocessed_root,
                     std::size_t max_sequences = 100);

    // Generate sliding windows for training
    std::vector<FrameWindow> windows(std::size_t window_size, std::size_t stride) const;

    // Convert frame to energy map for quantum foam
    std::vector<double> frameToEnergyMap(const Sequence& seq,
                                         const Frame& frame,
                                         int grid_width,
                                         int grid_height) const;

    // Aggregate labels in window
    FrameLabel aggregateLabels(const FrameWindow& window) const;

    // Action name <-> index mapping
    static int actionNameToIndex(const std::string& name);
    static std::string actionIndexToName(int index);

    const std::vector<Sequence>& sequences() const { return sequences_; }

private:
    std::vector<Sequence> sequences_;
    std::unordered_map<std::string, int> action_to_index_;

    void initializeActionMapping();
    Sequence loadBinarySequence(const std::string& bin_path);
};

} // namespace QESN
