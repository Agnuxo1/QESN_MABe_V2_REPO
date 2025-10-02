// QESN-MABe V2: Quantum Energy State Network for Mouse Behavior Classification
// Author: Francisco Angulo de Lafuente
// License: MIT
// Simplified binary data loader (no Apache Arrow dependency)

#include "io/dataset_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

namespace QESN {

// 37 MABe 2022 behavior classes
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

// Approximate frequencies from MABe 2022 dataset
const std::vector<int> MABE_ACTION_FREQUENCIES = {
    523, 2847, 1245, 89, 412,
    167, 734, 234, 98, 287,
    156, 892, 445, 78, 123,
    3, 234, 345, 189, 567,
    234, 67, 345, 12, 456,
    1234, 234, 2345, 567, 1678,
    45, 8901, 1234, 789, 234,
    123, 345
};

void MABeDatasetLoader::initializeActionMapping() {
    action_to_index_.clear();
    for (std::size_t i = 0; i < MABE_ACTION_NAMES.size(); ++i) {
        action_to_index_[MABE_ACTION_NAMES[i]] = static_cast<int>(i);
    }
}

int MABeDatasetLoader::actionNameToIndex(const std::string& name) {
    static std::unordered_map<std::string, int> mapping;
    if (mapping.empty()) {
        for (std::size_t i = 0; i < MABE_ACTION_NAMES.size(); ++i) {
            mapping[MABE_ACTION_NAMES[i]] = static_cast<int>(i);
        }
    }
    auto it = mapping.find(name);
    return (it != mapping.end()) ? it->second : -1;
}

std::string MABeDatasetLoader::actionIndexToName(int index) {
    if (index >= 0 && index < static_cast<int>(MABE_ACTION_NAMES.size())) {
        return MABE_ACTION_NAMES[index];
    }
    return "unknown";
}

void MABeDatasetLoader::loadDataset(const std::string& preprocessed_root,
                                     std::size_t max_sequences) {
    initializeActionMapping();
    sequences_.clear();

    std::cout << "Loading preprocessed dataset from: " << preprocessed_root << std::endl;

    // Find all .bin files in preprocessed directory
    std::vector<fs::path> bin_files;
    for (const auto& entry : fs::recursive_directory_iterator(preprocessed_root)) {
        if (entry.path().extension() == ".bin") {
            bin_files.push_back(entry.path());
        }
    }

    std::cout << "Found " << bin_files.size() << " binary files" << std::endl;

    // Load sequences
    std::size_t loaded = 0;
    for (const auto& bin_path : bin_files) {
        if (loaded >= max_sequences) {
            break;
        }

        try {
            std::cout << "Loading sequence " << (loaded + 1) << "/" << max_sequences
                      << ": " << bin_path.filename() << "..." << std::flush;

            Sequence seq = loadBinarySequence(bin_path.string());
            sequences_.push_back(std::move(seq));
            loaded++;

            std::cout << " OK (" << sequences_.back().frames.size() << " frames)" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << " FAILED: " << e.what() << std::endl;
        }
    }

    std::cout << "Loaded " << sequences_.size() << " sequences successfully." << std::endl;

    // Print summary
    std::size_t total_frames = 0;
    std::size_t total_labels = 0;
    for (const auto& seq : sequences_) {
        total_frames += seq.frames.size();
        total_labels += seq.labels.size();
    }

    std::cout << "Total frames: " << total_frames << std::endl;
    std::cout << "Total labels: " << total_labels << std::endl;
}

Sequence MABeDatasetLoader::loadBinarySequence(const std::string& bin_path) {
    std::ifstream file(bin_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open binary file: " + bin_path);
    }

    Sequence sequence;

    // Read header: video_id
    uint32_t video_id_len;
    file.read(reinterpret_cast<char*>(&video_id_len), sizeof(uint32_t));
    std::vector<char> video_id_buf(video_id_len);
    file.read(video_id_buf.data(), video_id_len);
    sequence.video_id = std::string(video_id_buf.begin(), video_id_buf.end());

    // Read lab_id
    uint32_t lab_id_len;
    file.read(reinterpret_cast<char*>(&lab_id_len), sizeof(uint32_t));
    std::vector<char> lab_id_buf(lab_id_len);
    file.read(lab_id_buf.data(), lab_id_len);
    sequence.lab_id = std::string(lab_id_buf.begin(), lab_id_buf.end());

    // Read metadata
    file.read(reinterpret_cast<char*>(&sequence.width), sizeof(int));
    file.read(reinterpret_cast<char*>(&sequence.height), sizeof(int));
    file.read(reinterpret_cast<char*>(&sequence.fps), sizeof(double));

    // Read dimensions
    uint32_t num_frames, num_mice, num_keypoints;
    file.read(reinterpret_cast<char*>(&num_frames), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&num_mice), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&num_keypoints), sizeof(uint32_t));

    // Read keypoints (flattened array)
    std::size_t total_elements = static_cast<std::size_t>(num_frames) * num_mice * num_keypoints * 3;
    std::vector<float> keypoints_flat(total_elements);
    file.read(reinterpret_cast<char*>(keypoints_flat.data()), total_elements * sizeof(float));

    // Reconstruct frames
    sequence.frames.resize(num_frames);
    std::size_t idx = 0;

    for (uint32_t frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        Frame& frame = sequence.frames[frame_idx];
        frame.frame_index = static_cast<int>(frame_idx);
        frame.timestamp = static_cast<double>(frame_idx) / sequence.fps;

        for (uint32_t mouse_idx = 0; mouse_idx < num_mice && mouse_idx < Frame::MAX_MICE; ++mouse_idx) {
            for (uint32_t kp_idx = 0; kp_idx < num_keypoints && kp_idx < Frame::MAX_KEYPOINTS; ++kp_idx) {
                frame.mice[mouse_idx][kp_idx].x = keypoints_flat[idx++];
                frame.mice[mouse_idx][kp_idx].y = keypoints_flat[idx++];
                frame.mice[mouse_idx][kp_idx].confidence = keypoints_flat[idx++];
            }
        }
    }

    // Read labels
    uint32_t num_labels;
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(uint32_t));

    for (uint32_t i = 0; i < num_labels; ++i) {
        uint32_t frame_idx;
        int32_t action_idx;

        file.read(reinterpret_cast<char*>(&frame_idx), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&action_idx), sizeof(int32_t));

        FrameLabel label;
        label.action_index = action_idx;
        label.action_name = actionIndexToName(action_idx);
        label.agent_id = 0;  // Simplified for binary format
        label.target_id = 0;

        sequence.labels[frame_idx] = label;
    }

    file.close();
    return sequence;
}

std::vector<FrameWindow> MABeDatasetLoader::windows(std::size_t window_size, std::size_t stride) const {
    std::vector<FrameWindow> result;

    for (const auto& sequence : sequences_) {
        if (sequence.frames.size() < window_size) {
            continue;
        }

        for (std::size_t start = 0; start + window_size <= sequence.frames.size(); start += stride) {
            FrameWindow window;
            window.sequence = &sequence;
            window.start_index = start;
            window.size = window_size;

            result.push_back(window);
        }
    }

    return result;
}

std::vector<double> MABeDatasetLoader::frameToEnergyMap(const Sequence& seq,
                                                         const Frame& frame,
                                                         int grid_width,
                                                         int grid_height) const {
    std::size_t grid_size = static_cast<std::size_t>(grid_width) * static_cast<std::size_t>(grid_height);
    std::vector<double> energy_map(grid_size, 0.0);

    // Inject energy for each keypoint
    for (int mouse = 0; mouse < Frame::MAX_MICE; ++mouse) {
        for (int kp = 0; kp < Frame::MAX_KEYPOINTS; ++kp) {
            const Keypoint& keypoint = frame.mice[mouse][kp];

            // Skip invalid keypoints
            if (std::isnan(keypoint.x) || std::isnan(keypoint.y) || keypoint.confidence < 0.1) {
                continue;
            }

            // Keypoints are already normalized (0-1) by Python preprocessor
            double nx = std::clamp(keypoint.x, 0.0, 0.999);
            double ny = std::clamp(keypoint.y, 0.0, 0.999);

            // Map to grid
            int gx = static_cast<int>(nx * grid_width);
            int gy = static_cast<int>(ny * grid_height);
            std::size_t grid_idx = static_cast<std::size_t>(gy) * grid_width + gx;

            // Accumulate energy (weighted by confidence)
            energy_map[grid_idx] += keypoint.confidence;
        }
    }

    // Normalize energy map
    double total_energy = 0.0;
    for (double e : energy_map) {
        total_energy += e;
    }

    if (total_energy > 0.0) {
        for (double& e : energy_map) {
            e /= total_energy;
        }
    }

    return energy_map;
}

FrameLabel MABeDatasetLoader::aggregateLabels(const FrameWindow& window) const {
    const Sequence& sequence = *window.sequence;

    // Use label from middle frame
    std::size_t middle_frame = window.start_index + window.size / 2;

    auto it = sequence.labels.find(static_cast<int>(middle_frame));
    if (it != sequence.labels.end()) {
        return it->second;
    }

    // If no label at middle, search nearby frames
    for (std::size_t offset = 0; offset < window.size / 4; ++offset) {
        // Try frames before middle
        if (middle_frame >= offset) {
            auto it_before = sequence.labels.find(static_cast<int>(middle_frame - offset));
            if (it_before != sequence.labels.end()) {
                return it_before->second;
            }
        }

        // Try frames after middle
        if (middle_frame + offset < sequence.frames.size()) {
            auto it_after = sequence.labels.find(static_cast<int>(middle_frame + offset));
            if (it_after != sequence.labels.end()) {
                return it_after->second;
            }
        }
    }

    // No label found - return default
    FrameLabel default_label;
    default_label.action_index = -1;
    default_label.action_name = "unlabeled";
    default_label.agent_id = 0;
    default_label.target_id = 0;

    return default_label;
}

} // namespace QESN
