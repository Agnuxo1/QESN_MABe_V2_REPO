// QESN-MABe V2: Quantum Energy State Network for Mouse Behavior Classification
// Author: Francisco Angulo de Lafuente
// Simplified main using preprocessed binary data

#include <iostream>
#include <string>
#include <memory>
#include <stdexcept>

#include "io/dataset_loader.h"
#include "training/trainer.h"

void printUsage(const char* program_name) {
    std::cout << "QESN-MABe V2: Quantum Energy State Network\n";
    std::cout << "Author: Francisco Angulo de Lafuente\n\n";
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Required:\n";
    std::cout << "  --data <path>          Path to preprocessed data directory\n\n";
    std::cout << "Training:\n";
    std::cout << "  --epochs <n>           Number of epochs (default: 30)\n";
    std::cout << "  --batch <n>            Batch size (default: 32)\n";
    std::cout << "  --lr <f>               Learning rate (default: 0.001)\n";
    std::cout << "  --window <n>           Window size (default: 30)\n";
    std::cout << "  --stride <n>           Stride (default: 15)\n";
    std::cout << "  --max-sequences <n>    Max sequences (default: 100)\n\n";
    std::cout << "Output:\n";
    std::cout << "  --checkpoints <path>   Checkpoint directory (default: checkpoints)\n";
    std::cout << "  --best <path>          Best model path (default: checkpoints/best_model.bin)\n";
    std::cout << "  --export <path>        Export directory (default: kaggle)\n\n";
    std::cout << "Physics:\n";
    std::cout << "  --coupling <f>         Coupling strength (default: 0.10)\n";
    std::cout << "  --diffusion <f>        Diffusion rate (default: 0.05)\n";
    std::cout << "  --decay <f>            Decay rate (default: 0.01)\n";
    std::cout << "  --noise <f>            Quantum noise (default: 0.0005)\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " --data data/preprocessed --epochs 30\n\n";
}

struct Args {
    std::string data_path;
    std::string checkpoint_dir = "checkpoints";
    std::string best_model_path = "checkpoints/best_model.bin";
    std::string export_dir = "kaggle";

    std::size_t epochs = 30;
    std::size_t batch_size = 32;
    std::size_t window_size = 30;
    std::size_t stride = 15;
    std::size_t max_sequences = 100;

    double learning_rate = 0.001;
    double coupling_strength = 0.10;
    double diffusion_rate = 0.05;
    double decay_rate = 0.01;
    double quantum_noise = 0.0005;

    bool show_help = false;
};

Args parseArgs(int argc, char* argv[]) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            args.show_help = true;
            return args;
        }

        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for: " + arg);
        }

        std::string value = argv[i + 1];

        if (arg == "--data") {
            args.data_path = value;
            ++i;
        } else if (arg == "--checkpoints") {
            args.checkpoint_dir = value;
            ++i;
        } else if (arg == "--best") {
            args.best_model_path = value;
            ++i;
        } else if (arg == "--export") {
            args.export_dir = value;
            ++i;
        } else if (arg == "--epochs") {
            args.epochs = std::stoull(value);
            ++i;
        } else if (arg == "--batch") {
            args.batch_size = std::stoull(value);
            ++i;
        } else if (arg == "--window") {
            args.window_size = std::stoull(value);
            ++i;
        } else if (arg == "--stride") {
            args.stride = std::stoull(value);
            ++i;
        } else if (arg == "--max-sequences") {
            args.max_sequences = std::stoull(value);
            ++i;
        } else if (arg == "--lr") {
            args.learning_rate = std::stod(value);
            ++i;
        } else if (arg == "--coupling") {
            args.coupling_strength = std::stod(value);
            ++i;
        } else if (arg == "--diffusion") {
            args.diffusion_rate = std::stod(value);
            ++i;
        } else if (arg == "--decay") {
            args.decay_rate = std::stod(value);
            ++i;
        } else if (arg == "--noise") {
            args.quantum_noise = std::stod(value);
            ++i;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    return args;
}

int main(int argc, char* argv[]) {
    std::cout << "\n========================================\n";
    std::cout << "QESN-MABe V2\n";
    std::cout << "Quantum Energy State Network\n";
    std::cout << "========================================\n\n";
    std::cout << "Author: Francisco Angulo de Lafuente\n";
    std::cout << "GitHub: https://github.com/Agnuxo1\n";
    std::cout << "License: MIT\n\n";

    try {
        Args args = parseArgs(argc, argv);

        if (args.show_help) {
            printUsage(argv[0]);
            return 0;
        }

        if (args.data_path.empty()) {
            std::cerr << "Error: --data is required\n\n";
            printUsage(argv[0]);
            return 1;
        }

        // Print config
        std::cout << "Configuration:\n";
        std::cout << "  Data: " << args.data_path << "\n";
        std::cout << "  Max sequences: " << args.max_sequences << "\n";
        std::cout << "  Epochs: " << args.epochs << "\n";
        std::cout << "  Batch: " << args.batch_size << "\n";
        std::cout << "  Window: " << args.window_size << " frames\n";
        std::cout << "  Stride: " << args.stride << " frames\n";
        std::cout << "  Learning rate: " << args.learning_rate << "\n\n";

        // Load dataset
        std::cout << "Loading dataset...\n";
        auto loader = std::make_shared<QESN::MABeDatasetLoader>();
        loader->loadDataset(args.data_path, args.max_sequences);
        std::cout << "\n";

        // Setup hyperparameters
        QESN::HyperParams params;
        params.epochs = args.epochs;
        params.batch_size = args.batch_size;
        params.window_size = args.window_size;
        params.stride = args.stride;
        params.learning_rate = args.learning_rate;
        params.coupling_strength = args.coupling_strength;
        params.diffusion_rate = args.diffusion_rate;
        params.decay_rate = args.decay_rate;
        params.quantum_noise = args.quantum_noise;

        // Create trainer
        QESN::QESNTrainer trainer(params, loader);

        // Train
        std::cout << "Starting training...\n\n";
        trainer.train(args.checkpoint_dir, args.best_model_path, args.export_dir);

        std::cout << "\n========================================\n";
        std::cout << "Training complete!\n";
        std::cout << "========================================\n";
        std::cout << "Checkpoints: " << args.checkpoint_dir << "\n";
        std::cout << "Best model: " << args.best_model_path << "\n";
        std::cout << "Export: " << args.export_dir << "\n\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << "\n\n";
        return 1;
    }
}
