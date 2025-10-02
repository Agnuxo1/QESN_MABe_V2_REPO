#include "core/quantum_foam.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace QESN {

QuantumFoam2D::QuantumFoam2D()
    : width_(0),
      height_(0),
      coupling_strength_(0.12),
      diffusion_rate_(0.06),
      decay_rate_(0.015),
      quantum_noise_(0.0005) {}

QuantumFoam2D::QuantumFoam2D(std::size_t width, std::size_t height) : QuantumFoam2D() {
    initialise(width, height);
}

void QuantumFoam2D::initialise(std::size_t width, std::size_t height) {
    width_ = width;
    height_ = height;

    grid_.clear();
    grid_.resize(width_);
    for (std::size_t x = 0; x < width_; ++x) {
        grid_[x].clear();
        grid_[x].resize(height_);
        for (std::size_t y = 0; y < height_; ++y) {
            Position2D pos{static_cast<double>(x) / static_cast<double>(std::max<std::size_t>(1, width_ - 1)),
                           static_cast<double>(y) / static_cast<double>(std::max<std::size_t>(1, height_ - 1))};
            grid_[x][y] = std::make_unique<QuantumNeuron>(pos);
        }
    }

    energy_field_.assign(width_, std::vector<double>(height_, 0.0));
    updateEnergyField();
}

void QuantumFoam2D::reset() {
    for (auto& column : grid_) {
        for (auto& neuron : column) {
            neuron->reset();
        }
    }
    updateEnergyField();
}

QuantumNeuron& QuantumFoam2D::neuron(std::size_t x, std::size_t y) {
    return *grid_.at(x).at(y);
}

const QuantumNeuron& QuantumFoam2D::neuron(std::size_t x, std::size_t y) const {
    return *grid_.at(x).at(y);
}

std::vector<QuantumNeuron*> QuantumFoam2D::neighbours(std::size_t x, std::size_t y, std::size_t radius) {
    std::vector<QuantumNeuron*> result;
    if (width_ == 0 || height_ == 0) {
        return result;
    }

    std::ptrdiff_t ix = static_cast<std::ptrdiff_t>(x);
    std::ptrdiff_t iy = static_cast<std::ptrdiff_t>(y);
    std::ptrdiff_t r = static_cast<std::ptrdiff_t>(radius);

    for (std::ptrdiff_t dx = -r; dx <= r; ++dx) {
        for (std::ptrdiff_t dy = -r; dy <= r; ++dy) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            std::ptrdiff_t nx = ix + dx;
            std::ptrdiff_t ny = iy + dy;
            if (isValid(nx, ny)) {
                result.push_back(grid_[static_cast<std::size_t>(nx)][static_cast<std::size_t>(ny)].get());
            }
        }
    }
    return result;
}

void QuantumFoam2D::timeStep(double dt) {
    propagate(dt);
    applyDiffusion(dt);
    evolvePhases(dt);
    updateEnergyField();
}

void QuantumFoam2D::propagate(double dt) {
    if (width_ == 0 || height_ == 0) {
        return;
    }

    for (std::size_t x = 0; x < width_; ++x) {
        for (std::size_t y = 0; y < height_; ++y) {
            QuantumNeuron& current = *grid_[x][y];
            current.decay(decay_rate_ * dt);

            if (x + 1 < width_) {
                QuantumNeuron& right = *grid_[x + 1][y];
                current.transferEnergy(right, coupling_strength_ * dt);
            }
            if (y + 1 < height_) {
                QuantumNeuron& up = *grid_[x][y + 1];
                current.transferEnergy(up, coupling_strength_ * dt);
            }
        }
    }
}

void QuantumFoam2D::applyDiffusion(double dt) {
    if (width_ == 0 || height_ == 0) {
        return;
    }

    for (std::size_t x = 0; x < width_; ++x) {
        for (std::size_t y = 0; y < height_; ++y) {
            grid_[x][y]->diffuseEnergy(neighbours(x, y), diffusion_rate_ * dt);
        }
    }
}

void QuantumFoam2D::evolvePhases(double dt) {
    static std::mt19937 rng{std::random_device{}()};
    std::normal_distribution<double> noise{0.0, quantum_noise_};

    for (std::size_t x = 0; x < width_; ++x) {
        for (std::size_t y = 0; y < height_; ++y) {
            QuantumNeuron& q = *grid_[x][y];
            auto neigh = neighbours(x, y);
            q.evolve(dt, neigh);
            q.addPhase(noise(rng));
        }
    }
}

void QuantumFoam2D::injectEnergy(std::size_t x, std::size_t y, double energy) {
    if (!isValid(static_cast<std::ptrdiff_t>(x), static_cast<std::ptrdiff_t>(y))) {
        return;
    }
    grid_[x][y]->injectEnergy(energy);
    updateEnergyField();
}

void QuantumFoam2D::injectGaussian(double center_x, double center_y, double sigma, double amplitude) {
    if (width_ == 0 || height_ == 0) {
        return;
    }

    double sigma_sq = sigma * sigma;
    for (std::size_t x = 0; x < width_; ++x) {
        for (std::size_t y = 0; y < height_; ++y) {
            double dx = (static_cast<double>(x) + 0.5) - center_x;
            double dy = (static_cast<double>(y) + 0.5) - center_y;
            double distance_sq = dx * dx + dy * dy;
            double energy = amplitude * std::exp(-distance_sq / (2.0 * sigma_sq));
            if (energy > 1e-6) {
                grid_[x][y]->injectEnergy(energy);
            }
        }
    }
    updateEnergyField();
}

EnergyGrid QuantumFoam2D::energySnapshot() const {
    return energy_field_;
}

double QuantumFoam2D::totalEnergy() const {
    double total = 0.0;
    for (const auto& column : energy_field_) {
        total += std::accumulate(column.begin(), column.end(), 0.0);
    }
    return total;
}

double QuantumFoam2D::averageCoherence() const {
    if (width_ == 0 || height_ == 0) {
        return 0.0;
    }
    double total = 0.0;
    for (const auto& column : grid_) {
        for (const auto& neuron : column) {
            total += neuron->measureCoherence();
        }
    }
    return total / static_cast<double>(width_ * height_);
}

std::vector<double> QuantumFoam2D::observe() const {
    std::vector<double> observation;
    observation.reserve(width_ * height_);
    for (const auto& column : energy_field_) {
        observation.insert(observation.end(), column.begin(), column.end());
    }
    return observation;
}

std::vector<double> QuantumFoam2D::observeGaussian(std::size_t radius) const {
    if (width_ == 0 || height_ == 0) {
        return {};
    }
    std::vector<double> result;
    result.reserve(width_ * height_);
    const double sigma = static_cast<double>(radius) * 0.5 + 1.0;

    for (std::size_t x = 0; x < width_; ++x) {
        for (std::size_t y = 0; y < height_; ++y) {
            double weighted = 0.0;
            double norm = 0.0;
            for (std::ptrdiff_t dx = -static_cast<std::ptrdiff_t>(radius); dx <= static_cast<std::ptrdiff_t>(radius); ++dx) {
                for (std::ptrdiff_t dy = -static_cast<std::ptrdiff_t>(radius); dy <= static_cast<std::ptrdiff_t>(radius); ++dy) {
                    std::ptrdiff_t nx = static_cast<std::ptrdiff_t>(x) + dx;
                    std::ptrdiff_t ny = static_cast<std::ptrdiff_t>(y) + dy;
                    if (!isValid(nx, ny)) {
                        continue;
                    }
                    double distance_sq = static_cast<double>(dx * dx + dy * dy);
                    double weight = std::exp(-distance_sq / (2.0 * sigma * sigma));
                    weighted += energy_field_[static_cast<std::size_t>(nx)][static_cast<std::size_t>(ny)] * weight;
                    norm += weight;
                }
            }
            result.push_back(norm > 0.0 ? weighted / norm : 0.0);
        }
    }
    return result;
}

EnergyPattern2D QuantumFoam2D::detectPattern() const {
    if (width_ < 4 || height_ < 4) {
        return EnergyPattern2D::None;
    }

    double total = totalEnergy();
    if (total < 1e-3) {
        return EnergyPattern2D::None;
    }

    double center_energy = 0.0;
    std::size_t cx0 = width_ / 4;
    std::size_t cx1 = width_ - cx0;
    std::size_t cy0 = height_ / 4;
    std::size_t cy1 = height_ - cy0;
    for (std::size_t x = cx0; x < cx1; ++x) {
        for (std::size_t y = cy0; y < cy1; ++y) {
            center_energy += energy_field_[x][y];
        }
    }
    double center_ratio = center_energy / total;

    if (center_ratio > 0.45) {
        return EnergyPattern2D::Mount;
    }

    double horizontal_gradient = 0.0;
    for (std::size_t x = 1; x < width_; ++x) {
        for (std::size_t y = 0; y < height_; ++y) {
            double diff = energy_field_[x][y] - energy_field_[x - 1][y];
            horizontal_gradient += diff * diff;
        }
    }
    double vertical_gradient = 0.0;
    for (std::size_t y = 1; y < height_; ++y) {
        for (std::size_t x = 0; x < width_; ++x) {
            double diff = energy_field_[x][y] - energy_field_[x][y - 1];
            vertical_gradient += diff * diff;
        }
    }

    double gradient_ratio = (horizontal_gradient + vertical_gradient) / (total + 1e-6);
    if (gradient_ratio > 0.30) {
        return EnergyPattern2D::Attack;
    }

    if (center_ratio > 0.25) {
        return EnergyPattern2D::Investigation;
    }

    if (horizontal_gradient > vertical_gradient * 1.5 || vertical_gradient > horizontal_gradient * 1.5) {
        return EnergyPattern2D::Pursuit;
    }

    return EnergyPattern2D::None;
}

void QuantumFoam2D::serialize(std::ofstream& file) const {
    file.write(reinterpret_cast<const char*>(&width_), sizeof(std::size_t));
    file.write(reinterpret_cast<const char*>(&height_), sizeof(std::size_t));
    file.write(reinterpret_cast<const char*>(&coupling_strength_), sizeof(double));
    file.write(reinterpret_cast<const char*>(&diffusion_rate_), sizeof(double));
    file.write(reinterpret_cast<const char*>(&decay_rate_), sizeof(double));
    file.write(reinterpret_cast<const char*>(&quantum_noise_), sizeof(double));

    for (std::size_t x = 0; x < width_; ++x) {
        for (std::size_t y = 0; y < height_; ++y) {
            grid_[x][y]->serialize(file);
        }
    }
}

void QuantumFoam2D::deserialize(std::ifstream& file) {
    file.read(reinterpret_cast<char*>(&width_), sizeof(std::size_t));
    file.read(reinterpret_cast<char*>(&height_), sizeof(std::size_t));
    file.read(reinterpret_cast<char*>(&coupling_strength_), sizeof(double));
    file.read(reinterpret_cast<char*>(&diffusion_rate_), sizeof(double));
    file.read(reinterpret_cast<char*>(&decay_rate_), sizeof(double));
    file.read(reinterpret_cast<char*>(&quantum_noise_), sizeof(double));

    initialise(width_, height_);
    for (std::size_t x = 0; x < width_; ++x) {
        for (std::size_t y = 0; y < height_; ++y) {
            grid_[x][y]->deserialize(file);
        }
    }
    updateEnergyField();
}

bool QuantumFoam2D::isValid(std::ptrdiff_t x, std::ptrdiff_t y) const {
    return x >= 0 && y >= 0 && static_cast<std::size_t>(x) < width_ && static_cast<std::size_t>(y) < height_;
}

void QuantumFoam2D::updateEnergyField() {
    if (width_ == 0 || height_ == 0) {
        return;
    }
    for (std::size_t x = 0; x < width_; ++x) {
        for (std::size_t y = 0; y < height_; ++y) {
            energy_field_[x][y] = grid_[x][y]->measureEnergy();
        }
    }
}

} // namespace QESN
