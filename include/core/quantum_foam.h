#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <cstddef>

#include "quantum_neuron.h"

namespace QESN {

using EnergyGrid = std::vector<std::vector<double>>;

enum class EnergyPattern2D {
    None,
    Pursuit,
    Mount,
    Attack,
    Investigation
};

class QuantumFoam2D {
public:
    QuantumFoam2D();
    QuantumFoam2D(std::size_t width, std::size_t height);

    void initialise(std::size_t width, std::size_t height);
    void reset();

    std::size_t width() const { return width_; }
    std::size_t height() const { return height_; }

    QuantumNeuron& neuron(std::size_t x, std::size_t y);
    const QuantumNeuron& neuron(std::size_t x, std::size_t y) const;

    std::vector<QuantumNeuron*> neighbours(std::size_t x, std::size_t y, std::size_t radius = 1);

    void timeStep(double dt);
    void propagate(double dt);
    void applyDiffusion(double dt);
    void evolvePhases(double dt);

    void injectEnergy(std::size_t x, std::size_t y, double energy);
    void injectGaussian(double center_x, double center_y, double sigma, double amplitude);

    void setCouplingStrength(double strength) { coupling_strength_ = strength; }
    void setDiffusionRate(double rate) { diffusion_rate_ = rate; }
    void setDecayRate(double rate) { decay_rate_ = rate; }
    void setQuantumNoise(double noise) { quantum_noise_ = noise; }

    double couplingStrength() const { return coupling_strength_; }
    double diffusionRate() const { return diffusion_rate_; }
    double decayRate() const { return decay_rate_; }
    double quantumNoise() const { return quantum_noise_; }

    EnergyGrid energySnapshot() const;
    double totalEnergy() const;
    double averageCoherence() const;

    std::vector<double> observe() const;
    std::vector<double> observeGaussian(std::size_t radius) const;

    EnergyPattern2D detectPattern() const;

    void serialize(std::ofstream& file) const;
    void deserialize(std::ifstream& file);

private:
    std::vector<std::vector<std::unique_ptr<QuantumNeuron>>> grid_;
    std::size_t width_;
    std::size_t height_;

    double coupling_strength_;
    double diffusion_rate_;
    double decay_rate_;
    double quantum_noise_;

    EnergyGrid energy_field_;

    bool isValid(std::ptrdiff_t x, std::ptrdiff_t y) const;
    void updateEnergyField();
};

} // namespace QESN
