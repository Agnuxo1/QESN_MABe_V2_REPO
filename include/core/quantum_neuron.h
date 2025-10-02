#pragma once

#include <complex>
#include <vector>
#include <array>
#include <random>
#include <fstream>
#include <cmath>

namespace QESN {

using Complex = std::complex<double>;
constexpr Complex I(0.0, 1.0);

struct Position2D {
    double x;
    double y;

    Position2D() : x(0.0), y(0.0) {}
    Position2D(double px, double py) : x(px), y(py) {}

    double distance(const Position2D& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        return std::sqrt(dx * dx + dy * dy);
    }
};

enum SpinState {
    SPIN_UP = 0,
    SPIN_DOWN = 1
};

class QuantumNeuron {
public:
    static constexpr std::size_t MEMORY_SIZE = 90;

    QuantumNeuron();
    explicit QuantumNeuron(const Position2D& position);

    void evolve(double dt, const std::vector<QuantumNeuron*>& neighbours);
    void applyUnitary(const std::array<std::array<Complex, 2>, 2>& unitary);

    void injectEnergy(double amount);
    void addPhase(double phase_delta);
    void decay(double rate);
    double transferEnergy(QuantumNeuron& neighbour, double coupling_strength);
    void diffuseEnergy(const std::vector<QuantumNeuron*>& neighbours, double diffusion_rate);

    void entangleWith(QuantumNeuron& partner, double strength);
    void applyDecoherence(double rate);
    void createSuperposition();

    SpinState collapseState();
    double measureEnergy() const { return energy_level_; }
    double measureCoherence() const { return coherence_; }
    Complex amplitude(SpinState state) const { return quantum_state_[state]; }

    void setPhase(double phase);
    void setPosition(const Position2D& position) { position_ = position; }
    Position2D position() const { return position_; }

    void updateMemory();
    double averageEnergyHistory() const;

    void addConnection(std::size_t neuron_id, double weight);
    void removeConnection(std::size_t neuron_id);
    const std::vector<std::size_t>& connections() const { return connections_; }
    const std::vector<double>& connectionWeights() const { return connection_weights_; }

    void serialize(std::ofstream& file) const;
    void deserialize(std::ifstream& file);

    void reset();
    void normaliseState();
    double probability(SpinState state) const;

private:
    std::array<Complex, 2> quantum_state_;
    double energy_level_;
    double phase_;
    double coherence_;

    std::array<double, MEMORY_SIZE> energy_history_;
    std::size_t history_index_;

    Position2D position_;
    std::vector<std::size_t> connections_;
    std::vector<double> connection_weights_;

    std::vector<std::size_t> entangled_partners_;
    double entanglement_strength_;

    static std::mt19937 rng_;
    static std::uniform_real_distribution<double> uniform_dist_;
};

} // namespace QESN
