#include "core/quantum_neuron.h"

#include <algorithm>
#include <numeric>

namespace QESN {

std::mt19937 QuantumNeuron::rng_{std::random_device{}()};
std::uniform_real_distribution<double> QuantumNeuron::uniform_dist_{0.0, 1.0};

QuantumNeuron::QuantumNeuron()
    : quantum_state_{Complex(1.0, 0.0), Complex(0.0, 0.0)},
      energy_level_(0.0),
      phase_(0.0),
      coherence_(1.0),
      history_index_(0),
      position_(0.0, 0.0),
      entanglement_strength_(0.0) {
    energy_history_.fill(0.0);
}

QuantumNeuron::QuantumNeuron(const Position2D& position) : QuantumNeuron() {
    position_ = position;
}

void QuantumNeuron::evolve(double dt, const std::vector<QuantumNeuron*>& neighbours) {
    Complex neighbour_influence{0.0, 0.0};

    for (const auto* neighbour : neighbours) {
        if (!neighbour) {
            continue;
        }
        double distance = position_.distance(neighbour->position_);
        if (distance < 1e-6) {
            continue;
        }
        double coupling = std::exp(-distance) * neighbour->energy_level_;
        double phase_delta = neighbour->phase_ - phase_;
        neighbour_influence += Complex(std::cos(phase_delta), std::sin(phase_delta)) * coupling;
    }

    Complex self_phase = std::exp(-I * phase_ * dt);
    for (auto& amplitude : quantum_state_) {
        amplitude *= self_phase;
        amplitude += neighbour_influence * dt * 0.05;
    }

    normaliseState();
    updateMemory();
}

void QuantumNeuron::applyUnitary(const std::array<std::array<Complex, 2>, 2>& unitary) {
    std::array<Complex, 2> new_state{};
    for (int i = 0; i < 2; ++i) {
        new_state[i] = unitary[i][0] * quantum_state_[0] + unitary[i][1] * quantum_state_[1];
    }
    quantum_state_ = new_state;
    normaliseState();
}

void QuantumNeuron::injectEnergy(double amount) {
    energy_level_ = std::clamp(energy_level_ + amount, 0.0, 1.0);
    coherence_ = std::clamp(coherence_ + amount * 0.2, 0.0, 1.0);
    updateMemory();
}

void QuantumNeuron::addPhase(double phase_delta) {
    phase_ += phase_delta;
    if (phase_ > 2.0 * M_PI) {
        phase_ = std::fmod(phase_, 2.0 * M_PI);
    }
    if (phase_ < 0.0) {
        phase_ += 2.0 * M_PI;
    }
}

void QuantumNeuron::decay(double rate) {
    energy_level_ *= (1.0 - rate);
    coherence_ *= (1.0 - rate * 0.5);
    if (energy_level_ < 1e-9) {
        energy_level_ = 0.0;
    }
    updateMemory();
}

double QuantumNeuron::transferEnergy(QuantumNeuron& neighbour, double coupling_strength) {
    double energy_difference = energy_level_ - neighbour.energy_level_;
    double transfer = energy_difference * coupling_strength;
    transfer = std::clamp(transfer, -energy_level_, energy_level_);

    energy_level_ -= transfer;
    neighbour.energy_level_ += transfer;
    energy_level_ = std::clamp(energy_level_, 0.0, 1.0);
    neighbour.energy_level_ = std::clamp(neighbour.energy_level_, 0.0, 1.0);
    return transfer;
}

void QuantumNeuron::diffuseEnergy(const std::vector<QuantumNeuron*>& neighbours, double diffusion_rate) {
    if (neighbours.empty()) {
        return;
    }

    double average_energy = 0.0;
    int counted = 0;
    for (const auto* neighbour : neighbours) {
        if (!neighbour) {
            continue;
        }
        average_energy += neighbour->energy_level_;
        ++counted;
    }

    if (counted == 0) {
        return;
    }

    average_energy /= static_cast<double>(counted);
    double delta = (average_energy - energy_level_) * diffusion_rate;
    energy_level_ += delta;
    energy_level_ = std::clamp(energy_level_, 0.0, 1.0);
}

void QuantumNeuron::entangleWith(QuantumNeuron& partner, double strength) {
    entangled_partners_.push_back(reinterpret_cast<std::size_t>(&partner));
    partner.entangled_partners_.push_back(reinterpret_cast<std::size_t>(this));

    entanglement_strength_ = strength;
    partner.entanglement_strength_ = strength;
    createSuperposition();
    partner.createSuperposition();
}

void QuantumNeuron::applyDecoherence(double rate) {
    coherence_ *= (1.0 - rate);
    coherence_ = std::clamp(coherence_, 0.0, 1.0);
    double shrink = 1.0 - rate;
    for (auto& amplitude : quantum_state_) {
        amplitude *= shrink;
    }
    normaliseState();
}

void QuantumNeuron::createSuperposition() {
    constexpr double inv_sqrt2 = 0.70710678118654752440;
    quantum_state_[0] = Complex(inv_sqrt2, 0.0);
    quantum_state_[1] = Complex(inv_sqrt2, 0.0);
}

SpinState QuantumNeuron::collapseState() {
    double p_up = probability(SPIN_UP);
    double sample = uniform_dist_(rng_);
    SpinState result = sample < p_up ? SPIN_UP : SPIN_DOWN;
    if (result == SPIN_UP) {
        quantum_state_[0] = Complex(1.0, 0.0);
        quantum_state_[1] = Complex(0.0, 0.0);
    } else {
        quantum_state_[0] = Complex(0.0, 0.0);
        quantum_state_[1] = Complex(1.0, 0.0);
    }
    return result;
}

void QuantumNeuron::setPhase(double phase) {
    phase_ = phase;
    if (phase_ > 2.0 * M_PI || phase_ < 0.0) {
        phase_ = std::fmod(phase_, 2.0 * M_PI);
        if (phase_ < 0.0) {
            phase_ += 2.0 * M_PI;
        }
    }
}

void QuantumNeuron::updateMemory() {
    energy_history_[history_index_] = energy_level_;
    history_index_ = (history_index_ + 1) % MEMORY_SIZE;
}

double QuantumNeuron::averageEnergyHistory() const {
    return std::accumulate(energy_history_.begin(), energy_history_.end(), 0.0) /
           static_cast<double>(MEMORY_SIZE);
}

void QuantumNeuron::addConnection(std::size_t neuron_id, double weight) {
    connections_.push_back(neuron_id);
    connection_weights_.push_back(weight);
}

void QuantumNeuron::removeConnection(std::size_t neuron_id) {
    auto it = std::find(connections_.begin(), connections_.end(), neuron_id);
    if (it == connections_.end()) {
        return;
    }
    std::size_t index = static_cast<std::size_t>(std::distance(connections_.begin(), it));
    connections_.erase(it);
    connection_weights_.erase(connection_weights_.begin() + static_cast<std::ptrdiff_t>(index));
}

void QuantumNeuron::serialize(std::ofstream& file) const {
    for (const auto& amplitude : quantum_state_) {
        double real = amplitude.real();
        double imag = amplitude.imag();
        file.write(reinterpret_cast<const char*>(&real), sizeof(double));
        file.write(reinterpret_cast<const char*>(&imag), sizeof(double));
    }

    file.write(reinterpret_cast<const char*>(&energy_level_), sizeof(double));
    file.write(reinterpret_cast<const char*>(&phase_), sizeof(double));
    file.write(reinterpret_cast<const char*>(&coherence_), sizeof(double));

    file.write(reinterpret_cast<const char*>(&position_.x), sizeof(double));
    file.write(reinterpret_cast<const char*>(&position_.y), sizeof(double));

    std::size_t connection_count = connections_.size();
    file.write(reinterpret_cast<const char*>(&connection_count), sizeof(std::size_t));
    for (std::size_t i = 0; i < connection_count; ++i) {
        file.write(reinterpret_cast<const char*>(&connections_[i]), sizeof(std::size_t));
        file.write(reinterpret_cast<const char*>(&connection_weights_[i]), sizeof(double));
    }
}

void QuantumNeuron::deserialize(std::ifstream& file) {
    for (auto& amplitude : quantum_state_) {
        double real = 0.0;
        double imag = 0.0;
        file.read(reinterpret_cast<char*>(&real), sizeof(double));
        file.read(reinterpret_cast<char*>(&imag), sizeof(double));
        amplitude = Complex(real, imag);
    }

    file.read(reinterpret_cast<char*>(&energy_level_), sizeof(double));
    file.read(reinterpret_cast<char*>(&phase_), sizeof(double));
    file.read(reinterpret_cast<char*>(&coherence_), sizeof(double));

    file.read(reinterpret_cast<char*>(&position_.x), sizeof(double));
    file.read(reinterpret_cast<char*>(&position_.y), sizeof(double));

    std::size_t connection_count = 0;
    file.read(reinterpret_cast<char*>(&connection_count), sizeof(std::size_t));
    connections_.resize(connection_count);
    connection_weights_.resize(connection_count);
    for (std::size_t i = 0; i < connection_count; ++i) {
        file.read(reinterpret_cast<char*>(&connections_[i]), sizeof(std::size_t));
        file.read(reinterpret_cast<char*>(&connection_weights_[i]), sizeof(double));
    }
    normaliseState();
}

void QuantumNeuron::reset() {
    quantum_state_[0] = Complex(1.0, 0.0);
    quantum_state_[1] = Complex(0.0, 0.0);
    energy_level_ = 0.0;
    phase_ = 0.0;
    coherence_ = 1.0;
    energy_history_.fill(0.0);
    history_index_ = 0;
    connections_.clear();
    connection_weights_.clear();
    entangled_partners_.clear();
    entanglement_strength_ = 0.0;
}

void QuantumNeuron::normaliseState() {
    double norm = std::sqrt(std::norm(quantum_state_[0]) + std::norm(quantum_state_[1]));
    if (norm < 1e-12) {
        quantum_state_[0] = Complex(1.0, 0.0);
        quantum_state_[1] = Complex(0.0, 0.0);
        return;
    }
    double inv_norm = 1.0 / norm;
    for (auto& amplitude : quantum_state_) {
        amplitude *= inv_norm;
    }
}

double QuantumNeuron::probability(SpinState state) const {
    return std::norm(quantum_state_[state]);
}

} // namespace QESN
