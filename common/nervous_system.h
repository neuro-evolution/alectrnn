#ifndef ALECTRNN_COMMON_NERVOUS_SYSTEM_H_
#define ALECTRNN_COMMON_NERVOUS_SYSTEM_H_

#include <vector>
#include <cstddef>
#include "network_generator.h"

namespace ctrnn {

double sigmoid(double x);

class NeuralNetwork {
  public:
    NeuralNetwork(const std::vector<std::vector<InEdge>>& neuron_neighbors,
        const std::vector<std::vector<InEdge>>& neuron_sensors, 
        std::size_t num_sensors, double step_size);
    ~NeuralNetwork();

    void EulerStep();
    void setNeuronBias(std::size_t neuron, double bias);
    void setNeuronState(std::size_t neuron, double state);
    void setNeuronGain(std::size_t neuron, double gain);
    void setNeuronTau(std::size_t neuron, double tau);
    void setSensorState(std::size_t sensor, double state);
    void Reset();
    double getNeuronState(std::size_t neuron) const;
    double getNeuronOutput(std::size_t neuron) const;

  private:
    double step_size_;
    double epsilon_;
    std::size_t num_neurons_;
    std::vector<double> sensor_states_;
    std::vector<double> neuron_states_;
    std::vector<double> neuron_outputs_;
    std::vector<double> neuron_biases_;
    std::vector<double> neuron_gains_;
    std::vector<double> neuron_rtaus_;
    const std::vector<std::vector<InEdge>>& neuron_neighbors_;
    const std::vector<std::vector<InEdge>>& neuron_sensors_;
};

}

#endif /* ALECTRNN_COMMON_NERVOUS_SYSTEM_H_ */
