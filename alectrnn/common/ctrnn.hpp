/*
 * ctrnn.h
 *
 *  Created on: Sep 2, 2017
 *      Author: Nathaniel Rodriguez
 *
 * A class used to define a CTRNN neural network.
 * num_neurons_ = total number of neurons in network
 * num_sensor_neurons_ = the number of neurons that will recieve sensor input
 * number of sensor states determined from input vector
 */

#ifndef ALECTRNN_COMMON_CTRNN_H_
#define ALECTRNN_COMMON_CTRNN_H_

#include <vector>
#include <cstddef>
#include "network_constructor.hpp"

namespace ctrnn {

class NeuralNetwork {
  public:
    NeuralNetwork(const std::vector<std::vector<InEdge>>& neuron_neighbors,
        const std::vector<std::vector<InEdge>>& neuron_sensors, 
        std::size_t num_sensors, float step_size);
    ~NeuralNetwork();

    void EulerStep();
    void setNeuronBias(std::size_t neuron, float bias);
    void setNeuronState(std::size_t neuron, float state);
    void setNeuronGain(std::size_t neuron, float gain);
    void setNeuronTau(std::size_t neuron, float tau);
    void setSensorState(std::size_t sensor, float state);
    void Reset();
    float getNeuronState(std::size_t neuron) const;
    float getNeuronOutput(std::size_t neuron) const;

  private:
    float step_size_;
    float epsilon_;
    std::size_t num_sensor_neurons_;
    std::size_t num_neurons_;
    std::vector<float> sensor_states_;
    std::vector<float> neuron_states_;
    std::vector<float> neuron_outputs_;
    std::vector<float> neuron_biases_;
    std::vector<float> neuron_gains_;
    std::vector<float> neuron_rtaus_;
    const std::vector<std::vector<InEdge>>& neuron_neighbors_;
    const std::vector<std::vector<InEdge>>& neuron_sensors_;
};

}

#endif /* ALECTRNN_COMMON_CTRNN_H_ */
