/*
 * nervous_system.cpp
 *
 *  Created on: Sep 2, 2017
 *      Author: Nathaniel Rodriguez
 *
 * A class used to define a CTRNN neural network.
 *
 */

#include <vector>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include "nervous_system.h"
#include "network_generator.h"

namespace ctrnn {

double sigmoid(double x) {
  return 1 / (1 + std::exp(-x));
}

NeuralNetwork::NeuralNetwork(const std::vector<std::vector<InEdge>>& neuron_neighbors,
              const std::vector<std::vector<InEdge>>& neuron_sensors,
              std::size_t num_sensors, double step_size)
    : neuron_neighbors_(neuron_neighbors), neuron_sensors_(neuron_sensors) {

  step_size_ = step_size;
  epsilon_ = 0.000000001; // small value to add to tau so it is > 0
  num_neurons_ = neuron_neighbors.size();
  sensor_states_.resize(num_sensors);
  neuron_states_.resize(neuron_neighbors.size());
  neuron_outputs_.resize(neuron_neighbors.size());
  neuron_biases_.resize(neuron_neighbors.size());
  neuron_gains_.resize(neuron_neighbors.size());
  neuron_rtaus_.resize(neuron_neighbors.size());
}

NeuralNetwork::~NeuralNetwork() {
}

void NeuralNetwork::EulerStep() {
  // Update neuron states
  double input(0.0);
  for (std::size_t iii = 0; iii < num_neurons_; iii++) {
    input = 0.0;
    for (auto iter_neuron_source = neuron_neighbors_[iii].begin();
          iter_neuron_source != neuron_neighbors_[iii].end(); 
          ++iter_neuron_source) {
      input += iter_neuron_source->weight 
                * neuron_outputs_[iter_neuron_source->source];
    }
    for (auto iter_sensor_source = neuron_sensors_[iii].begin();
          iter_sensor_source != neuron_sensors_[iii].end(); 
          ++iter_sensor_source) {
      input += iter_sensor_source->weight 
                * sensor_states_[iter_sensor_source->source];
    }
    neuron_states_[iii] += step_size_ * neuron_rtaus_[iii] 
                            * (input - neuron_states_[iii]);
  }
  // Update neuron outputs
  for (std::size_t iii = 0; iii < num_neurons_; iii++) {
    neuron_outputs_[iii] = sigmoid(neuron_gains_[iii] 
                            * (neuron_states_[iii] * neuron_biases_[iii]));
  }
}

void NeuralNetwork::setNeuronBias(std::size_t neuron, double bias) {
  neuron_biases_[neuron] = bias;
}

void NeuralNetwork::setNeuronState(std::size_t neuron, double state) {
  neuron_states_[neuron] = state;
}

void NeuralNetwork::setNeuronGain(std::size_t neuron, double gain) {
  neuron_gains_[neuron] = gain;
}

void NeuralNetwork::setNeuronTau(std::size_t neuron, double tau) {
  if (tau < 0) {
    throw std::invalid_argument( "received negative time constant" );
  }
  else {
    neuron_rtaus_[neuron] = 1. / (tau + epsilon_);
  }
}

void NeuralNetwork::setSensorState(std::size_t sensor, double state) {
  sensor_states_[sensor] = state;
}

void NeuralNetwork::Reset() {
  for (auto iter_states = sensor_states_.begin();
      iter_states != sensor_states_.end(); ++iter_states) {
    *iter_states = 0.0;
  }
  for (auto iter_states = neuron_states_.begin();
      iter_states != sensor_states_.end(); ++iter_states) {
    *iter_states = 0.0;
  }
  for (auto iter_outputs = neuron_outputs_.begin();
      iter_outputs != neuron_outputs_.end(); ++iter_outputs) {
    *iter_outputs = 0.0;
  }
}

double NeuralNetwork::getNeuronState(std::size_t neuron) const {
  return neuron_states_[neuron];
}

double NeuralNetwork::getNeuronOutput(std::size_t neuron) const {
  return neuron_outputs_[neuron];
}

}
