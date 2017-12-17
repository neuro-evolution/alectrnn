/*
 * ctrnn.cpp
 *
 *  Created on: Sep 2, 2017
 *      Author: Nathaniel Rodriguez
 *
 * A class used to define a CTRNN neural network.
 * In Euler updates the first X neurons of the network are used as input
 * neurons. These correspond with the outer vector of the neuron sensors
 * (the inner vector being a vector of sensor weights).
 *
 */

#include <vector>
#include <cstddef>
#include <stdexcept>
#include "ctrnn.hpp"
#include "network_constructor.hpp"
#include "utilities.hpp"

namespace ctrnn {

NeuralNetwork::NeuralNetwork(
              const std::vector<std::vector<InEdge>>& neuron_neighbors,
              const std::vector<std::vector<InEdge>>& neuron_sensors,
              std::size_t num_sensors, float step_size)
    : neuron_neighbors_(neuron_neighbors), neuron_sensors_(neuron_sensors) {

  step_size_ = step_size;
  epsilon_ = 0.00001; // small value to add to tau so it is > 0
  num_sensor_neurons_ = neuron_sensors.size();
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
  float input(0.0);
  for (std::size_t iii = 0; iii < num_neurons_; iii++) {
    input = 0.0;
    for (auto iter_neuron_source = neuron_neighbors_[iii].begin();
          iter_neuron_source != neuron_neighbors_[iii].end(); 
          ++iter_neuron_source) {
      input += iter_neuron_source->weight 
                * neuron_outputs_[iter_neuron_source->source];
    }
    if (iii < num_sensor_neurons_) {
      for (auto iter_sensor_source = neuron_sensors_[iii].begin();
            iter_sensor_source != neuron_sensors_[iii].end();
            ++iter_sensor_source) {
        input += iter_sensor_source->weight
                  * sensor_states_[iter_sensor_source->source];
      }
    }
    neuron_states_[iii] += step_size_ * neuron_rtaus_[iii] 
                            * (input - neuron_states_[iii]);
    neuron_states_[iii] = utilities::BoundState(neuron_states_[iii]);
  }
  // Update neuron outputs
  for (std::size_t iii = 0; iii < num_neurons_; iii++) {
    neuron_outputs_[iii] = sigmoid(neuron_gains_[iii] 
                            * (neuron_states_[iii] + neuron_biases_[iii]));
    neuron_outputs_[iii] = utilities::BoundState(neuron_outputs_[iii]);
  }
}

void NeuralNetwork::setNeuronBias(std::size_t neuron, float bias) {
  neuron_biases_[neuron] = bias;
}

void NeuralNetwork::setNeuronState(std::size_t neuron, float state) {
  neuron_states_[neuron] = state;
}

void NeuralNetwork::setNeuronGain(std::size_t neuron, float gain) {
  neuron_gains_[neuron] = gain;
}

void NeuralNetwork::setNeuronTau(std::size_t neuron, float tau) {
  if (tau < 0.0) {
    throw std::invalid_argument( "received negative time constant" );
  }
  else {
    neuron_rtaus_[neuron] = 1. / (tau + epsilon_);
  }
}

void NeuralNetwork::setSensorState(std::size_t sensor, float state) {
  sensor_states_[sensor] = state;
}

void NeuralNetwork::Reset() {
  // Reset the sensor states to 0
  for (auto iter_states = sensor_states_.begin();
      iter_states != sensor_states_.end(); ++iter_states) {
    *iter_states = 0.0;
  }
  // Reset neurons states to 0
  for (auto iter_states = neuron_states_.begin();
      iter_states != neuron_states_.end(); ++iter_states) {
    *iter_states = 0.0;
  }
  // Reset neuron outputs to 0
  for (auto iter_outputs = neuron_outputs_.begin();
      iter_outputs != neuron_outputs_.end(); ++iter_outputs) {
    *iter_outputs = 0.0;
  }
}

float NeuralNetwork::getNeuronState(std::size_t neuron) const {
  return neuron_states_[neuron];
}

float NeuralNetwork::getNeuronOutput(std::size_t neuron) const {
  return neuron_outputs_[neuron];
}

}
