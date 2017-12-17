/*
 * hybrid_nervous_system.cpp
 *
 *  Created on: Dec 5, 2017
 *      Author: Nathaniel Rodriguez
 *
 * A class used to define a neural network with CNN and RNN components.
 *
 */

#ifndef NN_HYBRID_NERVOUS_SYSTEM_H_
#define NN_HYBRID_NERVOUS_SYSTEM_H_

namespace nervous_system {

#include "nervous_system.hpp"

class HybridNeuralNetwork : public NervousSystem {
  public:
    HybridNeuralNetwork();//TBD
    ~HybridNeuralNetwork();

    void Step();
    // Set Bias
    // Set Gain
    // Set Tau

    void Reset();
    // Get state
    // Get output

    // Something about padding -> goes into update but deteremines whether sizes work
    // need to check if sizes work and send error if not

  protected:
    float step_size_;
    float epsilon_;

    // Not sure if copy params of just... idk ... might as well....?
    // std::vector<float> neuron_states_;
    // std::vector<float> neuron_outputs_;
    // std::vector<float> neuron_biases_;
    // std::vector<float> neuron_gains_;
    // std::vector<float> neuron_rtaus_;
};

} // End hybrid namespace

#endif /* NN_HYBRID_NERVOUS_SYSTEM_H_ */