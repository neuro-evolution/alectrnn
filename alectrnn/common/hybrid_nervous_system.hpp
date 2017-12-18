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
    void Reset();

    // Something about padding -> goes into update but deteremines whether sizes work
    // need to check if sizes work and send error (or just assert fail?) if not

  protected:


};

} // End hybrid namespace

#endif /* NN_HYBRID_NERVOUS_SYSTEM_H_ */