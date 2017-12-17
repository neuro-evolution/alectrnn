#ifndef NN_ACTIVATOR_H_
#define NN_ACTIVATOR_H_

// We really want to shuttle neuron activation function stuff here...
// including parameters

// conv will have an output buff automatically
// need to include buffer for ctrnn update as well
// layer should only have state

namespace nervous_system {

enum ACTIVATOR_TYPE {
  BASE,
  IDENTITY,
  CTRNN,
  SPIKE
};

// Abstract update_functor

// Identity update

// ctrnn update functor

// Spiking update functor

} // End hybrid namespace

#endif /* NN_ACTIVATOR_H_ */