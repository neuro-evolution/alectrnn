#ifndef NN_ACTIVATOR_H_
#define NN_ACTIVATOR_H_

// We really want to shuttle neuron activation function stuff here...
// including parameters

// conv will have an output buff automatically
// need to include buffer for ctrnn update as well
// layer should only have state

#include <vector>
#include <cstddef>
#include "multi_array.hpp"

namespace nervous_system {

enum ACTIVATOR_TYPE {
  BASE,
  IDENTITY,
  CTRNN,
  SPIKE
};

template<typename TReal>
class Activator {
  public:
    Activator();
    virtual ~Activator();
    virtual TReal operator()();
    virtual void Configure(); // Just needs to set base_ in ArraySlice
    virtual std::size_t GetParameterCount();

    ACTIVATOR_TYPE GetActivatorType() const {
      return activator_type_;
    }

  protected:
    ACTIVATOR_TYPE activator_type_;
    std::vector<multi_array::ArraySlice> parameter_slices_;
    std::size_t parameter_count_;
};

template<typename TReal>
class IdentityActivator : public Activator<TReal> {
  public:
    Identity();
    ~Identity();

    TReal operator()();
    void Configure();
    std::size_t GetParameterCount();

  protected:

};

template<typename TReal>
class Sigmoid : public Activator<TReal> {
  public:
    Sigmoid();
    ~Sigmoid();

    TReal operator()();
    void Configure();
    std::size_t GetParameterCount();

  protected:

// ctrnn update functor ///... diff params for each neuron require knowing # of neurons, so layer deal with? 
    // Set Bias
    // Set Gain
    // Set Tau
    // float step_size_;
    // float epsilon_;
};

// Spiking update functor
template<typename TReal>
class Iaf : public Activator<TReal> {
  public:
    Iaf();
    ~Iaf();

    TReal operator()();
    void Configure();
    std::size_t GetParameterCount();

  protected:
};

} // End hybrid namespace

#endif /* NN_ACTIVATOR_H_ */