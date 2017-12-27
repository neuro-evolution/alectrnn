#ifndef NN_ACTIVATOR_H_
#define NN_ACTIVATOR_H_

// We really want to shuttle neuron activation function stuff here...
// including parameters

// conv will have an output buff automatically
// need to include buffer for ctrnn update as well
// layer should only have state

#include <vector>
#include <cstddef>
#include <cassert>
#include "multi_array.hpp"
#include "utilities.hpp"

namespace nervous_system {

enum ACTIVATOR_TYPE {
  BASE,
  IDENTITY,
  CTRNN,
  IAF
};

template<typename TReal>
class Activator {
  public:
    Activator() {
      activator_type_ = ACTIVATOR_TYPE.BASE;
      parameter_count_ = 0;
    }
    virtual ~Activator()=default;

    /*
     * This operator must take the host's state and perform the specified
     * state updates on that state.
     */
    virtual void operator()(multi_array::Tensor<TReal>& state, const multi_array::Tensor<TReal>& input_buffer)=0;

    /*
     * When given a set of parameters, it will use them to set the 
     * Must provide an assert statement that requires # given parameters
     * to == parameter_count
     */
    virtual void Configure(const multi_array::ConstArraySlice<TReal>& parameters)=0; // Just needs to set base_ in ConstArraySlice
    
    std::size_t GetParameterCount() const {
      return parameter_count_;
    }

    ACTIVATOR_TYPE GetActivatorType() const {
      return activator_type_;
    }

  protected:
    ACTIVATOR_TYPE activator_type_;
    std::size_t parameter_count_;
};

template<typename TReal>
class IdentityActivator : public Activator<TReal> {
  public:
    Identity() {
      parameter_count_ = 0;
      activator_type_ = ACTIVATOR_TYPE.IDENTITY;
    }

    ~Identity()=default;

    void operator()(multi_array::Tensor<TReal>& state, const multi_array::Tensor<TReal>& input_buffer) {
      for (Index iii = 0; iii < state.size(); iii++) {
        state[iii] = input_buffer[iii];
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
    }
};

template<typename TReal>
class CTRNNActivator : public Activator<TReal> {
  public:
    typedef Index std::size_t;
    CTRNNActivator(std::size_t num_states) : num_states_(num_states) {
      // for step_size[1], bias[N] and rtau[N]
      parameter_count_ = 1 + num_states * 2;
      activator_type_ = ACTIVATOR_TYPE.CTRNN;
    }

    ~CTRNNActivator()=default;

    void operator()(multi_array::Tensor<TReal>& state, const multi_array::Tensor<TReal>& input_buffer) {
      for (Index iii = 0; iii < num_states_; iii++) {
        state[iii] += step_size_ * rtaus[iii] * 
            (-state[iii] + utilities::sigmoid(biases[iii] + input_buffer[iii]));
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      assert(parameters.size() == parameter_count_);
      biases_ = multi_array::ConstArraySlice<TReal>(parameters.data(), 
                parameters.start(), num_states, parameters.stride());
      rtaus_ = multi_array::ConstArraySlice<TReal>(parameters.data(), 
                parameters.start() + num_states, num_states, 
                parameters.stride());
      step_size_ = parameters[parameters.size()-1];
    }

  protected:
    multi_array::ConstArraySlice<TReal> biases;
    multi_array::ConstArraySlice<TReal> rtaus;
    TReal step_size_;
    std::size_t num_states_;
};

// Spiking update functor
template<typename TReal>
class IafActivator : public Activator<TReal> {
  public:
    typedef Index std::size_t;

    IafActivator(std::size_t num_states) : num_states_(num_states) {
      // for step_size[1], bias[N] and rtau[N]
      parameter_count_ = 1 + num_states * 2;
      activator_type_ = ACTIVATOR_TYPE.IAF;
    }

    ~IafActivator()=default;

    void operator()(multi_array::Tensor<TReal>& state, const multi_array::Tensor<TReal>& input_buffer) {

    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {

    }

  protected:
    multi_array::ConstArraySlice<TReal> range_;
    multi_array::ConstArraySlice<TReal> rtaus_;
    multi_array::ConstArraySlice<TReal> reset_;
    // Need Peak for value at burst
    // Need memory vector that holds last time for reset
    // Need reset time parameter
    TReal step_size_;
    std::size_t num_states_;
};

} // End nervous_system namespace

#endif /* NN_ACTIVATOR_H_ */