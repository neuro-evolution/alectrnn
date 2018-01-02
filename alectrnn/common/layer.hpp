#ifndef NN_LAYER_H_
#define NN_LAYER_H_

#include <vector>
#include <cstddef>
#include <cassert>
#include <initializer_list>
#include "activator.hpp"
#include "integrator.hpp"
#include "multi_array.hpp"

namespace nervous_system {

enum LAYER_TYPE {
  BASE,
  CONV,
  RECURENT,
  RESERVOIR,
  MOTOR
};

/*
 * Layers take ownership of the integrators and activations functions they use.
 * This is necessary as the Layer will be the only remaining access point to 
 * those objects once it is pushed off to the nervous system.
 */
template<typename TReal>
class Layer {
  public:
    typedef std::size_t Index;

    Layer(Integrator<TReal>* back_integrator, 
          Integrator<TReal>* self_integrator,
          Activator<TReal>* activation_function) 
          : back_integrator_(back_integrator), 
            self_integrator_(self_integrator), 
            activation_function_(activation_function) {
      parameter_count_ = back_integrator_->GetParameterCount() 
                       + self_integrator_->GetParameterCount()
                       + activation_function_->GetParameterCount();
    }

    Layer(const std::vector<Index>& shape, 
          Integrator<TReal>* back_integrator, 
          Integrator<TReal>* self_integrator,
          Activator<TReal>* activation_function) 
          : Layer(back_integrator, self_integrator, activation_function),
          layer_state_(shape), input_buffer_(shape), shape_(shape) {
    }

    virtual ~Layer() {
      delete back_integrator_;
      delete self_integrator_;
      delete activation_function_;
    }

    /*
     * Update neuron state. Calls both integrator and activator.
     */
    virtual void operator()(const Layer<TReal>& prev_layer) {
      // First clear input buffer
      input_buffer_.Fill(0.0);

      // Call back integrator first to resolve input from prev layer
      back_integrator_->(prev_layer.state(), input_buffer_);

      // Resolve self-connections if there are any
      self_integrator_->(input_buffer_, input_buffer_);

      // Apply activation and update state
      activation_function_->(layer_state_, input_buffer_);
    }

    /*
     * Passes the needed parameters to the Layer - Should be Slice with 
     * parameter_count_ in size. Layer will then make and assign Slices to
     * The activation_function and Integrator function
     */
    virtual void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      assert(parameters.size() == parameter_count_);
      multi_array::ConstArraySlice<TReal>() back_slice(parameters.data(),
        parameters.start(), 
        back_integrator_->GetParameterCount(), 
        parameters.stride());
      back_integrator_->Configure(back_slice);

      multi_array::ConstArraySlice<TReal>() self_slice(back_slice.data(),
        back_slice.start() + back_slice.stride() * back_slice.size(),
        self_integrator_->GetParameterCount(),
        back_slice.stride());
      self_integrator_->Configure(self_slice);

      multi_array::ConstArraySlice<TReal>() activ_slice(self_slice.data(),
        self_slice.start() + self_slice.stride() * self_slice.size(),
        activation_function_->GetParameterCount(),
        self_slice.stride());
      activation_function_->Configure(activ_slice);
    }

    virtual void Reset() {
      layer_state_.Fill(0.0);
      input_buffer_.Fill(0.0);
      activation_function_->Reset();
    }

    std::size_t GetParameterCount() const {
      return parameter_count_;
    }

    std::size_t NumNeurons() const {
      return layer_state_.size();
    }

    template<typename T>
    void SetNeuronState(Index neuron, T value) {
      layer_state_[neuron] = value;
    }

    const multi_array::Tensor<TReal>& state() const {
      return layer_state_;
    }

    multi_array::Tensor<TReal>& state() {
      return layer_state_;
    }

    const std::vector<Index>& shape() const {
      return shape_;
    }

  protected:
    multi_array::Tensor<TReal> layer_state_;
    // holds input values used to update the layer's state
    multi_array::Tensor<TReal> input_buffer_;
    // calculates inputs from other layers and applies them to input buffer
    Integrator<TReal>* back_integrator_;
    // claculates inputs from neurons within the layer and applies them to input buffer
    Integrator<TReal>* self_integrator_;
    // updates the layer's state using the input buffer (may also contain internal state)
    Activator<TReal>* activation_function_;
    std::size_t parameter_count_;
    std::vector<Index> shape_;
};

template<typename TReal>
class InputLayer : public Layer<TReal> {
  public:
    InputLayer(const std::vector<Index>& shape) 
        : shape_(shape), layer_state_(shape) {
      back_integrator_ = nullptr;
      self_integrator_ = nullptr;
      activation_function_ = nullptr;
      parameter_count_ = 0;
    }

    InputLayer(const std::initializer_list<Index>& shape) 
        : shape_(shape), layer_state_(shape) {
      back_integrator_ = nullptr;
      self_integrator_ = nullptr;
      activation_function_ = nullptr;
      parameter_count_ = 0;
    }

    void operator()(const Layer<TReal>& prev_layer) {}

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {}

    void Reset() { layer_state_.Fill(0.0); }
};

template<typename TReal>
class ConvLayer : public Layer<TReal> {
  // only might need to make it easier to build... maybe...
  // since it could generate the activators and integrators inside
};

template<typename TReal>
class RecurrentLayer : public Layer<TReal> {

};

template<typename TReal>
class ReservoirLayer : public Layer<TReal> {

};

template<typename TReal>
class MotorLayer : public Layer<TReal> {
  public:
    typedef std::size_t Index;

    MotorLayer(Index num_outputs, Index num_inputs, 
        Activator<TReal>* activation_function) 
        : activation_function_(activation_function) {
      back_integrator_ = new MotorLayer(num_outputs, num_inputs);
      self_integrator_ = nullptr;
      parameter_count_ = activation_function_->GetParameterCount();
    }

    void operator()(const Layer<TReal>& prev_layer) {
      // First clear input buffer
      input_buffer_.Fill(0.0);
      // Call back integrator first to resolve input from prev layer
      back_integrator_->(prev_layer.state(), input_buffer_);
      // Apply activation and update state
      activation_function_->(layer_state_, input_buffer_);
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      assert(parameter_count_ == parameters.size());
      back_integrator_->Configure(parameters);
    }
};

} // End nervous_system namespace

#endif /* NN_LAYER_H_ */