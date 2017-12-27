#ifndef NN_LAYER_H_
#define NN_LAYER_H_

#include <vector>
#include <cstddef>
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

template<typename TReal>
class Layer {
  public:
    typedef std::size_t Index;

    Layer();
    virtual ~Layer();

    /*
     * Should return the number of dimensions which derives from derived type.
     */
    virtual Index NumDimensions() const=0;
    
    /*
     * Update neuron state. Calls both integrator and activator.
     */
    virtual void operator()(const Layer<TReal>&)=0;

    /*
     * Passes the needed parameters to the Layer - Should be Slice with 
     * parameter_count_ in size. Layer will then make and assign Slices to
     * The activation_function and Integrator function
     */
    void Configure(const multi_array::ArraySlice<TReal>& parameters) {
      // Need to make them at first... but would like to just use (data) and pass the change down
    }

    LAYER_TYPE GetLayerType() const {
      return layer_type_;
    }

    std::size_t GetParameterCount() const {
      return parameter_count_;
    }

    std::size_t NumNeurons() const {
      return layer_state_.size();
    }

    template<typename T>
    void SetNeuronState(Index neuron, T value) {
      layer_state_[neuron] = static_cast<TReal>(value);
    }

  protected:
    LAYER_TYPE layer_type_;
    multi_array::Tensor<TReal> layer_state_;
    multi_array::Tensor<TReal> input_buffer_;
    Integrator& back_integrator_;
    Integrator& self_integrator_;
    Activator<TReal>& activation_function_;
    std::size_t parameter_count_;
};

template<typename TReal>
class InputLayer : public Layer<TReal> {

};

template<typename TReal>
class ConvLayer : public Layer<TReal> {

};

template<typename TReal>
class RecurrentLayer : public Layer<TReal> {

};

template<typename TReal>
class ReservoirLayer : public Layer<TReal> {

};

template<typename TReal>
class MotorLayer : public Layer<TReal> {

};

} // End nervous_system namespace

#endif /* NN_LAYER_H_ */