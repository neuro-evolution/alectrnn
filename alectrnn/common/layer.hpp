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
  RESERVOIR, // Difference between REC/RES, is that RES has default weights, not drawn from params
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
    virtual Index NumDimensions() const;
    
    /*
     * Update neuron state. Calls both integrator and activator.
     */
    virtual void operator()(const Layer<TReal>&);

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

    std::size_t GetNumParams() const {
      return parameter_count_;
    }

  protected:
    LAYER_TYPE layer_type_;
    Activator* activation_function_;
    Integrator* integration_function_;
    std::size_t parameter_count_;
};

template<typename TReal, std::size_t NumDim>
class InputLayer : public Layer<TReal> {

  protected:
    MultiArray<TReal, NumDim>* state_;
};

template<typename TReal, std::size_t NumDim>
class ConvLayer : public Layer<TReal> {

  protected:
    MultiArray<TReal, NumDim>* state_;
};

template<typename TReal, std::size_t NumDim>
class RecurrentLayer : public Layer<TReal> {

  protected:
    MultiArray<TReal, NumDim>* state_;
};

template<typename TReal, std::size_t NumDim>
class ReservoirLayer : public Layer<TReal> {

  protected:
    MultiArray<TReal, NumDim>* state_;
};

template<typename TReal>
class MotorLayer : public Layer<TReal> {

  protected:
    MultiArray<TReal, 1>* state_;
};

} // End hybrid namespace

#endif /* NN_LAYER_H_ */