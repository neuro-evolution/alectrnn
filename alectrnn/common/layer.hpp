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
  REC,
  RES, // Difference between REC/RES, is that RES has default weights, not drawn from params
  MOTOR
};

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
     * Return a view into the states of the neurons
     */
    template<typename T, std::size_t NumDim>
    virtual ArrayView<T, NumDim> NeuronView();
    
    /*
     * Update neuron state. Calls both integrator and activator.
     */
    virtual void operator()(const Layer& layer);

    template<typename T>
    virtual void Configure(const T* parameters);///???????

    LAYER_TYPE GetLayerType() const {
      return layer_type_;
    }

  protected:
    LAYER_TYPE layer_type_;
    Activator* activation_function_;
    Integrator* integration_function_;
};

template<typename TReal, std::size_t NumDim>
class InputLayer : public Layer {

  protected:
    MultiArray<TReal, NumDim>* state_;
};

template<typename TReal, std::size_t NumDim>
class ConvLayer : public Layer {

  protected:
    MultiArray<TReal, NumDim>* state_;
};

template<typename TReal, std::size_t NumDim>
class RecurrentLayer : public Layer {

  protected:
    MultiArray<TReal, NumDim>* state_;
};

template<typename TReal, std::size_t NumDim>
class ReservoirLayer : public Layer {

  protected:
    MultiArray<TReal, NumDim>* state_;
};

template<typename TReal>
class MotorLayer : public Layer {

  protected:
    MultiArray<TReal, 1>* state_;
};

} // End hybrid namespace

#endif /* NN_LAYER_H_ */