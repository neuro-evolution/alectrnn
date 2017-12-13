#ifndef NN_LAYER_H_
#define NN_LAYER_H_

#include <vector>
#include <cstddef>
#include <initializer_list>

namespace hybrid {

enum LAYER_TYPE {
  BASE,
  CONV,
  REC,
  MOTOR
};

/* 
 * Abstract Layer class
 * Builds a contiguous array for states
 * Major axis is the first layer_dimension (which also specifies the dim of
 * the layer). Minor axis are all following dimensions.
 */

//////////// Make # dimensions be a type so that I can 
/////////// do template magic to overload ()
template <typename TReal>
class Layer {
  public:
    // Could take a vector called shape and use accumulate multiply for size
    Layer(std::initializer_list<std::size_t> dimensions) : layer_type_(BASE) {
      std::size_t size = 0;
      for (auto dim : dimensions) {
        layer_dimensions_.push_back(dim);
        size *= dim;
      }
      neuron_states_.resize(size);
    }

    virtual ~Layer();

    std::size_t size() const {
      return neuron_states_.size();
    }

    std::size_t GetNumDimensions() const {
      return layer_dimensions_.size();
    }

    const std::vector<std::size_t>& GetDimensions() const {
      return layer_dimensions_;
    }

    LAYER_TYPE GetLayerType() const {
      return layer_type_;
    }

    TReal& operator [](std::size_t index) {
      return neuron_states_[index];
    }

    const TReal& operator [](std::size_t index) const {
      return neuron_states_[index];
    }

    TReal& at(std::size_t index) {
      return neuron_states_.at(index);
    }

    const TReal& at(std::size_t index) const {
      return neuron_states_.at(index)
    }

  private:
    std::vector<TReal> neuron_states_;
    std::vector<std::size_t> layer_dimensions_;
    LAYER_TYPE layer_type_;
};

// Conv layer class
class ConvLayer : public Layer {

};

// RNN layer class
class RecurrentLayer : public Layer {

};

// Output layer class
class MotorLayer : public Layer {

};

// Input layer class???

} // End hybrid namespace

#endif /* NN_LAYER_H_ */