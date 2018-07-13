
#ifndef NN_NERVOUS_SYSTEM_H_
#define NN_NERVOUS_SYSTEM_H_

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <initializer_list>
#include "layer.hpp"
#include "multi_array.hpp"
#include "parameter_types.hpp"

namespace nervous_system {

/*
 * NervousSystem takes ownership of the layers it is given.
 */
template<typename TReal>
class NervousSystem {
  public:
    typedef std::size_t Index ;

    NervousSystem(const std::vector<Index>& input_shape) : parameter_count_(0) {
      network_layers_.push_back(new InputLayer<TReal>(input_shape));
    }

    // If NervousSystem is owner, it needs this destructor
    ~NervousSystem() {
      for (auto obj_ptr = network_layers_.begin(); obj_ptr != network_layers_.end(); ++obj_ptr) {
        delete *obj_ptr;
      }
    }

    void Step() {
      for (Index iii = 1; iii < network_layers_.size(); ++iii) {
        (*network_layers_[iii])(network_layers_[iii-1]);
      }
    }
    
    void Reset() {
      for (auto iter = network_layers_.begin(); iter != network_layers_.end(); ++iter) {
        (*iter)->Reset();
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      if (parameters.size() != parameter_count_) {
        throw std::invalid_argument("NervousSystem received parameters with"
                                    " the wrong number of elements");
      }
      Index slice_start(0);
      for (auto layer_ptr = network_layers_.begin()+1; 
          layer_ptr != network_layers_.end(); ++layer_ptr) {
        (*layer_ptr)->Configure(parameters.slice(
          slice_start, (*layer_ptr)->GetParameterCount()));
        slice_start += parameters.stride() * (*layer_ptr)->GetParameterCount();
      }
    }

    /*
     * First layer is the input layer, its states are set here
     */
    template<typename T>
    void SetInput(const std::vector<T>& inputs) {
      for (Index iii = 0; iii < network_layers_[0]->NumNeurons(); iii++)
      {
        network_layers_[0]->SetNeuronState(iii, inputs[iii]);
      }
    }

    /*
     * Get state for a layer
     */
    multi_array::Tensor<TReal>& GetLayerState(const Index layer) {
      return network_layers_[layer]->state();
    }

    /*
     * It is assumed that the last layer is the output of the network
     * Its states will be returned as a const Tensor reference.
     */
    const multi_array::Tensor<TReal>& GetOutput() const {
      return network_layers_[network_layers_.size()-1]->state();
    }

    /* Pushes a layer onto the neural network, taking ownership and incrementing
     * the parameter count */
    void AddLayer(Layer<TReal>* layer) {
      network_layers_.push_back(layer);
      parameter_count_ += layer->GetParameterCount();
    }

    std::size_t GetParameterCount() const {
      return parameter_count_;
    }

    /*
     * Returns a coding for the parameters and what order they are expected
     */
    std::vector<PARAMETER_TYPE> GetParameterLayout() const {
      std::vector<PARAMETER_TYPE> layout(parameter_count_);
      Index parameter_id = 0;
      for (auto layer_ptr = network_layers_.begin()+1; 
          layer_ptr != network_layers_.end(); ++layer_ptr) {
        std::vector<PARAMETER_TYPE> layer_layout = (*layer_ptr)->GetParameterLayout();
        for (auto layout_ptr = layer_layout.begin();
            layout_ptr != layer_layout.end(); ++layout_ptr) {
          layout[parameter_id] = *layout_ptr;
          ++parameter_id;
        }
      }
      return layout;
    }

    /*
     * Loops through each layer and has it construct its portion of the weight
     * normalization factor vector, then writes those elements onto the
     * vector that will be returned.
     */
    std::vector<float> GetWeightNormalizationFactors() const {
      std::vector<float> normalization_factors(parameter_count_);

      Index parameter_id = 0;
      for (auto layer_prt : network_layers_) {
        std::vector<float> layer_factors = layer_prt->GetWeightNormalizationFactors();
        for (auto factor : layer_factors) {
          normalization_factors[parameter_id] = factor;
          ++parameter_id;
        }
      }

      return normalization_factors;
    }

    const Layer<TReal>& operator[](Index index) const {
      return *network_layers_[index];
    }

    Index size() const {
      return network_layers_.size();
    }

  protected:
    std::size_t parameter_count_;
    std::vector< Layer<TReal>* > network_layers_;
};

} // End nervous_system namespace

#endif /* NN_NERVOUS_SYSTEM_H_ */