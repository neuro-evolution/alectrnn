
#ifndef NN_NERVOUS_SYSTEM_H_
#define NN_NERVOUS_SYSTEM_H_

namespace nervous_system {

#include <cstddef>
#include <cassert>
#include <vector>
#include <initializer_list>
#include "layer.hpp"
#include "multi_array.hpp"

/*
 * TODO:... figure out if python capsule knows if NervousSystem is using the pointers... I don't think it does
 * this could be a problem, becaues Idk if the capsule can be made to give up ownership. Maybe manual incref/decref
 * in destructor? needs to import c-api though :(
 * Maybe don't give capsule a.... destructor? THis could be dangerous if we want to keep the layers in python...
 * oh dear :(
 */
template<typename TReal>
class NervousSystem {
  public:
    typedef Index std::size_t;

    NervousSystem(const std::vector<Index>& input_shape) : parameter_count_(0) {
      network_layers_.push_back(new InputLayer<TReal>(input_shape));
    }

    ~NervousSystem() {
      for (auto::iterator obj_ptr = network_layers_.begin(); obj_ptr != network_layers_.end(); ++obj_ptr) {
        delete *obj_ptr;
      }
    }

    void Step() {
      for (Index iii = 1; iii < network_layers_.size(); ++iii) {
        network_layers_[iii]->(*(network_layers_[iii-1]));
      }
    }
    
    void Reset() {
      for (auto::iterator iter = network_layers_.begin(); iter != network_layers_.end(); ++iter) {
        iter->Reset();
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      assert(parameters.size() == parameter_count_);
      Index slice_start = parameters.start();
      for (auto::iterator layer_ptr = network_layers_.begin()+1; 
          layer_ptr != network_layers_.end(); ++layer_ptr) {
        layer_ptr->Configure(multi_array::ConstArraySlice<TReal>(
          parameters.data(),
          slice_start,
          layer_ptr->GetParameterCount(),
          parameters.stride()));
        slice_start += parameters.stride() * layer_ptr->GetParameterCount();
      }
    }

    template<typename T>
    void SetInput(const std::vector<T>& inputs) {
      for (Index iii = 0; iii < network_layers_[0].NumNeurons(); iii++)
      {
        network_layers_[0].SetNeuronState(iii, inputs[iii]);
      }
    }

    void AddLayer(Layer<TReal>* layer) {
      network_layers_.push_back(layer);
      parameter_count_ += layer->GetParameterCount();
    }

    std::size_t GetParameterCount() const {
      return parameter_count_;
    }

  protected:
    std::vector< Layer<TReal>* > network_layers_;
    parameter_count_;
};

} // End nervous_system namespace

#endif /* NN_NERVOUS_SYSTEM_H_ */