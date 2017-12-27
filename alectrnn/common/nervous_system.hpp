
#ifndef NN_NERVOUS_SYSTEM_H_
#define NN_NERVOUS_SYSTEM_H_

namespace nervous_system {

#include <cstddef>
#include <vector>
#include "layer.hpp"

template<typename TReal>
class NervousSystem {
  public:
    typedef Index std::size_t;

    NervousSystem(); // During construction, don't let user pick input layer, just input dims, always make input layer
    ~NervousSystem();
    void Step() {
      // Loop through all layers following input (iii=1) (input is set separately) (or it chould be in step set)
        // At (i+1)th layer, use Layer(ith-Layer) to update state of Layer using ith-layer inputs
          // Layer will automatically get layer's state, and it will automatically call integrators and then activator
    }
    void Reset();

    template<typename T>
    void SetInput(const std::vector<T>& inputs) {
      for (Index iii = 0; iii < network_layers_[0].NumNeurons(); iii++)
      {
        network_layers_[0].SetNeuronState(iii, inputs[iii]);
      }
    }  

  protected:
    std::vector< Layer<TReal> > network_layers_;

};

} // End nervous_system namespace

#endif /* NN_NERVOUS_SYSTEM_H_ */