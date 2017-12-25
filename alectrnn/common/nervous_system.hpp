
#ifndef NN_NERVOUS_SYSTEM_H_
#define NN_NERVOUS_SYSTEM_H_

namespace nervous_system {

#include <cstddef>
#include <vector>
#include "layer.hpp"

template<typename TReal>
class NervousSystem {
  public:
    NervousSystem(); // During construction, don't let user pick input layer, just input dims, always make input layer
    virtual ~NervousSystem();
    virtual void Step() {
      // Loop through all layers following input (iii=1) (input is set separately) (or it chould be in step set)
        // At (i+1)th layer, use Layer(ith-Layer) to update state of Layer using ith-layer inputs
          // Layer will automatically get layer's state, and it will automatically call integrator and then activator
    }
    virtual void Reset();
    virtual void SetInput(); // Updates state of input layer (first layer)

  protected:
    std::vector< Layer<TReal> > network_layers_;

};

} // End nervous_system namespace

#endif /* NN_NERVOUS_SYSTEM_H_ */