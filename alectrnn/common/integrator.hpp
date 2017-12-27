#ifndef NN_INTEGRATOR_H_
#define NN_INTEGRATOR_H_

#include <cstddef>
#include "multi_array.hpp"
#include "graphs.hpp"

namespace nervous_system {

enum INTEGRATOR_TYPE {
  BASE,
  NONE,
  CONV,
  NET,
  RESERVOIR
};

template<typename TReal>
class Integrator {
  public:
    Integrator() {
      integrator_type_ = INTEGRATOR_TYPE.BASE;
      parameter_count_ = 0;
    }
    virtual ~Integrator();
    virtual void operator()(multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state)=0;
    virtual void Configure(const multi_array::ArraySlice<TReal>& parameters)=0;

    std::size_t GetParameterCount() const {
      return parameter_count_;
    }

    INTEGRATOR_TYPE GetIntegratorType() const {
      return integrator_type_;
    }

  protected:
    INTEGRATOR_TYPE integrator_type_;
    std::size_t parameter_count_;
}

// None integrator - does nothing
template<typename TReal>
class NoneIntegrator : public Integrator<TReal> {
  public:
    NoneIntegrator() : Integrator() { integrator_type_ = INTEGRATOR_TYPE.NONE }
    ~NoneIntegrator()=default;

    void operator()(multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state) {}
    void Configure(const multi_array::ArraySlice<TReal>& parameters) {}
}

// Conv integrator - uses implicit structure
// FIX: Not sure if I need NumDim... issue is that on CONSTRUCTION, I know layer type... but... not after
// ALso. layer can't know type of generic Integrator... So the builder of both would need to know NumDim (maybe in case switch??)
template<typename TReal>
class Conv3DIntegrator : public Integrator {

  protected:
  // Put buffers here
}

// Network integrator -- uses explicit unweighted structure
template<typename TReal>
class NetIntegrator : public Integrator<TReal> {

  protected:
    // Need network w/ structure here
}

// Reservoir -- uses explicit weighted structure
template<typename TReal>
class ReservoirIntegrator : public Integrator<TReal> {

  protected:
    // Need network w/ structure here
}

} // End nervous_system namespace

#endif /* NN_INTEGRATOR_H_ */