#ifndef NN_INTEGRATOR_H_
#define NN_INTEGRATOR_H_

#include <cstddef>

namespace nervous_system {

enum INTEGRATOR_TYPE {
  BASE,
  IDENTITY,
  CONV,
  NET,
  RESERVOIR
};

// Abstract integrator
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

// Identity integrator
class IdentityIntegrator : public Integrator {
  public:
    IdentityIntegrator();
    ~IdentityIntegrator();
    
};

// None integrator - does nothing
class NoneIntegrator : public Integrator {

}

// Conv integrator - uses implicit structure
class ConvIntegrator : public Integrator {

  protected:
  // Put buffers here
}

// Network integrator -- uses explicit unweighted structure
class NetIntegrator : public Integrator {

  protected:
    // Need network w/ structure here
}

// Reservoir -- uses explicit weighted structure
class ReservoirIntegrator : public Integrator {

  protected:
    // Need network w/ structure here
}

} // End nervous_system namespace

#endif /* NN_INTEGRATOR_H_ */