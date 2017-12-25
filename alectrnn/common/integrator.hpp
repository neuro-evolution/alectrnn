#ifndef NN_INTEGRATOR_H_
#define NN_INTEGRATOR_H_

#include <cstddef>

namespace nervous_system {

enum INTEGRATOR_TYPE {
  BASE,
  CONV
};

// Abstract integrator
class Integrator {
  public:
    Integrator();
    virtual ~Integrator();
    virtual void operator()();
    virtual void Configure();

    INTEGRATOR_TYPE GetIntegratorType() const { //might just swap out for some funct call to type
      return ;//////////not sure if even need
    }

  protected:
    INTEGRATOR_TYPE integrator_type_;
}

// Identity integrator
class IdentityIntegrator : public Integrator {
  public:
    IdentityIntegrator();
    ~IdentityIntegrator();
    
};

// Conv integrator

// Structured integrator


// Reservoir Integrator ?? Might be responsible for not changing weights?

} // End nervous_system namespace

#endif /* NN_INTEGRATOR_H_ */