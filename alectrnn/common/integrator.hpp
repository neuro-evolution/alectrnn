#ifndef NN_INTEGRATOR_H_
#define NN_INTEGRATOR_H_

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

    INTEGRATOR_TYPE GetIntegratorType() const { //might just swap out for some funct call to type
      return ;//////////not sure if even need
    }

  protected:
    INTEGRATOR_TYPE integrator_type_;
}

// Conv integrator

// Structured integrator

// Identity integrator

// Reservoir Integrator ?? Might be responsible for not changing weights?

} // End nervous_system namespace

#endif /* NN_INTEGRATOR_H_ */