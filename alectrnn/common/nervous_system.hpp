
#ifndef NN_NERVOUS_SYSTEM_H_
#define NN_NERVOUS_SYSTEM_H_

namespace nervous_system {

#include <cstddef>

class NervousSystem {
  public:
    NervousSystem();
    virtual ~NervousSystem();
    virtual void Step();

  protected:

};

} // End nervous_system namespace

#endif /* NN_NERVOUS_SYSTEM_H_ */