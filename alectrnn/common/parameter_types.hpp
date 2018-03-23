/*
 * Hold enumerated types for different kinds of parameters. When parameter
 * configuration are returned from the NervousSystem they are returned
 * as an integer array of types coded according the Enum below. Whenever a new
 * integrator/activator is added, if it introduces new types they should be
 * appended below.
 */

#ifndef NN_PARAMETER_TYPES_H_
#define NN_PARAMETER_TYPES_H_

namespace nervous_system {

enum PARAMETER_TYPE {
  BIAS, // CTRNN activator
  RTAUS, // CTRNN activator
  WEIGHT, // ALL2ALL, CONV3D, RECURRENT integrators
  RANGE, // difference between reset value and threshold for IF models
  REFRACTORY, // refractory period for IF models
  RESISTANCE // resistance for IF models
};

} // End nervous_system namespace

#endif /* NN_PARAMETER_TYPES_H_ */