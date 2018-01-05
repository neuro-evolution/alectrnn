#ifndef LAYER_GENERATOR_H_
#define LAYER_GENERATOR_H_

#include <Python.h>
#include "activator.hpp"
#include "integrator.hpp"

nervous_system::Activator<float>* ActivatorParser(nervous_system::ACTIVATOR_TYPE type, PyObject* args);
nervous_system::Integrator<float>* IntegratorParser(nervous_system::INTEGRATOR_TYPE type, PyObject* args);

#endif /* LAYER_GENERATOR_H_ */