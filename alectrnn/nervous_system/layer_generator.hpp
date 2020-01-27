#ifndef LAYER_GENERATOR_H_
#define LAYER_GENERATOR_H_

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "activator.hpp"
#include "integrator.hpp"

nervous_system::Activator<float>* ActivatorParser(nervous_system::ACTIVATOR_TYPE type, PyObject* args);
nervous_system::Integrator<float>* IntegratorParser(nervous_system::INTEGRATOR_TYPE type, PyObject* args);
PyMODINIT_FUNC PyInit_layer_generator(void);

#endif /* LAYER_GENERATOR_H_ */
