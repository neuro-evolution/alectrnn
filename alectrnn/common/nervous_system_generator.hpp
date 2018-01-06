/*
 * nervous_system_generator.cpp
 *
 *  Created on: Jan 2, 2018
 *      Author: Nathaniel Rodriguez
 *
 */

#ifndef NERVOUS_SYSTEM_GENERATOR_H_
#define NERVOUS_SYSTEM_GENERATOR_H_

#include <Python.h>
#include <cstddef>
#include <vector>
#include "nervous_system.hpp"

nervous_system::NervousSystem<float>* ParseLayers(std::vector<std::size_t> shape, PyObject* args);
PyMODINIT_FUNC PyInit_NervousSystem_generator(void);

#endif /* NERVOUS_SYSTEM_GENERATOR_H_ */