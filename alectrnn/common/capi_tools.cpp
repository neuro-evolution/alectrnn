/*
 * capi_tools.cpp
 *
 *  Created on: Sep 2, 2017
 *      Author: Nathaniel Rodriguez
 *
 * A utility functions for use by other classes or functions.
 */

#include <Python.h>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "arrayobject.hpp"

namespace alectrnn {

float *PyArrayToCArray(PyArrayObject *py_array) {
  return (float *) py_array->data;
}

}
