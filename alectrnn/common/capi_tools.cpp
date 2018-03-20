/*
 * capi_tools.cpp
 *
 *  Created on: Sep 2, 2017
 *      Author: Nathaniel Rodriguez
 *
 * A utility functions for use by other classes or functions.
 * data in python array is char* and is supposed to be casted to desired type 
 * based on the old c-style cast (as the c-api is in c). 
 * To get desired behavior I perform a reinterpret_cast explicitly.
 */

#include <Python.h>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include "numpy/arrayobject.h"
#include "multi_array.hpp"
#include "capi_tools.hpp"

namespace alectrnn {

float *PyArrayToCArray(PyArrayObject *py_array) {
  int array_type = PyArray_TYPE(py_array);
  if (array_type != NPY_FLOAT32) {
    throw std::invalid_argument("Numpy array must be of type npy_float32");
  }
  return reinterpret_cast<float *>(py_array->data);
}

std::uint64_t* uInt64PyArrayToCArray(PyArrayObject *py_array) {
  // Check that type is actually uint64 before reinterpret
  int array_type = PyArray_TYPE(py_array);
  if (array_type != NPY_UINT64) {
    throw std::invalid_argument("Numpy array must be of type npy_uint64");
  }
  return reinterpret_cast<std::uint64_t*>(py_array->data);
}

}
