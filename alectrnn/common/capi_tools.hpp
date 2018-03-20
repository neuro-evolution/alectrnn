/*
 * capi_tools.h
 *
 *  Created on: Sep 2, 2017
 *      Author: Nathaniel Rodriguez
 *
 * A utility functions for use by other classes or functions.
 * Make sure to do:  import_array(); for whatever module this is being used
 * in in order to activate the API.
 */

#ifndef ALECTRNN_COMMON_CAPI_TOOLS_H_
#define ALECTRNN_COMMON_CAPI_TOOLS_H_

#include <Python.h>
#include <stdexcept>
#include <iostream>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include "numpy/arrayobject.h"
#include "multi_array.hpp"

namespace alectrnn {

float *PyArrayToCArray(PyArrayObject *py_array);
std::uint64_t* uInt64PyArrayToCArray(PyArrayObject *py_array);

/* PyArray data has to be reinterpret cased to a corresponding c-type
 * before it can be safely casted and copied to the vector.
 * T should be any numerical type. */
template<typename T>
std::vector<T> uInt64PyArrayToVector(PyArrayObject *py_array) {
  // type check before reinterpret
  int array_type = PyArray_TYPE(py_array);
  if (array_type != NPY_UINT64) {
    std::cerr << " Numpy array type: " << array_type << std::endl;
    std::cerr << " Required type: " << NPY_UINT64 << std::endl;
    throw std::invalid_argument("Numpy array of wrong type. Requires npy_uint64");
  }

  std::size_t num_elements = 1;
  if (!(py_array->nd >= 0)) {
    throw std::invalid_argument("Numpy array shouldn't have negative dimensions");
  }
  for (std::size_t iii = 0; iii < py_array->nd; ++iii) {
    num_elements *= py_array->dimensions[iii];
  }
  std::vector<T> vec(num_elements);
  std::uint64_t* data_array = reinterpret_cast<std::uint64_t*>(py_array->data);
  for (std::size_t iii = 0; iii < num_elements; ++iii) {
    vec[iii] = static_cast<T>(data_array[iii]);
  }
  return vec;
}

template<typename T, std::size_t NumDims>
multi_array::SharedMultiArray<T, NumDims> PyArrayToSharedMultiArray(PyArrayObject *py_array) {
  /* The python data has to be reinterpret casted to the corresponding c-type
   * while Array will allocate new memory in the desired type */
  return multi_array::SharedMultiArray<T, NumDims>(reinterpret_cast<T*>(py_array->data),
    multi_array::Array<std::size_t, NumDims>(reinterpret_cast<std::uint64_t*>(
                                             py_array->dimensions)));
}

} // End alectrnn namespace

#endif /* ALECTRNN_COMMON_CAPI_TOOLS_H_ */