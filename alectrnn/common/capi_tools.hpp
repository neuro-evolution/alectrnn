/*
 * capi_tools.h
 *
 *  Created on: Sep 2, 2017
 *      Author: Nathaniel Rodriguez
 *
 * A utility functions for use by other classes or functions.
 */

#ifndef ALECTRNN_COMMON_CAPI_TOOLS_H_
#define ALECTRNN_COMMON_CAPI_TOOLS_H_

#include <Python.h>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include "numpy/arrayobject.h"
#include "multi_array.hpp"

namespace alectrnn {

float *PyArrayToCArray(PyArrayObject *py_array);
PyObject *ConvertTensorToPyArray(const multi_array::Tensor<float>& tensor);

template<typename T>
std::vector<T> PyArrayToVector(PyArrayObject *py_array) {
  std::size_t num_elements = 1;
  for (std::size_t iii = 0; iii < py_array->nd; ++iii) {
    num_elements *= py_array->dimensions[iii];
  }
  std::vector<T> vec(num_elements);
  T* data_array = reinterpret_cast<T*>(py_array->data);
  for (std::size_t iii = 0; iii < num_elements; ++iii) {
    vec[iii] = data_array[iii];
  }
  return vec;
}

template<typename T, std::size_t NumDims>
multi_array::SharedMultiArray<T, NumDims> PyArrayToSharedMultiArray(PyArrayObject *py_array) {
  return multi_array::SharedMultiArray<T, NumDims>((T *) py_array->data, 
    multi_array::Array<std::size_t, NumDims>(py_array->dimensions));
}

} // End alectrnn namespace



#endif /* ALECTRNN_COMMON_CAPI_TOOLS_H_ */
