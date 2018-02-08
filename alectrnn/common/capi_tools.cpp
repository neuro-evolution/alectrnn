/*
 * capi_tools.cpp
 *
 *  Created on: Sep 2, 2017
 *      Author: Nathaniel Rodriguez
 *
 * A utility functions for use by other classes or functions.
 * data in python array is char* and is supposed to be casted to desired type 
 * based on the old c-style cast (as the c-api is in c). 
 * To get desired behavior I perform a reintrepet_cast explicitly.
 */

#include <Python.h>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "numpy/arrayobject.h"
#include "multi_array.hpp"
#include "capi_tools.hpp"

namespace alectrnn {

float *PyArrayToCArray(PyArrayObject *py_array) {
  return reinterpret_cast<float *>(py_array->data);
}

PyObject *ConvertTensorToPyArray(const multi_array::Tensor<float>& tensor)
{
  std::vector<npy_intp> shape(tensor.ndimensions());
  for (std::size_t iii = 0; iii < tensor.ndimensions(); ++iii) {
    shape[iii] = tensor.shape()[iii];
  }
  PyObject* py_array = PyArray_SimpleNew(tensor.ndimensions(), shape.data(), NPY_FLOAT32);
  PyArrayObject *np_array = reinterpret_cast<PyArrayObject*>(py_array);
  npy_float32* data = reinterpret_cast<npy_float32*>(np_array->data);
  for (std::size_t iii = 0; iii < tensor.size(); ++iii) {
    data[iii] = tensor[iii];
  }
  return py_array;
}

}
