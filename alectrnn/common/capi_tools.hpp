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
#include "arrayobject.h"
#include "capi_tools.hpp"
#include "multi_array.hpp"

namespace alectrnn {

float *PyArrayToCArray(PyArrayObject *py_array);

template<typename T, std::size_t NumDims>
multi_array::SharedMultiArray<T, NumDims> PyArrayToSharedMultiArray(PyArrayObject *py_array) {
  return multi_array::SharedMultiArray<T, NumDims>((T *) py_array->data, 
    multi_array::Array<std::size_t, NumDims>(py_array->dimensions));
}

}



#endif /* ALECTRNN_COMMON_CAPI_TOOLS_H_ */
