/*
 * utilities.cpp
 *
 *  Created on: Sep 2, 2017
 *      Author: Nathaniel Rodriguez
 *
 * A utility functions for use by other classes or functions.
 *
 */

#include <Python.h>
#include "arrayobject.h"

namespace alectrnn {

double *PyArrayToCArray(PyArrayObject *py_array) {
  return (double *) py_array->data;
}

}
