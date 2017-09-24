/*
 * utilities.h
 *
 *  Created on: Sep 2, 2017
 *      Author: Nathaniel Rodriguez
 *
 * A utility functions for use by other classes or functions.
 *
 */

#ifndef ALECTRNN_COMMON_UTILITIES_H_
#define ALECTRNN_COMMON_UTILITIES_H_

#include <Python.h>
#include "arrayobject.h"

namespace alectrnn {

double *PyArrayToCArray(PyArrayObject *py_array);

}



#endif /* ALECTRNN_COMMON_UTILITIES_H_ */
