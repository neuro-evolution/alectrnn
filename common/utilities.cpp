#include <Python.h>
#include "arrayobject.h"

namespace alectrnn {

double *PyArrayToCArray(PyArrayObject *py_array) {
  return (double *) py_array->data;
}

}
