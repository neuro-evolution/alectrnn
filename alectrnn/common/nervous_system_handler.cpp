/*
 * nervous_system_handler.cpp
 *
 *  Created on: Jan 2, 2018
 *      Author: Nathaniel Rodriguez
 *
 * Allows calling and returning information from the NervousSystem in for Python
 */

#include <Python.h>
#include <cstddef>
#include <iostream>
#include <vector>
#include "nervous_system_handler.hpp"
#include "layer.hpp"
#include "nervous_system.hpp"

/*
 * Create a new py_function Create for each NervousSystem you want to add
 */
static PyObject *CreateNervousSystem(PyObject *self, PyObject *args, PyObject *kwargs) {
  static char *keyword_list[] = {"ale", "input_shape", "layers", NULL};

  PyObject* ale_capsule;
  PyArrayObject* input_shape;
  PyObject* layers_tuple;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", keyword_list,
      &ale_capsule, &input_shape, &layers_tuple)) {
    std::cout << "Error parsing CreateLayer arguments" << std::endl;
    return NULL;
  }

  if (!PyCapsule_IsValid(ale_capsule, "ale_generator.ale"))
  {
    std::cout << "Invalid pointer to ALE returned from capsule,"
        " or is not a capsule." << std::endl;
    return NULL;
  }
  ALEInterface* ale = static_cast<ALEInterface*>(PyCapsule_GetPointer(
      ale_capsule, "ale_generator.ale"));

  std::vector<std::size_t> shape = PyArrayToVector(input_shape);
  nervous_system::NervousSystem<float>* nervous_system = ParseLayers(shape, layers_tuple);

  PyObject* nervous_system_capsule = PyCapsule_New(static_cast<void*>(nervous_system),
                            "nervous_system_generator.nn", DeleteNervousSystem);
  return nervous_system_capsule;
}

/*
 * Add new commands in additional lines below:
 */
static PyMethodDef NervousSystemHandlerMethods[] = {
  { "CreateNervousSystem", (PyCFunction) NervousSystem,
          METH_VARARGS | METH_KEYWORDS,
          "Returns a handle to a NervousSystem"},
      //Additional layers here, make sure to add includes top
  { NULL, NULL, 0, NULL}
};

static struct PyModuleDef NervousSystemHandlerModule = {
  PyModuleDef_HEAD_INIT,
  "nervous_system_generator",
  "Returns a handle to a Nervous System",
  -1,
  NervousSystemHandlerMethods
};

PyMODINIT_FUNC PyInit_nn_generator(void) {
  return PyModule_Create(&NervousSystemHandlerModule);
}
