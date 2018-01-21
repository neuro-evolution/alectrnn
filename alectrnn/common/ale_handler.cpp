/*
 * ale_handler.cpp
 *
 *  Created on: Jan 8, 2018
 *      Author: Nathaniel Rodriguez
 *
 * Calls functions from ALE and returns information to python
 *
 */

#include <Python.h>
#include "ale_generator.hpp"
#include <ale_interface.hpp>
#include <iostream>

static PyObject *NumOutputs(PyObject *self, PyObject *args, PyObject *kwargs) {
  static char *keyword_list[] = {"ale", NULL};

  PyObject *ale_capsule;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", keyword_list, &ale_capsule)){
    std::cerr << "Error parsing NumOutput arguments" << std::endl;
    return NULL;
  }

  if (!PyCapsule_IsValid(ale_capsule, "ale_generator.ale"))
  {
    std::cerr << "Invalid pointer to ALE returned from capsule,"
        " or is not a capsule." << std::endl;
    return NULL;
  }
  ALEInterface* ale = static_cast<ALEInterface*>(PyCapsule_GetPointer(
      ale_capsule, "ale_generator.ale"));

  int num_outputs = ale->getMinimalActionSet().size();

  return Py_BuildValue("i", num_outputs);
}

static PyMethodDef ALEHandlerMethods[] = {
  { "NumOutputs", (PyCFunction) NumOutputs, METH_VARARGS | METH_KEYWORDS,
      "Returns number of controller outputs"},
  { NULL, NULL, 0, NULL}
};

static struct PyModuleDef ALEHandlerModule = {
  PyModuleDef_HEAD_INIT,
  "ale_handler",
  "Allows manipulating the ALE environment in python.",
  -1,
  ALEHandlerMethods
};

PyMODINIT_FUNC PyInit_ale_handler(void) {
  return PyModule_Create(&ALEHandlerModule);
}
