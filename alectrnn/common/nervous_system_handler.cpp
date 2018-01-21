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

static PyObject *GetParameterCount(PyObject *self, PyObject *args, PyObject *kwargs) {
  static char *keyword_list[] = {"neural_network", NULL};

  PyObject* nn_capsule;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", keyword_list,
      &nn_capsule)) {
    std::cerr << "Error parsing GetParameterCount arguments" << std::endl;
    return NULL;
  }

  if (!PyCapsule_IsValid(nn_capsule, "nervous_system_generator.nn"))
  {
    std::cerr << "Invalid pointer to NN returned from capsule,"
        " or is not a capsule." << std::endl;
    return NULL;
  }
  NervousSystem* nn = static_cast<NervousSystem*>(PyCapsule_GetPointer(
      nn_capsule, "nervous_system_generator.nn"));

  int num_params = nn->GetParameterCount();

  return Py_BuildValue("i", num_params);
}

static PyObject *GetSize(PyObject *self, PyObject *args, PyObject *kwargs) {
  static char *keyword_list[] = {"neural_network", NULL};

  PyObject* nn_capsule;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", keyword_list,
      &nn_capsule)) {
    std::cerr << "Error parsing GetParameterCount arguments" << std::endl;
    return NULL;
  }

  if (!PyCapsule_IsValid(nn_capsule, "nervous_system_generator.nn"))
  {
    std::cerr << "Invalid pointer to NN returned from capsule,"
        " or is not a capsule." << std::endl;
    return NULL;
  }
  NervousSystem* nn = static_cast<NervousSystem*>(PyCapsule_GetPointer(
      nn_capsule, "nervous_system_generator.nn"));

  int nn_size = nn->size();

  return Py_BuildValue("i", nn_size);
}

/*
 * Add new commands in additional lines below:
 */
static PyMethodDef NervousSystemHandlerMethods[] = {
  { "GetParameterCount", (PyCFunction) GetParameterCount,
          METH_VARARGS | METH_KEYWORDS,
          "Returns # parameters"},
  { "GetSize", (PyCFunction) GetSize,
          METH_VARARGS | METH_KEYWORDS,
          "Returns # layers in network"},
  { NULL, NULL, 0, NULL}
};

static struct PyModuleDef NervousSystemHandlerModule = {
  PyModuleDef_HEAD_INIT,
  "nervous_system_handler",
  "Access and modification function for nervous system",
  -1,
  NervousSystemHandlerMethods
};

PyMODINIT_FUNC PyInit_nn_handler(void) {
  return PyModule_Create(&NervousSystemHandlerModule);
}
