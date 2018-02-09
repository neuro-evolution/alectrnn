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
#include "numpy/arrayobject.h"
#include "layer.hpp"
#include "nervous_system.hpp"
#include "parameter_types.hpp"

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
  nervous_system::NervousSystem<float>* nn = 
      static_cast<nervous_system::NervousSystem<float>*>(
      PyCapsule_GetPointer(nn_capsule, "nervous_system_generator.nn"));

  int num_params = nn->GetParameterCount();

  return Py_BuildValue("i", num_params);
}

static PyObject *GetParameterLayout(PyObject *self, PyObject *args, PyObject *kwargs) {
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
  nervous_system::NervousSystem<float>* nn = 
      static_cast<nervous_system::NervousSystem<float>*>(
      PyCapsule_GetPointer(nn_capsule, "nervous_system_generator.nn"));
  std::vector<nervous_system::PARAMETER_TYPE> par_types = nn->GetParameterLayout();
  PyObject* parameter_layout = ConvertParameterTypesToPyArray(par_types);
  return parameter_layout;
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
  nervous_system::NervousSystem<float>* nn = 
      static_cast<nervous_system::NervousSystem<float>*>(
      PyCapsule_GetPointer(nn_capsule, "nervous_system_generator.nn"));

  int nn_size = nn->size();

  return Py_BuildValue("i", nn_size);
}

PyObject* ConvertParameterTypesToPyArray(const std::vector<nervous_system::PARAMETER_TYPE>& par_types) {
  // Need a temp shape pointer for numpy array
  npy_intp vector_size = par_types.size();
  npy_intp* shape_ptr = &vector_size;
  // Create numpy array and recast for assignment
  PyObject* py_array = PyArray_SimpleNew(1, shape_ptr, NPY_INT);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(py_array);
  npy_int* data = reinterpret_cast<npy_int*>(np_array->data);

  // Copy vect data to numpy array
  for (std::size_t iii = 0; iii < par_types.size(); ++iii) {
    data[iii] = static_cast<npy_int>(par_types[iii]);
  }
  return py_array;
}

/*
 * Add new commands in additional lines below:
 */
static PyMethodDef NervousSystemHandlerMethods[] = {
  { "GetParameterCount", (PyCFunction) GetParameterCount,
          METH_VARARGS | METH_KEYWORDS,
          "Returns # parameters"},
  {"GetParameterLayout", (PyCFunction) GetParameterLayout,
          METH_VARARGS | METH_KEYWORDS,
          "Returns numpy array of parameter layout (see parameter_types for code)"},
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
  import_array();
  return PyModule_Create(&NervousSystemHandlerModule);
}
