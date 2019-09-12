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
#include <utility>
#include "nervous_system_handler.hpp"
#include "numpy/arrayobject.h"
#include "../common/multi_array.hpp"
#include "../nervous_system/state_logger.hpp"
#include "layer.hpp"
#include "nervous_system.hpp"
#include "parameter_types.hpp"
#include "../common/capi_tools.hpp"

static PyObject *RunNeuralNetwork(PyObject *self, PyObject *args, PyObject *kwargs) {
  static char *keyword_list[] = {"neural_network", "inputs", "parameters"};
  PyObject* nn_capsule;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", keyword_list,
      &nn_capsule)) {
    std::cerr << "Error parsing RunNeuralNetwork arguments" << std::endl;
    return NULL;
  }

  // handle neural_network
  if (!PyCapsule_IsValid(nn_capsule, "nervous_system_generator.nn"))
  {
    std::cerr << "Invalid pointer to NN returned from capsule,"
        " or is not a capsule." << std::endl;
    return NULL;
  }
  nervous_system::NervousSystem<float>* nn =
      static_cast<nervous_system::NervousSystem<float>*>(
      PyCapsule_GetPointer(nn_capsule, "nervous_system_generator.nn"));

  // handle inputs
  // need to take numpy array and convert to ...
  // fuck idk it needs a vector<float> as input, but... that do
  // one option is to make a vector and copy values in and just do that each
  // iteration... probably the easiest way.

  // handle parameters
  float* cparameter_array(alectrnn::PyArrayToCArray(py_parameter_array));
  nn->Configure(multi_array::ConstArraySlice<float>(
    cparameter_array, 0, nn->GetParameterCount(), 1));

  // create StateLogger
  log = nervous_system::StateLogger<float>(nn);

  // run NN on inputs
  for (blah)
  {
    nn->SetInput(); // vector<floats>
    nn->Step();
    log(nn);
  }

  // convert log to numpy array... thought we got the whole thing not just layer
  // WE WANT FULL OUTPUT FOR REAL RUN AND FORCE INPUT... for simplices
  PyObject* np_history = ConvertLogToPyArray(log.GetLayerHistory(layer_index));

  return np_history;
}

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

static PyObject *GetWeightNormalizationFactors(PyObject *self, PyObject *args, PyObject *kwargs) {
  static char *keyword_list[] = {"neural_network", NULL};

  PyObject* nn_capsule;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", keyword_list,
                                   &nn_capsule)) {
    std::cerr << "Error parsing GetWeightNormalizationFactors arguments" << std::endl;
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

  // call function and get vector
  std::vector<float> normalization_factors = nn->GetWeightNormalizationFactors();
  // convert vector<float> -> numpy array float32
  return ConvertFloatVectorToPyFloat32Array(normalization_factors);
}

PyObject* ConvertFloatVectorToPyFloat32Array(const std::vector<float>& vec) {
  // Need a temp shape pointer for numpy array
  npy_intp vector_size = vec.size();
  npy_intp* shape_ptr = &vector_size;

  // Create numpy array and recast for assignment
  PyObject* py_array = PyArray_SimpleNew(1, shape_ptr, NPY_FLOAT32);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(py_array);
  npy_float32* data = reinterpret_cast<npy_float32*>(np_array->data);

  // Copy vect data to numpy array
  for (std::size_t iii = 0; iii < vec.size(); ++iii) {
    data[iii] = static_cast<npy_float32>(vec[iii]);
  }
  return py_array;
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
  {
    "GetWeightNormalizationFactors", (PyCFunction) GetWeightNormalizationFactors,
          METH_VARARGS | METH_KEYWORDS,
          "Returns a numpy array with normalization factors for each parameter"},
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
