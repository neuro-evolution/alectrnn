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
  static char *keyword_list[] = {"neural_network", "inputs", "parameters", NULL};
  PyObject* nn_capsule;
  PyArrayObject* inputs;
  PyArrayObject* py_parameter_array;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", keyword_list,
      &nn_capsule, &inputs, &py_parameter_array)) {
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
  nn->Reset();

  // handle parameters
  float* cparameter_array(alectrnn::PyArrayToCArray(py_parameter_array));
  nn->Configure(multi_array::ConstArraySlice<float>(
    cparameter_array, 0, nn->GetParameterCount(), 1));

  // create StateLogger
  nervous_system::StateLogger<float> log = nervous_system::StateLogger<float>(*nn);

  // run NN on inputs
  std::vector<float> nn_input(PyArray_DIM(inputs, 1));
  // npy_intp stride = PyArray_STRIDE(inputs, 0);
  for (npy_intp i = 0; i < PyArray_DIM(inputs, 0); i++)
  {
    for (npy_intp j = 0; j < PyArray_DIM(inputs, 1); ++j)
    {
        nn_input[j] = static_cast<float>(*reinterpret_cast<npy_float32*>(PyArray_GETPTR2(inputs, i, j)));
    }
    nn->SetInput(nn_input);
    nn->Step();
    log(*nn);
  }

  // convert log to tuple of numpy arrays for each layer
  PyObject *py_layers = PyTuple_New(static_cast<npy_intp>(log.size()));
  for (std::size_t layer_index = 0; layer_index < log.size(); ++layer_index)
  {
      const std::vector<multi_array::Tensor<float>>& history(log.GetLayerHistory(layer_index));
      std::vector<npy_intp> shape(1+history[0].ndimensions());
      shape[0] = history.size();
      for (std::size_t iii = 0; iii < history[0].ndimensions(); ++iii) {
        shape[iii+1] = history[0].shape()[iii];
      }
      PyObject* py_array = PyArray_SimpleNew(shape.size(), shape.data(), NPY_FLOAT32);
      PyArrayObject *np_array = reinterpret_cast<PyArrayObject*>(py_array);
      npy_float32* data = reinterpret_cast<npy_float32*>(np_array->data);
      std::vector<std::size_t> strides(shape.size());
      multi_array::CalculateStrides(shape.data(), strides.data(), shape.size());
      for (std::size_t iii = 0; iii < history.size(); ++iii) {
        for (std::size_t jjj = 0; jjj < history[iii].size(); ++jjj) {
          data[iii * strides[0] + jjj] = history[iii][jjj];
        }
      }
      PyTuple_SetItem(py_layers, layer_index, py_array);
  }

  return py_layers;
}

static PyObject *GetLayerStateSize(PyObject *self, PyObject *args, PyObject *kwargs)
{
  static char *keyword_list[] = {"neural_network", "layer", NULL};

  PyObject* nn_capsule;
  int layer_index;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", keyword_list,
                                   &nn_capsule, &layer_index)) {
    std::cerr << "Error parsing GetLayerStateSize arguments" << std::endl;
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

  int state_size = nn->GetLayerState().size();

  return Py_BuildValue("i", state_size);
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

static PyObject *GetParameterLayerIndices(PyObject *self, PyObject *args, PyObject *kwargs) {
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
  std::vector<int> par_indices = nn->GetParameterLayerIndices();
  PyObject* parameter_indices = ConvertToNumpyIntArray(par_indices);
  return parameter_indices;
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

PyObject* ConvertToNumpyIntArray(const std::vector<int>& indices) {
  // Need a temp shape pointer for numpy array
  npy_intp vector_size = indices.size();
  npy_intp* shape_ptr = &vector_size;
  // Create numpy array and recast for assignment
  PyObject* py_array = PyArray_SimpleNew(1, shape_ptr, NPY_INT);
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(py_array);
  npy_int* data = reinterpret_cast<npy_int*>(np_array->data);

  // Copy vect data to numpy array
  for (std::size_t iii = 0; iii < indices.size(); ++iii) {
    data[iii] = static_cast<npy_int>(indices[iii]);
  }
  return py_array;
}

/*
 * Add new commands in additional lines below:
 */
static PyMethodDef NervousSystemHandlerMethods[] = {
  { "RunNeuralNetwork", (PyCFunction) RunNeuralNetwork,
          METH_VARARGS | METH_KEYWORDS,
          "Evaluates NN on given inputs"},
  { "GetParameterCount", (PyCFunction) GetParameterCount,
          METH_VARARGS | METH_KEYWORDS,
          "Returns # parameters"},
  { "GetLayerStateSize", (PyCFunction) GetLayerStateSize,
    METH_VARARGS | METH_KEYWORDS,
    "Returns state size of a given layer"},
  {"GetParameterLayout", (PyCFunction) GetParameterLayout,
          METH_VARARGS | METH_KEYWORDS,
          "Returns numpy array of parameter layout (see parameter_types for code)"},
  {"GetParameterLayerIndices", (PyCFunction) GetParameterLayerIndices,
          METH_VARARGS | METH_KEYWORDS,
          "Returns numpy array of integers for each layer the parameter belongs too."},
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
