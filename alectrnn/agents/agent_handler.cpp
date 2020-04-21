/*
 * agent_handler.cpp
 *
 *  Created on: Jan 20, 2018
 *      Author: Nathaniel Rodriguez
 *
 */

#include <Python.h>
#include <cstddef>
#include <iostream>
#include "../common/multi_array.hpp"
#include "numpy/arrayobject.h"
#include "nervous_system.hpp"
#include "agent_handler.hpp"
#include "player_agent.hpp"
// Add includes to agents you wish to add below:
#include "nervous_system_agent.hpp"

static PyObject *GetScreenHistory(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {

  static char *keyword_list[] = {"agent", NULL};

  PyObject *agent_capsule;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", keyword_list,
                                   &agent_capsule)){
    std::cerr << "Error parsing GetScreenHistory arguments" << std::endl;
    return NULL;
  }

  if (!PyCapsule_IsValid(agent_capsule, "agent_generator.agent"))
  {
    std::cerr << "Invalid pointer to Agent returned from capsule,"
    " or is not a capsule." << std::endl;
    return NULL;
  }
  alectrnn::NervousSystemAgent* agent = static_cast<alectrnn::NervousSystemAgent*>(
  PyCapsule_GetPointer(agent_capsule, "agent_generator.agent"));

  PyObject* np_history = ConvertLogToPyArray(agent->GetScreenLog().GetHistory());

  return np_history;
}

/*
 * Returns the layer history from a NervousSystemAgent
 */
static PyObject *GetLayerHistory(PyObject *self, PyObject *args,
    PyObject *kwargs) {

  static char *keyword_list[] = {"agent", "layer", NULL};

  PyObject *agent_capsule;
  int layer_index;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", keyword_list, 
    &agent_capsule, &layer_index)){
    std::cerr << "Error parsing GetLayerHistory arguments" << std::endl;
    return NULL;
  }

  if (!PyCapsule_IsValid(agent_capsule, "agent_generator.agent"))
  {
    std::cerr << "Invalid pointer to Agent returned from capsule,"
        " or is not a capsule." << std::endl;
    return NULL;
  }
  alectrnn::NervousSystemAgent* agent = static_cast<alectrnn::NervousSystemAgent*>(
      PyCapsule_GetPointer(agent_capsule, "agent_generator.agent"));

  PyObject* np_history = ConvertLogToPyArray(agent->GetLog().GetLayerHistory(layer_index));

  return np_history;
}

static PyObject* GetTimeSeriesSize(PyObject* self, PyObject* args, PyObject* kwargs)
{
  static char *keyword_list[] = {"agent", "layer", NULL};
  PyObject *agent_capsule;
  int layer_index;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", keyword_list,
                                   &agent_capsule, &layer_index)) {
    std::cout << "Invalid argument in put into GetTimeSeriesSize!" << std::endl;
    return NULL;
  }

  if (!PyCapsule_IsValid(agent_capsule, "agent_generator.agent"))
  {
    std::cerr << "Invalid pointer to Agent returned from capsule,"
                 " or is not a capsule." << std::endl;
    return NULL;
  }

  alectrnn::NervousSystemAgent* agent = static_cast<alectrnn::NervousSystemAgent*>(
      PyCapsule_GetPointer(agent_capsule, "agent_generator.agent"));

  int duration = agent->GetLog().time_series_size(layer_index);
  return Py_BuildValue("i", duration);
}

static PyObject* AssignLayerHistory(PyObject* self, PyObject* args, PyObject* kwargs)
{
  static char *keyword_list[] = {"agent", "layer", "array", NULL};
  PyObject *agent_capsule;
  int layer_index;
  PyArrayObject* py_log;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OiO", keyword_list,
                                   &agent_capsule, &layer_index, &py_log)) {
    std::cout << "Invalid argument in put into AssignLayerHistory!" << std::endl;
    return NULL;
  }

  if (!PyCapsule_IsValid(agent_capsule, "agent_generator.agent"))
  {
    std::cerr << "Invalid pointer to Agent returned from capsule,"
                 " or is not a capsule." << std::endl;
    return NULL;
  }

  alectrnn::NervousSystemAgent* agent = static_cast<alectrnn::NervousSystemAgent*>(
      PyCapsule_GetPointer(agent_capsule, "agent_generator.agent"));

  const std::vector<multi_array::Tensor<float>>& history = agent->GetLog().GetLayerHistory(layer_index);
  std::vector<npy_intp> shape(1+history[0].ndimensions());
  shape[0] = history.size();
  for (std::size_t iii = 0; iii < history[0].ndimensions(); ++iii)
  {
    shape[iii+1] = history[0].shape()[iii];
  }
  npy_float32* data = reinterpret_cast<npy_float32*>(py_log->data);
  std::vector<std::size_t> strides(shape.size());
  multi_array::CalculateStrides(shape.data(), strides.data(), shape.size());
  for (std::size_t iii = 0; iii < history.size(); ++iii) {
    for (std::size_t jjj = 0; jjj < history[iii].size(); ++jjj) {
      data[iii * strides[0] + jjj] = history[iii][jjj];
    }
  }

  Py_RETURN_NONE;
}

PyObject *ConvertLogToPyArray(const std::vector<multi_array::Tensor<float>>& history) {
  // Determine the new shape from the layer shape + the temporal dimension
  std::vector<npy_intp> shape(1+history[0].ndimensions());
  shape[0] = history.size();
  for (std::size_t iii = 0; iii < history[0].ndimensions(); ++iii) {
    shape[iii+1] = history[0].shape()[iii];
  }

  // Build and fill python array
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
  return py_array;
}

static PyMethodDef AgentHandlerMethods[] = {
  { "GetLayerHistory", (PyCFunction) GetLayerHistory,
          METH_VARARGS | METH_KEYWORDS,
          "Returns PyArray of agent state history"},
  { "GetTimeSeriesSize", (PyCFunction) GetTimeSeriesSize,
  METH_VARARGS | METH_KEYWORDS,
  "Get duration of agent state history"},
  { "AssignLayerHistory", (PyCFunction) AssignLayerHistory,
      METH_VARARGS | METH_KEYWORDS,
      "Sets PyArray of agent state history"},
  { "GetScreenHistory", (PyCFunction) GetScreenHistory,
    METH_VARARGS | METH_KEYWORDS,
    "Returns PyArray of ALE screen history" },
      //Additional agents here, make sure to add includes top
  { NULL, NULL, 0, NULL}
};

static struct PyModuleDef AgentHandlerModule = {
  PyModuleDef_HEAD_INIT,
  "agent_handler",
  "Returns a handle to an Agent",
  -1,
  AgentHandlerMethods
};

PyMODINIT_FUNC PyInit_agent_handler(void) {
  import_array();
  return PyModule_Create(&AgentHandlerModule);
}
