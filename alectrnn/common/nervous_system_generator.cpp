/*
 * nervous_system_generator.cpp
 *
 *  Created on: Jan 2, 2018
 *      Author: Nathaniel Rodriguez
 *
 */

#include <Python.h>
#include <ale_interface.hpp>
#include <cstddef>
#include <iostream>
#include <vector>
#include "agent_generator.hpp"
#include "layer.hpp"
#include "nervous_system.hpp"

// Load in ALE, a couple numpy arrays and variables needed for NN creation
// this should include Input sizes

// Create NN
// Return NN pointer

// Create additional function that returns the # of parameters needed -> each layer will need a param Count member

/*
 * DeleteLayer can be shared among the Layers as a destructor
 */
static void DeleteNervousSystem(PyObject *nervous_system_capsule) {
  delete (nervous_system::NervousSystem<float> *)PyCapsule_GetPointer(
        nervous_system_capsule, "nervous_system_generator.nn");
}

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

nervous_system::NervousSystem<float>* ParseLayers(std::vector<std::size_t> shape, PyObject* args) {
  nervous_system::NervousSystem<float>* nervous_system = new nervous_system::NervousSystem<float>(shape);
  Py_ssize_t num_layers = PyTuple_Size(args);
  for (Py_ssize_t iii = 0; iii < num_layers; ++iii) {
    PyObject* layer_capsule = PyTuple_GetItem(args, iii);

    if (!PyCapsule_IsValid(layer_capsule, "layer_generator.layer"))
    {
      std::cout << "Invalid pointer to Layer returned from capsule,"
          " or is not a capsule." << std::endl;
      return NULL;
    }

    nervous_system::Layer<float>* layer = static_cast<nervous_system::Layer<float>*>(
      PyCapsule_GetPointer(layer_capsule, "layer_generator.layer"));
    nervous_system.AddLayer(layer);
  }

  return nervous_system;
}

/*
 * Add new NervousSystems in additional lines below:
 */
static PyMethodDef NervousSystemMethods[] = {
  { "CreateNervousSystem", (PyCFunction) NervousSystem,
          METH_VARARGS | METH_KEYWORDS,
          "Returns a handle to a NervousSystem"},
      //Additional layers here, make sure to add includes top
  { NULL, NULL, 0, NULL}
};

static struct PyModuleDef NervousSystemModule = {
  PyModuleDef_HEAD_INIT,
  "nervous_system_generator",
  "Returns a handle to a Layer",
  -1,
  NervousSysMethods
};

PyMODINIT_FUNC PyInit_NervousSystem_generator(void) {
  return PyModule_Create(&NervousSystemModule);
}
