/*
 * layer_generator.cpp
 *
 *  Created on: Jan 2, 2018
 *      Author: Nathaniel Rodriguez
 *
 * The layer_generator is a python extension used to create layers in a NN.
 */

#include <Python.h>
#include <cassert>
#include <ale_interface.hpp>
#include <cstddef>
#include <iostream>
#include "arrayobject.h"
#include "layer.hpp"
#include "capi_tools.hpp"
#include "multi_array.hpp"

/*
 * DeleteLayer can be shared among the Layers as a destructor
 */
static void DeleteLayer(PyObject *layer_capsule) {
  delete (nervous_system::Layer *)PyCapsule_GetPointer(
        layer_capsule, "layer_generator.layer");
}

// Integrators
// ALL2ALL (#_states in layer, # states in prev layer)
// Conv3D (# filters, filter shape, layer_shape(depends on prev layer shape & stride), stride)
// Net (Nx2)
// Reservoir (Nx2), (Nx1)

// Activators
// CTRNN (# states, step size)
// CTRNNConv (shape, step_size)

// Layers
// Layer (back*, self*, activation*, shape)
// Motor (# in, # out, activation*)
// Input (shape) -- made by nervous_system IGNORE

// CreateFunction
// Layer {"back": TYPE, (TUPLE); "self": TYPE, (TUPLE); "act": TYPE, (TUPLE), "shape": (TUPLE)}
// Motor ("ale": ALE, "in": NUM, "act": TYPE, (TUPLE))


/*
 * Create a new py_function CreateXXXLayer for each layer you want to add
 */
static PyObject *CreateLayer(PyObject *self, PyObject *args, PyObject *kwargs) {
  static char *keyword_list[] = {"back_integrator", "back_integrator_args",
    "self_integrator", "self_integrator_args", 
    "activator_type", "activator_args", "shape", NULL};

  int back_integrator_type;
  PyObject* back_integrator_args;
  int self_integrator;
  PyObject* self_integrator_args;
  int activator_type;
  PyObject* activator_args;
  PyArrayObject* shape;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iOiOiOO", keyword_list,
      &back_integrator_type, &back_integrator_args, &self_integrator,
      &self_integrator_args, &activator_type, &activator_args,
      &shape)) {
    std::cout << "Error parsing Agent arguments" << std::endl;
    return NULL;
  }



  // stuff for making integrator/activator/layer here
  nervous_system:: *layer = new ?????;

  PyObject* layer_capsule = PyCapsule_New(static_cast<void*>(layer),
                                "layer_generator.layer", DeleteLayer);
  return layer_capsule;
}

static PyObject *CreateMotorLayer(PyObject *self, PyObject *args, PyObject *kwargs) {
  static char *keyword_list[] = {"ale", "num_inputs", "activator_type", "activator_args", NULL};

  PyObject* ale_capsule;
  float step_size;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Of", keyword_list,
      &ale_capsule, &step_size)) {
    std::cout << "Error parsing Agent arguments" << std::endl;
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

  // stuff for making integrator/activator/layer here
  nervous_system:: *layer = new ?????;

  PyObject* layer_capsule = PyCapsule_New(static_cast<void*>(layer),
                                "layer_generator.layer", DeleteLayer);
  return layer_capsule;
}

nervous_system::Activator<float>* ActivatorParser(nervous_system::ACTIVATOR_TYPE type, PyObject* args) {

  nervous_system::Activator<float>* new_activator;  
  switch(type) {
    case IDENTITY:
      new_activator = new IdentityActivator<float>();
      break;
    case CTRNN:
      int num_states;
      float step_size;
      if (!PyArg_ParseTuple(args, "if", &num_states, &step_size)) {
        std::cout << "Error parsing Activator arguments" << std::endl;
        assert(0);
      }
      new_activator = new CTRNNActivator<float>(num_states, step_size);
      break;
    case CONV_CTRNN:
      PyArrayObject* shape;
      float step_size;
      if (!PyArg_ParseTuple(args, "Of", &shape, &step_size)) {
        std::cout << "Error parsing Activator arguments" << std::endl;
        assert(0);
      }
      new_activator = new Conv3DCTRNNActivator<float>(multi_array::Array<std::size_t,3>(shape->data), step_size);
      break;
    default:
      std::cout << "Activator case not supported... exiting." << std::endl;
      assert(0);
  }

  return new_activator;
}

nervous_system::Integrator<float>* IntegratorParser(nervous_system::INTEGRATOR_TYPE type, PyObject* args) {

  nervous_system::Integrator<float>* new_integrator;  
  switch(type) {
    case NONE:
      new_integrator = new NoneIntegrator<float>();
      break;
    case ALL2ALL:
      int num_states;
      int num_prev_states;
      if (!PyArg_ParseTuple(args, "ii", &num_states, &num_prev_states)) {
        std::cout << "Error parsing Integrator arguments" << std::endl;
        assert(0);
      }
      new_integrator = new All2AllIntegrator<float>(num_states, num_prev_states);
      break;
    case CONV:
      int num_filters;
      PyArrayObject* filter_shape;
      PyArrayObject* layer_shape;
      int stride;
      if (!PyArg_ParseTuple(args, "iOOi", &num_filters, &filter_shape,
        &layer_shape, &stride)) {
        std::cout << "Error parsing Integrator arguments" << std::endl;
        assert(0);
      }
      new_integrator = new Conv3DIntegrator<float>(num_filters,
        multi_array::Array<std::size_t,3>(filter_shape->data),
        multi_array::Array<std::size_t,3>(layer_shape->data),
        stride);
      break;
    case NET:
      int num_nodes;
      PyArrayObject* edge_list; // Nx2 dimensional array
      if (!PyArg_ParseTuple(args, "iO", &num_nodes, &edge_list)) {
        std::cout << "Error parsing Integrator arguments" << std::endl;
        assert(0);
      }
      new_integrator = new NetIntegrator<float>(graphs::ConvertEdgeListToDiGraph(
        num_nodes, PyArrayToSharedMultiArray<float,2>(edge_list)));
      break;
    case RESERVOIR:
      int num_nodes;
      PyArrayObject* edge_list; // Nx2 dimensional array
      PyArrayObject* weights; // Nx1 dimensional array
      if (!PyArg_ParseTuple(args, "iOO", &num_nodes, &edge_list, &weights)) {
        std::cout << "Error parsing Integrator arguments" << std::endl;
        assert(0);
      }
      new_integrator = new NetIntegrator<float>(graphs::ConvertEdgeListToDiGraph(
        num_nodes, PyArrayToSharedMultiArray<float,2>(edge_list),
        PyArrayToSharedMultiArray<float,1>(weights)));
      break;
    default:
      std::cout << "Integrator case not supported... exiting." << std::endl;
      assert(0);
  }

  return new_integrator;
}

/*
 * Add new layer in additional lines below:
 */
static PyMethodDef LayerMethods[] = {
  { "CreateLayer", (PyCFunction) CreateLayer,
          METH_VARARGS | METH_KEYWORDS,
          "Returns a handle to a Layer"},
  { "CreateMotorLayer", (PyCFunction) CreateMotorLayer,
          METH_VARARGS | METH_KEYWORDS,
          "Returns a handle to a MotorLayer"},
      //Additional layers here, make sure to add includes top
  { NULL, NULL, 0, NULL}
};

static struct PyModuleDef LayerModule = {
  PyModuleDef_HEAD_INIT,
  "layer_generator",
  "Returns a handle to a Layer",
  -1,
  LayerMethods
};

PyMODINIT_FUNC PyInit_layer_generator(void) {
  return PyModule_Create(&LayerModule);
}
