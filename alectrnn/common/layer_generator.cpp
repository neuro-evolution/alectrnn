/*
 * layer_generator.cpp
 *
 *  Created on: Jan 2, 2018
 *      Author: Nathaniel Rodriguez
 *
 * The layer_generator is a python extension used to create layers in a NN.
 * Integrators
 * ALL2ALL: (#_states in layer, # states in prev layer)
 * Conv3D: (# filters, filter shape, layer_shape, prev_layer_shape, stride)
 * Recurrent: graph - as (Nx2)
 * Reservoir: weighted graph - as (Nx2), (Nx1)
 *
 * Activators
 * CTRNN: (# states, step size)
 * CTRNNConv: (shape, step_size)
 *
 * Layers
 * Layer (back*, self*, activation*, shape)
 * Motor (# in, # out, activation*)
 * Input (shape) -- made by nervous_system IGNORE
 *
 * CreateFunction
 * Layer {"back": TYPE, (TUPLE); "self": TYPE, (TUPLE); "act": TYPE, (TUPLE), "shape": (TUPLE)}
 * Motor ("ale": ALE, "in": NUM, "act": TYPE, (TUPLE))
 */

#include <Python.h>
#include <cassert>
#include <ale_interface.hpp>
#include <cstddef>
#include <inttypes.h>
#include <iostream>
#include "layer_generator.hpp"
#include "arrayobject.h"
#include "layer.hpp"
#include "graphs.hpp"
#include "capi_tools.hpp"
#include "multi_array.hpp"
#include "activator.hpp"
#include "integrator.hpp"

/*
 * DeleteLayer can be shared among the Layers as a destructor
 * Nothing is deleted because ownership is supposed to be passed to
 * the NervousSystem, which will handle garbage collection.
 */
static void DeleteLayer(PyObject *layer_capsule) {
  // delete (nervous_system::Layer *)PyCapsule_GetPointer(
  //       layer_capsule, "layer_generator.layer");
}

/*
 * Create a new py_function CreateXXXLayer for each layer you want to add
 */
static PyObject *CreateLayer(PyObject *self, PyObject *args, PyObject *kwargs) {
  static char *keyword_list[] = {"back_integrator", "back_integrator_args",
    "self_integrator", "self_integrator_args", 
    "activator_type", "activator_args", "shape", NULL};

  int back_integrator_type;
  PyObject* back_integrator_args;
  int self_integrator_type;
  PyObject* self_integrator_args;
  int activator_type;
  PyObject* activator_args;
  PyArrayObject* shape;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iOiOiOO", keyword_list,
      &back_integrator_type, &back_integrator_args, &self_integrator_type,
      &self_integrator_args, &activator_type, &activator_args,
      &shape)) {
    std::cerr << "Error parsing CreateLayer arguments" << std::endl;
    return NULL;
  }

  // Call parsers -> they create NEW integrators and activators
  layer_shape = alectrnn::PyArrayToVector<float>(shape);
  nervous_system::Integrator<float>* back_integrator = IntegratorParser(
    back_integrator_type, back_integrator_args);
  nervous_system::Integrator<float>* self_integrator = IntegratorParser(
    self_integrator_type, self_integrator_args);
  nervous_system::Activator<float>* activator = ActivatorParser(
    activator_type, activator_args);

  // Ownership is transfered to new layer
  nervous_system::Layer<float>* layer = new nervous_system::Layer<float>(
    layer_shape, back_integrator, self_integrator, activator);

  PyObject* layer_capsule = PyCapsule_New(static_cast<void*>(layer),
                                "layer_generator.layer", DeleteLayer);
  return layer_capsule;
}

static PyObject *CreateMotorLayer(PyObject *self, PyObject *args, PyObject *kwargs) {
  static char *keyword_list[] = {"num_outputs", "num_inputs", 
                                "activator_type", "activator_args", NULL};

  int num_outputs;
  int num_inputs;
  int activator_type;
  PyObject* activator_args;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiiO", keyword_list,
      &num_outputs, &num_inputs, &activator_type, &activator_args)) {
    std::cerr << "Error parsing CreateMotorLayer arguments" << std::endl;
    return NULL;
  }

  nervous_system::Activator<float>* activator = ActivatorParser(
    activator_type, activator_args);

  nervous_system::Layer<float>* layer = new nervous_system::MotorLayer<float>(
    num_outputs, num_inputs, activator);

  PyObject* layer_capsule = PyCapsule_New(static_cast<void*>(layer),
                                "layer_generator.layer", DeleteLayer);
  return layer_capsule;
}

nervous_system::Activator<float>* ActivatorParser(nervous_system::ACTIVATOR_TYPE type, PyObject* args) {

  nervous_system::Activator<float>* new_activator;  
  switch(type) {
    case nervous_system::IDENTITY_ACTIVATOR:
      new_activator = new nervous_system::IdentityActivator<float>();
      break;

    case nervous_system::CTRNN_ACTIVATOR:
      int num_states;
      float step_size;
      if (!PyArg_ParseTuple(args, "if", &num_states, &step_size)) {
        std::cerr << "Error parsing Activator arguments" << std::endl;
        assert(0);
      }
      new_activator = new nervous_system::CTRNNActivator<float>(num_states, step_size);
      break;

    case nervous_system::CONV_CTRNN_ACTIVATOR:
      PyArrayObject* shape;
      float step_size;
      if (!PyArg_ParseTuple(args, "Of", &shape, &step_size)) {
        std::cerr << "Error parsing Activator arguments" << std::endl;
        assert(0);
      }
      new_activator = new nervous_system::Conv3DCTRNNActivator<float>(
        multi_array::Array<std::size_t,3>(shape->data), step_size);
      break;

    default:
      std::cerr << "Activator case not supported... exiting." << std::endl;
      assert(0);
  }

  return new_activator;
}

nervous_system::Integrator<float>* IntegratorParser(nervous_system::INTEGRATOR_TYPE type, PyObject* args) {

  nervous_system::Integrator<float>* new_integrator;  
  switch(type) {
    case nervous_system::NONE_INTEGRATOR:
      new_integrator = new nervous_system::NoneIntegrator<float>();
      break;

    case nervous_system::ALL2ALL_INTEGRATOR:
      int num_states;
      int num_prev_states;
      if (!PyArg_ParseTuple(args, "ii", &num_states, &num_prev_states)) {
        std::cerr << "Error parsing Integrator arguments" << std::endl;
        assert(0);
      }
      new_integrator = new nervous_system::All2AllIntegrator<float>(
        num_states, num_prev_states);
      break;

    case nervous_system::CONV_INTEGRATOR:
      PyArrayObject* filter_shape;
      PyArrayObject* layer_shape;
      PyArrayObject* prev_layer_shape;
      int stride;
      if (!PyArg_ParseTuple(args, "OOOi", &filter_shape,
        &layer_shape, &prev_layer_shape, &stride)) {
        std::cerr << "Error parsing Integrator arguments" << std::endl;
        assert(0);
      }
      new_integrator = new nervous_system::Conv3DIntegrator<float>(
        multi_array::Array<std::size_t,3>(filter_shape->data),
        multi_array::Array<std::size_t,3>(layer_shape->data),
        multi_array::Array<std::size_t,3>(prev_layer_shape->data),
        stride);
      break;

    case nervous_system::RECURRENT_INTEGRATOR:
      int num_nodes;
      PyArrayObject* edge_list; // Nx2 dimensional array
      if (!PyArg_ParseTuple(args, "iO", &num_nodes, &edge_list)) {
        std::cerr << "Error parsing Integrator arguments" << std::endl;
        assert(0);
      }
      new_integrator = new nervous_system::RecurrentIntegrator<float>(
        graphs::ConvertEdgeListToPredecessorGraph(
        num_nodes, alectrnn::PyArrayToSharedMultiArray<std::uint64_t,2>(edge_list)));
      break;

    case nervous_system::RESERVOIR_INTEGRATOR:
      int num_nodes;
      PyArrayObject* edge_list; // Nx2 dimensional array
      PyArrayObject* weights; // Nx1 dimensional array
      if (!PyArg_ParseTuple(args, "iOO", &num_nodes, &edge_list, &weights)) {
        std::cerr << "Error parsing Integrator arguments" << std::endl;
        assert(0);
      }
      new_integrator = new nervous_system::ReservoirIntegrator<float>(
        graphs::ConvertEdgeListToPredecessorGraph(
        num_nodes, alectrnn::PyArrayToSharedMultiArray<std::uint64_t,2>(edge_list),
        alectrnn::PyArrayToSharedMultiArray<float,1>(weights)));
      break;

    default:
      std::cerr << "Integrator case not supported... exiting." << std::endl;
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
