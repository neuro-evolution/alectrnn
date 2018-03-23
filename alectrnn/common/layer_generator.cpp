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
#include <stdexcept>
#include <cstddef>
#include <cinttypes>
#include <iostream>
#include <vector>
#include "layer_generator.hpp"
#include "numpy/arrayobject.h"
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
  std::vector<std::size_t> layer_shape = alectrnn::uInt64PyArrayToVector<std::size_t>(shape);
  nervous_system::Integrator<float>* back_integrator = IntegratorParser(
    (nervous_system::INTEGRATOR_TYPE) back_integrator_type, back_integrator_args);
  nervous_system::Integrator<float>* self_integrator = IntegratorParser(
    (nervous_system::INTEGRATOR_TYPE) self_integrator_type, self_integrator_args);
  nervous_system::Activator<float>* activator = ActivatorParser(
    (nervous_system::ACTIVATOR_TYPE) activator_type, activator_args);

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
    (nervous_system::ACTIVATOR_TYPE) activator_type, activator_args);

  nervous_system::Layer<float>* layer = new nervous_system::MotorLayer<float>(
    num_outputs, num_inputs, activator);

  PyObject* layer_capsule = PyCapsule_New(static_cast<void*>(layer),
                                "layer_generator.layer", DeleteLayer);
  return layer_capsule;
}

nervous_system::Activator<float>* ActivatorParser(nervous_system::ACTIVATOR_TYPE type, PyObject* args) {
  /*
   * CONV activators should take a shape argument
   * Non-conv activators should take a # states argument
   * Identity is an exception
   */

  nervous_system::Activator<float>* new_activator;  
  switch(type) {
    case nervous_system::IDENTITY_ACTIVATOR: {
      new_activator = new nervous_system::IdentityActivator<float>();
      break;
    }

    case nervous_system::CTRNN_ACTIVATOR: {
      int num_states;
      float step_size;
      if (!PyArg_ParseTuple(args, "if", &num_states, &step_size)) {
        std::cerr << "Error parsing Activator arguments" << std::endl;
        throw std::invalid_argument("CTRNN_Activator couldn't parse tuple");
      }
      new_activator = new nervous_system::CTRNNActivator<float>(num_states, step_size);
      break;
    }

    case nervous_system::CONV_CTRNN_ACTIVATOR: {
      PyArrayObject* shape;
      float step_size;
      if (!PyArg_ParseTuple(args, "Of", &shape, &step_size)) {
        std::cerr << "Error parsing Activator arguments" << std::endl;
        throw std::invalid_argument("CONV CTRNN_Activator couldn't parse tuple");
      }

      // Make sure the numpy array is the correct size
      npy_intp num_shape_elements = PyArray_SIZE(shape);
      if (num_shape_elements != 3) {
        throw std::invalid_argument("Invalid number of shape elements for"
                                    " CONV CTRNN ACTIVATOR (needs 3)");
      }

      new_activator = new nervous_system::Conv3DCTRNNActivator<float>(
        multi_array::Array<std::size_t,3>(alectrnn::uInt64PyArrayToCArray(
        shape)), step_size);
      break;
    }

    case nervous_system::IAF_ACTIVATOR: {
      int num_states;
      float step_size;
      float peak;
      float reset;
      if (!PyArg_ParseTuple(args, "ifff", &num_states, &step_size,
                            &peak, &reset)) {
        std::cerr << "Error parsing Activator arguments" << std::endl;
        throw std::invalid_argument("IAF Activator couldn't parse tuple");
      }

      new_activator = new nervous_system::IafActivator<float>(
        num_states, step_size, peak, reset);
      break;
    }

    case nervous_system::CONV_IAF_ACTIVATOR: {
      PyArrayObject* shape;
      int num_states;
      float step_size;
      float peak;
      float reset;
      if (!PyArg_ParseTuple(args, "Oifff", &shape, &num_states, &step_size,
                            &peak, &reset)) {
        std::cerr << "Error parsing Activator arguments" << std::endl;
        throw std::invalid_argument("IAF Activator couldn't parse tuple");
      }

      // Make sure the numpy array is the correct size
      npy_intp num_shape_elements = PyArray_SIZE(shape);
      if (num_shape_elements != 3) {
        throw std::invalid_argument("Invalid number of shape elements for"
                                    " CONV IAF ACTIVATOR (needs 3)");
      }

      new_activator = new nervous_system::Conv3DIafActivator<float>(
        multi_array::Array<std::size_t,3>(alectrnn::uInt64PyArrayToCArray(
        shape)), step_size, peak, reset);
      break;
    }

    default: {
      std::cerr << "Activator case not supported... exiting." << std::endl;
      throw std::invalid_argument("Not an activator");
    }
  }

  return new_activator;
}

nervous_system::Integrator<float>* IntegratorParser(nervous_system::INTEGRATOR_TYPE type, PyObject* args) {

  nervous_system::Integrator<float>* new_integrator;  
  switch(type) {
    case nervous_system::NONE_INTEGRATOR: {
      new_integrator = new nervous_system::NoneIntegrator<float>();
      break;
    }

    case nervous_system::ALL2ALL_INTEGRATOR: {
      int num_states;
      int num_prev_states;
      if (!PyArg_ParseTuple(args, "ii", &num_states, &num_prev_states)) {
        std::cerr << "Error parsing Integrator arguments" << std::endl;
        throw std::invalid_argument("ALL2ALL Integrator failed to parse"
                                    " tuples");
      }
      new_integrator = new nervous_system::All2AllIntegrator<float>(
        num_states, num_prev_states);
      break;
    }

    case nervous_system::CONV_INTEGRATOR: {
      PyArrayObject* filter_shape;
      PyArrayObject* layer_shape;
      PyArrayObject* prev_layer_shape;
      int stride;
      if (!PyArg_ParseTuple(args, "OOOi", &filter_shape,
        &layer_shape, &prev_layer_shape, &stride)) {
        std::cerr << "Error parsing Integrator arguments" << std::endl;
        throw std::invalid_argument("CONV_INTEGRATOR failed to parse tuples");
      }

      // Make sure the numpy arrays are the correct size
      npy_intp num_filter_elements = PyArray_SIZE(filter_shape);
      if (num_filter_elements != 3) {
        std::cerr << "num of filter elements: " << num_filter_elements << std::endl;
        throw std::invalid_argument("filter has wrong number of elements (needs 3)");
      }
      npy_intp num_layer_elements = PyArray_SIZE(layer_shape);
      if (num_layer_elements != 3) {
        std::cerr << "num of layer elements: " << num_layer_elements << std::endl;
        throw std::invalid_argument("layer has wrong number of elements (needs 3)");
      }
      npy_intp num_prev_layer_elements = PyArray_SIZE(prev_layer_shape);
      if (num_prev_layer_elements != 3) {
        std::cerr << "num of prev layer elements: " << num_prev_layer_elements << std::endl;
        throw std::invalid_argument("prev layer has wrong number of elements (needs 3)");
      }

      new_integrator = new nervous_system::Conv3DIntegrator<float>(
        multi_array::Array<std::size_t,3>(
        alectrnn::uInt64PyArrayToCArray(filter_shape)),
        multi_array::Array<std::size_t,3>(
        alectrnn::uInt64PyArrayToCArray(layer_shape)),
        multi_array::Array<std::size_t,3>(
        alectrnn::uInt64PyArrayToCArray(prev_layer_shape)),
        stride);
      break;
    }

    case nervous_system::RECURRENT_INTEGRATOR: {
      int num_nodes;
      PyArrayObject* edge_list; // Nx2 dimensional array
      if (!PyArg_ParseTuple(args, "iO", &num_nodes, &edge_list)) {
        std::cerr << "Error parsing Integrator arguments" << std::endl;
        throw std::invalid_argument("RECCURENT INTEGRATOR ERROR");
      }

      // Make sure numpy array has correct shape
      int edge_list_ndims = PyArray_NDIM(edge_list);
      if (edge_list_ndims != 2) {
        std::cerr << "edge list dimensions: " << edge_list_ndims << std::endl;
        throw std::invalid_argument("edge list has invalid # of dimensions (needs 2)");
      }
      npy_intp* edge_list_shape = PyArray_SHAPE(edge_list);
      if (edge_list_shape[1] != 2) {
        std::cerr << "edge list shape[1]: " << edge_list_shape[1] << std::endl;
        std::cerr << "edge list shape[1]: REQUIRES " << 2 << std::endl;
        throw std::invalid_argument("edge list is the wrong size");
      }

      new_integrator = new nervous_system::RecurrentIntegrator<float>(
        graphs::ConvertEdgeListToPredecessorGraph(
        num_nodes, alectrnn::PyArrayToSharedMultiArray<std::uint64_t,2>(edge_list)));
      break;
    }

    case nervous_system::RESERVOIR_INTEGRATOR: {
      int num_nodes;
      PyArrayObject* edge_list; // Nx2 dimensional array
      PyArrayObject* weights; // N element array
      if (!PyArg_ParseTuple(args, "iOO", &num_nodes, &edge_list, &weights)) {
        std::cerr << "Error parsing Integrator arguments" << std::endl;
        throw std::invalid_argument("FAILURE IN RESERVOIR INTEGRATOR");
      }

      // Make sure numpy arrays have correct shape
      int edge_list_ndims = PyArray_NDIM(edge_list);
      if (edge_list_ndims != 2) {
        std::cerr << "edge list ndims: " << edge_list_ndims << std::endl;
        std::cerr << "edge list ndims: REQUIRES " << 2 << std::endl;
        throw std::invalid_argument("edge list has wrong dimensions");
      }
      npy_intp* edge_list_shape = PyArray_SHAPE(edge_list);
      if (edge_list_shape[1] != 2) {
        std::cerr << "edge list shape[1]: " << edge_list_shape[1] << std::endl;
        std::cerr << "edge list shape[1]: REQUIRES " << 2 << std::endl;
        throw std::invalid_argument("edge list has wrong shape");
      }
      int weights_ndims = PyArray_NDIM(weights);
      if (weights_ndims != 1) {
        throw std::invalid_argument("weights needs to be a 1D array");
      }

      npy_intp* weights_shape = PyArray_SHAPE(weights);
      if (weights_shape[0] != edge_list_shape[0]) {
        std::cerr << "edge list shape[0]: " << edge_list_shape[0] << std::endl;
        std::cerr << "edge list shape[0]: REQUIRES " << weights_shape[0] << std::endl;
        throw std::invalid_argument("Need same number of weights as edges");
      }

      new_integrator = new nervous_system::ReservoirIntegrator<float>(
        graphs::ConvertEdgeListToPredecessorGraph(
        num_nodes, alectrnn::PyArrayToSharedMultiArray<std::uint64_t,2>(edge_list),
        alectrnn::PyArrayToSharedMultiArray<float,1>(weights)));
      break;
    }

    default: {
      std::cerr << "Integrator case not supported... exiting." << std::endl;
      throw std::invalid_argument("Unsupported integrator value");
    }
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
  import_array();
  return PyModule_Create(&LayerModule);
}
