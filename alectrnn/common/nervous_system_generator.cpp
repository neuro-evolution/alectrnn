/*
 * nervous_system_generator.cpp
 *
 *  Created on: Jan 2, 2018
 *      Author: Nathaniel Rodriguez
 *
 * Currently, unfortunately, PyCapsules are used to give those pointers.
 * NervousSystem TAKES OWNERSHIP of the Layers, the PyCapsules are not
 * the owners and will not destroy them once they go out of scope in Python.
 * The capsules act only to pass the Layers to the NervousSystem, and should
 * be discarded and can be considered temporary.
 */

#include <Python.h>
#include <cstddef>
#include <iostream>
#include <vector>
#include "nervous_system_generator.hpp"
#include "layer.hpp"
#include "nervous_system.hpp"
#include "capi_tools.hpp"

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
  static char *keyword_list[] = {"input_shape", "layers", NULL};

  PyArrayObject* input_shape;
  PyObject* layers_tuple;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", keyword_list,
      &input_shape, &layers_tuple)) {
    std::cerr << "Error parsing CreateLayer arguments" << std::endl;
    return NULL;
  }

  std::vector<std::size_t> shape = alectrnn::uInt64PyArrayToVector<std::size_t>(input_shape);
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
      std::cerr << "Invalid pointer to Layer returned from capsule,"
          " or is not a capsule." << std::endl;
      return NULL;
    }

    nervous_system::Layer<float>* layer = static_cast<nervous_system::Layer<float>*>(
      PyCapsule_GetPointer(layer_capsule, "layer_generator.layer"));

    nervous_system->AddLayer(layer);
  }

  return nervous_system;
}

/*
 * Add new NervousSystems in additional lines below:
 */
static PyMethodDef NervousSystemMethods[] = {
  { "CreateNervousSystem", (PyCFunction) CreateNervousSystem,
          METH_VARARGS | METH_KEYWORDS,
          "Returns a handle to a NervousSystem"},
      //Additional layers here, make sure to add includes top
  { NULL, NULL, 0, NULL}
};

static struct PyModuleDef NervousSystemModule = {
  PyModuleDef_HEAD_INIT,
  "nervous_system_generator",
  "Returns a handle to a Nervous System",
  -1,
  NervousSystemMethods
};

PyMODINIT_FUNC PyInit_nn_generator(void) {
  import_array();
  return PyModule_Create(&NervousSystemModule);
}
