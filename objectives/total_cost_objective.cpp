/*
 * total_reward_objective.cpp
 *
 *  Created on: Sep 2, 2017
 *      Author: nathaniel
 *
 */

#include <Python.h>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <ale_interface.hpp>
#include "arrayobject.h"
#include "total_cost_objective.h"
#include "../controllers/controller.h"
#include "../agents/player_agent.h"
#include "../agents/ctrnn_agent.h"
#include "../common/utilities.h"
#include "../common/screen_preprocessing.h"

static PyObject *TotalCostObjective(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  char *keyword_list[] = {"parameters", "ale", "num_neurons", NULL};

  PyArrayObject* py_parameter_array;
  PyObject* ale_capsule;
  std::size_t num_neurons;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!Oi", keyword_list,
      &PyArray_Type, &py_parameter_array, &ale_capsule, &num_neurons)){
    return NULL;
  }

  ALEInterface* ale = (ALEInterface *)PyCapsule_GetPointer(ale_capsule,
      "ale_generator.ale");
  double* cparameter_array(alectrnn::PyArrayToCArray(py_parameter_array));
  double total_cost(alectrnn::CalculateTotalCost(ale, cparameter_array,
                                                 num_neurons));

  return Py_BuildValue("d", total_cost);
}

static PyMethodDef TotalCostMethods[] = {
  { "TotalCostObjective", (PyCFunction) TotalCostObjective,
      METH_VARARGS | METH_KEYWORDS,
      "Objective function that sums game reward"},
  { NULL, NULL, 0, NULL}
};

static struct PyModuleDef TotalCostModule = {
  PyModuleDef_HEAD_INIT,
  "total_reward_objective",
  "Objective function that sums game reward",
  -1,
  TotalCostMethods
};

PyMODINIT_FUNC PyInit_total_reward_objective(void) {
  return PyModule_Create(&TotalCostModule);
}

namespace alectrnn {

double CalculateTotalCost(ALEInterface *ale, double* parameters,
    std::size_t num_neurons) {

  PlayerAgent* agent = new CtrnnAgent(ale, parameters, num_neurons);
  Controller* game_controller = new Controller(ale, agent);
  game_controller->Run();
  double total_cost(-(double)game_controller->getCumulativeScore());

  delete agent;
  delete game_controller;

  return total_cost;
}

}
