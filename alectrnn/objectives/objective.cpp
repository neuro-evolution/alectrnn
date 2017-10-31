/*
 * total_reward_objective.cpp
 *
 *  Created on: Sep 2, 2017
 *      Author: Nathaniel Rodriguez
 *
 * A Python extension module that creates a callable function from python for
 * use by optimization algorithms. It requires a parameter argument,
 * as well as a handler (python capsule) to the ALE environment, and
 * a handler to an agent.
 *
 */

#include <Python.h>
#include <vector>
#include <memory>
#include <cstdint>
#include <cstddef>
#include <ale_interface.hpp>
#include <iostream>
#include "arrayobject.h"
#include "objective.h"
#include "../controllers/controller.h"
#include "../agents/player_agent.h"
#include "../common/utilities.h"

static PyObject *TotalCostObjective(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  static char *keyword_list[] = {"parameters", "ale", "agent", NULL};

  PyArrayObject* py_parameter_array;
  PyObject* ale_capsule;
  PyObject* agent_capsule;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", keyword_list,
      &py_parameter_array, &ale_capsule, &agent_capsule)) {
    std::cout << "Invalid argument in put into objective!" << std::endl;
    return NULL;
  }

  if (!PyCapsule_IsValid(ale_capsule, "ale_generator.ale") ||
      !PyCapsule_IsValid(agent_capsule, "agent_generator.agent"))
  {
    std::cout << "Invalid pointer to returned from capsule,"
        " or is not correct capsule." << std::endl;
    return NULL;
  }

  ALEInterface* ale = static_cast<ALEInterface*>(PyCapsule_GetPointer(
      ale_capsule, "ale_generator.ale"));
  alectrnn::PlayerAgent* player_agent =
      static_cast<alectrnn::PlayerAgent*>(PyCapsule_GetPointer(agent_capsule,
          "agent_generator.agent"));
  double* cparameter_array(alectrnn::PyArrayToCArray(py_parameter_array));
  double total_cost(alectrnn::CalculateTotalCost(cparameter_array, ale,
      player_agent));

  return Py_BuildValue("d", total_cost);
}

static PyMethodDef ObjectiveMethods[] = {
  { "TotalCostObjective", (PyCFunction) TotalCostObjective,
      METH_VARARGS | METH_KEYWORDS,
      "Objective function that sums game reward"},
      //Additional objectives here
  { NULL, NULL, 0, NULL}
};

static struct PyModuleDef ObjectiveModule = {
  PyModuleDef_HEAD_INIT,
  "objective",
  "Objective function that sums game reward",
  -1,
  ObjectiveMethods
};

PyMODINIT_FUNC PyInit_objective(void) {
  return PyModule_Create(&ObjectiveModule);
}

namespace alectrnn {

double CalculateTotalCost(const double* parameters, ALEInterface *ale,
    PlayerAgent* agent) {

  agent->Configure(parameters);
  std::unique_ptr<Controller> game_controller =
      std::make_unique<Controller>(ale, agent);
  game_controller->Run();
  double total_cost(-(double)game_controller->getCumulativeScore());

  return total_cost;
}

}
