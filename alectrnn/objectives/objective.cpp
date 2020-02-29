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
#include "numpy/arrayobject.h"
#include "objective.hpp"
#include "../controllers/controller.hpp"
#include "../agents/player_agent.hpp"
#include "../agents/nervous_system_agent.hpp"
#include "../nervous_system/nervous_system.hpp"
#include "../common/capi_tools.hpp"
#include "../nervous_system/integrator.hpp"
#include "../common/multi_array.hpp"

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

  float* cparameter_array(alectrnn::PyArrayToCArray(py_parameter_array));
  float total_cost(0);
  // int needs_api = PyDataType_REFCHK(PyArray_DESCR(self));
  // NPY_BEGIN_THREADS_DEF;
  // if (!needs_api)
  // {
  //   NPY_BEGIN_THREADS; //NPY_BEGIN_ALLOW_THREADS;
  // }
  PyGILState_STATE gstate = PyGILState_Ensure();
  PyGILState_Release(gstate);

  total_cost = alectrnn::CalculateTotalCost(cparameter_array, ale,
                                            player_agent);

  // if (!needs_api)
  // {
  //   NPY_END_THREADS; //NPY_END_ALLOW_THREADS;
  // }
  return Py_BuildValue("f", total_cost);
}

/*
 * This objective adds the negative of the score to the # edge * scaling factor
 *
 * Note: The number of edges is calculated. If you want to scale this
 * according to the total number of possible edges, then factor that into the
 * scaling parameter
 *
 * Note: This only supports TRUNCATED_RECURRENT_INTEGRATOR, as it is the only
 * one with weight truncation. As new ones are added ill expand the code.
 */
static PyObject *ScoreAndConnectionCostObjective(PyObject *self, PyObject *args,
                                    PyObject *kwargs)
{
  static char *keyword_list[] = {"parameters", "ale", "agent",
                                 "cc_scale", NULL};

  PyArrayObject* py_parameter_array;
  PyObject* ale_capsule;
  PyObject* agent_capsule;
  double cc_scale;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOd", keyword_list,
                                   &py_parameter_array, &ale_capsule,
                                   &agent_capsule, &cc_scale)) {
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

  float* cparameter_array(alectrnn::PyArrayToCArray(py_parameter_array));
  float total_cost(alectrnn::CalculateTotalCost(cparameter_array, ale,
                                                player_agent));

  // CalculateConnectionCost Needs to be called second so that CalculatTotalCost
  // can configure the agent first.
  // Note: s&cc only compatible with NervousSystemAgents atm
  std::uint64_t connection_cost(alectrnn::CalculateConnectionCost(
    dynamic_cast<alectrnn::NervousSystemAgent*>(player_agent)));
  total_cost += static_cast<float>(cc_scale * connection_cost);
  return Py_BuildValue("f", total_cost);
}

static PyMethodDef ObjectiveMethods[] = {
  { "TotalCostObjective", (PyCFunction) TotalCostObjective,
      METH_VARARGS | METH_KEYWORDS,
      "Objective function that sums game reward"},
  { "ScoreAndConnectionCostObjective", (PyCFunction) ScoreAndConnectionCostObjective,
    METH_VARARGS | METH_KEYWORDS,
    "Objective function that sums game reward and penalizes connections"},
      //Additional objectives here
  { NULL, NULL, 0, NULL}
};

static struct PyModuleDef ObjectiveModule = {
  PyModuleDef_HEAD_INIT,
  "objective",
  "Objective functions",
  -1,
  ObjectiveMethods
};

PyMODINIT_FUNC PyInit_objective(void) {
  import_array();
  return PyModule_Create(&ObjectiveModule);
}

namespace alectrnn {

/*
 * Adds up the negative of the score of the agent on the atari game.
 */
float CalculateTotalCost(const float* parameters, ALEInterface *ale,
    PlayerAgent* agent) {

  agent->Configure(parameters);
  Controller game_controller = Controller(ale, agent);
  game_controller.Run();
  float total_cost(-(float)game_controller.getCumulativeScore());

  return total_cost;
}

/*
 * Adds up the number of weights whose absolute value is greater than the
 * threshold. Such weights are considered non-zero and count as a connection.
 * Hence, this function adds up the number of connections.
 */
std::uint64_t CalculateConnectionCost(alectrnn::NervousSystemAgent* agent) {

  std::uint64_t cost = 0;

  // Loop through the agent's nervous system layers
  const nervous_system::NervousSystem<float>& neural_net = agent->GetNeuralNet();
  for (std::size_t iii = 0; iii < neural_net.size(); ++iii) {

    if (neural_net[iii].GetBackIntegrator() != nullptr) {
      if (neural_net[iii].GetBackIntegrator()->GetIntegratorType()
          == nervous_system::TRUNCATED_RECURRENT_INTEGRATOR) {
        // Get the number of connections present
        const nervous_system::TruncatedRecurrentIntegrator<float>* integrator =
        dynamic_cast<const nervous_system::TruncatedRecurrentIntegrator<float>*>(
        neural_net[iii].GetBackIntegrator());
        cost += CalculateNumConnection(integrator->GetWeights(), integrator->GetWeightThreshold());
      }
    }

    if (neural_net[iii].GetSelfIntegrator() != nullptr) {
      if (neural_net[iii].GetSelfIntegrator()->GetIntegratorType()
          == nervous_system::TRUNCATED_RECURRENT_INTEGRATOR) {
        // Get the number of connections present
        const nervous_system::TruncatedRecurrentIntegrator<float>* integrator =
        dynamic_cast<const nervous_system::TruncatedRecurrentIntegrator<float>*>(
        neural_net[iii].GetSelfIntegrator());
        cost += CalculateNumConnection(integrator->GetWeights(), integrator->GetWeightThreshold());
      }
    }
  }

  return cost;
}

std::uint64_t CalculateNumConnection(const multi_array::ConstArraySlice<float>& weights,
                                     float weight_threshold) {
  std::uint64_t num_connections = 0;
  for (std::size_t iii = 0; iii < weights.size(); ++iii) {
    if ((weights[iii] > weight_threshold && weights[iii] >= 0) ||
        (weights[iii] < -weight_threshold && weights[iii] <= 0)) {
      num_connections += 1;
    }
  }

  return num_connections;
}

}
