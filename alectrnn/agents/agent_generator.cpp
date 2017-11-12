/*
 * agent_generator.cpp
 *
 *  Created on: Sep 11, 2017
 *      Author: Nathaniel Rodriguez
 *
 * The agent_generator is a python extension used to pass an agent pointer
 * in a capsule to python, where it can be returned and used by other c++
 * functions, such as the conntroller class or objective function.
 *
 * Add any additional agents to this file. Notes below show where information
 * needs to be added for each new agent. The destructor provided is generic to
 * the PlayerAgent class and can be used by any of the agents for destruction.
 *
 * Ideally we want a wrapper PlayerAgent class that holds the ale capsule so
 * that on creation it increments the reference to the ale capsule and
 * decrements it on destruction.
 * Py_DECREF(ale_capsule);
 * Py_XINCREF(ale_capsule);
 */

#include <Python.h>
#include <ale_interface.hpp>
#include <cstddef>
#include <iostream>
#include "agent_generator.h"
#include "player_agent.h"
// Add includes to agents you wish to add below:
#include "ctrnn_agent.h"

/*
 * DeleteAgent can be shared among the agents as a destructor
 */
static void DeleteAgent(PyObject *agent_capsule) {
  delete (alectrnn::PlayerAgent *)PyCapsule_GetPointer(
        agent_capsule, "agent_generator.agent");
}

/*
 * Create a new py_function CreateXXXAgent for each agent you want to add
 */
static PyObject *CreateCtrnnAgent(PyObject *self, PyObject *args,
    PyObject *kwargs) {
  static char *keyword_list[] = {"ale", "num_neurons", "num_sensor_neurons",
                          "input_screen_width", "input_screen_height",
                          "use_color", "step_size", NULL};

  PyObject* ale_capsule;
  int num_neurons;
  int num_sensor_neurons;
  int input_screen_width;
  int input_screen_height;
  int use_color; //int instead of bool because python api can't deal with bool
  float step_size;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oiiiiif", keyword_list,
      &ale_capsule, &num_neurons, &num_sensor_neurons,
      &input_screen_width, &input_screen_height, &use_color, &step_size)) {
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

  alectrnn::PlayerAgent *agent = new alectrnn::CtrnnAgent(ale,
      static_cast<std::size_t>(num_neurons),
      static_cast<std::size_t>(num_sensor_neurons),
      static_cast<std::size_t>(input_screen_width),
      static_cast<std::size_t>(input_screen_height),
      static_cast<bool>(use_color), step_size);

  PyObject* agent_capsule = PyCapsule_New(static_cast<void*>(agent),
                                "agent_generator.agent", DeleteAgent);
  return agent_capsule;
}

/*
 * Add new agents in additional lines below:
 */
static PyMethodDef AgentMethods[] = {
  { "CreateCtrnnAgent", (PyCFunction) CreateCtrnnAgent,
          METH_VARARGS | METH_KEYWORDS,
          "Returns a handle to a CtrnnAgent"},
      //Additional agents here, make sure to add includes top
  { NULL, NULL, 0, NULL}
};

static struct PyModuleDef AgentModule = {
  PyModuleDef_HEAD_INIT,
  "agent_generator",
  "Returns a handle to an Agent",
  -1,
  AgentMethods
};

PyMODINIT_FUNC PyInit_agent_generator(void) {
  return PyModule_Create(&AgentModule);
}
