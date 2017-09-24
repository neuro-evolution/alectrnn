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
 */

#include <Python.h>
#include <ale_interface.hpp>
#include <cstddef>
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
  char *keyword_list[] = {"ale", "num_neurons", "num_sensor_neurons",
                          "input_screen_width", "input_screen_height",
                          "use_color", "step_size", NULL};

  PyObject* ale_capsule;
  std::size_t num_neurons;
  std::size_t num_sensor_neurons;
  std::size_t input_screen_width;
  std::size_t input_screen_height;
  bool use_color;
  double step_size;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oiiiipd", keyword_list,
      &ale_capsule, &num_neurons, &num_sensor_neurons,
      &input_screen_width, &input_screen_height, &use_color, &step_size)){
    return NULL;
  }

  ALEInterface* ale = (ALEInterface*)PyCapsule_GetPointer(ale_capsule,
      "ale_generator.ale");

  alectrnn::PlayerAgent *agent = new alectrnn::CtrnnAgent(ale, num_neurons,
      num_sensor_neurons, input_screen_width, input_screen_height,
      use_color, step_size);

  PyObject* agent_capsule;
  agent_capsule = PyCapsule_New((void *) ale, "agent_generator.agent",
                                DeleteAgent);
  return Py_BuildValue("O", agent_capsule);
}

/*
 * Add new agents in additional lines below:
 */
static PyMethodDef AgentMethods[] = {
  { "CreatCtrnnAgent", (PyCFunction) CreateCtrnnAgent,
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



