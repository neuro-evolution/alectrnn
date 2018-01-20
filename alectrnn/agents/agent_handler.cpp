/*
 * agent_handler.cpp
 *
 *  Created on: Jan 20, 2018
 *      Author: Nathaniel Rodriguez
 *
 */

#include <Python.h>
#include <ale_interface.hpp>
#include <cstddef>
#include <iostream>
#include "nervous_system.hpp"
#include "agent_handler.hpp"
#include "player_agent.hpp"
// Add includes to agents you wish to add below:
#include "ctrnn_agent.hpp"
#include "nervous_system_agent.hpp"

/*
 * Create a new py_function for each Agent type to handle
 */
static PyObject *GetAgentHistory(PyObject *self, PyObject *args,
    PyObject *kwargs) {

}

static PyMethodDef AgentHandlerMethods[] = {
  { "GetAgentHistory", (PyCFunction) GetAgentHistory,
          METH_VARARGS | METH_KEYWORDS,
          "Returns PyArray to agent state history"},
      //Additional agents here, make sure to add includes top
  { NULL, NULL, 0, NULL}
};

static struct PyModuleDef AgentHandlerModule = {
  PyModuleDef_HEAD_INIT,
  "agent_handler",
  "Returns a handle to an Agent",
  -1,
  AgentHandlerMethods
};

PyMODINIT_FUNC PyInit_agent_handler(void) {
  return PyModule_Create(&AgentHandlerModule);
}
