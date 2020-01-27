/*
 * agent_handler.h
 *
 *  Created on: Jan 20, 2018
 *      Author: Nathaniel Rodriguez
 *
 *
 */

#ifndef AGENTS_AGENT_HANDLER_H_
#define AGENTS_AGENT_HANDLER_H_

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "../common/multi_array.hpp"

PyMODINIT_FUNC PyInit_agent_handler(void);
PyObject *ConvertLogToPyArray(const std::vector<multi_array::Tensor<float>>& history);

#endif /* AGENTS_AGENT_HANDLER_H_ */
