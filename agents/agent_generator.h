/*
 * agent_generator.h
 *
 *  Created on: Sep 1, 2017
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

#ifndef AGENTS_AGENT_GENERATOR_H_
#define AGENTS_AGENT_GENERATOR_H_

#include <Python.h>

PyMODINIT_FUNC PyInit_agent_generator(void);

#endif /* AGENTS_AGENT_GENERATOR_H_ */
