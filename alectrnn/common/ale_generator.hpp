/*
 * ale_generator.h
 *
 *  Created on: Sep 2, 2017
 *      Author: Nathaniel Rodriguez
 *
 * ALE or Arcade-Learning-Environment wrapper for Python.
 * It generates a new environment and returns the pointer to the environment
 * in a Python Capsule for use in almost every other class. ALE is passed to
 * both the agent and controller classes for access to the ALE environment.
 *
 */

#ifndef ALECTRNN_COMMON_ALE_GENERATOR_H_
#define ALECTRNN_COMMON_ALE_GENERATOR_H_

#include <Python.h>

PyMODINIT_FUNC PyInit_ale_generator(void);

#endif /* ALECTRNN_COMMON_ALE_GENERATOR_H_ */
