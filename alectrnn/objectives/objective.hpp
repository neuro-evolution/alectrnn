/*
 * objective.h
 *
 *  Created on: Sep 2, 2017
 *      Author: Nathaniel Rodriguez
 *
 * A Python extension module that creates a callable function from python for
 * use by optimization algorithms. It requires a parameter argument,
 * as well as a handler (python capsule) to the ALE environment, and
 * a handler to an agent.

 */

#ifndef ALECTRNN_OBJECTIVES_OBJECTIVE_H_
#define ALECTRNN_OBJECTIVES_OBJECTIVE_H_

#include <Python.h>
#include <ale_interface.hpp>
#include "../agents/player_agent.hpp"

PyMODINIT_FUNC PyInit_objective(void);

namespace alectrnn {

float CalculateTotalCost(const float* parameters, ALEInterface *ale,
    PlayerAgent* agent);

}

#endif /* OBJECTIVES_OBJECTIVE_H_ */
