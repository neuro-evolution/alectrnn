/*
 * total_reward_objective.h
 *
 *  Created on: Sep 2, 2017
 *      Author: nathaniel
 */

#ifndef ALECTRNN_OBJECTIVES_TOTAL_COST_OBJECTIVE_H_
#define ALECTRNN_OBJECTIVES_TOTAL_COST_OBJECTIVE_H_

#include <Python.h>
#include <ale_interface.hpp>
#include "../agents/player_agent.h"

PyMODINIT_FUNC PyInit_total_cost_objective(void);

namespace alectrnn {

double CalculateTotalCost(ALEInterface *ale, double* parameters,
    PlayerAgent* agent);

}

#endif /* OBJECTIVES_TOTAL_COST_OBJECTIVE_H_ */
