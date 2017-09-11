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
#include <cstddef>

PyMODINIT_FUNC PyInit_total_cost_objective(void);

namespace alectrnn {

double CalculateTotalCost(ALEInterface *ale, double* parameters,
    std::size_t num_neurons);

}

#endif /* OBJECTIVES_TOTAL_COST_OBJECTIVE_H_ */
