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

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <ale_interface.hpp>
#include <cstdint>
#include "../agents/player_agent.hpp"
#include "../common/multi_array.hpp"
#include "../agents/nervous_system_agent.hpp"

PyMODINIT_FUNC PyInit_objective(void);

namespace alectrnn {

float CalculateTotalCost(const float* parameters, ALEInterface *ale,
                         PlayerAgent* agent);
std::uint64_t CalculateConnectionCost(alectrnn::NervousSystemAgent* agent);
std::uint64_t CalculateNumConnection(const multi_array::ConstArraySlice<float>& weights,
                                     float weight_threshold);
}

#endif /* ALECTRNN_OBJECTIVES_OBJECTIVE_H_ */
