/*
 * network_constructor.h
 *
 *  Created on: Sep 2, 2017
 *      Author: Nathaniel Rodriguez
 *
 * Defines a couple functions that can be used to generate all-to-all networks.
 * New functions may be added in the future to create more specific structures.
 * The InEdge struct is used as the basic block for NeuralNetworks.
 *
 */

#ifndef ALECTRNN_COMMON_NETWORK_CONSTRUCTOR_H_
#define ALECTRNN_COMMON_NETWORK_CONSTRUCTOR_H_

#include <cstddef>
#include <vector>

namespace ctrnn {

struct InEdge {
  InEdge();
  InEdge(int source_node, float source_weight);
  int source;
  float weight;
};

std::vector< std::vector<InEdge> > All2AllNetwork(std::size_t num_nodes,
    const float *weights);
std::vector< std::vector<InEdge> > All2AllNetwork(std::size_t num_nodes,
    float default_weight=1.0);
std::vector< std::vector<InEdge> > FullSensorNetwork(std::size_t num_sensors,
    std::size_t num_nodes, const float *weights);
std::vector< std::vector<InEdge> > FullSensorNetwork(std::size_t num_sensors,
    std::size_t num_nodes, float default_weight=1.0);
void FillFullSensorNetwork(std::vector< std::vector<InEdge> > &network,
                           const float *weights);
void FillAll2AllNetwork(std::vector< std::vector<InEdge> > &network,
                        const float *weights);
}

#endif /* ALECTRNN_COMMON_NETWORK_CONSTRUCTOR_H_ */
