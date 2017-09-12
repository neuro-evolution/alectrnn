#ifndef ALECTRNN_COMMON_NETWORK_GENERATOR_H_
#define ALECTRNN_COMMON_NETWORK_GENERATOR_H_

#include <cstddef>
#include <vector>

namespace ctrnn {

struct InEdge;
std::vector< std::vector<InEdge> > All2AllNetwork(std::size_t num_nodes,
    const double *weights);
std::vector< std::vector<InEdge> > All2AllNetwork(std::size_t num_nodes,
    double default_weight=1.0);
std::vector< std::vector<InEdge> > FullSensorNetwork(std::size_t num_sensors,
    std::size_t num_nodes, const double *weights);
std::vector< std::vector<InEdge> > FullSensorNetwork(std::size_t num_sensors,
    std::size_t num_nodes, double default_weight=1.0);
void FillFullSensorNetwork(std::vector< std::vector<InEdge> > &network,
                           const double *weights);
void FillAll2AllNetwork(std::vector< std::vector<InEdge> > &network,
                        const double *weights);
}

#endif /* ALECTRNN_COMMON_NETWORK_GENERATOR_H_ */
