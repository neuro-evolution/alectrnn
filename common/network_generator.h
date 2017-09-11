#ifndef ALECTRNN_COMMON_NETWORK_GENERATOR_H_
#define ALECTRNN_COMMON_NETWORK_GENERATOR_H_

#include <cstddef>
#include <vector>

namespace ctrnn {

struct InEdge;
std::vector< std::vector<InEdge> > All2AllNetwork(std::size_t num_nodes,
    const double *weights);
std::vector< std::vector<InEdge> > FullSensorNetwork(std::size_t num_sensors,
    std::size_t num_nodes, const double *weights);

}

#endif /* ALECTRNN_COMMON_NETWORK_GENERATOR_H_ */
