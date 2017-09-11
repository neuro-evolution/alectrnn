#include <cstddef>
#include <vector>
#include "network_generator.h"

namespace ctrnn {

struct InEdge {
  InEdge(int source_node, double source_weight) 
      : source(source_node), weight(source_weight) {}
  int source;
  double weight;
};

// Creates a vector for all the in coming edges in an all-to-all network
std::vector< std::vector<InEdge> > All2AllNetwork(std::size_t num_nodes,
    const double *weights) {
  std::vector< std::vector<InEdge> > node_neighbors(num_nodes);
  for (std::size_t iii = 0; iii < num_nodes; iii++) {
    std::vector<InEdge> neighbors(num_nodes);
    for (std::size_t jjj = 0; jjj < num_nodes; jjj++) {
      neighbors[jjj] = InEdge(jjj, weights[iii * num_nodes + jjj]);
    }
    node_neighbors[iii] = neighbors;
  }
  return node_neighbors;
}

std::vector< std::vector<InEdge> > FullSensorNetwork(std::size_t num_sensors,
    std::size_t num_nodes, const double *weights) {
  std::vector<std::vector<InEdge> > node_sensors(num_nodes);
  for (std::size_t iii = 0; iii < num_nodes; iii++) {
    std::vector<InEdge> sensors(num_sensors);
    for (std::size_t jjj = 0; jjj < num_sensors; jjj++) {
      sensors[jjj] = InEdge(jjj, weights[iii * num_nodes + jjj]);
    }
    node_sensors[iii] = sensors;
  }
  return node_sensors;
}

}
