#ifndef GRAPHS_H_
#define GRAPHS_H_

namespace graphs {

#include <cstddef>
#include <vector>

template<TReal>
struct WeightedInEdge {

  WeightedInEdge() {
    source = 0;
    weight = 0;
  }

  WeightedInEdge(std::size_t source_node, TReal source_weight) : 
      source(source_node), weight(source_weight) {}

  std::size_t source;
  TReal weight;
};

template<typename Link>
class Graph {
  public:
    typedef std::size_t NodeID;
    typedef std::size_t Index;

    Graph(const std::vector< std::vector<Link> >& graph) : graph_(graph) {
      num_nodes_ = graph_.size();
      num_edges_ = CalcNumEdges();
    }

    Graph(const Graph<Link>& other) : graph_(other.graph_),
        num_nodes_(other.num_nodes_), num_edges_(other.graph_) {}

    Graph(Graph<Link>&& other) : graph_(std::move(other.graph_)),
        num_nodes_(std::move(other.num_nodes_)), 
        num_edges_(std::move(other.num_edges_)) {}

    Graph<Link>& operator=(const Graph<Link>& other) {
      graph_ = other.graph_;
      num_nodes_ = other.num_nodes_;
      num_edges_ = other.num_edges_;
      return *this;
    }

    Graph<Link>& operator=(Graph<Link>&& other) {
      graph_ = std::move(other.graph_);
      num_nodes_ = std::move(other.num_nodes_);
      num_edges_ = std::move(other.num_edges_);
      return *this;
    }

    std::size_t CalcNumEdges() {
      std::size_t sum = 0;
      for (NodeID iii = 0; iii < graph_.size(); iii++) {
        sum += graph_[iii].size();
      }

      return sum;
    }

    std::size_t GetNumNodes() {
      return num_nodes_;
    }

    std::vector<Link>& GetNeighbors(NodeID node) {
      return graph_[node];
    }

  protected:
    std::vector< std::vector<Link> > graph_;
    std::size_t num_nodes_;
    std::size_t num_edges_;
}

typedef typename Graph<WeightedInEdge<float> > WeightedNeighborGraph;
typedef typename Graph<std::size_t> UnWeightedNeighborGraph;

} // End graphs namespace

#endif /* GRAPHS_H_ */