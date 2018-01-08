#ifndef GRAPHS_H_
#define GRAPHS_H_

#include <cstddef>
#include <vector>
#include "multi_array.hpp"

namespace graphs {

typedef std::size_t NodeID;
typedef std::size_t Index;

template<typename TReal>
struct EdgeTail {

  EdgeTail() {
    source = 0;
    weight = 1;
  }

  EdgeTail(NodeID source_node, TReal source_weight) : 
      source(source_node), weight(source_weight) {}

  NodeID source;
  TReal weight;
};

template<typename TReal>
class DiGraph {
  public:
    typedef std::size_t NodeID;
    typedef std::size_t Index;

    /*
     * When building from an edge list it is assumed nodes are labelled
     * contiguously starting from 0 to num_nodes-1
     */
    DiGraph(NodeID num_nodes)
        : num_nodes_(num_nodes) {
      graph_.resize(num_nodes);
      num_edges_ = 0;
      for (NodeID iii = 0; iii < num_nodes; ++iii) {
        graph_[iii].resize(0);
      }
    }

    DiGraph(const std::vector< std::vector<EdgeTail<TReal>> >& graph) : graph_(graph) {
      num_nodes_ = graph_.size();
      num_edges_ = CalcNumEdges();
    }

    DiGraph(const DiGraph& other) : graph_(other.graph_),
        num_nodes_(other.num_nodes_), num_edges_(other.graph_) {}

    DiGraph(DiGraph&& other) : graph_(std::move(other.graph_)),
        num_nodes_(std::move(other.num_nodes_)), 
        num_edges_(std::move(other.num_edges_)) {}

    DiGraph& operator=(const DiGraph& other) {
      graph_ = other.graph_;
      num_nodes_ = other.num_nodes_;
      num_edges_ = other.num_edges_;
      return *this;
    }

    DiGraph& operator=(DiGraph&& other) {
      graph_ = std::move(other.graph_);
      num_nodes_ = std::move(other.num_nodes_);
      num_edges_ = std::move(other.num_edges_);
      return *this;
    }

    std::size_t CalcNumEdges() const {
      std::size_t sum = 0;
      for (NodeID iii = 0; iii < graph_.size(); iii++) {
        sum += graph_[iii].size();
      }

      return sum;
    }

    std::size_t NumEdges() const {
      return num_edges_;
    }

    std::size_t NumNodes() const {
      return num_nodes_;
    }

    const std::vector<EdgeTail<TReal>>& InNeighbors(NodeID node) const {
      return graph_[node];
    }

    const EdgeTail<TReal>& Edge(NodeID source, NodeID target) const {
      return graph_[source][target];
    }

    void AddEdge(NodeID source, NodeID target, TReal weight=1) {
      graph_[target].push_back(EdgeTail<TReal>(source, weight));
    }

  protected:
    std::vector< std::vector<EdgeTail<TReal>> > graph_;
    std::size_t num_nodes_;
    std::size_t num_edges_;
}

/*
 * edge list may exclude end nodes if they aren't connected, so it is
 * insufficient to just pass a node list.
 * First dimension is NumEdges, second dimension is [source, target]
 * If there are weights then pass weight array (must be same size as edges)
 */

template<typename Integer>
DiGraph ConvertEdgeListToDiGraph(NodeID num_nodes,
    const multi_array::MultiArray<Integer, 2>& edge_list) {
  graph = DiGraph(num_nodes);
  ArrayView<Integer, 2> edge_view(edge_list);
  for (Index iii = 0; iii < edge_view.extent(0); ++iii) {
    graph.AddEdge(edge_view[iii][1], edge_view[iii][0]);
  }

  return graph;
}

template<typename Integer, typename TReal>
DiGraph ConvertEdgeListToDiGraph(NodeID num_nodes,
    const multi_array::MultiArray<Integer, 2>& edge_list, 
    const multi_array::MultiArray<TReal, 1>& weights) {
  graph = DiGraph(num_nodes);
  ArrayView<Integer, 2> edge_view(edge_list);
  ArrayView<TReal, 1> weight_view(weights);
  for (Index iii = 0; iii < edge_view.extent(0); ++iii) {
    graph.AddEdge(edge_view[iii][1], edge_view[iii][0], weight_view[iii]);
  }

  return graph;
}

} // End graphs namespace

#endif /* GRAPHS_H_ */