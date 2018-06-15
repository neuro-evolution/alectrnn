#ifndef GRAPHS_H_
#define GRAPHS_H_

#include <cstddef>
#include <vector>
#include <Eigen/Sparse>
#include "multi_array.hpp"

namespace graphs {

typedef std::size_t NodeID;
typedef std::size_t Index;

template<typename TReal=void>
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

template<>
struct EdgeTail<void> {

  EdgeTail() {
    source = 0;
  }

  EdgeTail(NodeID source_node) : 
      source(source_node) {}

  NodeID source;
};

template<typename TReal=void>
class PredecessorGraph {
  public:
    typedef std::size_t NodeID;
    typedef std::size_t Index;

    /*
     * When building from an edge list it is assumed nodes are labelled
     * contiguously starting from 0 to num_nodes-1
     */
    PredecessorGraph() : PredecessorGraph(0) {
    }

    PredecessorGraph(NodeID num_nodes)
        : num_nodes_(num_nodes) {
      graph_.resize(num_nodes);
      num_edges_ = 0;
      for (NodeID iii = 0; iii < num_nodes; ++iii) {
        graph_[iii].resize(0);
      }
    }

    PredecessorGraph(const std::vector< std::vector<EdgeTail<TReal>> >& graph) : graph_(graph) {
      num_nodes_ = graph_.size();
      num_edges_ = CalcNumEdges();
    }

    PredecessorGraph(const PredecessorGraph<TReal>& other) : graph_(other.graph_),
        num_nodes_(other.num_nodes_), num_edges_(other.num_edges_) {}

    PredecessorGraph(PredecessorGraph<TReal>&& other) : graph_(std::move(other.graph_)),
        num_nodes_(std::move(other.num_nodes_)), 
        num_edges_(std::move(other.num_edges_)) {}

    PredecessorGraph<TReal>& operator=(const PredecessorGraph<TReal>& other) {
      graph_ = other.graph_;
      num_nodes_ = other.num_nodes_;
      num_edges_ = other.num_edges_;
      return *this;
    }

    PredecessorGraph<TReal>& operator=(PredecessorGraph<TReal>&& other) {
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

    const std::vector<EdgeTail<TReal>>& Predecessors(NodeID node) const {
      return graph_[node];
    }

    void AddEdge(NodeID source, NodeID target, TReal weight=1) {
      // It is assumed the Node should exist, so new nodes are added
      // up to the NodeID of the source or target
      if (source >= num_nodes_) {
        std::size_t num_new_nodes = source - num_nodes_ + 1;
        for (Index iii = 0; iii < num_new_nodes; ++iii) {
          graph_.push_back(std::vector<EdgeTail<TReal>>(0));
          ++num_nodes_;
        }
      }

      if (target >= num_nodes_) {
        std::size_t num_new_nodes = target - num_nodes_ + 1;
        for (Index iii = 0; iii < num_new_nodes; ++iii) {
          graph_.push_back(std::vector<EdgeTail<TReal>>(0));
          ++num_nodes_;
        }
      }

      graph_[target].push_back(EdgeTail<TReal>(source, weight));
      ++num_edges_;
    }

  protected:
    std::vector< std::vector<EdgeTail<TReal>> > graph_;
    std::size_t num_nodes_;
    std::size_t num_edges_;
};

template<>
class PredecessorGraph<void> {
  public:
    typedef std::size_t NodeID;
    typedef std::size_t Index;

    /*
     * When building from an edge list it is assumed nodes are labelled
     * contiguously starting from 0 to num_nodes-1
     */
    PredecessorGraph() : PredecessorGraph(0) {
    }

    PredecessorGraph(NodeID num_nodes)
        : num_nodes_(num_nodes) {
      graph_.resize(num_nodes);
      num_edges_ = 0;
      for (NodeID iii = 0; iii < num_nodes; ++iii) {
        graph_[iii].resize(0);
      }
    }

    PredecessorGraph(const std::vector< std::vector<EdgeTail<>> >& graph) : graph_(graph) {
      num_nodes_ = graph_.size();
      num_edges_ = CalcNumEdges();
    }

    PredecessorGraph(const PredecessorGraph& other) : graph_(other.graph_),
        num_nodes_(other.num_nodes_), num_edges_(other.num_edges_) {}

    PredecessorGraph(PredecessorGraph&& other) : graph_(std::move(other.graph_)),
        num_nodes_(std::move(other.num_nodes_)), 
        num_edges_(std::move(other.num_edges_)) {}

    PredecessorGraph& operator=(const PredecessorGraph& other) {
      graph_ = other.graph_;
      num_nodes_ = other.num_nodes_;
      num_edges_ = other.num_edges_;
      return *this;
    }

    PredecessorGraph& operator=(PredecessorGraph&& other) {
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

    const std::vector<EdgeTail<>>& Predecessors(NodeID node) const {
      return graph_[node];
    }

    // Automatically resizes graph if new nodes are introduced
    void AddEdge(NodeID source, NodeID target) {
      // It is assumed the Node should exist, so new nodes are added
      // up to the NodeID of the source or target
      if (source >= num_nodes_) {
        std::size_t num_new_nodes = source - num_nodes_ + 1;
        for (Index iii = 0; iii < num_new_nodes; ++iii) {
          graph_.push_back(std::vector<EdgeTail<>>(0));
          ++num_nodes_;
        }
      }

      if (target >= num_nodes_) {
        std::size_t num_new_nodes = target - num_nodes_ + 1;
        for (Index iii = 0; iii < num_new_nodes; ++iii) {
          graph_.push_back(std::vector<EdgeTail<>>(0));
          ++num_nodes_;
        }
      }

      graph_[target].push_back(EdgeTail<>(source));
      ++num_edges_;
    }

  protected:
    std::vector< std::vector<EdgeTail<>> > graph_;
    std::size_t num_nodes_;
    std::size_t num_edges_;
};

/*
 * edge list may exclude end nodes if they aren't connected, so it is
 * insufficient to just pass a node list.
 * First dimension is NumEdges, second dimension is [source, target]
 * If there are weights then pass weight array (must be same size as edges)
 */

template<typename Integer, template<typename, Index> class multi>
PredecessorGraph<> ConvertEdgeListToPredecessorGraph(const multi<Integer, 2>& edge_list) {
  PredecessorGraph<> graph = PredecessorGraph<>();
  const multi_array::ArrayView<Integer, 2> edge_view = edge_list.accessor();
  for (Index iii = 0; iii < edge_view.extent(0); ++iii) {
    graph.AddEdge(edge_view[iii][0], edge_view[iii][1]);
  }

  return graph;
}

template<typename Integer, typename TReal, template<typename, Index> class multi>
PredecessorGraph<TReal> ConvertEdgeListToPredecessorGraph(const multi<Integer, 2>& edge_list,
    const multi<TReal, 1>& weights) {
  PredecessorGraph<TReal> graph = PredecessorGraph<TReal>();
  const multi_array::ArrayView<Integer, 2> edge_view = edge_list.accessor();
  const multi_array::ArrayView<TReal, 1> weight_view = weights.accessor();
  for (Index iii = 0; iii < edge_view.extent(0); ++iii) {
    graph.AddEdge(edge_view[iii][0], edge_view[iii][1], weight_view[iii]);
  }

  return graph;
}

/*
 * Converts an edge list into a sparse matrix. The tail_size, is the number
 * of nodes that act as the source of the links, and head_size is the number
 * of nodes that act as the sink of the links. For a standard graph this is
 * just the total number of nodes in the graph, else if the graph is
 * bipartite, then one will be larger than the other.
 * For edge list, the order of the list should be row-major
 * with shape Ex2 with major axis as the edge with (tail, head) so that
 * X[edge#][0]=tail, X[edge#][1]=head
 *
 * Because Eigen uses Fortran's column major format, the ordering will be
 * reversed in order to preserve the proper direction of the links in the
 * sparse matrix.
 *
 * The resulting sparse matrix shape is: Head x Tail
 * Where Tail == # cols, and Head == # rows
 */
template<typename Integer, typename TReal, template<typename, Index> class multi>
Eigen::SparseMatrix<TReal> ConvertEdgeListToSparseMatrix(const multi<Integer, 2>& edge_list,
                                                         const int num_tail_nodes,
                                                         const int num_head_nodes) {

  const auto edge_view = edge_list.accessor();
  std::vector<Eigen::Triplet<TReal>> matrix_elements(edge_view.extent(0));
  for (std::size_t i = 0; i < edge_view.extent(0); ++i) {
    matrix_elements[i] = Eigen::Triplet<TReal>(edge_view[i][1], edge_view[i][0], 1);
  }
  Eigen::SparseMatrix<TReal> graph(num_head_nodes, num_tail_nodes);
  graph.setFromTriplets(matrix_elements.begin(), matrix_elements.end());

  return graph;
};

template<typename Integer, typename TReal, template<typename, Index> class multi>
Eigen::SparseMatrix<TReal> ConvertEdgeListToSparseMatrix(const multi<Integer, 2>& edge_list,
                                                         const int num_tail_nodes,
                                                         const int num_head_nodes,
                                                         const multi<TReal, 1>& weights) {

  const auto edge_view = edge_list.accessor();
  const auto weight_view = weights.accessor();
  std::vector<Eigen::Triplet<TReal>> matrix_elements(edge_view.extent(0));
  for (std::size_t i = 0; i < edge_view.extent(0); ++i) {
    matrix_elements[i] = Eigen::Triplet<TReal>(edge_view[i][1], edge_view[i][0],
                                               weight_view[i]);
  }
  Eigen::SparseMatrix<TReal> graph(num_head_nodes, num_tail_nodes);
  graph.setFromTriplets(matrix_elements.begin(), matrix_elements.end());

  return graph;
};

} // End graphs namespace

#endif /* GRAPHS_H_ */