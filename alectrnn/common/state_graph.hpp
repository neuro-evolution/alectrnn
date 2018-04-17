/*
 * Created by nathaniel on 4/13/18.
 *
 */

#ifndef STATE_GRAPH_HPP
#define STATE_GRAPH_HPP

#include <cstddef>
#include <vector>
#include "multi_array.hpp"

namespace graphs {

/*
 * PredecessorStateGraph holds state information of the nodes contiguously
 * in memory for iteration over node and node neighbors.
 * The graph is built from an edge list where it is assumed nodes are labelled
 * contiguously starting from 0 to num_nodes-1 so that they match the index
 * location in the arrays.
 */
//template<typename TState, typename TNodeID>
//class PredecessorStateGraph {
//  public:
//    typedef std::size_t Index;
//    typedef std::size_t Size;
//
//    // Support for multi_array
//    template <template<typename, Index> class multi>
//    PredecessorStateGraph(Size num_nodes, const multi<TNodeID, 2>& edge_list) {
//
//      const multi_array::ArrayView<TNodeID, 2> edge_view = edge_list.accessor();
//      for (Index iii = 0; iii < edge_view.extent(0); ++iii) {
//        // Build the in-degree of each node
//        // setup node neighbor states states
//
//        // Seem like it should have a predecessor graph inside it...
//          // need to know how states are updated -> requires connectivity info
//          // which is not implicit in node neighbor states
//      }
//    }
//
//    ~PredecessorStateGraph();
//
//    void UpdateState() {
//      // loops through states in tensor,
//      // assigns such states to corresponding points in irregular matrix
//    }
//
//  protected:
//    Size num_nodes_;
//    multi_array::IrregularMatrix<TState> node_neighbor_states_;
//};
//
//template<typename TState, typename TNodeID, typename TWeight>
//class PredecessorStateGraph {
//  public:
//
//  protected:
//    multi_array::IrregularMatrix<TState> node_neighbor_states_;
//    multi_array::IrregularMatrix<TState> node_neighbor_weights_;
//};

} // end namespace graphs

#endif // STATE_GRAPH_HPP
