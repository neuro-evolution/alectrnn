/*
 * The Conv integrators will implicitly determine the structure of the previous
 * layer, as they have to be structured in a specific way. The Net and reservoir
 * however, specify the connections either within or between layers.
 * Within a layer, a neighbor graph is used, where sources of the connections
 * correspond with nodes in the same graph.
 * Between layers, the sources of the connections correspond to the nodes
 * of the previous layer, while the destination corresponds with nodes in the
 * current layer.
 */

#ifndef NN_INTEGRATOR_H_
#define NN_INTEGRATOR_H_

#include <cstddef>
#include <vector>
#include <cassert>
#include "multi_array.hpp"
#include "graphs.hpp"

namespace nervous_system {

enum INTEGRATOR_TYPE {
  BASE,
  NONE,
  CONV,
  NET,
  RESERVOIR
};

// Abstract base class
template<typename TReal>
class Integrator {
  public:
    Integrator() {
      integrator_type_ = INTEGRATOR_TYPE.BASE;
      parameter_count_ = 0;
    }
    virtual ~Integrator();
    virtual void operator()(multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state)=0;
    virtual void Configure(const multi_array::ConstArraySlice<TReal>& parameters)=0;

    std::size_t GetParameterCount() const {
      return parameter_count_;
    }

    INTEGRATOR_TYPE GetIntegratorType() const {
      return integrator_type_;
    }

  protected:
    INTEGRATOR_TYPE integrator_type_;
    std::size_t parameter_count_;
};

// None integrator - does nothing
template<typename TReal>
class NoneIntegrator : public Integrator<TReal> {
  public:
    NoneIntegrator() : Integrator() { integrator_type_ = INTEGRATOR_TYPE.NONE }
    ~NoneIntegrator()=default;

    void operator()(multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state) {}
    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {}
};

// Integrator that has All2All connectivity with previous layer
template<typename TReal>
class All2AllIntegrator : public Integrator<TReal> {
  public:
    All2AllIntegrator(Index num_states, Index num_prev_states) 
        : num_states_(num_states), num_prev_states_(num_prev_states) {
      parameter_count_ = num_states_ * num_prev_states_;
    }

    void operator()(multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state) {
      assert((src_state.size() == num_prev_states_) && (tar_state.size() == num_states));
      Index weight_id = 0;
      for (Index iii = 0; iii < tar_state.size(); ++iii) {
        TReal cumulative_sum = 0.0;
        for (Index jjj = 0; jjj < src_state.size(); ++jjj) {
          cumulative_sum += src_state[jjj] * weights_[weight_id];
          ++weight_id;
        }
        tar_state[iii] = cumulative_sum;
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      assert(parameters.size() == parameter_count_);
      weights_ = multi_array::ConstArraySlice<TReal>(
                  parameters.data(),
                  parameters.start(),
                  parameter_count_,
                  parameters.stride());
    }

  protected:
    Index num_states_;
    Index num_prev_states_;
    multi_array::ConstArraySlice<TReal> weights_;
}

/*
 * Conv integrator - uses implicit structure
 * Uses separable filters, so # params for a KxHxW kernel is K*(H+W)
 */
template<typename TReal>
class Conv3DIntegrator : public Integrator<TReal> {
  public:
    typedef std::size_t Index;

    Conv3DIntegrator(Index num_filters, 
        const multi_array::Array<Index,3>& filter_shape, 
        const multi_array::Array<Index,3>& layer_shape, Index stride)
        : filter_shape_(filter_shape), num_filters_(num_filters), stride_(stride) {
      major_filter_param_count_ = filter_shape[0] * filter_shape[2];
      minor_filter_param_count_ = filter_shape[0] * filter_shape[2];
      parameter_count_ = num_filters * (major_filter_param_count_ + minor_filter_param_count_);
      integrator_type_ = INTEGRATOR_TYPE.CONV;
      state_buffer1_ = multi_array::Tensor<TReal>(layer_shape);
      state_buffer2_ = multi_array::Tensor<TReal>(layer_shape);
      filter_parameters_major_({num_filters, filter_shape[0]});
      filter_parameters_minor_({num_filters, filter_shape[0]});
    }
    ~Conv3DIntegrator()=default;

    void operator()(const multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state) {
      const multi_array::ArrayView<TReal, 3> src_view = src_state.accessor<3>();
      multi_array::ArrayView<TReal, 3> tar_view = tar_state.accessor<3>();
      multi_array::ArrayView<TReal, 3> buffer1_view = state_buffer1_.accessor<3>();
      multi_array::ArrayView<TReal, 3> buffer2_view = state_buffer2_.accessor<3>();
      multi_array::ArrayView<multi_array::ConstArraySlice<TReal>, 2> major_view(filter_parameters_major_);
      multi_array::ArrayView<multi_array::ConstArraySlice<TReal>, 2> minor_view(filter_parameters_minor_);

      // for each filter
      for (Index iii = 0; iii < num_filters_; iii++) {
        multi_array::ArrayView<TReal, 2> tar_image = tar_view[iii];
        // for each input filter
        for (Index jjj = 0; jjj < src_state.shape()[0]; jjj++) {
          // Carry out separable conv on filter
          const multi_array::ArrayView<TReal, 2> src_image = src_view[jjj];
          multi_array::ArrayView<TReal, 2> buffer1_image = buffer1_view[jjj];
          multi_array::ArrayView<TReal, 2> buffer2_image = buffer2_view[jjj];
          Convolve2D(src_view, buffer1_image, buffer2_image,
                     major_view[iii][jjj], minor_view[iii][jjj], stride_);
        }
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      assert(parameters.size() == parameter_count_);
      multi_array::ArrayView<<multi_array::ConstArraySlice<TReal>, 2> major_view(filter_parameters_major_);
      multi_array::ArrayView<<multi_array::ConstArraySlice<TReal>, 2> minor_view(filter_parameters_minor_);
      for (Index filter = 0; filter < num_filters_; filter++) {
        for (Index iii = 0; iii < filter_shape_[0]; iii++) {
          major_view[filter][iii] = multi_array::ConstArraySlice<TReal>(
            parameters.data(), 
            parameters.start() + parameters.stride() * filter * iii * filter_shape_[2], 
            filter_shape_[2], parameters.stride());
          
          minor_view[filter][iii] = multi_array::ConstArraySlice<TReal>(
            parameters.data(), 
            parameters.start() + parameters.stride() * filter * iii * filter_shape_[1], 
            filter_shape_[1], parameters.stride());
        }
      }
    }

    void Convolve2D(const multi_array::ArrayView<TReal, 2>& src,
        multi_array::ArrayView<TReal, 2>& tar, 
        multi_array::ArrayView<TReal, 2>& buffer,
        const multi_array::ConstArraySlice<TReal> &kernel_minor, 
        const multi_array::ConstArraySlice<TReal> &kernel_major,
        Index stride) {

      // If stride is larger than the kernel, cumulative sum is reset
      if (stride < kernel_major.size()) {
        // Accumulate through major axis
        Index half_kernel = kernel_major.size() / 2;
        // Minor axis loop
        for (Index iii = 0; iii < src.extent(0); iii++) {
          TReal cumulative_sum = 0.0;
          // Initiate sum
          for (Index jjj = 0; jjj < kernel_major.size(); jjj++) {
            // Add out-of-bounds components
            for (Index kkk = 0; kkk < half_kernel; kkk++) {
              cumulative_sum += kernel_major[kkk] * src[iii][0];
            }
            // Add in-bounds components
            for (Index kkk = half_kernel; kkk < kernel_major.size(); kkk++) {
              cumulative_sum += kernel_major[kkk] * src[iii][kkk - half_kernel];
            }
          }
          buffer[iii][0] = cumulative_sum;

          // Roll sum (out-of-bounds components)
          Index jjj = stride;
          for (; jjj < half_kernel+stride; jjj+=stride) {
            for (Index kkk = 0; kkk < stride; ++kkk) {
              cumulative_sum += kernel_major[kernel_major.size() - kkk - 1] * src[iii][jjj + half_kernel - kkk]
                              - kernel_major[kkk] * src[iii][0];
            }
            buffer[iii][jjj] = cumulative_sum;
          }

          //  Roll sum (in-bounds components)
          for (; jjj < src.extent(1)-half_kernel; jjj+=stride) {
            for (Index kkk = 0; kkk < stride; +++kkk) {
              cumulative_sum += kernel_major[kernel_major.size() - kkk - 1] * src[iii][jjj + half_kernel - kkk]
                              - kernel_major[kkk] * src[iii][jjj - half_kernel - 1 - kkk];
            }
            buffer[iii][jjj] = cumulative_sum;
          }

          // Roll sum (out-of-bounds end components)
          for (; jjj < src.extent(1); jjj+=stride) {
            for (Index kkk = 0; kkk < stride; +++kkk) {
              cumulative_sum += kernel_major[kernel_major.size() - kkk - 1] * src[iii][src.extent(1)-1 - kkk]
                              - kernel_major[kkk] * src[iii][jjj - half_kernel - 1 - kkk];
            }
            buffer[iii][jjj] = cumulative_sum;
          }
        }
      }
      else {

      }

      if (stride < kernel_minor.size()) {
        // Accumulate through minor axis
        Index half_kernel = kernel_minor.size() / 2;
        for (Index iii = 0; iii < buffer.extent(1); iii++) {
          TReal cumulative_sum = 0.0;
          // Initiate sum
          for (Index jjj = 0; jjj < kernel_minor.size(); jjj++) {
            // Add out-of-bounds components
            for (Index kkk = 0; kkk < half_kernel; kkk++) {
              cumulative_sum += kernel_minor[kkk] * buffer[0][iii];
            }
            // Add in-bounds components
            for (Index kkk = half_kernel; kkk < kernel_minor.size(); kkk++) {
              cumulative_sum += kernel_minor[kkk] * buffer[kkk - half_kernel][iii];
            }
          }
          tar[0][iii] = cumulative_sum;

          // Roll sum (out-of-bounds components)
          jjj = stride;
          for (; jjj < half_kernel+1; jjj+=stride) {
            for (Index kkk = 0; kkk < stride; +++kkk) {
              cumulative_sum += kernel_major[kernel_minor.size() - kkk - 1] * buffer[jjj + half_kernel - kkk][iii]
                              - kernel_minor[kkk] * buffer[0][iii];
            }
            tar[jjj][iii] = cumulative_sum;
          }

          //  Roll sum (in-bounds components)
          for (; jjj < buffer.extent(0)-half_kernel; jjj+=stride) {
            for (Index kkk = 0; kkk < stride; +++kkk) {
              cumulative_sum += kernel_major[kernel_minor.size() - kkk - 1] * buffer[jjj + half_kernel - kkk][iii]
                              - kernel_minor[kkk] * buffer[jjj - half_kernel - 1 - kkk][iii];
            }
            tar[jjj][iii] = cumulative_sum;
          }

          // Roll sum (out-of-bounds end components)
          for (; jjj < buffer.extent(0); jjj+=stride) {
            for (Index kkk = 0; kkk < stride; +++kkk) {
              cumulative_sum += kernel_major[kernel_minor.size() - kkk - 1] * buffer[src.extent(0)-1][iii]
                              - kernel_minor[kkk] * buffer[jjj - half_kernel - 1 - kkk][iii];
            }
            tar[jjj][iii] = cumulative_sum;
          }
        }
      }
      else {

      }
    }

  protected:
    multi_array::Array<Index, 3> filter_shape_;
    Index num_filters_;
    multi_array::Tensor<TReal> state_buffer1_;
    multi_array::Tensor<TReal> state_buffer2_;
    multi_array::MultiArray<multi_array::ConstArraySlice<TReal>, 2> filter_parameters_major_;
    multi_array::MultiArray<multi_array::ConstArraySlice<TReal>, 2> filter_parameters_minor_;
    Index major_filter_param_count_;
    Index minor_filter_param_count_;
    Index stride_;
};

// Network integrator -- uses explicit unweighted structure
template<typename TReal>
class NetIntegrator : public Integrator<TReal> {
  public:

    NetIntegrator(const graphs::UnWeightedNeighborGraph& network) 
        : network_(network) {
      integrator_type_ = INTEGRATOR_TYPE.NET;
      parameter_count_ = network_.NumEdges();
    }

    ~NetIntegrator()=default;

    void operator()(multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state) {
      
      assert(tar_state.size() == network_.NumNodes());
      Index edge_id = 0;
      for (Index node = 0; node < network_.NumNodes(); ++node) {
        for (Index iii = 0; iii < network_.Neighbors(node).size(); ++iii) {
          tar_state[node] += src_state[network_.Neighbors(node)[iii].source] * weights_[edge_id];
          ++edge_id;
        }
      }
      assert(edge_id == network_.NumEdges());
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      assert(parameters.size() == parameter_count_);
      weights_ = multi_array::ConstArraySlice<TReal>(
                  parameters.data(),
                  parameters.start(),
                  parameter_count_,
                  parameters.stride());
    }

  protected:
    graphs::UnWeightedNeighborGraph network_;
    multi_array::ConstArraySlice<TReal> weights_;
};

// Reservoir -- uses explicit weighted structure
template<typename TReal>
class ReservoirIntegrator : public Integrator<TReal> {
  public:
    typedef typename graphs::Graph<WeightedInEdge<TReal> > WeightedNeighborGraph;
    typedef std::size_t Index;
    ReservoirIntegrator(const WeightedNeighborGraph& network)
        : network_(network) {
      integrator_type_ = INTEGRATOR_TYPE.RESERVOIR;
      parameter_count_ = 0;
    }

    ~ReservoirIntegrator()=default;

    void operator()(multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state) {
      assert(tar_state.size() == network_.NumNodes());
      for (Index node = 0; node < network_.NumNodes(); ++node) {
        for (Index iii = 0; iii < network_.Neighbors(node).size(); ++iii) {
          tar_state[node] += src_state[network_.Neighbors(node)[iii].source] * network_.Neighbors(node)[iii].weight;
        }
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {}

  protected:
    WeightedNeighborGraph network_;
};

} // End nervous_system namespace

#endif /* NN_INTEGRATOR_H_ */