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

#include <iostream> /////////////////////////////// TESTING

namespace nervous_system {

enum INTEGRATOR_TYPE {
  BASE,
  NONE,
  ALL2ALL,
  CONV,
  NETWORK,
  RESERVOIR
};

// Abstract base class
template<typename TReal>
class Integrator {
  public:
    Integrator() {
      integrator_type_ = BASE;
      parameter_count_ = 0;
    }
    virtual ~Integrator()=default;
    virtual void operator()(const multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state)=0;
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
  typedef Integrator<TReal> super_type;
  public:
    NoneIntegrator() : super_type() { super_type::integrator_type_ = NONE; }
    ~NoneIntegrator()=default;

    void operator()(const multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state) {}
    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {}
};

// Integrator that has All2All connectivity with previous layer
template<typename TReal>
class All2AllIntegrator : public Integrator<TReal> {
  typedef Integrator<TReal> super_type;
  typedef std::size_t Index;
  public:
    All2AllIntegrator(Index num_states, Index num_prev_states) 
        : num_states_(num_states), num_prev_states_(num_prev_states) {
      super_type::parameter_count_ = num_states_ * num_prev_states_;
    }

    void operator()(const multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state) {
      assert((src_state.size() == num_prev_states_) && (tar_state.size() == num_states_));
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
      assert(parameters.size() == super_type::parameter_count_);
      weights_ = multi_array::ConstArraySlice<TReal>(parameters);
    }

  protected:
    Index num_states_;
    Index num_prev_states_;
    multi_array::ConstArraySlice<TReal> weights_;
};

/*
 * Conv integrator - uses implicit structure
 * Uses separable filters, so # params for a KxHxW kernel is K*(H+W)
 */
template<typename TReal>
class Conv3DIntegrator : public Integrator<TReal> {
  typedef Integrator<TReal> super_type;
  typedef std::size_t Index;
  public:
    /*
     * F = layer.shape[0]
     * H/W = filter.shape[1/2]
     * D = filter.shape[0]
     * filter_shape and layer_shape implicitly contain most the information for
     * the shape of the previous layer. Because this is a separable filter
     * the # of parameters for each filter is (W * D) + (H * D).
     * This convolution uses spatially and depthwise separable convolutions.
     * First F HxW convolutions are applied to each input dimension D.
     * Then F Dx1x1 filters are applied to the results.
     * Principle behind DxHxW + DxFx1x1 is that the same filter is used on all
     * layers, and then a 1x1 filter is used to do a weighted sum. This is
     * opposed to having a separate filer for each input channel 
     * (each 1x1 needs D parameters, with a total of F 1x1 filters)
     */
    Conv3DIntegrator(const multi_array::Array<Index,3>& filter_shape, 
        const multi_array::Array<Index,3>& layer_shape, 
        const multi_array::Array<Index,3>& prev_layer_shape, Index stride)
        : num_filters_(layer_shape[0]), filter_shape_(filter_shape), stride_(stride),
        filter_parameters_major_({num_filters_}),
        filter_parameters_minor_({num_filters_}),
        channel_weights_({num_filters_}) {

      assert(filter_shape[0] == prev_layer_shape[0]);
      super_type::parameter_count_ = num_filters_ 
        * (filter_shape_[2] + filter_shape_[1] + filter_shape_[0]);
      super_type::integrator_type_ = CONV;
      firstpass_buffer_ = multi_array::Tensor<TReal>(
                            {prev_layer_shape[1], layer_shape[2]});
      secondpass_buffer_ = multi_array::Tensor<TReal>(
                            {layer_shape[1], layer_shape[2]});
    }
    ~Conv3DIntegrator()=default;

    void operator()(const multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state) {

      const multi_array::TensorView<TReal> src_view = src_state.accessor();
      multi_array::TensorView<TReal> tar_view = tar_state.accessor();
      multi_array::ArrayView<multi_array::ConstArraySlice<TReal>, 1> major_view(filter_parameters_major_);
      multi_array::ArrayView<multi_array::ConstArraySlice<TReal>, 1> minor_view(filter_parameters_minor_);
      multi_array::ArrayView<multi_array::ConstArraySlice<TReal>, 1> channel_view(channel_weights_);

      // for each filter
      for (Index iii = 0; iii < num_filters_; iii++) {
        multi_array::TensorView<TReal> tar_image = tar_view[iii];
        // for each input channel
        for (Index jjj = 0; jjj < src_state.shape()[0]; jjj++) {
          // Carry out separable conv on layer of inputs
          const multi_array::TensorView<TReal> src_image = src_view[jjj];
          multi_array::TensorView<TReal> secondpass_image = secondpass_buffer_.accessor();
          multi_array::TensorView<TReal> firstpass_image = firstpass_buffer_.accessor();
          Convolve2D(src_image, secondpass_image, firstpass_image,
                     major_view[iii], minor_view[iii], stride_);

          for (Index kkk = 0; kkk < tar_state.shape()[1]; ++kkk) {
            for (Index lll = 0; lll < tar_state.shape()[2]; ++lll) {
              tar_image[kkk][lll] += secondpass_image[kkk][lll] * channel_weights_[iii][jjj];
            }
          }
        }
      }
    }

    void Convolve2D(const multi_array::TensorView<TReal>& src,
        multi_array::TensorView<TReal>& tar, 
        multi_array::TensorView<TReal>& buffer,
        const multi_array::ConstArraySlice<TReal> &kernel_minor, 
        const multi_array::ConstArraySlice<TReal> &kernel_major,
        Index stride) {

      // If stride is larger than the kernel, cumulative sum is reset
      if (stride < kernel_major.size()) {
        // Accumulate through major axis
        Index half_kernel = kernel_major.size() / 2;
        // Minor axis loop
        TReal cumulative_sum = 0.0;
        Index output_index = 0;
        for (Index iii = 0; iii < src.extent(0); ++iii) {
          cumulative_sum = 0.0;
          output_index = 0;
          // Initiate sum
          // Add out-of-bounds components
          for (Index kkk = 0; kkk < half_kernel; kkk++) {
            cumulative_sum += kernel_major[kkk] * src[iii][0];
          }
          // Add in-bounds components
          for (Index kkk = half_kernel; kkk < kernel_major.size(); kkk++) {
            cumulative_sum += kernel_major[kkk] * src[iii][kkk - half_kernel];
          }
          buffer[iii][output_index++] = cumulative_sum;

          // Roll sum (out-of-bounds components)
          Index jjj = stride;
          for (; jjj < half_kernel+stride; jjj+=stride) {
            for (Index kkk = 0; kkk < stride; ++kkk) {
              cumulative_sum += kernel_major[kernel_major.size() - kkk - 1] * src[iii][jjj + half_kernel - kkk]
                              - kernel_major[kkk] * src[iii][0];
            }
            buffer[iii][output_index++] = cumulative_sum;
          }

          //  Roll sum (in-bounds components)
          for (; jjj < src.extent(1)-half_kernel; jjj+=stride) {
            for (Index kkk = 0; kkk < stride; ++kkk) {
              cumulative_sum += kernel_major[kernel_major.size() - kkk - 1] * src[iii][jjj + half_kernel - kkk]
                              - kernel_major[kkk] * src[iii][jjj - half_kernel - 1 - kkk];
            }
            buffer[iii][output_index++] = cumulative_sum;
          }

          // Roll sum (out-of-bounds end components)
          for (; jjj < src.extent(1); jjj+=stride) {
            for (Index kkk = 0; kkk < stride; ++kkk) {
              cumulative_sum += kernel_major[kernel_major.size() - kkk - 1] * src[iii][src.extent(1)-1]
                              - kernel_major[kkk] * src[iii][jjj - half_kernel - 1 - kkk];
            }
            buffer[iii][output_index++] = cumulative_sum;
          }
        }
      }
      else {
        // Accumulate through major axis
        Index half_kernel = kernel_major.size() / 2;
        // Minor axis loop
        TReal cumulative_sum = 0.0;
        Index output_index = 0;
        for (Index iii = 0; iii < src.extent(0); ++iii) {
          cumulative_sum = 0.0;
          output_index = 0;
          // Initiate sum
          // Add out-of-bounds components
          for (Index kkk = 0; kkk < half_kernel; kkk++) {
            cumulative_sum += kernel_major[kkk] * src[iii][0];
          }
          // Add in-bounds components
          for (Index kkk = half_kernel; kkk < kernel_major.size(); kkk++) {
            cumulative_sum += kernel_major[kkk] * src[iii][kkk - half_kernel];
          }
          buffer[iii][output_index++] = cumulative_sum;

          //  Roll sum (in-bounds components)
          Index jjj = stride;
          for (; jjj < src.extent(1)-half_kernel; jjj+=stride) {
            cumulative_sum = 0.0;
            for (Index kkk = 0; kkk < kernel_major.size(); ++kkk) {
              cumulative_sum += kernel_major[kkk] * src[iii][jjj - half_kernel + kkk];
            }
            buffer[iii][output_index++] = cumulative_sum;
          }

          // Roll sum (out-of-bounds end components)
          for (; jjj < src.extent(1); jjj+=stride) {
            cumulative_sum = 0.0;
            // Add in-bounds components
            for (Index kkk = 0; kkk < half_kernel+1; kkk++) {
              cumulative_sum += kernel_major[kkk] * src[iii][jjj - half_kernel + kkk];
            }
            // Add out-of-bounds components
            for (Index kkk = half_kernel+1; kkk < kernel_major.size(); kkk++) {
              cumulative_sum += kernel_major[kkk] * src[iii][src.extent(1)-1];
            }
            buffer[iii][output_index++] = cumulative_sum;
          }
        }
      }
      if (stride < kernel_minor.size()) {
        // Accumulate through minor axis
        TReal cumulative_sum = 0.0;
        Index output_index = 0;
        Index half_kernel = kernel_minor.size() / 2;
        for (Index iii = 0; iii < buffer.extent(1); iii++) {
          cumulative_sum = 0.0;
          output_index = 0;
          // Initiate sum
          // Add out-of-bounds components
          for (Index kkk = 0; kkk < half_kernel; kkk++) {
            cumulative_sum += kernel_minor[kkk] * buffer[0][iii];
          }
          // Add in-bounds components
          for (Index kkk = half_kernel; kkk < kernel_minor.size(); kkk++) {
            cumulative_sum += kernel_minor[kkk] * buffer[kkk - half_kernel][iii];
          }
          tar[output_index++][iii] = cumulative_sum;

          // Roll sum (out-of-bounds components)
          Index jjj = stride;
          for (; jjj < half_kernel+stride; jjj+=stride) {
            for (Index kkk = 0; kkk < stride; ++kkk) {
              cumulative_sum += kernel_major[kernel_minor.size() - kkk - 1] * buffer[jjj + half_kernel - kkk][iii]
                              - kernel_minor[kkk] * buffer[0][iii];
              std::cout<<"\t\tcum:" << cumulative_sum << " ADDat:" << jjj + half_kernel - kkk
              << " SUBat:" << 0 <<std::endl;
            }
            tar[output_index++][iii] = cumulative_sum;
          }

          //  Roll sum (in-bounds components)
          for (; jjj < buffer.extent(0)-half_kernel; jjj+=stride) {
            for (Index kkk = 0; kkk < stride; ++kkk) {
              cumulative_sum += kernel_major[kernel_minor.size() - kkk - 1] * buffer[jjj + half_kernel - kkk][iii]
                              - kernel_minor[kkk] * buffer[jjj - half_kernel - 1 - kkk][iii];
            }
            tar[output_index++][iii] = cumulative_sum;
          }

          // Roll sum (out-of-bounds end components)
          for (; jjj < buffer.extent(0); jjj+=stride) {
            for (Index kkk = 0; kkk < stride; ++kkk) {
              cumulative_sum += kernel_major[kernel_minor.size() - kkk - 1] * buffer[src.extent(0)-1][iii]
                              - kernel_minor[kkk] * buffer[jjj - half_kernel - 1 - kkk][iii];
            }
            tar[output_index++][iii] = cumulative_sum;
          }
        }
      }
      else {

        // Accumulate through major axis
        Index half_kernel = kernel_minor.size() / 2;
        TReal cumulative_sum = 0.0;
        Index output_index = 0;
        // Minor axis loop
        for (Index iii = 0; iii < buffer.extent(1); ++iii) {
          cumulative_sum = 0.0;
          output_index = 0;
          // Initiate sum
          // Add out-of-bounds components
          for (Index kkk = 0; kkk < half_kernel; kkk++) {
            cumulative_sum += kernel_minor[kkk] * buffer[0][iii];
          }
          // Add in-bounds components
          for (Index kkk = half_kernel; kkk < kernel_minor.size(); kkk++) {
            cumulative_sum += kernel_minor[kkk] * buffer[kkk - half_kernel][iii];
          }
          tar[output_index++][iii] = cumulative_sum;

          //  Roll sum (in-bounds components)
          Index jjj = stride;
          for (; jjj < buffer.extent(0)-half_kernel; jjj+=stride) {
            cumulative_sum = 0.0;
            for (Index kkk = 0; kkk < kernel_minor.size(); ++kkk) {
              cumulative_sum += kernel_minor[kkk] * buffer[jjj - half_kernel + kkk][iii];
            }
            tar[output_index++][iii] = cumulative_sum;
          }

          // Roll sum (out-of-bounds end components)
          for (; jjj < buffer.extent(0); jjj+=stride) {
            cumulative_sum = 0.0;
            // Add in-bounds components
            for (Index kkk = 0; kkk < half_kernel+1; kkk++) {
              cumulative_sum += kernel_minor[kkk] * buffer[jjj - half_kernel + kkk][iii];
            }
            // Add out-of-bounds components
            for (Index kkk = half_kernel+1; kkk < kernel_minor.size(); kkk++) {
              cumulative_sum += kernel_minor[kkk] * buffer[buffer.extent(0)-1][iii];
            }
            tar[output_index++][iii] = cumulative_sum;
          }
        }
      }
      for (Index iii = 0; iii < tar.extent(0); ++iii) {
        for (Index jjj = 0; jjj < tar.extent(1); ++jjj) {

          std::cout << tar[iii][jjj] << " ";
        }
        std::cout << std::endl;//
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {

      assert(parameters.size() == super_type::parameter_count_);
      multi_array::ArrayView< multi_array::ConstArraySlice<TReal>, 1> major_view(filter_parameters_major_);
      multi_array::ArrayView< multi_array::ConstArraySlice<TReal>, 1> minor_view(filter_parameters_minor_);
      multi_array::ArrayView< multi_array::ConstArraySlice<TReal>, 1> channel_view(channel_weights_);

      Index start(0);
      for (Index filter = 0; filter < num_filters_; filter++) {
        major_view[filter] = parameters.slice(start, filter_shape_[2]);
        start += parameters.stride() * filter_shape_[2];

        minor_view[filter] = parameters.slice(start, filter_shape_[1]);
        start += parameters.stride() * filter_shape_[1];

        channel_view[filter] = parameters.slice(start, filter_shape_[0]);
        start += parameters.stride() * filter_shape_[0];
      }
    }

  protected:
    Index num_filters_;
    multi_array::Array<Index, 3> filter_shape_;
    Index stride_;
    multi_array::MultiArray<multi_array::ConstArraySlice<TReal>, 1> filter_parameters_major_;
    multi_array::MultiArray<multi_array::ConstArraySlice<TReal>, 1> filter_parameters_minor_;
    multi_array::MultiArray<multi_array::ConstArraySlice<TReal>, 1> channel_weights_;
    multi_array::Tensor<TReal> secondpass_buffer_;
    multi_array::Tensor<TReal> firstpass_buffer_;
};

// Network integrator -- uses explicit unweighted structure
template<typename TReal>
class NetworkIntegrator : public Integrator<TReal> {
  typedef Integrator<TReal> super_type;
  typedef std::size_t Index;
  public:

    NetworkIntegrator(const graphs::PredecessorGraph<>& network) 
        : network_(network) {
      super_type::integrator_type_ = NETWORK;
      super_type::parameter_count_ = network_.NumEdges();
    }

    ~NetworkIntegrator()=default;

    void operator()(const multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state) {
      
      assert(tar_state.size() == network_.NumNodes());
      Index edge_id = 0;
      for (Index node = 0; node < network_.NumNodes(); ++node) {
        for (Index iii = 0; iii < network_.Predecessors(node).size(); ++iii) {
          tar_state[node] += src_state[network_.Predecessors(node)[iii].source] * weights_[edge_id];
          ++edge_id;
        }
      }
      assert(edge_id == network_.NumEdges());
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      assert(parameters.size() == super_type::parameter_count_);
      weights_ = multi_array::ConstArraySlice<TReal>(
                  parameters.data(),
                  parameters.start(),
                  super_type::parameter_count_,
                  parameters.stride());
    }

  protected:
    graphs::PredecessorGraph<> network_;
    multi_array::ConstArraySlice<TReal> weights_;
};

// Reservoir -- uses explicit weighted structure
template<typename TReal>
class ReservoirIntegrator : public Integrator<TReal> {
  typedef Integrator<TReal> super_type;
  typedef std::size_t Index;
  public:
    ReservoirIntegrator(const graphs::PredecessorGraph<TReal>& network)
        : network_(network) {
      super_type::integrator_type_ = RESERVOIR;
      super_type::parameter_count_ = 0;
    }

    ~ReservoirIntegrator()=default;

    void operator()(const multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state) {
      assert(tar_state.size() == network_.NumNodes());
      for (Index node = 0; node < network_.NumNodes(); ++node) {
        for (Index iii = 0; iii < network_.Predecessors(node).size(); ++iii) {
          tar_state[node] += src_state[network_.Predecessors(node)[iii].source] * network_.Predecessors(node)[iii].weight;
        }
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {}

  protected:
    graphs::PredecessorGraph<TReal> network_;
};

} // End nervous_system namespace

#endif /* NN_INTEGRATOR_H_ */