/*
 * The Conv integrators will implicitly determine the structure of the previous
 * layer, as they have to be structured in a specific way. The Net and reservoir
 * however, specify the connections either within or between layers.
 * Within a layer, a neighbor graph is used, where sources of the connections
 * correspond with nodes in the same graph.
 * Between layers, the sources of the connections correspond to the nodes
 * of the previous layer, while the destination corresponds with nodes in the
 * current layer.
 *
 * Integrators don't actually have to know the real size/shape of the previous
 * layer. Any size/shape can be given so long as the # elements is less than
 * then number in the previous layer so that non-existent elements are not
 * accessed. The integrator will create its own view into the previous layer
 * with the shape provided it, regardless of whether that layer has that shape.
 */

#ifndef NN_INTEGRATOR_H_
#define NN_INTEGRATOR_H_

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <functional>
#include "multi_array.hpp"
#include "graphs.hpp"
#include "parameter_types.hpp"
#include "utilities.hpp"

namespace nervous_system {

enum INTEGRATOR_TYPE {
  BASE_INTEGRATOR,
  NONE_INTEGRATOR,
  ALL2ALL_INTEGRATOR,
  CONV_INTEGRATOR,
  RECURRENT_INTEGRATOR,
  RESERVOIR_INTEGRATOR,
  RESERVOIR_HYBRID
};

// Abstract base class
template<typename TReal>
class Integrator {
  public:
    Integrator() {
      integrator_type_ = BASE_INTEGRATOR;
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

    virtual std::vector<PARAMETER_TYPE> GetParameterLayout() const=0;

  protected:
    INTEGRATOR_TYPE integrator_type_;
    std::size_t parameter_count_;
};

// None integrator - does nothing
template<typename TReal>
class NoneIntegrator : public Integrator<TReal> {
  typedef Integrator<TReal> super_type;
  public:
    NoneIntegrator() : super_type() { super_type::integrator_type_ = NONE_INTEGRATOR; }
    ~NoneIntegrator()=default;

    void operator()(const multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state) {}
    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {}

    std::vector<PARAMETER_TYPE> GetParameterLayout() const {
      return std::vector<PARAMETER_TYPE>(super_type::parameter_count_);
    }
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
      if (!((src_state.size() == num_prev_states_) && (tar_state.size() == num_states_))) {
        std::cerr << "src state size: " << src_state.size() << std::endl;
        std::cerr << "prev state size: " << num_prev_states_ << std::endl;
        std::cerr << "tar state size: " << tar_state.size() << std::endl;
        std::cerr << "state size: " << num_states_ << std::endl;
        throw std::invalid_argument("src state size and prev state size "
                                    "must be equal. tar state size and state"
                                    " size must be equal");
      }
      Index weight_id = 0;
      for (Index iii = 0; iii < tar_state.size(); ++iii) {
        TReal cumulative_sum = 0.0;
        for (Index jjj = 0; jjj < src_state.size(); ++jjj) {
          cumulative_sum += src_state[jjj] * weights_[weight_id];
          ++weight_id;
        }
        tar_state[iii] = utilities::BoundState(cumulative_sum);
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      if (parameters.size() != super_type::parameter_count_) {
        std::cerr << "parameter size: " << parameters.size() << std::endl;
        std::cerr << "parameter count: " << super_type::parameter_count_ << std::endl;
        throw std::invalid_argument("Wrong number of parameters");
      }
      weights_ = multi_array::ConstArraySlice<TReal>(parameters);
    }

    std::vector<PARAMETER_TYPE> GetParameterLayout() const {
      std::vector<PARAMETER_TYPE> layout(super_type::parameter_count_);
      for (Index iii = 0; iii < super_type::parameter_count_; ++iii) {
        layout[iii] = WEIGHT;
      }
      return layout;
    }

  protected:
    Index num_states_;
    Index num_prev_states_;
    multi_array::ConstArraySlice<TReal> weights_;
};

/*
 * Conv integrator - uses implicit structure
 * Uses separable filters, so # params for a KxHxW kernel is K*(H+W)
 *
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
template<typename TReal>
class Conv3DIntegrator : public Integrator<TReal> {
  typedef Integrator<TReal> super_type;
  typedef std::size_t Index;
  public:
    Conv3DIntegrator(const multi_array::Array<Index,3>& filter_shape, 
        const multi_array::Array<Index,3>& layer_shape, 
        const multi_array::Array<Index,3>& prev_layer_shape, Index stride)
        : num_filters_(layer_shape[0]), layer_shape_(layer_shape), 
        prev_layer_shape_(prev_layer_shape),
        filter_shape_(filter_shape), stride_(stride),
        filter_parameters_major_({num_filters_}),
        filter_parameters_minor_({num_filters_}),
        channel_weights_({num_filters_}) {

      min_src_size_ = std::accumulate(prev_layer_shape_.begin(), 
        prev_layer_shape_.end(), 1, std::multiplies<TReal>());
      min_tar_size_ = std::accumulate(layer_shape_.begin(), layer_shape_.end(), 
        1, std::multiplies<TReal>());
      // NumDim should be 3 for stride calculation
      multi_array::CalculateStrides(layer_shape_.data(), layer_strides_.data(), 3);
      multi_array::CalculateStrides(prev_layer_shape_.data(), prev_layer_strides_.data(), 3);
      if (filter_shape[0] != prev_layer_shape[0]) {
        std::cerr << "first filter shape: " << filter_shape[0] << std::endl;
        std::cerr << "first prev layer shape: " << prev_layer_shape[0] << std::endl;
        throw std::invalid_argument("First dimensions must be equal.");
      }

      super_type::parameter_count_ = num_filters_ 
        * (filter_shape_[2] + filter_shape_[1] + filter_shape_[0]);
      super_type::integrator_type_ = CONV_INTEGRATOR;
      firstpass_buffer_ = multi_array::Tensor<TReal>(
                            {prev_layer_shape[1], layer_shape[2]});
      secondpass_buffer_ = multi_array::Tensor<TReal>(
                            {layer_shape[1], layer_shape[2]});
    }
    ~Conv3DIntegrator()=default;

    void operator()(const multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state) {
      /*
       * tar_state needs to have AT LEAST as many elements as required by
       * layer_shape, but it doesn't not need to have the same shape.
       * tar_state will be re-interpreted as the same shape as layer_shape
       * src_state will be reinterpreted as the same shape as prev_layer_shape
       */
      if (src_state.size() < min_src_size_) {
        throw std::invalid_argument("Src state too small for integrator");
      }
      const multi_array::TensorView<TReal> src_view(src_state.data(), 
        prev_layer_strides_.data(), prev_layer_shape_.data());

      if (tar_state.size() < min_tar_size_) {
        throw std::invalid_argument("tar state too small for integrator");
      }
      multi_array::TensorView<TReal> tar_view(tar_state.data(), 
        layer_strides_.data(), layer_shape_.data());

      multi_array::ArrayView<multi_array::ConstArraySlice<TReal>, 1> major_view(filter_parameters_major_);
      multi_array::ArrayView<multi_array::ConstArraySlice<TReal>, 1> minor_view(filter_parameters_minor_);
      multi_array::ArrayView<multi_array::ConstArraySlice<TReal>, 1> channel_view(channel_weights_);

      // for each filter
      for (Index iii = 0; iii < num_filters_; iii++) {
        multi_array::TensorView<TReal> tar_image = tar_view[iii];
        // for each input channel
        for (Index jjj = 0; jjj < src_view.extent(0); jjj++) {
          // Carry out separable conv on layer of inputs
          const multi_array::TensorView<TReal> src_image = src_view[jjj];
          multi_array::TensorView<TReal> secondpass_image = secondpass_buffer_.accessor();
          multi_array::TensorView<TReal> firstpass_image = firstpass_buffer_.accessor();
          Convolve2D(src_image, secondpass_image, firstpass_image,
                     major_view[iii], minor_view[iii], stride_);

          for (Index kkk = 0; kkk < tar_view.extent(1); ++kkk) {
            for (Index lll = 0; lll < tar_view.extent(2); ++lll) {
              tar_image[kkk][lll] += secondpass_image[kkk][lll] * channel_weights_[iii][jjj];
              tar_image[kkk][lll] = utilities::BoundState<TReal>(tar_image[kkk][lll]);
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
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {

      if (parameters.size() != super_type::parameter_count_) {
        std::cerr << "parameter size: " << parameters.size() << std::endl;
        std::cerr << "parameter count: " << super_type::parameter_count_ << std::endl;
        throw std::invalid_argument("Wrong number of parameters");
      }
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

    std::vector<PARAMETER_TYPE> GetParameterLayout() const {
      std::vector<PARAMETER_TYPE> layout(super_type::parameter_count_);
      for (Index iii = 0; iii < super_type::parameter_count_; ++iii) {
        layout[iii] = WEIGHT;
      }
      return layout;
    }

  protected:
    Index num_filters_;
    multi_array::Array<Index, 3> layer_shape_;
    multi_array::Array<Index, 3> prev_layer_shape_;
    multi_array::Array<Index, 3> filter_shape_;
    Index stride_;
    Index min_src_size_;
    Index min_tar_size_;
    multi_array::Array<Index, 3> prev_layer_strides_;
    multi_array::Array<Index, 3> layer_strides_;
    multi_array::MultiArray<multi_array::ConstArraySlice<TReal>, 1> filter_parameters_major_;
    multi_array::MultiArray<multi_array::ConstArraySlice<TReal>, 1> filter_parameters_minor_;
    multi_array::MultiArray<multi_array::ConstArraySlice<TReal>, 1> channel_weights_;
    multi_array::Tensor<TReal> secondpass_buffer_;
    multi_array::Tensor<TReal> firstpass_buffer_;
};

// Network integrator -- uses explicit unweighted structure
template<typename TReal>
class RecurrentIntegrator : public Integrator<TReal> {
  typedef Integrator<TReal> super_type;
  typedef std::size_t Index;
  public:

    RecurrentIntegrator(const graphs::PredecessorGraph<>& network) 
        : network_(network) {
      super_type::integrator_type_ = RECURRENT_INTEGRATOR;
      super_type::parameter_count_ = network_.NumEdges();
    }

    ~RecurrentIntegrator()=default;

    void operator()(const multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state) {
      
      if (tar_state.size() != network_.NumNodes()) {
        std::cerr << "tar state size: " << tar_state.size() << std::endl;
        std::cerr << "network size: " << network_.NumNodes() << std::endl;
        throw std::invalid_argument("tar state must have the same number of "
                                    "nodes as the network");
      }
      Index edge_id = 0;
      for (Index node = 0; node < network_.NumNodes(); ++node) {
        for (Index iii = 0; iii < network_.Predecessors(node).size(); ++iii) {
          tar_state[node] += src_state.at(network_.Predecessors(node)[iii].source) * weights_[edge_id];
          tar_state[node] = utilities::BoundState(tar_state[node]);
          ++edge_id;
        }
      }
      if (edge_id != network_.NumEdges()) {
        throw std::runtime_error("Miss match between number of edges and the"
                                 " number integrated");
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      if (parameters.size() != super_type::parameter_count_) {
        std::cerr << "parameter size: " << parameters.size() << std::endl;
        std::cerr << "parameter count: " << super_type::parameter_count_ << std::endl;
        throw std::invalid_argument("Wrong number of parameters");
      }
      weights_ = parameters.slice(0, super_type::parameter_count_);
    }

    std::vector<PARAMETER_TYPE> GetParameterLayout() const {
      std::vector<PARAMETER_TYPE> layout(super_type::parameter_count_);
      for (Index iii = 0; iii < super_type::parameter_count_; ++iii) {
        layout[iii] = WEIGHT;
      }
      return layout;
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
      super_type::integrator_type_ = RESERVOIR_INTEGRATOR;
      super_type::parameter_count_ = 0;
    }

    ~ReservoirIntegrator()=default;

    void operator()(const multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state) {
      if (tar_state.size() != network_.NumNodes()) {
        std::cerr << "tar state size: " << tar_state.size() << std::endl;
        std::cerr << "network size: " << network_.NumNodes() << std::endl;
        throw std::invalid_argument("tar state must have the same number of "
                                    "nodes as the network");
      }
      for (Index node = 0; node < network_.NumNodes(); ++node) {
        for (Index iii = 0; iii < network_.Predecessors(node).size(); ++iii) {
          tar_state[node] += src_state.at(network_.Predecessors(node)[iii].source)
                          * network_.Predecessors(node)[iii].weight;
          tar_state[node] = utilities::BoundState(tar_state[node]);
        }
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {}

    std::vector<PARAMETER_TYPE> GetParameterLayout() const {
      return std::vector<PARAMETER_TYPE>(super_type::parameter_count_);
    }

  protected:
    graphs::PredecessorGraph<TReal> network_;
};

// A hybrid reservoir, where only some of the weights are trained
// template<typename TReal>
// class ReservoirHybridIntegrator : public Integrator<TReal> {
//   typedef Integrator<TReal> super_type;
//   typedef std::size_t Index;
//   public:

//   protected:
//     graphs::PredecessorGraph<TReal> network_;
//     multi_array::ConstArraySlice<TReal> weights_;
// };

} // End nervous_system namespace

#endif /* NN_INTEGRATOR_H_ */