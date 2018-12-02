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
#include <utility>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "../common/multi_array.hpp"
#include "../common/graphs.hpp"
#include "parameter_types.hpp"
#include "../common/utilities.hpp"

namespace nervous_system {

enum INTEGRATOR_TYPE {
  BASE_INTEGRATOR,
  NONE_INTEGRATOR,
  ALL2ALL_INTEGRATOR,
  CONV_INTEGRATOR,
  RECURRENT_INTEGRATOR,
  RESERVOIR_INTEGRATOR,
  RESERVOIR_HYBRID,
  TRUNCATED_RECURRENT_INTEGRATOR,
  CONV_EIGEN_INTEGRATOR,
  ALL2ALL_EIGEN_INTEGRATOR,
  RECURRENT_EIGEN_INTEGRATOR,
  RESERVOIR_EIGEN_INTEGRATOR,
  REWARD_MODULATED_INTEGRATOR,
  REWARD_MODULATED_ALL2ALL_INTEGRATOR,
  REWARD_MODULATED_RECURRENT_INTEGRATOR,
  REWARD_MODULATED_CONV_INTEGRATOR
};

// Abstract base class
template<typename TReal>
class Integrator {
  public:
    typedef std::size_t Index;

    Integrator() {
      integrator_type_ = BASE_INTEGRATOR;
      parameter_count_ = 0;
    }
    virtual ~Integrator()=default;
    virtual void operator()(const multi_array::Tensor<TReal>& src_state,
                            multi_array::Tensor<TReal>& tar_state)=0;
    virtual void Configure(const multi_array::ConstArraySlice<TReal>& parameters)=0;

    std::size_t GetParameterCount() const {
      return parameter_count_;
    }

    INTEGRATOR_TYPE GetIntegratorType() const {
      return integrator_type_;
    }

    virtual std::vector<PARAMETER_TYPE> GetParameterLayout() const=0;

    /*
     * Returns a pair representing the start and stop indices for the parameters
     * associated with link weights. These should always be contiguous.
     */
    virtual std::pair<Index, Index> GetWeightIndexRange() const=0;

  protected:
    INTEGRATOR_TYPE integrator_type_;
    std::size_t parameter_count_;
};

template <typename TReal>
class RewardModulatedIntegrator : public virtual Integrator<TReal>
{
  public:
    typedef std::size_t Index;
    typedef Integrator<TReal> super_type;

    RewardModulatedIntegrator(const TReal learning_rate)
        : super_type(), learning_rate_(learning_rate) {
    }
    virtual ~RewardModulatedIntegrator()= default;
    virtual void UpdateWeights(const TReal reward,
                               const TReal reward_average,
                               const multi_array::Tensor<TReal>& src_state,
                               const multi_array::Tensor<TReal>& tar_state,
                               const multi_array::Tensor<TReal>& tar_state_averages)=0;
    virtual const multi_array::Tensor<TReal>& GetWeights() const=0;

  protected:
    const TReal learning_rate_;
};

// None integrator - does nothing
template<typename TReal>
class NoneIntegrator : public virtual Integrator<TReal> {
  public:
    typedef Integrator<TReal> super_type;
    typedef typename super_type::Index Index;

    NoneIntegrator() : super_type() { super_type::integrator_type_ = NONE_INTEGRATOR; }
    virtual ~NoneIntegrator()=default;

    virtual void operator()(const multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state) {}
    virtual void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {}

    virtual std::vector<PARAMETER_TYPE> GetParameterLayout() const {
      return std::vector<PARAMETER_TYPE>(super_type::parameter_count_);
    }

    virtual std::pair<Index, Index> GetWeightIndexRange() const {
      return std::make_pair(0, 0);
    }
};

// Integrator that has All2All connectivity with previous layer
template<typename TReal>
class All2AllIntegrator : public virtual Integrator<TReal> {
  public:
    typedef Integrator<TReal> super_type;
    typedef typename super_type::Index Index;

    All2AllIntegrator(Index num_states, Index num_prev_states)
        : num_states_(num_states), num_prev_states_(num_prev_states) {
      super_type::parameter_count_ = num_states_ * num_prev_states_;
      super_type::integrator_type_ = ALL2ALL_INTEGRATOR;
    }

    virtual void operator()(const multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state) {
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

    virtual void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      if (parameters.size() != super_type::parameter_count_) {
        std::cerr << "parameter size: " << parameters.size() << std::endl;
        std::cerr << "parameter count: " << super_type::parameter_count_ << std::endl;
        throw std::invalid_argument("Wrong number of parameters");
      }
      weights_ = multi_array::ConstArraySlice<TReal>(parameters);
    }

    virtual std::vector<PARAMETER_TYPE> GetParameterLayout() const {
      std::vector<PARAMETER_TYPE> layout(super_type::parameter_count_);
      for (Index iii = 0; iii < super_type::parameter_count_; ++iii) {
        layout[iii] = WEIGHT;
      }
      return layout;
    }

    virtual std::pair<Index, Index> GetWeightIndexRange() const {
      return std::make_pair(0, super_type::parameter_count_);
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
class Conv2DIntegrator : public virtual Integrator<TReal> {
  public:
    typedef Integrator<TReal> super_type;
    typedef typename super_type::Index Index;

    Conv2DIntegrator(const multi_array::Array<Index,3>& filter_shape,
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
    virtual ~Conv2DIntegrator()=default;

    virtual void operator()(const multi_array::Tensor<TReal>& src_state,
                            multi_array::Tensor<TReal>& tar_state) {
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

    virtual void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {

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

    virtual std::vector<PARAMETER_TYPE> GetParameterLayout() const {
      std::vector<PARAMETER_TYPE> layout(super_type::parameter_count_);
      for (Index iii = 0; iii < super_type::parameter_count_; ++iii) {
        layout[iii] = WEIGHT;
      }
      return layout;
    }

    virtual std::pair<Index, Index> GetWeightIndexRange() const {
      return std::make_pair(0, super_type::parameter_count_);
    }

    const multi_array::Array<Index, 3>& GetFilterShape() const {
      return filter_shape_;
    };

    Index GetMinTarSize() const {
      return min_tar_size_;
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
class RecurrentIntegrator : public virtual Integrator<TReal> {
  public:
    typedef Integrator<TReal> super_type;
    typedef typename super_type::Index Index;

    RecurrentIntegrator(const graphs::PredecessorGraph<>& network) 
        : network_(network) {
      super_type::integrator_type_ = RECURRENT_INTEGRATOR;
      super_type::parameter_count_ = network_.NumEdges();
    }

    virtual ~RecurrentIntegrator()=default;

    virtual void operator()(const multi_array::Tensor<TReal>& src_state,
                    multi_array::Tensor<TReal>& tar_state) {
      
      /*
       * Graph maybe a connector graph or an internal graph.
       * In case of connector, the num nodes doesn't need to match tar_state,
       * because predecessors should be empty for nodes not in tar_state.
       * However, src state does have to be checked using (at) when tar_state
       * is larger than src_state, to ensure nothing invalid is accessed.
       */
      Index edge_id = 0;
      for (Index node = 0; node < network_.NumNodes(); ++node) {
        for (Index iii = 0; iii < network_.Predecessors(node).size(); ++iii) {
          tar_state.at(node) += src_state.at(network_.Predecessors(node)[iii].source) * weights_[edge_id];
          tar_state[node] = utilities::BoundState(tar_state[node]);
          ++edge_id;
        }
      }
      if (edge_id != network_.NumEdges()) {
        throw std::runtime_error("Miss match between number of edges and the"
                                 " number integrated");
      }
    }

    virtual void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      if (parameters.size() != super_type::parameter_count_) {
        std::cerr << "parameter size: " << parameters.size() << std::endl;
        std::cerr << "parameter count: " << super_type::parameter_count_ << std::endl;
        throw std::invalid_argument("Wrong number of parameters");
      }
      weights_ = parameters.slice(0, super_type::parameter_count_);
    }

    virtual std::vector<PARAMETER_TYPE> GetParameterLayout() const {
      std::vector<PARAMETER_TYPE> layout(super_type::parameter_count_);
      for (Index iii = 0; iii < super_type::parameter_count_; ++iii) {
        layout[iii] = WEIGHT;
      }
      return layout;
    }

    const multi_array::ConstArraySlice<TReal>& GetWeights() const {
      return weights_;
    }

    virtual std::pair<Index, Index> GetWeightIndexRange() const {
      return std::make_pair(0, super_type::parameter_count_);
    }

    const graphs::PredecessorGraph<>& GetGraph() const {
      return network_;
    }

  protected:
    graphs::PredecessorGraph<> network_;
    multi_array::ConstArraySlice<TReal> weights_;
};

// Truncated Recurrent integrator sets weights to 0 during calculations if
// they are below the magnitude of the threshold.
template<typename TReal>
class TruncatedRecurrentIntegrator : public virtual RecurrentIntegrator<TReal> {
  public:
    typedef RecurrentIntegrator<TReal> super_type;
    typedef typename super_type::Index Index;

    TruncatedRecurrentIntegrator(const graphs::PredecessorGraph<>& network,
                                 TReal weight_threshold)
        : super_type(network), weight_threshold_(weight_threshold) {

      super_type::integrator_type_ = TRUNCATED_RECURRENT_INTEGRATOR;
    }

    virtual ~TruncatedRecurrentIntegrator()= default;

    virtual void operator()(const multi_array::Tensor<TReal>& src_state,
                    multi_array::Tensor<TReal>& tar_state) {

      Index edge_id = 0;
      for (Index node = 0; node < super_type::network_.NumNodes(); ++node) {
        for (Index iii = 0; iii < super_type::network_.Predecessors(node).size(); ++iii) {
          // Only carry out calculation if it exceeds magnitude of the threshold
          if ((super_type::weights_[edge_id] > weight_threshold_
               && super_type::weights_[edge_id] >= 0) ||
              (super_type::weights_[edge_id] < -weight_threshold_
               && super_type::weights_[edge_id] <= 0)) {
            tar_state.at(node) += src_state.at(super_type::network_.Predecessors(node)[iii].source)
                               * super_type::weights_[edge_id];
            tar_state[node] = utilities::BoundState(tar_state[node]);
          }
          ++edge_id;
        }
      }
      if (edge_id != super_type::network_.NumEdges()) {
        throw std::runtime_error("Miss match between number of edges and the"
                                 " number integrated");
      }
    }

    TReal GetWeightThreshold() const {
      return weight_threshold_;
    }

  protected:
    TReal weight_threshold_;
};

// Reservoir -- uses explicit weighted structure
template<typename TReal>
class ReservoirIntegrator : public virtual Integrator<TReal> {
  public:
    typedef Integrator<TReal> super_type;
    typedef typename super_type::Index Index;

    ReservoirIntegrator(const graphs::PredecessorGraph<TReal>& network)
        : network_(network) {
      super_type::integrator_type_ = RESERVOIR_INTEGRATOR;
      super_type::parameter_count_ = 0;
    }

    ~ReservoirIntegrator()=default;

    void operator()(const multi_array::Tensor<TReal>& src_state,
                    multi_array::Tensor<TReal>& tar_state) {

      for (Index node = 0; node < network_.NumNodes(); ++node) {
        for (Index iii = 0; iii < network_.Predecessors(node).size(); ++iii) {
          tar_state.at(node) += src_state.at(network_.Predecessors(node)[iii].source)
                              * network_.Predecessors(node)[iii].weight;
          tar_state[node] = utilities::BoundState(tar_state[node]);
        }
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {}

    std::vector<PARAMETER_TYPE> GetParameterLayout() const {
      return std::vector<PARAMETER_TYPE>(super_type::parameter_count_);
    }

    std::pair<Index, Index> GetWeightIndexRange() const {
      return std::make_pair(0, 0);
    }

    const graphs::PredecessorGraph<TReal>& GetGraph() const {
      return network_;
    }

  protected:
    graphs::PredecessorGraph<TReal> network_;
};

/*
 * Implements non-separable convolution using im2col and gemm through Eigen
 * Assumes NCHW memory layout
 * TODO: Support even filters
 */
template<typename TReal>
class ConvEigenIntegrator : public virtual Integrator<TReal> {
  public:
    typedef Integrator<TReal> super_type;
    typedef typename super_type::Index Index;
    typedef Eigen::Matrix<TReal, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef const Eigen::Matrix<TReal, Eigen::Dynamic, Eigen::Dynamic> ConstMatrix;
    typedef Eigen::Map<Matrix> MatrixView;
    typedef const Eigen::Map<ConstMatrix> ConstMatrixView;

    /*
     * Filter shape order: {depth, height, width}
     * Layer shape order: {# channels, height, width}
     * Prev Layer order: {# channels, height, width}
     * Note: This is a bit confusing because my tensors us C++ memory order
     * and Eigen uses Fortran. So I list shapes here with last element of shape
     * being the major axis. This is reversed in Eigen, where first is the major.
     */
    ConvEigenIntegrator(const multi_array::Array<Index,3>& filter_shape,
                        const multi_array::Array<Index,3>& layer_shape,
                        const multi_array::Array<Index,3>& prev_layer_shape,
                        Index stride)
                      : num_filters_(layer_shape[0]), layer_shape_(layer_shape),
                        prev_layer_shape_(prev_layer_shape),
                        filter_shape_(filter_shape), stride_(stride),
                        channels_(prev_layer_shape[0]),
                        height_(prev_layer_shape[1]),
                        width_(prev_layer_shape[2]),
                        kernel_h_(filter_shape[1]),
                        kernel_w_(filter_shape[2]),
                        pad_h_(kernel_h_/2),
                        pad_w_(kernel_w_/2),
                        channel_size_(((height_ + 2 * pad_h_ - kernel_h_)
                                       / stride_ + 1)
                                      * ((width_ + 2 * pad_w_ - kernel_w_)
                                         / stride_ + 1)),
                        buffer_state_(channel_size_,
                                      kernel_h_ * kernel_w_ * channels_) {
      super_type::parameter_count_ = kernel_w_ * kernel_h_ * channels_ * num_filters_;
      super_type::integrator_type_ = CONV_EIGEN_INTEGRATOR;
    }

    virtual ~ConvEigenIntegrator()=default;

    /*
     * Matrix: (# rows, # cols) -> column major
     * src shape: {height * width, channels}
     * tar shape: {(new height * new width), num_filters}
     * Note: If you check tensor shape it will be the reverse, since I use C's
     * order, while Eigen uses Fortran's. Sorry for the confusion, but Eigen
     * got added in latter and I am not sure why they choose Fortran's way for a
     * C++ API.
     */
    virtual void operator()(const multi_array::Tensor<TReal>& src_state,
                    multi_array::Tensor<TReal>& tar_state) override {

      utilities::Im2Col(src_state.data(), channels_, height_, width_,
                        kernel_h_, kernel_w_, pad_h_, pad_w_, stride_, stride_,
                        1, 1, buffer_state_.data());
      MatrixView output(tar_state.data(), channel_size_, num_filters_);
      ConstMatrixView params(weight_view_.data() + weight_view_.start(),
                             kernel_w_ * kernel_h_ * channels_,
                             num_filters_);
      output.noalias() = buffer_state_ * params;
    }

    virtual void Configure(const multi_array::ConstArraySlice<TReal>& parameters) override {
      weight_view_ = parameters;
    }

    virtual std::vector<PARAMETER_TYPE> GetParameterLayout() const override {
      std::vector<PARAMETER_TYPE> layout(super_type::parameter_count_);
      for (Index iii = 0; iii < super_type::parameter_count_; ++iii) {
        layout[iii] = WEIGHT;
      }
      return layout;
    }

    virtual std::pair<Index, Index> GetWeightIndexRange() const override {
      return std::make_pair(0, super_type::parameter_count_);
    };

  protected:
    const utilities::Integer num_filters_;
    const multi_array::Array<Index, 3> layer_shape_;
    const multi_array::Array<Index, 3> prev_layer_shape_;
    const multi_array::Array<Index, 3> filter_shape_;
    const utilities::Integer stride_;
    const utilities::Integer channels_;
    const utilities::Integer height_;
    const utilities::Integer width_;
    const utilities::Integer kernel_h_;
    const utilities::Integer kernel_w_;
    const utilities::Integer pad_h_;
    const utilities::Integer pad_w_;
    const utilities::Integer channel_size_;
    Matrix buffer_state_;
    multi_array::ConstArraySlice<TReal> weight_view_;
};

template <typename TReal>
class RewardModulatedConvIntegrator : public ConvEigenIntegrator<TReal>,
                                      public RewardModulatedIntegrator<TReal> {
  public:
    typedef RewardModulatedIntegrator<TReal> reward_modulator_type;
    typedef ConvEigenIntegrator<TReal> conv_type;
    typedef typename conv_type::Index Index;
    typedef typename conv_type::Matrix Matrix;
    typedef typename conv_type::ConstMatrix ConstMatrix;
    typedef typename conv_type::MatrixView MatrixView;
    typedef typename conv_type::ConstMatrixView ConstMatrixView;

    RewardModulatedConvIntegrator(const multi_array::Array<Index,3>& filter_shape,
                                  const multi_array::Array<Index,3>& layer_shape,
                                  const multi_array::Array<Index,3>& prev_layer_shape,
                                  Index stride, const TReal learning_rate)
        : conv_type(filter_shape, layer_shape, prev_layer_shape, stride),
          reward_modulator_type(learning_rate),
          weights_({conv_type::parameter_count_}) {
      conv_type::integrator_type_ = REWARD_MODULATED_CONV_INTEGRATOR;
    }

    void operator()(const multi_array::Tensor<TReal>& src_state,
                    multi_array::Tensor<TReal>& tar_state) override {

      utilities::Im2Col(src_state.data(), conv_type::channels_,
                        conv_type::height_, conv_type::width_,
                        conv_type::kernel_h_, conv_type::kernel_w_,
                        conv_type::pad_h_, conv_type::pad_w_,
                        conv_type::stride_, conv_type::stride_,
                        1, 1, conv_type::buffer_state_.data());
      MatrixView output(tar_state.data(), conv_type::channel_size_,
                        conv_type::num_filters_);
      ConstMatrixView params(weights_.data(),
                             conv_type::kernel_w_
                             * conv_type::kernel_h_
                             * conv_type::channels_,
                             conv_type::num_filters_);
      output.noalias() = conv_type::buffer_state_ * params;
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      if (parameters.size() != conv_type::parameter_count_) {
        std::cerr << "parameter size: " << parameters.size() << std::endl;
        std::cerr << "parameter count: " << conv_type::parameter_count_ << std::endl;
        throw std::invalid_argument("Wrong number of parameters");
      }

      for (Index iii = 0; iii < parameters.size(); ++iii) {
        weights_[iii] = parameters[iii];
      }
      conv_type::weight_view_ = multi_array::ConstArraySlice<TReal>(weights_.data(), 0,
                                                         weights_.size());
    }

    void UpdateWeights(const TReal reward,
                       const TReal reward_average,
                       const multi_array::Tensor<TReal>& src_state,
                       const multi_array::Tensor<TReal>& tar_state,
                       const multi_array::Tensor<TReal>& tar_state_averages) {

      MatrixView weights(weights_.data(),
                         conv_type::kernel_w_
                         * conv_type::kernel_h_
                         * conv_type::channels_,
                         conv_type::num_filters_);
      ConstMatrixView tar_view(tar_state.data(), conv_type::channel_size_,
                               conv_type::num_filters_);
      ConstMatrixView tar_avg_view(tar_state_averages.data(),
                                   conv_type::channel_size_,
                                   conv_type::num_filters_);

      std::cout << "weights dim: cols " << weights.cols() << " rows " << weights.rows() << std::endl;/////////////
      std::cout << "tar view dim: cols " << tar_view.cols() << " rows " << tar_view.rows() << std::endl;/////////////
      std::cout << "tar avg view dim: cols " << tar_avg_view.cols() << " rows " << tar_avg_view.rows() << std::endl;/////////////
      std::cout << "buff dim: cols " << conv_type::buffer_state_.cols() << " rows " << conv_type::buffer_state_.rows() << std::endl;/////////////

      // find max index tar state, updates single filter
      const Index max_neuron_index = utilities::IndexOfMaxElement(tar_state);
      // Get the filter index for the max neuron
      const Index max_filter = max_neuron_index / conv_type::channel_size_;
      // The buffer_state im2col matrix has a window of states from the previous
      // layer that will correspond to the states that need to be integrated
      // for a given position in the current layer (same for each channel)
      // This position is modulo the size of the channel:
      const Index max_neuron_window = max_neuron_index % conv_type::channel_size_;
      // Alternatively find max index tar state for each filter, updates each filter

      // Loops through src neuron states
      const TReal reward_modulated_learning_factor = (reward - reward_average)
                                                     * reward_modulator_type::learning_rate_;
      for (auto src_neuron = 0; src_neuron < conv_type::buffer_state_.cols(); ++src_neuron) {
        weights(src_neuron, max_filter) += conv_type::buffer_state_(max_neuron_window,
                                                                    src_neuron)
                                 * (tar_state[max_neuron_index]
                                    - tar_state_averages[max_neuron_index])
                                 * reward_modulated_learning_factor;
      }
    }

    const multi_array::Tensor<TReal>& GetWeights() const
    {
      return weights_;
    }

  protected:
    multi_array::Tensor<TReal> weights_;
};

/*
 * Implements an All2All integrator with an Eigen backend
 */
template<typename TReal>
class All2AllEigenIntegrator : public virtual Integrator<TReal> {
  public:
    typedef Integrator<TReal> super_type;
    typedef typename super_type::Index Index;
    typedef Eigen::Matrix<TReal, Eigen::Dynamic, 1> ColVector;
    typedef Eigen::Map<ColVector> ColVectorView;
    typedef const Eigen::Matrix<TReal, Eigen::Dynamic, 1> ConstColVector;
    typedef const Eigen::Map<ConstColVector> ConstColVectorView;
    typedef const Eigen::Matrix<TReal, Eigen::Dynamic, Eigen::Dynamic> ConstMatrix;
    typedef const Eigen::Map<ConstMatrix> ConstMatrixView;

    All2AllEigenIntegrator(Index num_states, Index num_prev_states)
    : num_states_(num_states), num_prev_states_(num_prev_states) {
      super_type::parameter_count_ = num_states_ * num_prev_states_;
      super_type::integrator_type_ = ALL2ALL_EIGEN_INTEGRATOR;
    }

    virtual void operator()(const multi_array::Tensor<TReal>& src_state,
                            multi_array::Tensor<TReal>& tar_state) {
      if (!((src_state.size() == num_prev_states_) && (tar_state.size() == num_states_))) {
        std::cerr << "src state size: " << src_state.size() << std::endl;
        std::cerr << "prev state size: " << num_prev_states_ << std::endl;
        std::cerr << "tar state size: " << tar_state.size() << std::endl;
        std::cerr << "state size: " << num_states_ << std::endl;
        throw std::invalid_argument("src state size and prev state size "
                                    "must be equal. tar state size and state"
                                    " size must be equal");
      }

      ColVectorView output(tar_state.data(), tar_state.size());
      output.noalias() = ConstMatrixView(weight_view_.data() + weight_view_.start(),
                                         tar_state.size(),
                                         src_state.size())
                         * ConstColVectorView(src_state.data(), src_state.size());
    }

    virtual void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      if (parameters.size() != super_type::parameter_count_) {
        std::cerr << "parameter size: " << parameters.size() << std::endl;
        std::cerr << "parameter count: " << super_type::parameter_count_ << std::endl;
        throw std::invalid_argument("Wrong number of parameters");
      }
      weight_view_ = parameters;
    }

    virtual std::vector<PARAMETER_TYPE> GetParameterLayout() const {
      std::vector<PARAMETER_TYPE> layout(super_type::parameter_count_);
      for (Index iii = 0; iii < super_type::parameter_count_; ++iii) {
        layout[iii] = WEIGHT;
      }
      return layout;
    }

    virtual std::pair<Index, Index> GetWeightIndexRange() const {
      return std::make_pair(0, super_type::parameter_count_);
    }

  protected:
    Index num_states_;
    Index num_prev_states_;
    multi_array::ConstArraySlice<TReal> weight_view_;
};

/*
 * Implements a recurrent integrator with Eigen
 */
template<typename TReal>
class RecurrentEigenIntegrator : public virtual Integrator<TReal> {
  public:
    typedef Integrator<TReal> super_type;
    typedef typename super_type::Index Index;
    typedef Eigen::Matrix<TReal, Eigen::Dynamic, 1> ColVector;
    typedef Eigen::Map<ColVector> ColVectorView;
    typedef const Eigen::Matrix<TReal, Eigen::Dynamic, 1> ConstColVector;
    typedef const Eigen::Map<ConstColVector> ConstColVectorView;
    typedef Eigen::SparseMatrix<TReal> SparseMatrix;
    typedef const Eigen::SparseMatrix<TReal> ConstSparseMatrix;
    typedef const Eigen::Map<ConstSparseMatrix> ConstSparseMatrixView;

    RecurrentEigenIntegrator(SparseMatrix network) : network_(std::move(network)) {
      network_.makeCompressed();
      super_type::integrator_type_ = RECURRENT_EIGEN_INTEGRATOR;
      super_type::parameter_count_ = network_.nonZeros();
    }

    virtual ~RecurrentEigenIntegrator()=default;

    virtual void operator()(const multi_array::Tensor<TReal>& src_state,
                            multi_array::Tensor<TReal>& tar_state) {

      if ((network_.cols() != src_state.size())
          && (network_.rows() != tar_state.size())) {
        throw std::invalid_argument("src state size and tar state size "
                                    "incompatible with network");
      }

      ConstSparseMatrixView weight_matrix(network_.rows(), network_.cols(),
                                          weight_view_.size(), network_.outerIndexPtr(),
                                          network_.innerIndexPtr(),
                                          weight_view_.data() + weight_view_.start(),
                                          network_.innerNonZeroPtr());
      ColVectorView output_vector(tar_state.data(), tar_state.size());
      ConstColVectorView src_vector(src_state.data(), src_state.size());

      output_vector.noalias() = weight_matrix * src_vector;
    }

    virtual void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      if (parameters.size() != super_type::parameter_count_) {
        std::cerr << "parameter size: " << parameters.size() << std::endl;
        std::cerr << "parameter count: " << super_type::parameter_count_ << std::endl;
        throw std::invalid_argument("Wrong number of parameters");
      }
      weight_view_ = parameters.slice(0, super_type::parameter_count_);
    }

    virtual std::vector<PARAMETER_TYPE> GetParameterLayout() const {
      std::vector<PARAMETER_TYPE> layout(super_type::parameter_count_);
      for (Index iii = 0; iii < super_type::parameter_count_; ++iii) {
        layout[iii] = WEIGHT;
      }
      return layout;
    }

    const multi_array::ConstArraySlice<TReal>& GetWeights() const {
      return weight_view_;
    }

    virtual std::pair<Index, Index> GetWeightIndexRange() const {
      return std::make_pair(0, super_type::parameter_count_);
    }

  protected:
    SparseMatrix network_;
    multi_array::ConstArraySlice<TReal> weight_view_;
};

/*
 * Implements a reservior layer using Eigen
 */
template<typename TReal>
class ReservoirEigenIntegrator : public virtual Integrator<TReal> {
  public:
    typedef Integrator<TReal> super_type;
    typedef typename super_type::Index Index;
    typedef Eigen::Matrix<TReal, Eigen::Dynamic, 1> ColVector;
    typedef Eigen::Map <ColVector> ColVectorView;
    typedef const Eigen::Matrix<TReal, Eigen::Dynamic, 1> ConstColVector;
    typedef const Eigen::Map <ConstColVector> ConstColVectorView;
    typedef Eigen::SparseMatrix <TReal> SparseMatrix;

    ReservoirEigenIntegrator(SparseMatrix network)
    : network_(std::move(network)) {
      network_.makeCompressed();
      super_type::integrator_type_ = RESERVOIR_EIGEN_INTEGRATOR;
      super_type::parameter_count_ = 0;
    }

    ~ReservoirEigenIntegrator()=default;

    void operator()(const multi_array::Tensor<TReal>& src_state,
                    multi_array::Tensor<TReal>& tar_state) {

      if ((network_.cols() != src_state.size())
          && (network_.rows() != tar_state.size())) {
        throw std::invalid_argument("src state size and tar state size "
                                    "incompatible with network");
      }

      ColVectorView output_vector(tar_state.data(), tar_state.size());
      ConstColVectorView src_vector(src_state.data(), src_state.size());
      output_vector.noalias() = network_ * src_vector;
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {}

    std::vector<PARAMETER_TYPE> GetParameterLayout() const {
      return std::vector<PARAMETER_TYPE>(super_type::parameter_count_);
    }

    std::pair<Index, Index> GetWeightIndexRange() const {
      return std::make_pair(0, 0);
    }

  protected:
    SparseMatrix network_;
};

template <typename TReal>
class RewardModulatedAll2AllIntegrator : public All2AllEigenIntegrator<TReal>,
                                         public RewardModulatedIntegrator<TReal> {
  public:
    typedef All2AllEigenIntegrator<TReal> all2all_type;
    typedef RewardModulatedIntegrator<TReal> reward_modulator_type;
    typedef typename all2all_type::Index Index;
    typedef typename all2all_type::ColVector ColVector;
    typedef typename all2all_type::ColVectorView ColVectorView;
    typedef typename all2all_type::ConstColVector ConstColVector;
    typedef typename all2all_type::ConstColVectorView ConstColVectorView;
    typedef typename all2all_type::ConstMatrix ConstMatrix;
    typedef typename all2all_type::ConstMatrixView ConstMatrixView;
    typedef Eigen::Matrix<TReal, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Map<Matrix> MatrixView;

    RewardModulatedAll2AllIntegrator(const Index num_states,
                                     const Index num_prev_states,
                                     const TReal learning_rate)
      : all2all_type(num_states, num_prev_states),
        reward_modulator_type(learning_rate),
        weights_({all2all_type::parameter_count_}) {
      all2all_type::integrator_type_ = REWARD_MODULATED_ALL2ALL_INTEGRATOR;
    }

    void operator()(const multi_array::Tensor<TReal>& src_state,
                    multi_array::Tensor<TReal>& tar_state) override
    {
      if (!((src_state.size() == all2all_type::num_prev_states_)
            && (tar_state.size() == all2all_type::num_states_))) {
        std::cerr << "src state size: " << src_state.size() << std::endl;
        std::cerr << "prev state size: " << all2all_type::num_prev_states_ << std::endl;
        std::cerr << "tar state size: " << tar_state.size() << std::endl;
        std::cerr << "state size: " << all2all_type::num_states_ << std::endl;
        throw std::invalid_argument("src state size and prev state size "
                                    "must be equal. tar state size and state"
                                    " size must be equal");
      }

      ColVectorView output(tar_state.data(), tar_state.size());
      output.noalias() = ConstMatrixView(weights_.data(),
                                         tar_state.size(),
                                         src_state.size())
                         * ConstColVectorView(src_state.data(), src_state.size());
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) override {
      if (parameters.size() != all2all_type::parameter_count_) {
        std::cerr << "parameter size: " << parameters.size() << std::endl;
        std::cerr << "parameter count: " << all2all_type::parameter_count_ << std::endl;
        throw std::invalid_argument("Wrong number of parameters");
      }
      for (Index iii = 0; iii < parameters.size(); ++iii) {
        weights_[iii] = parameters[iii];
      }
      all2all_type::weight_view_ = multi_array::ConstArraySlice<TReal>(weights_.data(), 0,
                                                         weights_.size());
    }

    void UpdateWeights(const TReal reward,
                       const TReal reward_average,
                       const multi_array::Tensor<TReal>& src_state,
                       const multi_array::Tensor<TReal>& tar_state,
                       const multi_array::Tensor<TReal>& tar_state_averages) override {

      ConstColVectorView src_view(src_state.data(), src_state.size());
      ConstColVectorView tar_view(tar_state.data(), tar_state.size());
      ConstColVectorView tar_avg_view(tar_state_averages.data(), tar_state_averages.size());
      MatrixView weight_view(weights_.data(), tar_state.size(), src_state.size());
      const TReal reward_modulated_learning_factor = (reward - reward_average)
                                                   * reward_modulator_type::learning_rate_;
      for (Index s = 0; s < src_state.size(); ++s) {
        for (Index t = 0; t < tar_state.size(); ++t) {
          weight_view(t, s) += src_view(s)
                               * (tar_view(t) - tar_avg_view(t))
                               * reward_modulated_learning_factor;
        }
      }
    }

    const multi_array::Tensor<TReal>& GetWeights() const
    {
      return weights_;
    }

  protected:
    multi_array::Tensor<TReal> weights_;
};

template <typename TReal>
class RewardModulatedRecurrentIntegrator : public RecurrentEigenIntegrator<TReal>,
                                           public RewardModulatedIntegrator<TReal> {
  public:
    typedef RecurrentEigenIntegrator<TReal> recurrent_type;
    typedef RewardModulatedIntegrator<TReal> reward_modulator_type;
    typedef typename recurrent_type::Index Index;
    typedef typename recurrent_type::ColVector ColVector;
    typedef typename recurrent_type::ColVectorView ColVectorView;
    typedef typename recurrent_type::ConstColVector ConstColVector;
    typedef typename recurrent_type::ConstColVectorView ConstColVectorView;
    typedef typename recurrent_type::SparseMatrix SparseMatrix;
    typedef typename Eigen::Map<SparseMatrix> SparseMatrixView;
    typedef typename recurrent_type::ConstSparseMatrix ConstSparseMatrix;
    typedef typename recurrent_type::ConstSparseMatrixView ConstSparseMatrixView;

    RewardModulatedRecurrentIntegrator(SparseMatrix network, const TReal learning_rate)
        : recurrent_type(network), reward_modulator_type(learning_rate),
          weights_({recurrent_type::parameter_count_}) {
      recurrent_type::integrator_type_ = REWARD_MODULATED_RECURRENT_INTEGRATOR;
    }

    void operator()(const multi_array::Tensor<TReal>& src_state,
                    multi_array::Tensor<TReal>& tar_state) override {

      if ((recurrent_type::network_.cols() != src_state.size())
          && (recurrent_type::network_.rows() != tar_state.size())) {
        throw std::invalid_argument("src state size and tar state size "
                                    "incompatible with network");
      }

      ConstSparseMatrixView weight_matrix(recurrent_type::network_.rows(),
                                          recurrent_type::network_.cols(),
                                          weights_.size(),
                                          recurrent_type::network_.outerIndexPtr(),
                                          recurrent_type::network_.innerIndexPtr(),
                                          weights_.data(),
                                          recurrent_type::network_.innerNonZeroPtr());
      ColVectorView output_vector(tar_state.data(), tar_state.size());
      ConstColVectorView src_vector(src_state.data(), src_state.size());

      output_vector.noalias() = weight_matrix * src_vector;
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) override {
      if (parameters.size() != recurrent_type::parameter_count_) {
        std::cerr << "parameter size: " << parameters.size() << std::endl;
        std::cerr << "parameter count: " << recurrent_type::parameter_count_ << std::endl;
        throw std::invalid_argument("Wrong number of parameters");
      }

      for (Index iii = 0; iii < parameters.size(); ++iii) {
        weights_[iii] = parameters[iii];
      }
      recurrent_type::weight_view_ = multi_array::ConstArraySlice<TReal>(weights_.data(), 0,
                                                         weights_.size());
    }

    void UpdateWeights(const TReal reward,
                       const TReal reward_average,
                       const multi_array::Tensor<TReal>& src_state,
                       const multi_array::Tensor<TReal>& tar_state,
                       const multi_array::Tensor<TReal>& tar_state_averages) override {

      SparseMatrixView weight_matrix(recurrent_type::network_.rows(),
                                     recurrent_type::network_.cols(),
                                     weights_.size(),
                                     recurrent_type::network_.outerIndexPtr(),
                                     recurrent_type::network_.innerIndexPtr(),
                                     weights_.data(),
                                     recurrent_type::network_.innerNonZeroPtr());
      ConstColVectorView src_view(src_state.data(), src_state.size());
      ConstColVectorView tar_view(tar_state.data(), tar_state.size());
      ConstColVectorView tar_avg_view(tar_state_averages.data(), tar_state_averages.size());
      const TReal reward_modulated_learning_factor = (reward - reward_average)
                                                     * reward_modulator_type::learning_rate_;
      for (Index s = 0; s < weight_matrix.outerSize(); ++s) {
        for (typename SparseMatrixView::InnerIterator it(weight_matrix, s); it; ++it) {
          it.valueRef() += src_view(s)
                           * (tar_view(it.row()) - tar_avg_view(it.row()))
                           * reward_modulated_learning_factor;
        }
      }
    }

    const multi_array::Tensor<TReal>& GetWeights() const override
    {
      return weights_;
    }

  protected:
    multi_array::Tensor<TReal> weights_;
};

} // End nervous_system namespace

#endif /* NN_INTEGRATOR_H_ */