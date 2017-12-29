#ifndef NN_INTEGRATOR_H_
#define NN_INTEGRATOR_H_

#include <cstddef>
#include <vector>
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

enum PADDING {
  NONE,
  SAME
}

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
}

// None integrator - does nothing
template<typename TReal>
class NoneIntegrator : public Integrator<TReal> {
  public:
    NoneIntegrator() : Integrator() { integrator_type_ = INTEGRATOR_TYPE.NONE }
    ~NoneIntegrator()=default;

    void operator()(multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state) {}
    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {}
}

/*
 * Conv integrator - uses implicit structure
 * Uses separable filters, so # params for a KxHxW kernel is K*(H+W)
 */
template<typename TReal>
class Conv3DIntegrator : public Integrator {
  public:
    typedef std::size_t Index;

    Conv3DIntegrator(Index num_filters, 
        const multi_array::Array<Index,3>& filter_shape, 
        const multi_array::Array<Index,3>& layer_shape, Index stride)
        : filter_shape_(filter_shape), num_filters_(num_filters), stride_(stride) {
      parameters_per_filter_ = filter_shape[0] * (filter_shape[1] + filter_shape[2]);
      parameter_count_ = num_filters * parameters_per_filter_;
      integrator_type_ = INTEGRATOR_TYPE.CONV;
      state_buffer_1 = multi_array::Tensor<TReal>(layer_shape);
      state_buffer_2 = multi_array::Tensor<TReal>(layer_shape);
      filter_parameters_.resize(num_filters);
    }
    ~Conv3DIntegrator()=default;

    void operator()(const multi_array::Tensor<TReal>& src_state, multi_array::Tensor<TReal>& tar_state) {
      const multi_array::ArrayView<TReal, 3> src_view = src_state.accessor<3>();
      multi_array::ArrayView<TReal, 3> tar_view = tar_state.accessor<3>();
      multi_array::ArrayView<TReal, 3> buffer_view = state_buffer_.accessor<3>();

      // Clear target state for accumulation
      ZeroArray(tar_view);

      // for each filter
      for (Index iii = 0; iii < num_filters_; iii++) {
        multi_array::ArrayView<TReal, 2> tar_image = tar_view[iii];
        // for each input filter
        for (Index jjj = 0; jjj < src_state.shape()[0]; jjj++) {
          // Carry out separable conv on filter
          const multi_array::ArrayView<TReal, 2> src_image = src_view[jjj];
          multi_array::ArrayView<TReal, 2> buffer_image = buffer_view[jjj];
          // call filter function len(shape times), one for each dim (starting with major)
            // Need to initiate rolling sum
            // loop through rolling sum before boundary conflicts (loop on major dim)
            // some function for handling boundary conflicts
          // not sure... should just add 2Dconvs for each filter?
        }
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      for (Index filter = 0; filter < num_filters_; filter++) {
        filter_parameters_[filter] = multi_array::ConstArraySlice<TReal>(
          parameters.data(), 
          parameters.start() + filter * parameters_per_filter_, 
          parameters_per_filter_, parameters.stride());
      }
    }

    void ZeroArray(multi_array::ArrayView<TReal, 3>& view) {
      for (Index iii = 0; iii < view.extent(0); iii++) {
        for (Index jjj = 0; jjj < view.extent(1); jjj++) {
          for (Index kkk = 0; kkk < view.extent(2); kkk++) {
            view[iii][jjj][kkk] = 0.0;
          }
        }
      }
    }

    void Convolve2D(const multi_array::ArrayView<TReal, 2>& src,
        multi_array::ArrayView<TReal, 2>& tar, 
        multi_array::ArrayView<TReal, 2>& buffer,
        const multi_array::ConstArraySlice<TReal> &kernel_minor, 
        const multi_array::ConstArraySlice<TReal> &kernel_major,
        Index stride) {

      // Accumulate through major axis
      Index half_kernel = kernel_major.size() / 2;
      TReal major_first = kernel_major[kernel_major.size()-1];
      TReal major_last = kernel_major[0];
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
        for (Index jjj = 1; jjj < half_kernel+1; jjj++) {
          cumulative_sum += major_last * src[iii][jjj + half_kernel]
                          - major_first * src[iii][0];
          buffer[iii][jjj] = cumulative_sum;
        }

        //  Roll sum (in-bounds components)
        for (Index jjj = half_kernel+1; jjj < src.extent(1)-half_kernel; jjj++) {
          cumulative_sum += major_last * src[iii][jjj + half_kernel]
                          - major_first * src[iii][jjj - half_kernel - 1];
          buffer[iii][jjj] = cumulative_sum;
        }

        // Roll sum (out-of-bounds end components)
        for (Index jjj = src.extent(1)-half_kernel; jjj < src.extent(1); jjj++) {
          cumulative_sum += major_last * src[iii][src.extent(1)-1]
                          - major_first * src[iii][jjj - half_kernel - 1];
          buffer[iii][jjj] = cumulative_sum;
        }
      }

      // Accumulate through minor axis
      Index half_kernel = kernel_minor.size() / 2;
      TReal major_first = kernel_minor[kernel_minor.size()-1];
      TReal major_last = kernel_major[0];
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
        for (Index jjj = 1; jjj < half_kernel+1; jjj++) {
          cumulative_sum += major_last * buffer[jjj + half_kernel][iii]
                          - major_first * buffer[0][iii];
          tar[jjj][iii] = cumulative_sum;
        }

        //  Roll sum (in-bounds components)
        for (Index jjj = half_kernel+1; jjj < buffer.extent(0)-half_kernel; jjj++) {
          cumulative_sum += major_last * buffer[jjj + half_kernel][iii]
                          - major_first * buffer[jjj - half_kernel - 1][iii];
          tar[jjj][iii] = cumulative_sum;
        }

        // Roll sum (out-of-bounds end components)
        for (Index jjj = buffer.extent(0)-half_kernel; jjj < buffer.extent(0); jjj++) {
          cumulative_sum += major_last * buffer[src.extent(0)-1][iii]
                          - major_first * buffer[jjj - half_kernel - 1][iii];
          tar[jjj][iii] = cumulative_sum;
        }
      }
    }

  protected:
    multi_array::Array<Index, 3> filter_shape_;
    Index num_filters_;
    multi_array::Tensor<TReal> state_buffer_1;
    multi_array::Tensor<TReal> state_buffer_2;
    std::vector<multi_array::ConstArraySlice<TReal> > filter_parameters_;
    Index parameters_per_filter_;
    Index stride_;
}

// Network integrator -- uses explicit unweighted structure
template<typename TReal>
class NetIntegrator : public Integrator<TReal> {

  protected:
    // Need network w/ structure here
}

// Reservoir -- uses explicit weighted structure
template<typename TReal>
class ReservoirIntegrator : public Integrator<TReal> {

  protected:
    // Need network w/ structure here
}

} // End nervous_system namespace

#endif /* NN_INTEGRATOR_H_ */