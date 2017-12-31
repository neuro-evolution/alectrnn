/*
 * Current CTRNN activator has independent parameters for each neuron.
 * Could add a dimension member that specifies the # dim of containing layer
 * so that the activator will pick uniform parameters for neurons according
 * to the most minor dimension. This way a Conv layer w/ 64 filters will have
 * 64 unique params instead of 64xWxH. For single dimensional layers it will
 * then just be a user choice at the layer level whether to pick the uniform
 * Activator or independent one.
 */

#ifndef NN_ACTIVATOR_H_
#define NN_ACTIVATOR_H_

#include <vector>
#include <cstddef>
#include <cassert>
#include <limits>
#include "multi_array.hpp"
#include "utilities.hpp"

namespace nervous_system {

enum ACTIVATOR_TYPE {
  BASE,
  IDENTITY,
  CTRNN,
  CONV_CTRNN,
  IAF
};

template<typename TReal>
class Activator {
  public:
    Activator() {
      activator_type_ = ACTIVATOR_TYPE.BASE;
      parameter_count_ = 0;
    }
    virtual ~Activator()=default;

    /*
     * This operator must take the host's state and perform the specified
     * state updates on that state.
     */
    virtual void operator()(multi_array::Tensor<TReal>& state, const multi_array::Tensor<TReal>& input_buffer)=0;

    /*
     * When given a set of parameters, it will use them to set the 
     * Must provide an assert statement that requires # given parameters
     * to == parameter_count
     */
    virtual void Configure(const multi_array::ConstArraySlice<TReal>& parameters)=0; // Just needs to set base_ in ConstArraySlice
    
    /*
     * Some activators may have internal states, these need to be resetable
     */
    virtual void Reset()=0;

    std::size_t GetParameterCount() const {
      return parameter_count_;
    }

    ACTIVATOR_TYPE GetActivatorType() const {
      return activator_type_;
    }

  protected:
    ACTIVATOR_TYPE activator_type_;
    std::size_t parameter_count_;
};

template<typename TReal>
class IdentityActivator : public Activator<TReal> {
  public:
    Identity() {
      parameter_count_ = 0;
      activator_type_ = ACTIVATOR_TYPE.IDENTITY;
    }

    ~Identity()=default;

    void operator()(multi_array::Tensor<TReal>& state, const multi_array::Tensor<TReal>& input_buffer) {
      for (Index iii = 0; iii < state.size(); iii++) {
        state[iii] = input_buffer[iii];
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
    }

    void Reset() {}
};

template<typename TReal>
class CTRNNActivator : public Activator<TReal> {
  public:
    typedef Index std::size_t;
    CTRNNActivator(std::size_t num_states, TReal step_size) : 
        num_states_(num_states), step_size_(step_size) {
      // bias[N] and rtau[N]
      parameter_count_ = num_states * 2;
      activator_type_ = ACTIVATOR_TYPE.CTRNN;
    }

    ~CTRNNActivator()=default;

    void operator()(multi_array::Tensor<TReal>& state, const multi_array::Tensor<TReal>& input_buffer) {
      for (Index iii = 0; iii < num_states_; iii++) {
        state[iii] += step_size_ * rtaus[iii] * 
            (-state[iii] + utilities::sigmoid(biases[iii] + input_buffer[iii]));
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      assert(parameters.size() == parameter_count_);
      biases_ = multi_array::ConstArraySlice<TReal>(parameters.data(), 
                parameters.start(), num_states, parameters.stride());
      rtaus_ = multi_array::ConstArraySlice<TReal>(parameters.data(), 
                parameters.start() + parameters.stride() * num_states, 
                num_states, 
                parameters.stride());
    }

    void Reset() {};

  protected:
    multi_array::ConstArraySlice<TReal> biases;
    multi_array::ConstArraySlice<TReal> rtaus;
    TReal step_size_;
    std::size_t num_states_;
};

/*
 * A uniform version of the CTRNN activator. All neurons in same filter
 * will share parameters. The first index of the layer is the # filters
 */
template<typename TReal>
class Conv3DCTRNNActivator : public Activator<TReal> {
  public:
    typedef Index std::size_t;

    Conv3DCTRNNActivator(const multi_array::Array<Index, 3>& shape, TReal step_size) : 
        step_size_(step_size), shape_(shape) {
      parameter_count_ = shape_[0] * 2;
      activator_type_ = ACTIVATOR_TYPE.CONV_CTRNN;
    }

    ~Conv3DCTRNNActivator()=default;

    void operator()(multi_array::Tensor<TReal>& state, const multi_array::Tensor<TReal>& input_buffer) {
      multi_array::ArrayView<TReal, 3> state_accessor = state.accessor<3>();
      const multi_array::ArrayView<TReal, 3> input_accessor = input_buffer.accessor<3>();

      for (Index filter = 0; filter < shape_[0]; filter++) {
        for (Index iii = 0; iii < shape_[1]; iii++) {
          for (Index jjj = 0; jjj < shape_[2]; jjj++) {
            state_accessor[filter][iii][jjj] += step_size_ 
              * rtaus[filter]
              * (-state_accessor[filter][iii][jjj] 
              + utilities::sigmoid(biases[filter] 
              + input_accessor[filter][iii][jjj]));
          }
        }
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      assert(parameters.size() == parameter_count_);
      biases_ = multi_array::ConstArraySlice<TReal>(parameters.data(), 
                parameters.start(), shape_[0], parameters.stride());
      rtaus_ = multi_array::ConstArraySlice<TReal>(parameters.data(), 
                parameters.start() + parameters.stride() * shape_[0], shape_[0], 
                parameters.stride());
    }

    void Reset() {};

  protected:
    multi_array::ConstArraySlice<TReal> biases;
    multi_array::ConstArraySlice<TReal> rtaus;
    TReal step_size_;
    multi_array::Array<Index, 3> shape_;
};

/*
 * Simulates a spiking neuron. It has its own internal state Tensor
 * in order to mimic the transmission of only super-threshold activity.
 * Sub-threshold activity is recorded in a local tensor, while the Layer
 * state only holds spikes.
 */
// template<typename TReal>
// class IafActivator : public Activator<TReal> {
//   public:
//     typedef Index std::size_t;

//     IafActivator(std::size_t num_states, TReal step_size) : 
//         num_states_(num_states), step_size_(step_size) {
//       // peak, refract, range, rtaus, and reset
//       parameter_count_ = 2 + num_states * 3;
//       activator_type_ = ACTIVATOR_TYPE.IAF;
//       last_spike_time_.resize(num_states_);
//       subthreshold_state_ = multi_array::Tensor<TReal>({num_states_});
//     }

//     ~IafActivator()=default;

//     void operator()(multi_array::Tensor<TReal>& state, const multi_array::Tensor<TReal>& input_buffer) {
      
//     }

//     void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {

//     }

//     void Reset() {
//       for (Index iii = 0; iii < num_states_; iii++) {
//         last_spike_time_[iii] = std::numeric_limits<TReal>::max();
//         subthreshold_state_[iii] = 0;
//       }
//     }

//   protected:
//     multi_array::ConstArraySlice<TReal> range_;
//     multi_array::ConstArraySlice<TReal> rtaus_;
//     multi_array::ConstArraySlice<TReal> reset_;
//     TReal peak_;
//     TReal refractory_period_;
//     TReal step_size_;
//     std::vector<TReal> last_spike_time_;
//     multi_array::Tensor<TReal> subthreshold_state_;
//     std::size_t num_states_;
// };

} // End nervous_system namespace

#endif /* NN_ACTIVATOR_H_ */