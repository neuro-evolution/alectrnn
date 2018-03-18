/*
 * Current CTRNN_ACTIVATOR activator has independent parameters for each neuron.
 * Could add a dimension member that specifies the # dim of containing layer
 * so that the activator will pick uniform parameters for neurons according
 * to the most minor dimension. This way a Conv layer w/ 64 filters will have
 * 64 unique params instead of 64xWxH. For single dimensional layers it will
 * then just be a user choice at the layer level whether to pick the uniform
 * Activator or independent one.
 *
 * CONV activators should take a shape argument
 * Non-conv activators should take a # states argument
 */

#ifndef NN_ACTIVATOR_H_
#define NN_ACTIVATOR_H_

#include <vector>
#include <cstddef>
#include <cassert>
#include <limits>
#include "multi_array.hpp"
#include "utilities.hpp"
#include "parameter_types.hpp"

namespace nervous_system {

enum ACTIVATOR_TYPE {
  BASE_ACTIVATOR,
  IDENTITY_ACTIVATOR,
  CTRNN_ACTIVATOR,
  CONV_CTRNN_ACTIVATOR,
  IAF_ACTIVATOR
};

template<typename TReal>
class Activator {
  public:
    Activator() {
      activator_type_ = BASE_ACTIVATOR;
      parameter_count_ = 0;
    }
    virtual ~Activator()=default;

    /*
     * This operator must take the host's state and perform the specified
     * state updates on that state.
     */
    virtual void operator()(multi_array::Tensor<TReal>& state, 
                            const multi_array::Tensor<TReal>& input_buffer)=0;

    /*
     * When given a set of parameters, it will use them to set the 
     * Must provide an assert statement that requires # given parameters
     * to == parameter_count
     */
    virtual void Configure(const multi_array::ConstArraySlice<TReal>& parameters)=0;
    
    /*
     * Some activators may have internal states, these need to be resetable
     */
    virtual void Reset()=0;

    std::size_t GetParameterCount() const {
      return parameter_count_;
    }

    /*
     * Should return a vector with parameter_types ordered according to
     * the order they are expected in the parameter slice.
     * If no parameters are used, the vector should be empty
     */
    virtual std::vector<PARAMETER_TYPE> GetParameterLayout() const=0;

    ACTIVATOR_TYPE GetActivatorType() const {
      return activator_type_;
    }

  protected:
    ACTIVATOR_TYPE activator_type_;
    std::size_t parameter_count_;
};

template<typename TReal>
class IdentityActivator : public Activator<TReal> {
  typedef Activator<TReal> super_type;
  typedef std::size_t Index;
  public:
    IdentityActivator() {
      super_type::parameter_count_ = 0;
      super_type::activator_type_ = IDENTITY_ACTIVATOR;
    }

    ~IdentityActivator()=default;

    void operator()(multi_array::Tensor<TReal>& state, 
                    const multi_array::Tensor<TReal>& input_buffer) {
      for (Index iii = 0; iii < state.size(); iii++) {
        state[iii] = input_buffer[iii];
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
    }

    std::vector<PARAMETER_TYPE> GetParameterLayout() const {
      return std::vector<PARAMETER_TYPE>(0);
    }

    void Reset() {}
};

template<typename TReal>
class CTRNNActivator : public Activator<TReal> {
  typedef Activator<TReal> super_type;
  typedef std::size_t Index;
  public:
    CTRNNActivator(std::size_t num_states, TReal step_size) : 
        num_states_(num_states), step_size_(step_size) {
      // bias[N] and rtau[N]
      super_type::parameter_count_ = num_states * 2;
      super_type::activator_type_ = CTRNN_ACTIVATOR;
    }

    ~CTRNNActivator()=default;

    void operator()(multi_array::Tensor<TReal>& state, 
                    const multi_array::Tensor<TReal>& input_buffer) {
      // Loop through all the neurons and apply the CTRNN update equation
      for (Index iii = 0; iii < num_states_; iii++) {
        state[iii] += step_size_ * rtaus_[iii] * (-state[iii] +
            utilities::sigmoid(biases_[iii] + input_buffer[iii]));
        state[iii] = utilities::BoundState(state[iii]);
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      assert(parameters.size() == super_type::parameter_count_);
      biases_ = parameters.slice(0, num_states_);
      rtaus_ = parameters.slice(parameters.stride() * num_states_, num_states_);
    }

    std::vector<PARAMETER_TYPE> GetParameterLayout() const {
      std::vector<PARAMETER_TYPE> layout(super_type::parameter_count_);
      for (Index iii = 0; iii < num_states_; ++iii) {
        layout[iii] = BIAS;
      }
      for (Index iii = num_states_; iii < super_type::parameter_count_; ++iii) {
        layout[iii] = RTAUS;
      }
      return layout;
    }

    void Reset() {};

  protected:
    multi_array::ConstArraySlice<TReal> biases_;
    multi_array::ConstArraySlice<TReal> rtaus_;
    std::size_t num_states_;
    TReal step_size_;
};

/*
 * A uniform version of the CTRNN_ACTIVATOR activator. All neurons in same filter
 * will share parameters. The first index of the layer is the # filters
 */
template<typename TReal>
class Conv3DCTRNNActivator : public Activator<TReal> {
  typedef Activator<TReal> super_type;
  typedef std::size_t Index;
  public:
    Conv3DCTRNNActivator(const multi_array::Array<Index, 3>& shape, TReal step_size) : 
        step_size_(step_size), shape_(shape) {
      super_type::parameter_count_ = shape_[0] * 2;
      super_type::activator_type_ = CONV_CTRNN_ACTIVATOR;
    }

    ~Conv3DCTRNNActivator()=default;

    void operator()(multi_array::Tensor<TReal>& state, const multi_array::Tensor<TReal>& input_buffer) {
      multi_array::TensorView<TReal> state_accessor = state.accessor();
      const multi_array::TensorView<TReal> input_accessor = input_buffer.accessor();

      for (Index filter = 0; filter < shape_[0]; filter++) {
        for (Index iii = 0; iii < shape_[1]; iii++) {
          for (Index jjj = 0; jjj < shape_[2]; jjj++) {
            state_accessor[filter][iii][jjj] += step_size_
              * rtaus_[filter]
              * (-state_accessor[filter][iii][jjj] 
              + utilities::sigmoid(biases_[filter] 
              + input_accessor[filter][iii][jjj]));
            state_accessor[filter][iii][jjj] = utilities::BoundState<TReal>(
                state_accessor[filter][iii][jjj]);
          }
        }
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      assert(parameters.size() == super_type::parameter_count_);
      biases_ = parameters.slice(0, shape_[0]);
      rtaus_ = parameters.slice(parameters.stride() * shape_[0], shape_[0]);
    }

    std::vector<PARAMETER_TYPE> GetParameterLayout() const {
      std::vector<PARAMETER_TYPE> layout(super_type::parameter_count_);
      for (Index iii = 0; iii < shape_[0]; ++iii) {
        layout[iii] = BIAS;
      }
      for (Index iii = shape_[0]; iii < super_type::parameter_count_; ++iii) {
        layout[iii] = RTAUS;
      }
      return layout;
    }

    void Reset() {};

  protected:
    multi_array::ConstArraySlice<TReal> biases_;
    multi_array::ConstArraySlice<TReal> rtaus_;
    TReal step_size_;
    multi_array::Array<Index, 3> shape_;
};

/*
 * Simulates a spiking neuron. It has its own internal state Tensor
 * in order to mimic the transmission of only super-threshold activity.
 * Sub-threshold activity is recorded in a local tensor, while the Layer
 * state only holds spikes.
 */
//template<typename TReal>
//class IafActivator : public Activator<TReal> {
//    typedef Activator<TReal> super_type;
//    typedef Index std::size_t;
//  public:
//
//    IafActivator(std::size_t num_states, TReal step_size) :
//        num_states_(num_states), step_size_(step_size) {
//      // peak, refract, range, rtaus_, and reset
//      parameter_count_ = 2 + num_states * 3;
//      activator_type_ = IAF_ACTIVATOR;
//      last_spike_time_.resize(num_states_);
//      subthreshold_state_ = multi_array::Tensor<TReal>({num_states_});
//    }
//
//    ~IafActivator()=default;
//
//    void operator()(multi_array::Tensor<TReal>& state,
//                    const multi_array::Tensor<TReal>& input_buffer) {
//      utilities::BoundState
//    }
//
//    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
//
//    }
//
//    void Reset() {
//      for (Index iii = 0; iii < num_states_; iii++) {
//        last_spike_time_[iii] = std::numeric_limits<TReal>::max();
//        subthreshold_state_[iii] = 0;
//      }
//    }
//
//  protected:
//    /* The threshold is range_ + reset_ */
//    multi_array::ConstArraySlice<TReal> range_;
//    /* rtaus_ is the inverse of the time-constant */
//    multi_array::ConstArraySlice<TReal> rtaus_;
//    /* The resting state voltage is reset_ */
//    multi_array::ConstArraySlice<TReal> reset_;
//    /* The action potential voltage is peak_ + range_ + reset_ */
//    multi_array::ConstArraySlice<TReal> peak_;
//    /* The neuron can spike again after refractory_period_ * (1 / step_size_) steps */
//    multi_array::ConstArraySlice<TReal> refractory_period_; //uh oh... need int, but only got real
//    TReal step_size_;
//
//    std::vector<TReal> refractory_steps_;
//    multi_array::Tensor<TReal> subthreshold_state_;
//    std::size_t num_states_;
//};

} // End nervous_system namespace

#endif /* NN_ACTIVATOR_H_ */