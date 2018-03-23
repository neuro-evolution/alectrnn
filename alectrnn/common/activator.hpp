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
#include <stdexcept>
#include <limits>
#include <iostream>
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
     * Must provide an throw statement that requires # given parameters
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
      if (!((state.size() == num_states_) && (input_buffer.size() == num_states_))) {
        std::cerr << "state size: " << state.size() << std::endl;
        std::cerr << "input size: " << input_buffer.size() << std::endl;
        std::cerr << "activator size: " << num_states_ << std::endl;
        throw std::invalid_argument("Incompatible states. State and input"
                                    " must be the same size as activator");
      }
      // Loop through all the neurons and apply the CTRNN update equation
      for (Index iii = 0; iii < num_states_; iii++) {
        state[iii] += step_size_ * rtaus_[iii] * (-state[iii] +
            utilities::sigmoid(biases_[iii] + input_buffer[iii]));
        state[iii] = utilities::BoundState(state[iii]);
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      if (parameters.size() != super_type::parameter_count_) {
        std::cerr << "parameter size: " << parameters.size() << std::endl;
        std::cerr << "parameter count: " << super_type::parameter_count_ << std::endl;
        throw std::invalid_argument("Number of parameters must equal parameter"
                                    " count");
      }
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
 * NOTE: WE should change this so that it takes a function so we don't have
 * to write this over and over again for each activator
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

      if (!((state.shape() == shape_) && (input_buffer.shape() == shape_))) {
        std::cerr << "Activator incompatible with state: " <<
                  (state.shape() != shape_) << std::endl;
        std::cerr << "Activator incompatible with input: " <<
                  (input_buffer.shape() != shape_) << std::endl;
        throw std::invalid_argument("ConvCTRNN State, input, and activator must"
                                    " all be the same shape");
      }

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
      if (parameters.size() != super_type::parameter_count_) {
        std::cerr << "parameter size: " << parameters.size() << std::endl;
        std::cerr << "parameter count: " << super_type::parameter_count_ << std::endl;
        throw std::invalid_argument("Parameter array must have the same"
                                    " number of elements as the count");
      }
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
template<typename TReal>
class IafActivator : public Activator<TReal> {
    typedef Activator<TReal> super_type;
    typedef std::size_t Index;
  public:

    IafActivator(std::size_t num_states, TReal step_size, TReal peak, TReal reset) :
        num_states_(num_states), step_size_(step_size), peak_(peak), reset_(reset) {
      // refract, range, rtaus_, and resistance
      super_type::parameter_count_ = num_states * 4;
      super_type::activator_type_ = IAF_ACTIVATOR;

      last_spike_time_.resize(num_states_);
      subthreshold_state_ = multi_array::Tensor<TReal>({num_states_});
      alpha_.resize(num_states_);
      vthresh_.resize(num_states_);
      Reset();
    }

    ~IafActivator()=default;

    void operator()(multi_array::Tensor<TReal>& state,
                    const multi_array::Tensor<TReal>& input_buffer) {
      if (!((state.size() == num_states_) && (input_buffer.size() == num_states_))) {
        std::cerr << "state size: " << state.size() << std::endl;
        std::cerr << "input size: " << input_buffer.size() << std::endl;
        std::cerr << "activator size: " << num_states_ << std::endl;
        throw std::invalid_argument("Incompatible states. State and input"
                                    " must be the same size as activator");
      }

      // update subthreshold state (if not in refractory period)
      for (Index iii = 0; iii < num_states_; ++iii) {
        // update refractory state:
        // increment the time since last spike by the simulation step_size
        last_spike_time_[iii] += step_size_;

        // if a spike can occur unclamp state and check for action potential
        if (last_spike_time_[iii] > refractory_period_[iii]) {
          // evaluates the equation: -rtaus * dT * (u - u_reset) + R * I)
          subthreshold_state_[iii] += alpha_[iii] * (subthreshold_state_[iii] - reset_)
                                      + resistance_[iii] * input_buffer[iii];
          subthreshold_state_[iii] = utilities::BoundState<TReal>(subthreshold_state_[iii]);

          /* check for action potential and reset last spike time if a
           * spike occurred. Also reset the membrane potential */
          if (subthreshold_state_[iii] > vthresh_[iii]) {
            state[iii] = peak_;
            subthreshold_state_[iii] = reset_;
            last_spike_time_[iii] = 0.0;
          }
          else {
            state[iii] = 0.0;
          }
        }
        else {
          state[iii] = 0.0;
        }
      }
    }

    void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      if (parameters.size() != super_type::parameter_count_) {
        std::cerr << "parameter size: " << parameters.size() << std::endl;
        std::cerr << "parameter count: " << super_type::parameter_count_ << std::endl;
        throw std::invalid_argument("Number of parameters must equal parameter"
                                    " count");
      }

      range_ = parameters.slice(0, num_states_);
      rtaus_ = parameters.slice(parameters.stride() * num_states_, num_states_);
      refractory_period_ = parameters.slice(2 * parameters.stride() * num_states_, num_states_);
      resistance_ = parameters.slice(3 * parameters.stride() * num_states_, num_states_);

      Reset();
      // Need to update the precomputed parameters
      for (Index iii = 0; iii < num_states_; ++iii) {
        alpha_[iii] = -step_size_ * rtaus_[iii];
        vthresh_[iii] = reset_ + range_[iii];
      }
    }

    std::vector<PARAMETER_TYPE> GetParameterLayout() const {
      std::vector<PARAMETER_TYPE> layout(super_type::parameter_count_);
      for (Index iii = 0; iii < num_states_; ++iii) {
        layout[iii] = RANGE;
      }
      for (Index iii = num_states_; iii < 2*num_states_; ++iii) {
        layout[iii] = RTAUS;
      }
      for (Index iii = 2*num_states_; iii < 3*num_states_; ++iii) {
        layout[iii] = REFRACTORY;
      }
      for (Index iii = 3*num_states_; iii < 4*num_states_; ++iii) {
        layout[iii] = RESISTANCE;
      }
      return layout;
    }

    void Reset() {
      for (Index iii = 0; iii < num_states_; ++iii) {
        last_spike_time_[iii] = std::numeric_limits<TReal>::max();
        subthreshold_state_[iii] = 0.0;
      }
    }

  protected:
    std::size_t num_states_;
    /* The threshold is range_ + reset_ */
    multi_array::ConstArraySlice<TReal> range_;
    /* rtaus_ is the inverse of the time-constant */
    multi_array::ConstArraySlice<TReal> rtaus_;
    /* The neuron can spike again after refractory_period_ / step_size_ steps
     * will allow spiking asap based on step-size */
    multi_array::ConstArraySlice<TReal> refractory_period_;
    /* The input resistance of the neuron */
    multi_array::ConstArraySlice<TReal> resistance_;
    TReal step_size_;
    TReal peak_;
    /* The resting state voltage is reset_ */
    TReal reset_;
    std::vector<TReal> last_spike_time_;
    // alpha == the pre-computed product of rtaus and step_size_ and -1
    std::vector<TReal> alpha_;
    // vthresh == the pre-computed reset threshold (reset_ + range_)
    std::vector<TReal> vthresh_;
    multi_array::Tensor<TReal> subthreshold_state_;
};

} // End nervous_system namespace

#endif /* NN_ACTIVATOR_H_ */