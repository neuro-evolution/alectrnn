#ifndef NN_LAYER_H_
#define NN_LAYER_H_

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <initializer_list>
#include <iostream>
#include <utility>
#include "../common/graphs.hpp"
#include "activator.hpp"
#include "integrator.hpp"
#include "../common/multi_array.hpp"
#include "parameter_types.hpp"

namespace nervous_system {

/*
 * Layers take ownership of the integrators and activations functions they use.
 * This is necessary as the Layer will be the only remaining access point to 
 * those objects once it is pushed off to the nervous system.
 */
template<typename TReal>
class Layer {
  public:
    typedef std::size_t Index;
    typedef float Factor;

    Layer() {
      back_integrator_ = nullptr;
      self_integrator_ = nullptr;
      activation_function_ = nullptr;
      parameter_count_ = 0;
    }

    Layer(const std::vector<Index>& shape, 
          Integrator<TReal>* back_integrator, 
          Integrator<TReal>* self_integrator,
          Activator<TReal>* activation_function) 
          : back_integrator_(back_integrator), 
            self_integrator_(self_integrator), 
            activation_function_(activation_function),
            layer_state_(shape), input_buffer_(shape), shape_(shape) {
      parameter_count_ = back_integrator_->GetParameterCount() 
                       + self_integrator_->GetParameterCount()
                       + activation_function_->GetParameterCount();
    }

    virtual ~Layer() {
      delete back_integrator_;
      delete self_integrator_;
      delete activation_function_;
    }

    /*
     * Update neuron state. Calls both integrator and activator.
     */
    virtual void operator()(const Layer<TReal>* prev_layer) {
      // First clear input buffer
      input_buffer_.Fill(0.0);

      // Call back integrator first to resolve input from prev layer
      (*back_integrator_)(prev_layer->state(), input_buffer_);

      // Resolve self-connections if there are any
      (*self_integrator_)(input_buffer_, input_buffer_);

      // Apply activation and update state
      (*activation_function_)(layer_state_, input_buffer_);
    }

    /*
     * Passes the needed parameters to the Layer - Should be Slice with 
     * parameter_count_ in size. Layer will then make and assign Slices to
     * The activation_function and Integrator function
     */
    virtual void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {

      if (parameters.size() != parameter_count_) {
        std::cerr << "parameter size: " << parameters.size() << std::endl;
        std::cerr << "parameter count: " << parameter_count_ << std::endl;
        throw std::invalid_argument("Wrong number of parameters given.");
      }

      back_integrator_->Configure(
        parameters.slice(0, back_integrator_->GetParameterCount()));

      self_integrator_->Configure(
        parameters.slice(parameters.stride() * back_integrator_->GetParameterCount(), 
                         self_integrator_->GetParameterCount()));

      activation_function_->Configure(
        parameters.slice(parameters.stride() * back_integrator_->GetParameterCount() 
                         + parameters.stride() * self_integrator_->GetParameterCount(),
                         activation_function_->GetParameterCount()));
    }

    virtual void Reset() {
      layer_state_.Fill(0.0);
      input_buffer_.Fill(0.0);
      activation_function_->Reset();
    }

    std::size_t GetParameterCount() const {
      return parameter_count_;
    }

    virtual std::vector<PARAMETER_TYPE> GetParameterLayout() const {

      std::vector<PARAMETER_TYPE> layout(parameter_count_);

      // layout produced in configure order: back->self->act
      Index order = 0;
      std::vector<PARAMETER_TYPE> back_layout = back_integrator_->GetParameterLayout();
      for (auto par_type_ptr = back_layout.begin(); 
          par_type_ptr != back_layout.end(); ++par_type_ptr) {
        layout[order] = *par_type_ptr;
        ++order;
      }
      std::vector<PARAMETER_TYPE> self_layout = self_integrator_->GetParameterLayout();
      for (auto par_type_ptr = self_layout.begin(); 
          par_type_ptr != self_layout.end(); ++par_type_ptr) {
        layout[order] = *par_type_ptr;
        ++order;
      }
      std::vector<PARAMETER_TYPE> act_layout = activation_function_->GetParameterLayout();
      for (auto par_type_ptr = act_layout.begin(); 
          par_type_ptr != act_layout.end(); ++par_type_ptr) {
        layout[order] = *par_type_ptr;
        ++order;
      }

      return layout;
    }

    /*
     * Constructs a normalization factor vector. Each element corresponds to
     * a parameter. Parameters associated with weights should be set to a
     * non-zero value == to the degree of the post-synaptic neuron.
     */
    virtual std::vector<Factor> GetWeightNormalizationFactors() const {
      std::vector<Factor> normalization_factors(parameter_count_);
      // initializes values to 0
      for (auto& factor : normalization_factors) {
        factor = 0.0;
      }

      EvaluateNormalizationFactors(normalization_factors, back_integrator_,
                                   self_integrator_, NumNeurons());

      return normalization_factors;
    }

    std::size_t NumNeurons() const {
      return layer_state_.size();
    }

    template<typename T>
    void SetNeuronState(Index neuron, T value) {
      layer_state_[neuron] = value;
    }

    const multi_array::Tensor<TReal>& state() const {
      return layer_state_;
    }

    multi_array::Tensor<TReal>& state() {
      return layer_state_;
    }

    const std::vector<Index>& shape() const {
      return shape_;
    }

    const Integrator<TReal>* GetBackIntegrator() const {
      return back_integrator_;
    }

    const Integrator<TReal>* GetSelfIntegrator() const {
      return self_integrator_;
    }

  protected:
    // calculates inputs from other layers and applies them to input buffer
    Integrator<TReal>* back_integrator_;
    // claculates inputs from neurons within the layer and applies them to input buffer
    Integrator<TReal>* self_integrator_;
    // updates the layer's state using the input buffer (may also contain internal state)
    Activator<TReal>* activation_function_;
    // maintains the current layer's state
    multi_array::Tensor<TReal> layer_state_;
    // holds input values used to update the layer's state
    multi_array::Tensor<TReal> input_buffer_;
    std::vector<Index> shape_;
    // Number of parameters required by layer
    std::size_t parameter_count_;
};

template <typename TReal>
class RewardModulatedLayer : public Layer<TReal> {
  public:
    typedef Layer<TReal> super_type;
    typedef typename super_type::Index Index;

    RewardModulatedLayer(const std::vector<Index>& shape,
                         RewardModulatedIntegrator<TReal>* back_integrator,
                         RewardModulatedIntegrator<TReal>* self_integrator,
                         Activator<TReal>* activation_function)
        : super_type(shape, back_integrator, self_integrator,
                     activation_function),
          reward_average_(0.0),
          activation_averages_(super_type::input_buffer_.size())  {
      super_type::parameter_count_ += 2; // reward and activation smoothing factors
      Reset();
    }

    virtual void Configure(const multi_array::ConstArraySlice<TReal>& parameters) override {
      super_type::Configure(parameters);
      reward_smoothing_factor_ = utilities::Wrap0to1(parameters[parameters.size()-2]);
      activation_smoothing_factor_ = utilities::Wrap0to1(parameters[parameters.size()-1]);
    }

    virtual std::vector<PARAMETER_TYPE> GetParameterLayout() const override {

      std::vector<PARAMETER_TYPE> layout = super_type::GetParameterLayout();
      layout.push_back(SMOOTHING);
      layout.push_back(SMOOTHING);
      return layout;
    }

    virtual void Reset() override {
      super_type::Reset();
      for (auto& avg : activation_averages_) {
        avg = 0;
      }
      reward_average_ = 0.0;
    }

    /*
     * Called after all integrators and activators have been called.
     */
    UpdateWeights(const TReal reward, const Layer<TReal>* prev_layer) {
      // call weight update function
      dynamic_cast<RewardModulatedIntegrator<TReal>*>(super_type::back_integrator_)->UpdateWeights(
          reward, reward_average_, prev_layer->state(), super_type::input_buffer_, activation_averages_);
      dynamic_cast<RewardModulatedIntegrator<TReal>*>(super_type::self_integrator_)->UpdateWeights(
          reward, reward_average_, prev_layer->state(), super_type::input_buffer_, activation_averages_);

      // update rolling avgerages
      reward_average_ = utilities::ExponentialRollingAverage(reward, reward_average_, reward_smoothing_factor_);
      for (Index i = 0; i < activation_averages_.size(); ++i) {
        activation_averages_[i] = utilities::ExponentialRollingAverage(super_type::input_buffer_[i],
                                                                       activation_averages_[i],
                                                                       activation_smoothing_factor_);
      }
    }

  protected:
    multi_array::Tensor<TReal> activation_averages_;
    TReal reward_average_;
    TReal reward_smoothing_factor_; // between [0,1]
    TReal activation_smoothing_factor_; // between [0,1]
};

template<typename TReal>
class InputLayer : public Layer<TReal> {
  public:
    typedef Layer<TReal> super_type;
    typedef typename super_type::Index Index;
    typedef typename super_type::Factor Factor;

    InputLayer() : super_type() {
    }

    InputLayer(const std::vector<Index>& shape) {
      super_type::shape_ = shape;
      super_type::layer_state_ = shape;
      super_type::back_integrator_ = nullptr;
      super_type::self_integrator_ = nullptr;
      super_type::activation_function_ = nullptr;
      super_type::parameter_count_ = 0;
    }

    InputLayer(const std::initializer_list<Index>& shape) {
      super_type::shape_ = shape;
      super_type::layer_state_ = shape;
      super_type::back_integrator_ = nullptr;
      super_type::self_integrator_ = nullptr;
      super_type::activation_function_ = nullptr;
      super_type::parameter_count_ = 0;
    }

    virtual void operator()(const Layer<TReal>* prev_layer) {}

    virtual void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {}

    virtual void Reset() { super_type::layer_state_.Fill(0.0); }

    virtual std::vector<PARAMETER_TYPE> GetParameterLayout() const {
      return std::vector<PARAMETER_TYPE>(0);
    }
};

template<typename TReal>
class MotorLayer : public Layer<TReal> {
  public:
    typedef Layer<TReal> super_type;
    typedef typename super_type::Index Index;
    typedef typename super_type::Factor Factor;

    MotorLayer() : super_type() {
    }

    MotorLayer(Index num_outputs, Index num_inputs, 
        Activator<TReal>* activation_function) {
      super_type::activation_function_ = activation_function;
      super_type::back_integrator_ = new nervous_system::All2AllIntegrator<TReal>(num_outputs, num_inputs);
      super_type::self_integrator_ = nullptr;
      super_type::parameter_count_ = super_type::activation_function_->GetParameterCount() 
                                   + super_type::back_integrator_->GetParameterCount();
      super_type::layer_state_ = multi_array::Tensor<TReal>({num_outputs});
      super_type::input_buffer_ = multi_array::Tensor<TReal>({num_outputs});
    }

    virtual ~MotorLayer() {};

    virtual void operator()(const Layer<TReal>* prev_layer) {
      // First clear input buffer
      super_type::input_buffer_.Fill(0.0);
      // Call back integrator first to resolve input from prev layer
      (*super_type::back_integrator_)(prev_layer->state(), super_type::input_buffer_);
      // Apply activation and update state
      (*super_type::activation_function_)(super_type::layer_state_, super_type::input_buffer_);
    }

    virtual void Configure(const multi_array::ConstArraySlice<TReal>& parameters) {
      if (super_type::parameter_count_ != parameters.size()) {
        std::cerr << "parameter size: " << parameters.size() << std::endl;
        std::cerr << "parameter count: " << super_type::parameter_count_ << std::endl;
        throw std::invalid_argument("Wrong number of parameters given");
      }
      // configure back integrator parameters
      super_type::back_integrator_->Configure(
        parameters.slice(0, super_type::back_integrator_->GetParameterCount()));
      // configure activation parameters
      super_type::activation_function_->Configure(
        parameters.slice(parameters.stride() * super_type::back_integrator_->GetParameterCount(),
                         super_type::activation_function_->GetParameterCount()));
    }

    virtual std::vector<PARAMETER_TYPE> GetParameterLayout() const {
      std::vector<PARAMETER_TYPE> layout(super_type::parameter_count_);

      // layout produced in configure order: back->act
      Index order = 0;
      std::vector<PARAMETER_TYPE> back_layout = super_type::back_integrator_->GetParameterLayout();
      for (auto par_type_ptr = back_layout.begin(); 
          par_type_ptr != back_layout.end(); ++par_type_ptr) {
        layout[order] = *par_type_ptr;
        ++order;
      }

      std::vector<PARAMETER_TYPE> act_layout = super_type::activation_function_->GetParameterLayout();
      for (auto par_type_ptr = act_layout.begin(); 
          par_type_ptr != act_layout.end(); ++par_type_ptr) {
        layout[order] = *par_type_ptr;
        ++order;
      }

      return layout;
    }
};

/*
 * A motor layer that has no internal memory, and whose outputs represents
 * a distribution, so that all state members sum to 1.
 */
template<typename TReal>
class SoftMaxMotorLayer : public MotorLayer<TReal> {
  public:
    typedef MotorLayer<TReal> super_type;
    typedef typename super_type::Index Index;
    typedef typename super_type::Factor Factor;

    SoftMaxMotorLayer() : super_type() {
    }

    SoftMaxMotorLayer(Index num_outputs, Index num_inputs, TReal temperature) {
      super_type::activation_function_ = new nervous_system::SoftMaxActivator<TReal>(temperature);
      super_type::back_integrator_ = new nervous_system::All2AllIntegrator<TReal>(num_outputs, num_inputs);
      super_type::self_integrator_ = nullptr;
      super_type::parameter_count_ = super_type::activation_function_->GetParameterCount()
                                     + super_type::back_integrator_->GetParameterCount();
      super_type::layer_state_ = multi_array::Tensor<TReal>({num_outputs});
      super_type::input_buffer_ = multi_array::Tensor<TReal>({num_outputs});
    }
};

/*
 * A motor layer that uses the Eigen A2A integrator.
 */
template<typename TReal>
class EigenMotorLayer : public MotorLayer<TReal> {
  public:
    typedef MotorLayer<TReal> super_type;
    typedef typename super_type::Index Index;
    typedef typename super_type::Factor Factor;

    EigenMotorLayer() : super_type() {
    }

    EigenMotorLayer(Index num_outputs, Index num_inputs,
                    Activator<TReal>* activation_function) {
      super_type::activation_function_ = activation_function;
      super_type::back_integrator_ = new nervous_system::All2AllEigenIntegrator<TReal>(num_outputs,
                                                                                       num_inputs);
      super_type::self_integrator_ = nullptr;
      super_type::parameter_count_ = super_type::activation_function_->GetParameterCount()
                                     + super_type::back_integrator_->GetParameterCount();
      super_type::layer_state_ = multi_array::Tensor<TReal>({num_outputs});
      super_type::input_buffer_ = multi_array::Tensor<TReal>({num_outputs});
    }
};

/*
 * Calculates the number of links the integrator has
 */
template <typename Factor, typename TReal>
Factor GetNumOfLinks(const Integrator<TReal>* integrator) {

  if (integrator == nullptr) {
    return 0; // If there is no integrator (which is valid) it has no links.
  }

  INTEGRATOR_TYPE integrator_type = integrator->GetIntegratorType();
  Factor num_links = 0;
  switch (integrator_type) {
    case ALL2ALL_INTEGRATOR: {
      num_links = integrator->GetParameterCount();
      break;
    }

    case NONE_INTEGRATOR: {
      break;
    }

    case CONV_INTEGRATOR: {
      const Conv2DIntegrator<TReal> *conv_integrator = dynamic_cast<const Conv2DIntegrator<TReal> *>(integrator);
      const multi_array::Array<std::size_t, 3> filter = conv_integrator->GetFilterShape();
      num_links = filter[0] * filter[1] * filter[2] * conv_integrator->GetMinTarSize();
      break;
    }

    case TRUNCATED_RECURRENT_INTEGRATOR:
    case RECURRENT_INTEGRATOR: {
      const RecurrentIntegrator<TReal> *recurrent_integrator = dynamic_cast<const RecurrentIntegrator<TReal> *>(integrator);
      const graphs::PredecessorGraph<> graph = recurrent_integrator->GetGraph();
      num_links = graph.NumEdges();
      break;
    }

    case RESERVOIR_INTEGRATOR: {
      const ReservoirIntegrator<TReal> *reservoir_integrator = dynamic_cast<const ReservoirIntegrator<TReal> *>(integrator);
      const graphs::PredecessorGraph<TReal> graph = reservoir_integrator->GetGraph();
      num_links = graph.NumEdges();
      break;
    }

    default:
      throw std::invalid_argument("Error: Integrator type not found.");
  }

  return num_links;
};

/*
 * Calculates the average degree by combining information from the back and
 * self integrators, and then assigns the values to the respective links
 */
template <typename Factor, typename TReal>
void EvaluateNormalizationFactors(std::vector<Factor>& normalization_factors,
                                  const Integrator<TReal>* back_integrator,
                                  const Integrator<TReal>* self_integrator,
                                  std::size_t layer_size) {

  std::size_t num_back_links = 0;
  std::pair<std::size_t, std::size_t> back_weight_slice{0,0};
  if (back_integrator != nullptr) {
    num_back_links = GetNumOfLinks<std::size_t, TReal>(back_integrator);
    back_weight_slice = back_integrator->GetWeightIndexRange();
  }

  std::size_t num_self_links = 0;
  std::pair<std::size_t, std::size_t> self_weight_slice{0,0};
  if (self_integrator != nullptr) {
    num_self_links = GetNumOfLinks<std::size_t, TReal>(self_integrator);
    self_weight_slice = self_integrator->GetWeightIndexRange();
  }

  // Because self params are added after back params, we have to offset the weight indices for self
  if (back_integrator != nullptr) {
    self_weight_slice.first += back_integrator->GetParameterCount();
    self_weight_slice.second += back_integrator->GetParameterCount();
  }

  // Calculate average degree and the assign it to the weight indices of self and back
  Factor average_degree = (num_back_links + num_self_links) / static_cast<double>(layer_size);
  for (std::size_t iii = 0; iii < normalization_factors.size(); ++iii) {
    if (((iii >= back_weight_slice.first) && (iii < back_weight_slice.second)) ||
       ((iii >= self_weight_slice.first) && (iii < self_weight_slice.second))) {
      normalization_factors[iii] = average_degree;
    }
  }
};

} // End nervous_system namespace

#endif /* NN_LAYER_H_ */