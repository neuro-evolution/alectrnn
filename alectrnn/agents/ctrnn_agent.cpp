/*
 * ctrnn_agent.cpp
 *  Created on: Sep 1, 2017
 *      Author: Nathaniel Rodriguez
 *
 * The CTRNN agent class is a specific PlayerAgent that creates a neural
 * network and returns actions based on the provided visual stimuli from
 * the ALE environment. This agent only needs to be created once, it has
 * a Configure function that can take parameters to reconfigure the
 * neural network weights.
 *
 */

#include <limits>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <memory>
#include <ale_interface.hpp>
#include "ctrnn_agent.h"
#include "player_agent.h"
#include "../common/network_generator.h"
#include "../common/screen_preprocessing.h"
#include "../common/nervous_system.h"
#include <iostream> ///test

namespace alectrnn {

CtrnnAgent::CtrnnAgent(ALEInterface* ale,
      std::size_t num_neurons, std::size_t num_sensor_neurons,
      std::size_t input_screen_width,
      std::size_t input_screen_height, bool use_color, float step_size,
      std::size_t update_rate)
    : PlayerAgent(ale), num_neurons_(num_neurons),
      num_sensor_neurons_(num_sensor_neurons), use_color_(use_color),
      input_screen_width_(input_screen_width),
      input_screen_height_(input_screen_height), is_configured_(false),
      update_rate_(update_rate) {
  // Reserve space for input screens
  if (!use_color_) {
    num_sensors_ = input_screen_width_ * input_screen_height_;
    full_screen_.resize(
        ale_->environment->getScreenHeight() *
        ale_->environment->getScreenWidth());
    buffer_screen_.resize(
        ale_->environment->getScreenHeight() *
        ale_->environment->getScreenWidth());
    downsized_screen_.resize(num_sensors_);
  }
  else {
    //r,g,b + luminance
    num_sensors_ = input_screen_width_ * input_screen_height_ * 4;
    full_screen_.resize(
        ale_->environment->getScreenHeight() *
        ale_->environment->getScreenWidth() * 4);
    buffer_screen_.resize(
        ale_->environment->getScreenHeight() *
        ale_->environment->getScreenWidth() * 4);
    downsized_screen_.resize(num_sensors_);
  }

  node_sensors_ = ctrnn::FullSensorNetwork(num_sensors_, num_sensor_neurons_);
  node_neighbors_ = ctrnn::All2AllNetwork(num_neurons_);
  agent_neural_system_ = std::make_unique<ctrnn::NeuralNetwork>(node_neighbors_,
      node_sensors_, num_sensors_, step_size);

  if (available_actions_.size() > num_neurons_)
  {
    throw std::invalid_argument( "To few neurons for # game outputs." );
  }
}

CtrnnAgent::CtrnnAgent(ALEInterface* ale, std::size_t num_neurons,
        std::size_t num_sensor_neurons, std::size_t input_screen_width,
        std::size_t input_screen_height, bool use_color, float step_size) 
          : CtrnnAgent(ale, num_neurons, num_sensor_neurons, 
            input_screen_width, input_screen_height, 
            use_color, step_size, 1) {
}

CtrnnAgent::~CtrnnAgent() {
}

void CtrnnAgent::Configure(const float *parameters) {
  /*
   * Parameters need to be decoded and the nervous system configured
   * We will assume we are getting valid parameters.
   * Note: assumes parameters is a contiguous c-style array
   *
   * TO DO: Need a size check, make sure parameters array size == # parameters
   */

  // configure bias
  const float *bias_parameters = parameters;
  for (std::size_t iii = 0; iii < num_neurons_; iii++) {
    agent_neural_system_->setNeuronBias(iii, bias_parameters[iii]);
  }

  // configure tau
  const float *tau_parameters = bias_parameters + num_neurons_;
  for (std::size_t iii = 0; iii < num_neurons_; iii++) {
    agent_neural_system_->setNeuronTau(iii, tau_parameters[iii]);
  }

  // configure gain
  const float *gain_parameters = tau_parameters + num_neurons_;
  for (std::size_t iii = 0; iii < num_neurons_; iii++) {
    agent_neural_system_->setNeuronGain(iii, gain_parameters[iii]);
  }

  // configure weights
  const float *circuit_parameters = gain_parameters + num_neurons_;
  ctrnn::FillAll2AllNetwork(node_neighbors_, circuit_parameters);
  const float *sensor_parameters = circuit_parameters + (num_neurons_*num_neurons_);
  ctrnn::FillFullSensorNetwork(node_sensors_, sensor_parameters);

  is_configured_ = true;
}

void CtrnnAgent::Reset() {
  PlayerAgent::Reset();
  agent_neural_system_->Reset();
}

Action CtrnnAgent::Act() {
  // Need to get the screen
  if (!use_color_) {
    ale_->getScreenGrayscale(full_screen_);
    // Need to downsize the screen
    ResizeGrayScreen(ale_->environment->getScreenWidth(),
                     ale_->environment->getScreenHeight(),
                     input_screen_width_,
                     input_screen_height_,
                     full_screen_,
                     downsized_screen_,
                     buffer_screen_);
  }
  else {
    /*
     * Not currently implemented
     */
  }

  // Update sensor state (cast uint8_t into float)
  for (std::size_t iii = 0; iii < num_sensors_; iii++) {
    agent_neural_system_->setSensorState(iii, (float)downsized_screen_[iii]);
  }
  // The neural network will be updates update_rate_ times before output is read
  std::cout << update_rate_ << std::endl;/////////test
  for (std::size_t iii = 0; iii < update_rate_; iii++) {
    std::cout << iii << " " << update_rate_ << std::endl;/////////test
    agent_neural_system_->EulerStep();
  }

  // Read values from last X neurons, X==LastNeuronIndex - Action#
  Action prefered_action(PLAYER_A_NOOP);
  float prefered_output(std::numeric_limits<float>::lowest());
  Action last_action(PLAYER_A_NOOP);
  float last_output(std::numeric_limits<float>::lowest());
  for (std::size_t iii = 0; iii < available_actions_.size(); iii++) {
    last_output = agent_neural_system_->getNeuronOutput(num_neurons_ - 1 -
                                          (std::size_t)available_actions_[iii]);
    last_action = available_actions_[iii];

    if (prefered_output < last_output) {
      prefered_output = last_output;
      prefered_action = last_action;
    }
  }
  return prefered_action;
}

}