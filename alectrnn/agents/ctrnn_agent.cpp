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

namespace alectrnn {

CtrnnAgent::CtrnnAgent(ALEInterface* ale,
      std::size_t num_neurons, std::size_t num_sensor_neurons,
      std::size_t input_screen_width,
      std::size_t input_screen_height, bool use_color, double step_size)
    : PlayerAgent(ale), num_neurons_(num_neurons),
      num_sensor_neurons_(num_sensor_neurons), use_color_(use_color),
      input_screen_width_(input_screen_width),
      input_screen_height_(input_screen_height), is_configured_(false) {
  // Reserve space for input screens
  if (!use_color_) {
    num_sensors_ = input_screen_width_ * input_screen_height_;
    full_screen_.resize(
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

CtrnnAgent::~CtrnnAgent() {
}

void CtrnnAgent::Configure(const double *parameters) {
  /*
   * Parameters need to be decoded and the nervous system configured
   * Since we will be using cmaes, we will let the parameter bounds be defined
   * there. So we will assume we are getting valid parameters.
   * Note: assumes parameters is a contiguous c-style array
   *
   * TO DO: Need a size check, make sure parameters array size == # parameters
   */

  // configure bias
  const double *bias_parameters = parameters;
  for (std::size_t iii = 0; iii < num_neurons_; iii++) {
    agent_neural_system_->setNeuronBias(iii, bias_parameters[iii]);
  }

  // configure tau
  const double *tau_parameters = bias_parameters + num_neurons_;
  for (std::size_t iii = 0; iii < num_neurons_; iii++) {
    agent_neural_system_->setNeuronTau(iii, tau_parameters[iii]);
  }

  // configure gain
  const double *gain_parameters = tau_parameters + num_neurons_;
  for (std::size_t iii = 0; iii < num_neurons_; iii++) {
    agent_neural_system_->setNeuronGain(iii, gain_parameters[iii]);
  }

  // configure weights
  const double *circuit_parameters = gain_parameters + num_neurons_;
  ctrnn::FillAll2AllNetwork(node_neighbors_, circuit_parameters);
  const double *sensor_parameters = circuit_parameters + (num_neurons_*num_neurons_);
  ctrnn::FillFullSensorNetwork(node_sensors_, sensor_parameters);
  // Reset nervous system
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
                     downsized_screen_);
  }
  else {
    /*
     * Not currently implemented
     */
  }

  // Update sensor state (cast uint8_t into double)
  for (std::size_t iii = 0; iii < num_sensors_; iii++) {
    agent_neural_system_->setSensorState(iii, (double)downsized_screen_[iii]);
  }

  agent_neural_system_->EulerStep();
  // Read values from last X neurons, X==LastNeuronIndex - Action#
  Action prefered_action(PLAYER_A_NOOP);
  double prefered_output(-100000.0);
  Action last_action(PLAYER_A_NOOP);
  double last_output(-100000.0);
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
