/*
 * ctrnn_agent.h
 *  Created on: Sep 1, 2017
 *      Author: Nathaniel Rodriguez
 *
 * The CTRNN agent class is a specific PlayerAgent that creates a neural
 * network and returns actions based on the provided visual stimuli from
 * the ALE environment. This agent only needs to be created once, it has
 * a Configure function that can take parameters to reconfigure the
 * neural network weights.
 *
 * update_rate_ = number of euler steps taken per screen input
 *                the screen remains still during these updates (default 1)
 * step_size = the euler step size for integration
 */

#ifndef ALECTRNN_CTRNN_AGENT_H_
#define ALECTRNN_CTRNN_AGENT_H_

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <ale_interface.hpp>
#include "player_agent.hpp"
#include "../common/network_generator.hpp"
#include "../common/nervous_system.hpp"

namespace alectrnn {

class CtrnnAgent : public PlayerAgent {
  public:
    CtrnnAgent(ALEInterface* ale, std::size_t num_neurons,
        std::size_t num_sensor_neurons, std::size_t input_screen_width,
        std::size_t input_screen_height, bool use_color, float step_size,
        std::size_t update_rate);
    CtrnnAgent(ALEInterface* ale, std::size_t num_neurons,
        std::size_t num_sensor_neurons, std::size_t input_screen_width,
        std::size_t input_screen_height, bool use_color, float step_size);
    ~CtrnnAgent();
    void Configure(const float *parameters);
    void Reset();

  protected:
    Action Act();

  private:
    std::unique_ptr<ctrnn::NeuralNetwork> agent_neural_system_;
    std::vector<std::vector<ctrnn::InEdge> > node_sensors_;
    std::vector<std::vector<ctrnn::InEdge> > node_neighbors_;
    std::vector<std::uint8_t> buffer_screen_;
    std::vector<std::uint8_t> full_screen_;
    std::vector<std::uint8_t> downsized_screen_;
    std::size_t num_neurons_;
    std::size_t num_sensor_neurons_;
    std::size_t input_screen_width_;
    std::size_t input_screen_height_;
    std::size_t num_sensors_;
    std::size_t update_rate_;
    bool use_color_;
    bool is_configured_;
};

}

#endif /* ALECTRNN_CTRNN_AGENT_H_ */
