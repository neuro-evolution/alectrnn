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
 */

#ifndef ALECTRNN_CTRNN_AGENT_H_
#define ALECTRNN_CTRNN_AGENT_H_

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <ale_interface.hpp>
#include "player_agent.h"
#include "../common/network_generator.h"
#include "../common/nervous_system.h"

namespace alectrnn {

class CtrnnAgent : public PlayerAgent {
  public:
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
    bool use_color_;
    bool is_configured_;
};

}

#endif /* ALECTRNN_CTRNN_AGENT_H_ */
