//
// Created by nathaniel on 6/8/18.
//

#include "shared_motor_agent.hpp"
#include <ale_interface.hpp>
#include "nervous_system_agent.hpp"
#include "../common/nervous_system.hpp"
#include "../common/multi_array.hpp"

namespace alectrnn {

SharedMotorAgent::SharedMotorAgent(ALEInterface *ale,
                                   nervous_system::NervousSystem<float> &neural_net,
                                   Index update_rate, bool is_logging)
    : super_type(ale, neural_net, update_rate, is_logging) {
}

Action SharedMotorAgent::Act() {
  super_type::UpdateScreen();
  super_type::StepNervousSystem();
  return GetActionFromNervousSystem();
}

Action SharedMotorAgent::GetActionFromNervousSystem() {

  // Read values from NN output and find preferred action
  Action preferred_action(PLAYER_A_NOOP);
  float preferred_output(std::numeric_limits<float>::lowest());
  Action last_action(PLAYER_A_NOOP);
  float last_output(std::numeric_limits<float>::lowest());
  const multi_array::Tensor<float>& output = super_type::neural_net_.GetOutput();
  for (std::size_t iii = 0; iii < super_type::available_actions_.size(); iii++) {
    // Motor outputs not in the minimal set are ignored.
    last_output = output[static_cast<std::size_t>(super_type::available_actions_[iii])];
    last_action = super_type::available_actions_[iii];

    if (preferred_output < last_output) {
      preferred_output = last_output;
      preferred_action = last_action;
    }
  }
  return preferred_action;
}

}