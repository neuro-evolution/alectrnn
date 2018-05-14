//
// Created by nathaniel on 5/13/18.
//

#include <ale_interface.hpp>
#include "soft_max_agent.hpp"

namespace alectrnn {

void SoftMaxAgent::seed(int new_seed) {
  rng_.seed(new_seed);
}

Action SoftMaxAgent::Act() {
  super_type::UpdateScreen();
  super_type::StepNervousSystem();
  return GetActionFromNervousSystem();
}

Action SoftMaxAgent::GetActionFromNervousSystem() {

  Action preferred_action(PLAYER_A_NOOP);
  float cdf = 0.0;
  float rv = rand_real_(rng_);
  const multi_array::Tensor<float>& output = super_type::neural_net_.GetOutput();
  for (std::size_t iii = 0; iii < super_type::available_actions_.size(); iii++) {
    cdf += output[iii];
    if (rv <= cdf) {
      return super_type::available_actions_[iii];
    }
  }

  return preferred_action;
}

}