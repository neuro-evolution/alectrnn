//
// Created by nathaniel on 9/15/18.
//

#include <ale_interface.hpp>
#include "shared_motor_agent.hpp"
#include "../nervous_system/nervous_system.hpp"
#include "reward_mod_agent.hpp"
#include "../nervous_system/layer.hpp"

namespace alectrnn
{

RewardModulatedAgent::RewardModulatedAgent(ALEInterface* ale,
                                           nervous_system::NervousSystem<float>& neural_net,
                                           Index update_rate, bool is_logging)
    : super_type(ale, neural_net, update_rate, is_logging)
{}

void RewardModulatedAgent::RewardFeedback(const int reward)
{
  for (std::size_t iii = 1; iii < neural_net_.size(); ++iii)
  {
    nervous_system::RewardModulatedLayer<float>& reward_layer =
      dynamic_cast<nervous_system::RewardModulatedLayer<float>&>(neural_net_[iii]);
    reward_layer.UpdateWeights(reward, &neural_net_[iii-1]);
  }
}

}