//
// Created by nathaniel on 9/15/18.
//

#ifndef ALECTRNN_REWARD_MOD_AGENT_HPP
#define ALECTRNN_REWARD_MOD_AGENT_HPP

#include <ale_interface.hpp>
#include "shared_motor_agent.hpp"
#include "../nervous_system/nervous_system.hpp"

namespace alectrnn
{

class RewardModulatedAgent : public SharedMotorAgent
{
  public:
    using super_type = SharedMotorAgent;
    using Index = super_type::Index;

    RewardModulatedAgent(ALEInterface* ale,
                         nervous_system::NervousSystem<float>& neural_net,
                         Index update_rate, bool is_logging);
    virtual ~RewardModulatedAgent()=default;

    virtual void RewardFeedback(const int reward);
};

}

#endif //ALECTRNN_REWARD_MOD_AGENT_HPP
