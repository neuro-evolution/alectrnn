//
// Created by nathaniel on 6/8/18.
//

#ifndef ALECTRNN_SHARED_MOTOR_AGENT_HPP
#define ALECTRNN_SHARED_MOTOR_AGENT_HPP

#include <ale_interface.hpp>
#include "nervous_system_agent.hpp"
#include "../nervous_system/nervous_system.hpp"

namespace alectrnn {

/*
 * This class implements a "shared" motor system, where it employs a motor
 * layer (which is checked on construction) that is the same size as the legal
 * action set in ALE. Actions are still only chosen from the minimal set for
 * any given game. Action take values from 0->LegalActionSetSize. The action
 * values correspond to indexes in the motor layer, so Ith action corresponds
 * to the action with value I. This allows you to keep a single motor layer
 * (and potentially a single network) between games.
 */
class SharedMotorAgent : public NervousSystemAgent {
  public:
    typedef NervousSystemAgent super_type;
    typedef typename super_type::Index Index;

    SharedMotorAgent(ALEInterface* ale,
                     nervous_system::NervousSystem<float>& neural_net,
                     Index update_rate, bool is_logging);

  protected:
    Action Act() override;
    Action GetActionFromNervousSystem();
};

}

#endif //ALECTRNN_SHARED_MOTOR_AGENT_HPP
