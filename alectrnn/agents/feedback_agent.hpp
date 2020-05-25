#ifndef ALECTRNN_FEEDBACK_AGENT_HPP
#define ALECTRNN_FEEDBACK_AGENT_HPP

#include <ale_interface.hpp>
#include "shared_motor_agent.hpp"
#include "../nervous_system/nervous_system.hpp"

namespace alectrnn
{

class FeedbackAgent : public SharedMotorAgent
{
  public:
    using super_type = SharedMotorAgent;
    using Index = super_type::Index;

    FeedbackAgent(ALEInterface* ale,
                  nervous_system::NervousSystem<float>& neural_net,
                  Index update_rate, bool is_logging,
                  Index motor_index, Index feedback_index);
    virtual ~FeedbackAgent()=default;

    virtual void RewardFeedback(const int reward);
  protected:
    Index motor_index_;
    Index feedback_index_;
};

}

#endif //ALECTRNN_FEEDBACK_AGENT_HPP
