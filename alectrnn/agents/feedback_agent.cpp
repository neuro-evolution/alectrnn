//
// Created by nathaniel on 9/15/18.
//

#include <ale_interface.hpp>
#include "shared_motor_agent.hpp"
#include "../nervous_system/nervous_system.hpp"
#include "feedback_agent.hpp"
#include "../nervous_system/layer.hpp"

namespace alectrnn
{

FeedbackAgent::FeedbackAgent(ALEInterface* ale,
                             nervous_system::NervousSystem<float>& neural_net,
                             Index update_rate, bool is_logging,
                             Index motor_index, Index feedback_index)
    : super_type(ale, neural_net, update_rate, is_logging),
      motor_index_(motor_index), feedback_index_(feedback_index)
{}

void FeedbackAgent::RewardFeedback(const int reward)
{
  nervous_system::FeedbackLayer<float>& feedback_layer =
    dynamic_cast<nervous_system::FeedbackLayer<float>&>(neural_net_[feedback_index_]);
  feedback_layer.update_feedback(reward, &neural_net_[motor_index_]);
}

}
