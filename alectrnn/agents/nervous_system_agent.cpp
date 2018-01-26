#include "nervous_system_agent.hpp"
#include "multi_array.hpp"
#include "player_agent.hpp"
#include "../common/screen_preprocessing.hpp"
#include "../common/nervous_system.hpp"
#include "../common/state_logger.hpp"

namespace alectrnn {

NervousSystemAgent::NervousSystemAgent(ALEInterface* ale, 
    nervous_system::NervousSystem<float>& neural_net) 
    : NervousSystemAgent(ale, neural_net, 1, false) {

}

NervousSystemAgent::NervousSystemAgent(ALEInterface* ale, 
    nervous_system::NervousSystem<float>& neural_net, 
    Index update_rate, bool is_logging) : PlayerAgent(ale), neural_net_(neural_net), 
    update_rate_(update_rate), is_logging_(is_logging) {

  is_configured_ = false;
  buffer_screen1_.resize(ale_->environment->getScreenHeight() *
        ale_->environment->getScreenWidth());
  buffer_screen2_.resize(ale_->environment->getScreenHeight() *
        ale_->environment->getScreenWidth());
  full_screen_.resize(ale_->environment->getScreenHeight() *
        ale_->environment->getScreenWidth());
  downsized_screen_.resize(neural_net_[0].NumNeurons());

  if (is_logging_) {
    log_ = nervous_system::StateLogger<float>(neural_net_);
  }
}

NervousSystemAgent::~NervousSystemAgent() {}

void NervousSystemAgent::Configure(const float *parameters) {
  // Assumed that parameters is a contiguous array with # elements == par count
  // User must make sure this holds, as the slices only garantee that it won't
  // exceed count
  neural_net_.Configure(multi_array::ConstArraySlice<float>(
    parameters, 0, neural_net_.GetParameterCount(), 1));
}

void NervousSystemAgent::Reset() {
  neural_net_.Reset();
}

Action NervousSystemAgent::Act() {

  // Need to get the screen
  ale_->getScreenGrayscale(full_screen_);
  // Need to downsize the screen
  ResizeGrayScreen(ale_->environment->getScreenWidth(),
                   ale_->environment->getScreenHeight(),
                   neural_net_[0].shape()[1],//  major input_screen_width_
                   neural_net_[0].shape()[0],//  minor input_screen_height_
                   full_screen_,
                   downsized_screen_,
                   buffer_screen1_,
                   buffer_screen2_);

  neural_net_.SetInput(downsized_screen_);
  // The neural network will be updates update_rate_ times before output is read
  for (std::size_t iii = 0; iii < update_rate_; iii++) {
    if (is_logging_) {
      log_(neural_net_);
    }
    neural_net_.Step();
  }

  // Read values from last X neurons, X==LastNeuronIndex - Action#
  Action prefered_action(PLAYER_A_NOOP);
  float prefered_output(std::numeric_limits<float>::lowest());
  Action last_action(PLAYER_A_NOOP);
  float last_output(std::numeric_limits<float>::lowest());
  const multi_array::Tensor<float>& output = neural_net_.GetOutput();
  for (std::size_t iii = 0; iii < available_actions_.size(); iii++) {
    last_output = output[iii];
    last_action = available_actions_[iii];

    if (prefered_output < last_output) {
      prefered_output = last_output;
      prefered_action = last_action;
    }
  }
  return prefered_action;
}

const nervous_system::StateLogger<float>& NervousSystemAgent::GetLog() const {
  return log_;
}

} // End namespace alectrnn