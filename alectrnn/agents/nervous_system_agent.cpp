#include <stdexcept>
#include "nervous_system_agent.hpp"
#include "player_agent.hpp"
#include "../common/multi_array.hpp"
#include "../common/screen_preprocessing.hpp"
#include "../common/nervous_system.hpp"
#include "../common/state_logger.hpp"

namespace alectrnn {

NervousSystemAgent::NervousSystemAgent(ALEInterface* ale, 
    nervous_system::NervousSystem<float>& neural_net) 
    : NervousSystemAgent(ale, neural_net, 1, false) {
}

/*
 * The neural network's first layer needs to be a 3-dimensional layer where the
 * first layer is the # of channels and following layers are the height and
 * width of the screen respectively. If gray-screen is used then there must be
 * only one channel.
 */
NervousSystemAgent::NervousSystemAgent(ALEInterface* ale, 
    nervous_system::NervousSystem<float>& neural_net, 
    Index update_rate, bool is_logging) : PlayerAgent(ale), neural_net_(neural_net), 
    update_rate_(update_rate), is_logging_(is_logging) {

  is_configured_ = false;

  // If grayscreen input
  // Check that NN input is correct dim and # channels
  if (!((neural_net_[0].shape().size() == 4) || (neural_net_[0].shape()[0] == 1))) {
    throw std::invalid_argument("NervousSystemAgent received screen input "
                                "dimensions that do not match available inputs."
                                " e.g. 4 and 1");
  }

  buffer_screen1_.resize(ale_->environment->getScreenHeight() *
        ale_->environment->getScreenWidth());
  buffer_screen2_.resize(ale_->environment->getScreenHeight() *
        ale_->environment->getScreenWidth());
  grey_screen_.resize(ale_->environment->getScreenHeight() *
        ale_->environment->getScreenWidth());
  downsized_screen_.resize(neural_net_[0].NumNeurons());

  // TO DO: Colored + luminance input (check use_color_ & nn dim/#channels)
  // ALE color axis: HxWx3
  // Luminance == ale->getScreenGrayscale
  // End shape will be HxWx4 -> needs to be reshaped to 4xHxW for NN

  if (is_logging_) {
    log_ = nervous_system::StateLogger<float>(neural_net_);
  }

  color_screen_.resize(3 * ale_->environment->getScreenHeight()
                       * ale_->environment->getScreenWidth());
  screen_log_ = ScreenLogger<float>({ale_->environment->getScreenHeight(),
                                     ale_->environment->getScreenWidth(), 3});
}

NervousSystemAgent::~NervousSystemAgent() {}

void NervousSystemAgent::Configure(const float *parameters) {
  // Assumed that parameters is a contiguous array with # elements == par count
  // User must make sure this holds, as the slices only guarantee that it won't
  // exceed count
  neural_net_.Configure(multi_array::ConstArraySlice<float>(
    parameters, 0, neural_net_.GetParameterCount(), 1));
}

void NervousSystemAgent::Reset() {
  PlayerAgent::Reset();
  neural_net_.Reset();
}

Action NervousSystemAgent::Act() {
  // Need to get the screen
  ale_->getScreenGrayscale(grey_screen_);

  if (is_logging_) {
    ale_->getScreenRGB(color_screen_);
    screen_log_(color_screen_);
  }

  // Need to downsize the screen
  ResizeGrayScreen(ale_->environment->getScreenWidth(),
                   ale_->environment->getScreenHeight(),
                   neural_net_[0].shape()[2],//  major input_screen_width_
                   neural_net_[0].shape()[1],//  minor input_screen_height_
                   grey_screen_,
                   downsized_screen_,
                   buffer_screen1_,
                   buffer_screen2_);
  neural_net_.SetInput(downsized_screen_);
  // The neural network will be updates update_rate_ times before output is read
  for (std::size_t iii = 0; iii < update_rate_; iii++) {
    neural_net_.Step();
    if (is_logging_) {
      log_(neural_net_);
    }
  }

  // Read values from last X neurons, X==LastNeuronIndex - Action#
  Action preferred_action(PLAYER_A_NOOP);
  float preferred_output(std::numeric_limits<float>::lowest());
  Action last_action(PLAYER_A_NOOP);
  float last_output(std::numeric_limits<float>::lowest());
  const multi_array::Tensor<float>& output = neural_net_.GetOutput();
  for (std::size_t iii = 0; iii < available_actions_.size(); iii++) {
    last_output = output[iii];
    last_action = available_actions_[iii];

    if (preferred_output < last_output) {
      preferred_output = last_output;
      preferred_action = last_action;
    }
  }
  return preferred_action;
}

const nervous_system::StateLogger<float>& NervousSystemAgent::GetLog() const {
  return log_;
}

const ScreenLogger<float>& NervousSystemAgent::GetScreenLog() const {
  return screen_log_;
}

} // End namespace alectrnn