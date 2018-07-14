#include <stdexcept>
#include <ale_interface.hpp>
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
 * width of the screen respectively. The # of channels represent temporal channels
 * where 0th channel is the most recent screen input, the ith is the ith most
 * recent screen. Note: these need not be contiguous screens, as controller
 * has control over what screens the NervousSystem receives.
 */
NervousSystemAgent::NervousSystemAgent(ALEInterface* ale, 
    nervous_system::NervousSystem<float>& neural_net, 
    Index update_rate, bool is_logging) : PlayerAgent(ale), neural_net_(neural_net), 
    update_rate_(update_rate), is_logging_(is_logging) {

  is_configured_ = false;

  // Check that NN input is correct dim
  if (neural_net_[0].shape().size() != 3) {
    throw std::invalid_argument("NervousSystemAgent needs input layer with "
                                "3 dimensions");
  }

  buffer_screen1_.resize(ale_->environment->getScreenHeight() *
        ale_->environment->getScreenWidth());
  buffer_screen2_.resize(ale_->environment->getScreenHeight() *
        ale_->environment->getScreenWidth());
  grey_screen_.resize(ale_->environment->getScreenHeight() *
        ale_->environment->getScreenWidth());
  downsized_screen_.resize(neural_net_[0].shape()[1] * neural_net_[0].shape()[2]);

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

void NervousSystemAgent::UpdateScreen() {
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
}

Action NervousSystemAgent::GetActionFromNervousSystem() {

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

void NervousSystemAgent::UpdateNervousSystemInput() {

  auto& input_state(neural_net_.GetLayerState(0));
  auto input_layer_view = input_state.accessor();
  Index filter = 1;
  // if temporal channels exist, move contents of each input forward one channel
  while (filter != input_layer_view.extent(0)) {
    for (Index iii = 0; iii < input_layer_view.extent(1); ++iii) {
      for (Index jjj = 0; jjj < input_layer_view.extent(2); ++jjj) {
        input_layer_view[filter-1][iii][jjj] = input_layer_view[filter][iii][jjj];
      }
    }
    ++filter;
  }

  // Write in new screen input into last input channel
  auto end_channel = input_layer_view.extent(0) - 1;
  auto screen_width = ale_->environment->getScreenWidth();
  for (Index iii = 0; iii < input_layer_view.extent(1); ++iii) {
    for (Index jjj = 0; jjj < input_layer_view.extent(2); ++jjj) {
      input_layer_view[end_channel][iii][jjj] = downsized_screen_[iii * input_layer_view.extent(2)
                                                                  + jjj];
    }
  }
}

void NervousSystemAgent::StepNervousSystem() {
  /*
   * Screen is updated once, but the network can be updated multiple times for
   * each screen update.
   */
  UpdateNervousSystemInput();
  for (std::size_t iii = 0; iii < update_rate_; iii++) {
    neural_net_.Step();
    if (is_logging_) {
      log_(neural_net_);
    }
  }
}

Action NervousSystemAgent::Act() {
  UpdateScreen();
  StepNervousSystem();
  return GetActionFromNervousSystem();
}

const nervous_system::StateLogger<float>& NervousSystemAgent::GetLog() const {
  return log_;
}

const ScreenLogger<float>& NervousSystemAgent::GetScreenLog() const {
  return screen_log_;
}

const nervous_system::NervousSystem<float>& NervousSystemAgent::GetNeuralNet() const {
  return neural_net_;
}

} // End namespace alectrnn