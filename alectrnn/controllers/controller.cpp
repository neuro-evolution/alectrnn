/*
 * internal_controller.cpp
 *
 *  Created on: Sep 10, 2017
 *      Author: nathaniel
 */

#include <ale_interface.hpp>
#include "../agents/player_agent.hpp"
#include "controller.hpp"
#include <string>
#include <sstream>
#include <iomanip>

namespace alectrnn {

Controller::Controller(ALEInterface* ale, PlayerAgent* agent)
      : ale_(ale), agent_(agent), episode_score_(0), episode_number_(0),
        cumulative_score_(0), frame_number_(0),
        frame_skip_(ale_->getInt("frame_skip")),
        max_num_frames_(ale_->getInt("max_num_frames")),
        max_num_episodes_(ale_->getInt("max_num_episodes")) {
  ale_->training_reset();
}

Controller::~Controller() {
}

void Controller::Run() {
  Action agent_action;
  bool first_step = true;
  ale_->training_reset();
  agent_->Reset();

  while (!IsDone()) {
    // Start a new episode: Check for terminal state
    if (ale_->environment->isTerminal()) {
      EpisodeEnd();
      first_step = true;
    }
    else {
      if (first_step) {
        // Start a new episode; obtain actions
        EpisodeStart(agent_action);
        first_step = false;
      }
      else {
        // Poll agents for actions
        EpisodeStep(agent_action);
      }

      // Apply said actions
      ApplyActions(agent_action);
      frame_number_ += frame_skip_;
      episode_score_ += ale_->romSettings->getReward();
      cumulative_score_ += ale_->romSettings->getReward();
    }

    if (ale_->getBool("print_screen")) {
      std::stringstream ss;
      ss << std::setw(10) << std::setfill('0') << frame_number_;
      std::string framename = ss.str();
      ale_->saveScreenPNG(framename + "_game_frame.png");
    }
  }
}

int Controller::getCumulativeScore() const {
  return cumulative_score_;
}

void Controller::EpisodeStart(Action& action) {
  // Poll the agent for first action
  action = agent_->EpisodeStart();
  episode_score_ = 0;
  episode_number_++;
}

void Controller::EpisodeStep(Action& action) {
  action = agent_->AgentStep();
}

void Controller::EpisodeEnd() {
  agent_->EpisodeEnd();
  ale_->training_reset();
}

void Controller::ApplyActions(Action& action) {
  // Perform different operations based on the first player's action
  switch (action) {
    case LOAD_STATE: // Load system state
      // Note - this does not reset the game screen;
      // so that the subsequent screen
      // is incorrect (in fact, two screens, due to colour averaging)
      ale_->environment->load();
      break;
    case SAVE_STATE: // Save system state
      ale_->environment->save();
      break;
    case SYSTEM_RESET:
      ale_->training_reset(); // i don't think episode gets incremented....
      break;
    default:
      // Pass action to emulator!
      ale_->environment->minimalAct(action, PLAYER_B_NOOP);
      break;
  }
}

bool Controller::IsDone() const {
  return (agent_->HasTerminated() ||
      (max_num_episodes_ > 0 && episode_number_ > max_num_episodes_) ||
      (max_num_frames_ > 0 &&
          frame_number_ >= max_num_frames_));
}

int Controller::GetEpisodeNumber() const {
  return episode_number_;
}

int Controller::GetFrameNumber() const {
  return frame_number_;
}

int Controller::GetEpisodeFrameNumber() const {
  return ale->getEpisodeFrameNumber();
}

}
