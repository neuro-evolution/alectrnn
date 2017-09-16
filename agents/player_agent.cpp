/*
 * player_agent.cpp
 *  Created on: Sep 1, 2017
 *      Author: Nathaniel Rodriguez
 *
 * The PlayerAgent is an abstract class use to define the basic functionality
 * of an agent for the ALE atari games. It is used by the Controller to play
 * the games. Derived classes MUST define an Act() function that returns
 * an ALE action.
 *
 * This is a simplified and slightly modified version of the google Xitari
 * PlayerAgent class.
 */

#include "player_agent.h"
#include <ale_interface.hpp>

namespace alectrnn {

PlayerAgent::PlayerAgent(ALEInterface* ale) : ale_(ale),
    frame_number_(0), episode_frame_number_(0), episode_number_(0),
    available_actions_(ale->romSettings->getMinimalActionSet()),
    has_terminated_(false) {
  max_num_frames_ = ale_->getInt("max_num_frames");
  max_num_episodes_ = ale_->getInt("max_num_episodes");
  max_num_frames_per_episode_ = ale_->getInt("max_num_frames_per_episode");
}

PlayerAgent::~PlayerAgent() {
}

Action PlayerAgent::AgentStep() {
  if (max_num_frames_ > 0 && frame_number_ >= max_num_frames_) {
    EndGame();
  }

  if (max_num_frames_per_episode_ > 0 &&
      episode_frame_number_ >= max_num_frames_per_episode_) {
    return RESET; //Pushes reset button on console (Action #40)
  }

  Action agent_action(Act());
  frame_number_++;
  episode_frame_number_++;

  return agent_action;
}

Action PlayerAgent::EpisodeStart() {
  episode_frame_number_ = 0;
  Action agent_action(Act());
  frame_number_++;
  episode_frame_number_++;
  return agent_action;
}

void PlayerAgent::EpisodeEnd() {
  episode_number_++;
  if (max_num_episodes_ > 0 && episode_number_ >= max_num_episodes_) {
    EndGame();
  }
}

bool PlayerAgent::HasTerminated() {
  return has_terminated_;
}

void PlayerAgent::EndGame() {
  has_terminated_ = true;
}

}
