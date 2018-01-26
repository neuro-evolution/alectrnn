/*
 * internal_controller.h
 *
 *  Created on: Sep 10, 2017
 *      Author: Nathaniel Rodriguez
 *
 * Setting max_num_frames = -1 allows agents to keep playing till they fail
 * Setting max_num_episodes = -1 allows agents to keep playing new games
 */

#ifndef ALECTRNN_CONTROLLERS_CONTROLLER_H_
#define ALECTRNN_CONTROLLERS_CONTROLLER_H_

#include <ale_interface.hpp>
#include "../agents/player_agent.hpp"

namespace alectrnn {

class Controller {
  public:
    Controller(ALEInterface* ale, PlayerAgent* agent);
    ~Controller();
    void Run();
    int getCumulativeScore() const;

  protected:
    void EpisodeStart(Action& action);
    void EpisodeStep(Action& action);
    void EpisodeEnd();
    void ApplyActions(Action& action);
    bool IsDone() const;

  protected:
    int max_num_frames_;
    int max_num_episodes_;
    int episode_score_;
    int cumulative_score_;
    int episode_number_;
    int frame_number_;
    PlayerAgent* agent_;
    ALEInterface* ale_;
};

}

#endif /* ALECTRNN_CONTROLLERS_CONTROLLER_H_ */
