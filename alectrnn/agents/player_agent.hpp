/*
 * player_agent.hpp
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

#ifndef ALECTRNN_AGENTS_PLAYER_AGENT_H_
#define ALECTRNN_AGENTS_PLAYER_AGENT_H_

#include <ale_interface.hpp>

namespace alectrnn {

class PlayerAgent {
  public:
    PlayerAgent(ALEInterface* ale);
    virtual ~PlayerAgent();

    /*
     * Function for calling the act() method in addition to keeping track of
     * # frames.
     */
    virtual Action AgentStep();

    /*
     * Called at beginning of game
     */
    virtual Action EpisodeStart();

    /*
     * Called at end of game
     */
    virtual void EpisodeEnd();
    /*
     * Returns True when agent is done playing.
     */
    virtual bool HasTerminated();

    /*
     * Configures the agent using search parameters
     */
    virtual void Configure(const float *parameters)=0;

    /*
     * Resets the agent's internal variables for a new set of episode runs
     * Make sure PlayerAgent::Reset() gets called in any derived class to ensure
     * that the frames and episode numbers are reset.
     */
    virtual void Reset();

    /*
     * Provides the option for agents to adjust their internal state given
     * the reward resulting from their actions. Default to no-op.
     */
    virtual void RewardFeedback(const int reward);

  protected:
    virtual Action Act()=0;
    void EndGame();

  protected:
    ALEInterface* ale_;
    const int frame_skip_;
    int max_num_frames_;
    int max_num_episodes_;
    int max_num_frames_per_episode_;
    int frame_number_;
    int episode_frame_number_;
    int episode_number_;
    ActionVect available_actions_;
    bool has_terminated_;
};

}

#endif /* ALECTRNN_AGENTS_PLAYER_AGENT_H_ */
