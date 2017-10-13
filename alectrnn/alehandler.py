"""A container class that holds the parameters for the various
objects in the alectrnn environment. It can use those parameters
to create handles to c++ objects which can be used by optimization
algorithms. It will also use the parameters to create a wrapper for
the objective function. All objective functions need an agent and ale.
Upon creation, this class will generate a new objective funtion with those
already given as arguments, allowing the returned function to be called
simply by giving the parameter vector."""

import alectrnn.ale_generator
import alectrnn.agent_generator
import alectrnn.total_cost_objective
import sys
from functools import partial

class ALEHandler:
    # The roms are located with the library
    rom_list = [] # TO DO get file locations and adjust init arguments
    # so that you can use game name or file name

    def __init__(self, rom, ale_seed, display_screen, sound, 
                color_avg, max_num_frames, max_num_episodes,
                max_num_frames_per_episode, agent_type, 
                agent_parameters, objective_type, objective_parameters):
        """
        ALE parameters:
          rom - rom filename
          ale_seed - integer type
          display_screen - boolean type
          sound - boolean type
          color_avg - boolean type (whether to average consecutive screens)
          max_num_frames - integer type
          max_num_episodes - integer type
          max_num_frames_per_episode - integer type

        Agent parameters:
          agent_type - "ctrnn"
          agent_parameters - dictionary of keyword arguments for the agent

            For "ctrnn": (num_neurons, num_sensor_neurons,
                          input_screen_width, input_screen_height,
                          use_color, step_size)

        Objective parameters:
        # Note: should not include agent/ALE/parameters, only configuration pars
          objective_type - "totalcost"
          objective_parameters - dictionary of keyword arguments for objective

            For "totalcost": none

        """

        self.rom = rom
        self.ale_seed = ale_seed
        self.display_screen = display_screen
        self.sound = sound
        self.color_avg = color_avg
        self.max_num_frames = max_num_frames
        self.max_num_episodes = max_num_episodes
        self.max_num_frames_per_episode = max_num_frames_per_episode
        self.agent_type = agent_type
        self.agent_parameters = agent_parameters
        self.objective_type = objective_type
        self.objective_parameters = objective_parameters

        # Create ALE handle
        self.ale = ale_generator.CreatALE(rom=self.rom, seed=self.ale_seed,
            display_screen=self.display_screen, sound=self.sound,
            color_avg=self.color_avg, max_num_frames=self.max_num_frames,
            max_num_episodes=self.max_num_episodes,
            max_num_frames_per_episode=self.max_num_frames_per_episode)

        # Create Agent handle
        if self.agent_type == "ctrnn":
            self.agent = agent_generator.CreatCtrnnAgent(self.ale, 
                                                        **self.agent_parameters)
        else:
            sys.exit("No agent by that name is implemented")

        # Create partial objective function
        if self.objective_type == "totalcost":
            self.objective = partial(total_cost_objective.TotalCostObjective, 
                                    ale=self.ale, agent=self.agent)

    def get_ale():
        return self.ale

    def get_objective():
        return self.objective

    def get_agent():
        return self.agent

if __name__ == '__main__':
    """
    example
    """
    pass