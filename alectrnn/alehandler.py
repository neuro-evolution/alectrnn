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
import alectrnn.objective
import sys
from functools import partial
from pkg_resources import resource_listdir
from pkg_resources import resource_filename

def generate_rom_dictionary():
    roms_list = resource_listdir("alectrnn", "roms")
    rom_path_list = [ resource_filename("alectrnn", "roms/" + rom) 
                        for rom in roms_list if ".bin" in rom ]
    rom_name_list = [ rom[:-4] for rom in roms_list if ".bin" in rom ]
    return { rom_name_list[i] : rom_path_list[i] 
            for i in range(len(rom_path_list)) }

class ALEHandler:

    installed_roms = generate_rom_dictionary()

    def __init__(self, rom, ale_seed, display_screen, sound, 
                color_avg, max_num_frames, max_num_episodes,
                max_num_frames_per_episode, agent_type, 
                objective_type,
                agent_parameters={},
                objective_parameters={},
                rom_file=""):
        """
        ALE parameters:
          rom - rom name (specify from list)
          rom_file - rom filename (specify for using your own roms)
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

        self.rom_path = ""
        if rom_file != "":
            self.rom_path = rom_file
        else:
            if rom in ALEHandler.installed_roms:
                self.rom_path = ALEHandler.installed_roms[rom]
            else:
                sys.exit("Error: " + rom + " is not installed.")

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
        self.ale = alectrnn.ale_generator.CreatALE(
            rom=self.rom_path, 
            seed=self.ale_seed, 
            display_screen=self.display_screen, 
            sound=self.sound,
            color_avg=self.color_avg, 
            max_num_frames=self.max_num_frames,
            max_num_episodes=self.max_num_episodes,
            max_num_frames_per_episode=self.max_num_frames_per_episode)

        # Create Agent handle
        if self.agent_type == "ctrnn":
            self.agent = alectrnn.agent_generator.CreateCtrnnAgent(self.ale, 
                                                        **self.agent_parameters)
        else:
            sys.exit("No agent by that name is implemented")

        # Create partial objective function
        if self.objective_type == "totalcost":
            self.objective = partial(alectrnn.objective.TotalCostObjective, 
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