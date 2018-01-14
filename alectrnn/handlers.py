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
import alectrnn.layer_generator
import alectrnn.nn_generator
import alectrnn.nn_handler
import alectrnn.ale_handler
import sys
from functools import partial
from pkg_resources import resource_listdir
from pkg_resources import resource_filename
from enum import Enum
import numpy as np

def generate_rom_dictionary():
    roms_list = resource_listdir("alectrnn", "roms")
    rom_path_list = [ resource_filename("alectrnn", "roms/" + rom) 
                        for rom in roms_list if ".bin" in rom ]
    rom_name_list = [ rom[:-4] for rom in roms_list if ".bin" in rom ]
    return { rom_name_list[i] : rom_path_list[i] 
            for i in range(len(rom_path_list)) }

class Handler:

    def __init__(self, handle_type, handle_parameters={}):

        self.handle_type = handle_type
        self.handle_parameters = handle_parameters
        self.handle = None

    def create(self):
        raise NotImplementedError

    def get(self):
        return self.handle

class ObjHandler(Handler):

    def __init__(self, ale, agent, obj_type, obj_parameters={}):
        """
        Objective parameters:
        # Note: should not include agent/ALE/parameters, only configuration pars
          objective_type - "totalcost"
          objective_parameters - dictionary of keyword arguments for objective

            For "totalcost": none
        """
        super(obj_type, obj_parameters)
        self.ale = ale
        self.agent = agent

    def create(self):
        if self.handle_type == "totalcost":
            self.handle = partial(alectrnn.objective.TotalCostObjective, 
                                    ale=self.ale, agent=self.agent)

class AgentHandler(Handler):

    def __init__(self, ale, agent_type, agent_parameters={}):
        """
        Agent parameters:
          agent_type - "ctrnn"
          agent_parameters - dictionary of keyword arguments for the agent

            For "ctrnn": (num_neurons, num_sensor_neurons,
                          input_screen_width, input_screen_height,
                          use_color, step_size)
        """
        super(agent_type, agent_parameters)
        self.ale = ale

    def create(self):
        # Create Agent handle
        if self.handle_type == "ctrnn":
            self.handle = alectrnn.agent_generator.CreateCtrnnAgent(self.ale, 
                                                        **self.handle_parameters)
        elif self.handle_type == "hybrid":
            self.handle = alectrnn.agent_generator.CreateHybridAgent(self.ale, 
                                                        **self.handle_parameters)
        else:
            sys.exit("No agent by that name is implemented")

class ALEHandler(Handler):

    installed_roms = generate_rom_dictionary()

    def __init__(self, rom, ale_seed,
                color_avg, max_num_frames, max_num_episodes,
                max_num_frames_per_episode,
                rom_file="",
                print_screen=False,
                display_screen=False,
                sound=False):
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
          print_screen - boolean type

        """
        super(None, None)
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
        self.print_screen = print_screen

    def create(self):
        # Create ALE handle
        self.handle = alectrnn.ale_generator.CreatALE(
            rom=self.rom_path, 
            seed=self.ale_seed, 
            display_screen=self.display_screen, 
            sound=self.sound,
            color_avg=self.color_avg, 
            max_num_frames=self.max_num_frames,
            max_num_episodes=self.max_num_episodes,
            max_num_frames_per_episode=self.max_num_frames_per_episode,
            print_screen=self.print_screen)

    def action_set_size(self):
        return ale_handler.NumOutputs(self.handle)

    @classmethod
    def print_available_roms(cls):
        print("Available roms:")
        roms = list(ALEHandler.installed_roms.keys())
        roms.sort(key=str.lower)
        for rom in roms:
            print("\t", rom)

class ACTIVATOR_TYPE(Enum):
    BASE=0
    IDENTITY=1
    CTRNN=2
    CONV_CTRNN=3
    IAF=4

class INTEGRATOR_TYPE(Enum):
    BASE=0
    NONE=1
    ALL2ALL=2
    CONV=3
    NET=4
    RESERVOIR=5

class NervousSystem:
    """
    Layer: 3x integ/act type, args (tuple)
    Motor: num_outputs, num_inputs, act type, args (tuple)

    if ACT CTRNN: i num_states, f step_size
    if ACT CONV_CTRNN: numpy array 3ele shape(float32), f step_size

    if int NONE: nothing, empty
    if int CONV: i #filters, np arr filter_shape, np arr layer_shape, i stride, 
    if int NET: i #nodes, np arr std::uint64_t edge_list
    if int RESERVOIR: i #nodes, np arr std::uint64_t edge_list, np arr f32 weights
    """

    def __init__(self, input_shape, num_outputs, step_size, nn_parameters):
        """
        nn_parameters - list of dictionaries. One for each layer.

        back_connections:
            (filter-dimensions, # filters)
            (#nodes, edge_list)[bipartite]

        self_connections:
            (#nodes, edge_list)[graph]
            (#nodes, edge_list), (weights)

        activator:
            (type)

        """

        sefl.nn_parameters = nn_parameters
        self.layers = []

    def add_motor_layer(self):
        pass




if __name__ == '__main__':
    """
    example
    """
    pass