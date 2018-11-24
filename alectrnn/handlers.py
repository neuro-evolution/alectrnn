"""A container class that holds the parameters for the various
objects in the alectrnn environment. It can use those parameters
to create handles to c++ objects which can be used by optimization
algorithms. It will also use the parameters to create a wrapper for
the objective function. All objective functions need an agent and ale.
Upon creation, this class will generate a new objective funtion with those
already given as arguments, allowing the returned function to be called
simply by giving the parameter vector."""

from alectrnn import ale_generator
from alectrnn import agent_generator
from alectrnn import objective
from alectrnn import ale_handler
from alectrnn import agent_handler
import sys
from functools import partial
from pkg_resources import resource_listdir
from pkg_resources import resource_filename


def generate_rom_dictionary():
    """
    Determines what roms are available
    :return: A dictionary keyed by rom name and valued by path
    """
    roms_list = resource_listdir("alectrnn", "roms")
    rom_path_list = [resource_filename("alectrnn", "roms/" + rom) 
                      for rom in roms_list if ".bin" in rom]
    rom_name_list = [rom[:-4] for rom in roms_list if ".bin" in rom]
    return {rom_name_list[i] : rom_path_list[i] 
            for i in range(len(rom_path_list))}


class Handler:
    """
    Parent class for the various Handlers. Holds parameters and defines a
    few functions for accessing the Python Capsules (handle).
    """
    def __init__(self, handle_type, handle_parameters=None):
        """
        :param handle_type: the type of handler being created
        :param handle_parameters: a dictionary of parameters needed for its creation
        """
        if handle_parameters is None:
            handle_parameters = {}

        self._handle_type = handle_type
        self._handle_parameters = handle_parameters
        self._handle = None
        self._handle_exists = False

    def create(self):
        """
        Should create the handle by calling the appropriate c-function and
        return nothing. Should also set the self._handle_exists flag to True
        :return: None
        """
        raise NotImplementedError

    @property
    def handle_type(self):
        return self._handle_type

    @handle_type.setter
    def handle_type(self, value):
        raise AttributeError("Not allowed to change type once set")

    @property
    def handle(self):
        if not self._handle_exists:
            raise AttributeError("Error: Handle hasn't been created yet. Call "
                                 "the created method first.")
        else:
            return self._handle

    @handle.setter
    def handle(self, value):
        raise AttributeError("Not allowed to set the handle")

    @property
    def parameters(self):
        return self._handle_parameters


class ObjectiveHandler(Handler):
    """
    Subclass of handler meant to create and manage the c++ objective function.
    """
    def __init__(self, ale, agent, obj_type, obj_parameters=None):
        """
        Objective parameters:
        # Note: should not include agent/ALE/parameters, only configuration pars
          obj_type - "totalcost"
          obj_parameters - dictionary of keyword arguments for objective

            For "totalcost": none
            For "s&cc": cc_scale
        """
        if obj_parameters is None:
            obj_parameters = {}

        super().__init__(obj_type, obj_parameters)
        self._ale = ale
        self._agent = agent

    def create(self):
        """
        If either ale or agent change in-place, the handler needs to be
        re-created in order to update the partial function.
        :return: None
        """
        if self._handle_type == "totalcost":
            self._handle = partial(objective.TotalCostObjective, 
                                   ale=self._ale, agent=self._agent)
            self._handle_exists = True
        elif self._handle_type == "s&cc":
            self._handle = partial(objective.ScoreAndConnectionCostObjective,
                ale=self._ale, agent=self._agent,
                cc_scale=self._handle_parameters['cc_scale'])
            self._handle_exists = True

        else:
            raise NotImplementedError

    @property
    def ale(self):
        return self._ale

    @ale.setter
    def ale(self, new_ale):
        self._ale = new_ale
        self.create()

    @property
    def agent(self):
        return self._agent

    @agent.setter
    def agent(self, new_agent):
        self._agent = new_agent
        self.create()


class AgentHandler(Handler):
    """
    Handler subclass meant for dealing with ALE agents
    """
    def __init__(self, ale, agent_type, agent_parameters=None):
        """
        Agent parameters:
          agent_type - "ctrnn"/"nervous_system"/"softmax"/"shared_motor"/"rewardmod"
          agent_parameters - dictionary of keyword arguments for the agent

            For "ctrnn": (num_neurons, num_sensor_neurons,
                          input_screen_width, input_screen_height,
                          use_color, step_size)

            For "nervous_system": (nervous_system, update_rate, logging)

            For "softmax": (nervous_system, update_rate, logging, seed)

            For "shared_motor": (nervous_system, update_rate, logging)

            For "rewardmod" : (nervous_system, update_rate, logging)
        """
        if agent_parameters is None:
            agent_parameters = {}

        super().__init__(agent_type, agent_parameters)
        self._ale = ale

    def create(self):
        """
        Creates the handle:
        Currently supports "ctrnn"/"nervous_system"/"softmax"/"shared_motor"/
        "rewardmod"
        """
        # Create Agent handle
        if self._handle_type == "ctrnn":
            self._handle = agent_generator.CreateCtrnnAgent(self._ale,
                                                            **self._handle_parameters)
        elif self._handle_type == "nervous_system":
            self._handle = agent_generator.CreateNervousSystemAgent(self._ale,
                                                                    **self._handle_parameters)
        elif self._handle_type == "softmax":
            self._handle = agent_generator.CreateSoftMaxAgent(self._ale,
                                                              **self._handle_parameters)
        elif self._handle_type == "shared_motor":
            self._handle = agent_generator.CreateSharedMotorAgent(self._ale,
                                                                  **self._handle_parameters)
        elif self._handle_type == "rewardmod":
            self._handle = agent_generator.CreateRewardModulatedAgent(self._ale,
                                                                      **self._handle_parameters)
        else:
            sys.exit("No agent by that name is implemented")
        self._handle_exists = True

    @property
    def ale(self):
        return self._ale

    @ale.setter
    def ale(self, new_ale):
        self._ale = new_ale
        self.create()


class LoggingAndHistoryMixin:

    def layer_history(self, layer_index):
        """
        Returns a numpy array with dimensions equal to the layer dimensions
        and # elements = # states in that layer.
        It will be of dtype=np.float32
        """
        if self._handle_parameters['logging']:
            return agent_handler.GetLayerHistory(self._handle, layer_index)
        else:
            raise AssertionError("Error: Logging not active, no history table")

    def screen_history(self):
        """
        :return: A numpy array with the history of the color screen of the
            emulator. First dimension is time, followed by HxWx3, where the
            elements are ordered as RGB. dtype=np.float32
        """
        if self._handle_parameters['logging']:
            return agent_handler.GetScreenHistory(self._handle)
        else:
            raise AssertionError("Error: Logging not active, no screen history")

    @property
    def logging(self):
        return self._handle_parameters['logging']

    @logging.setter
    def logging(self, is_logging):
        """
        Toggles the logging of the agent. Requires that the agent get re-made
        if the value changes
        :param is_logging: a boolean
        :return: None
        """
        if self._handle_parameters['logging'] != is_logging:
            self._handle_parameters['logging'] = int(is_logging)
            self.create()


class NervousSystemAgentHandler(AgentHandler, LoggingAndHistoryMixin):
    """
    Subclass of AgentHandler for easy creation and handling of NervousSystem
    agents, which have several distinct methods (e.g. layer_history and
    screen_history).
    """
    def __init__(self, ale, nervous_system, update_rate, logging=False):
        super().__init__(ale, "nervous_system", {'nervous_system': nervous_system,
                                                 'update_rate': update_rate,
                                                 'logging': int(logging)})


class SharedMotorAgentHandler(AgentHandler, LoggingAndHistoryMixin):
    """
    Subclass of AgentHandler for easy creation and handling of SharedMotor
    agents, which allow a single neural network to be used across games by
    using the legal action set size for the motor layer. Minimal action sets
    are still used to play the game.
    """
    def __init__(self, ale, nervous_system, update_rate, logging=False):
        super().__init__(ale, "shared_motor", {'nervous_system': nervous_system,
                                               'update_rate': update_rate,
                                               'logging': int(logging)})


class RewardModulatedAgentHandler(AgentHandler, LoggingAndHistoryMixin):
    """
    Subclass of AgentHandler for easy creation and handling of Rewardmod
    agents, which are a subclass of shared motor agents but which also
    use reward information to update parameters online.
    """
    def __init__(self, ale, nervous_system, update_rate, logging=False):
        super().__init__(ale, "rewardmod", {'nervous_system': nervous_system,
                                            'update_rate': update_rate,
                                            'logging': int(logging)})


class SoftMaxAgentHandler(AgentHandler, LoggingAndHistoryMixin):
    """
    Subclass of AgentHandler for easy creation and handling of SoftMaxAgentHandlers
    """
    def __init__(self, ale, nervous_system, update_rate, seed, logging=False):
        super().__init__(ale, "softmax", {'nervous_system': nervous_system,
                                           'update_rate': update_rate,
                                           'logging': int(logging),
                                           'seed': int(seed)})


class ALEHandler(Handler):
    """
    Handles the arcade-learning-environment. Starts up an ALE environment on
    creation.
    """
    installed_roms = generate_rom_dictionary()

    def __init__(self, rom, seed,
                 color_avg, max_num_frames, max_num_episodes,
                 max_num_frames_per_episode,
                 frame_skip=1,
                 print_screen=False,
                 display_screen=False,
                 sound=False,
                 system_reset_steps=4,
                 use_environment_distribution=True,
                 num_random_environments=30):
        """
        ALE parameters:
          rom - rom name (specify from list)
          seed - integer type
          display_screen - boolean type (default=False)
          frame_skip - number of frames to skip between NN evaluations (default 1)
          sound - boolean type (default=False)
          color_avg - boolean type (whether to average consecutive screens)
          max_num_frames - integer type
          max_num_episodes - integer type
          max_num_frames_per_episode - integer type
          print_screen - boolean type (default=False)
          system_reset_steps - int (default=4)
          use_environment_distribution - boolean (default True)
          num_random_environments - int (default 30)

        """

        if rom in ALEHandler.installed_roms:
            self.rom = rom
            self.rom_path = ALEHandler.installed_roms[rom]
        else:
            sys.exit("Error: " + rom + " is not installed.")

        super().__init__("ale", {'rom_path': self.rom_path,
                                 'seed': seed,
                                 'color_avg': color_avg,
                                 'frame_skip': frame_skip,
                                 'max_num_frames': max_num_frames,
                                 'max_num_episodes': max_num_episodes,
                                 'max_num_frames_per_episode': max_num_frames_per_episode,
                                 'print_screen': print_screen,
                                 'display_screen': display_screen,
                                 'sound': sound,
                                 'system_reset_steps': system_reset_steps,
                                 'use_environment_distribution': use_environment_distribution,
                                 'num_random_environments': num_random_environments})

    def set_parameters(self, **kwargs):
        """
        This function destroys the current handle object and makes a new ALE
        instance using updated parameter settings
        :param kwargs: any ale key word value pairs
        :return: None
        """

        # If the user decides to change the rom, we have to update
        # rom attributes, and add a rom_path
        if 'rom' in kwargs:
            if kwargs['rom'] in ALEHandler.installed_roms:
                self.rom = kwargs['rom']
                self.rom_path = ALEHandler.installed_roms[self.rom]
                kwargs.pop('rom')
                kwargs['rom_path'] = self.rom_path
            else:
                sys.exit("Error: " + kwargs['rom'] + " is not installed.")

        if self._handle_exists:
            self._handle_parameters.update(kwargs)
            self.create()
        else:
            self._handle_parameters.update(kwargs)

    def seed(self, integer):
        """
        This function destroys the current handle object and makes a new ALE
        instance using the new seed.
        :param integer: an integer value
        :return: None
        """

        if self._handle_exists:
            self._handle_parameters['seed'] = integer
            self.create()
        else:
            self._handle_parameters['seed'] = integer

    def create(self):
        # Create ALE handle
        self._handle = ale_generator.CreatALE(**self._handle_parameters)
        self._handle_exists = True

    def action_set_size(self):
        """
        :return: Minimal number of actions required to play the game
        """
        return ale_handler.NumOutputs(ale=self._handle)

    def legal_action_set_size(self):
        """
        :return: The number of legal acitons for the player, should be the same
            across games (18)
        """
        return ale_handler.LegalActionSetSize(ale=self._handle)

    @classmethod
    def print_available_roms(cls):
        print("Available roms:")
        roms = list(ALEHandler.installed_roms.keys())
        roms.sort(key=str.lower)
        for rom in roms:
            print("\t", rom)


if __name__ == '__main__':
    """
    example
    """
    pass
