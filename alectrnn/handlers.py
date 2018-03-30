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
from alectrnn import layer_generator
from alectrnn import nn_generator
from alectrnn import nn_handler
from alectrnn import ale_handler
from alectrnn import agent_handler
import sys
from functools import partial
from pkg_resources import resource_listdir
from pkg_resources import resource_filename
from enum import Enum
import numpy as np
import json


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
          objective_type - "totalcost"
          objective_parameters - dictionary of keyword arguments for objective

            For "totalcost": none
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
          agent_type - "ctrnn", "nervous_system"
          agent_parameters - dictionary of keyword arguments for the agent

            For "ctrnn": (num_neurons, num_sensor_neurons,
                          input_screen_width, input_screen_height,
                          use_color, step_size)
            For "nervous_system": (nervous_system, update_rate, logging)
        """
        if agent_parameters is None:
            agent_parameters = {}

        super().__init__(agent_type, agent_parameters)
        self._ale = ale

    def create(self):
        """
        Creates the handle:
        Currently supports "ctrnn" and "nervous_system"
        """
        # Create Agent handle
        if self._handle_type == "ctrnn":
            self._handle = agent_generator.CreateCtrnnAgent(self._ale,
                                                        **self._handle_parameters)
        elif self._handle_type == "nervous_system":
            self._handle = agent_generator.CreateNervousSystemAgent(self._ale,
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


class NervousSystemAgentHandler(AgentHandler):
    """
    Subclass of AgentHandler for easy creation and handling of NervousSystem
    agents, which have several distinct methods (e.g. layer_history and
    screen_history).
    """
    def __init__(self, ale, nervous_system, update_rate, logging=False):
        super().__init__(ale, "nervous_system", {'nervous_system': nervous_system, 
                                                 'update_rate': update_rate,
                                                 'logging': int(logging)})

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


class ALEHandler(Handler):
    """
    Handles the arcade-learning-environment. Starts up an ALE environment on
    creation.
    """
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
        super().__init__(None, None)
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

    def seed(self, integer):
        """
        This function destroys the current handle object and makes a new ALE
        instance using the new seed.
        :param integer: an integer value
        :return: None
        """

        if self._handle_exists:
            self.ale_seed = integer
            self.create()
        else:
            self.ale_seed = integer

    def create(self):
        # Create ALE handle
        self._handle = ale_generator.CreatALE(
            rom=self.rom_path, 
            seed=self.ale_seed, 
            display_screen=self.display_screen, 
            sound=self.sound,
            color_avg=self.color_avg, 
            max_num_frames=self.max_num_frames,
            max_num_episodes=self.max_num_episodes,
            max_num_frames_per_episode=self.max_num_frames_per_episode,
            print_screen=self.print_screen)
        self._handle_exists = True

    def action_set_size(self):
        return ale_handler.NumOutputs(ale=self._handle)

    @classmethod
    def print_available_roms(cls):
        print("Available roms:")
        roms = list(ALEHandler.installed_roms.keys())
        roms.sort(key=str.lower)
        for rom in roms:
            print("\t", rom)


def load_ALEHandler(json_filename):
    """
    creates an ale handler from parameters in the json file
    json file should be created from python dumps that will be read in
    as a dictionary with expected keys.
    Top level key should be ale_parameters
    :param json_filename: a json filename to be loaded
    :return: an ale_handler
    """

    configuration = json.load(open(json_filename, 'r'))
    return ALEHandler(**configuration['ale_parameters'])


def load_ObjectiveHandler(json_filename):
    """
    creates an objective handler from parameters in the json file
    json file should be created from python dumps and will expect a top
    level key of objective_parameters
    :param json_filename: a json filename to be loaded
    :return: an objective handler
    """

    configuration = json.load(open(json_filename, 'r'))
    return ObjectiveHandler(**configuration['objective_parameters'])


def load_AgentHandler(json_filename):
    """
    creates an agent handler from parameters in the json file. The json
    file should be created from python dumps and will expect a top level
    key of agent_parameters
    :param json_filename: a json filename to be loaded
    :return: an agent handler
    """

    configuration = json.load(open(json_filename, 'r'))
    raise NotImplementedError


def load_game_configuration(json_filename):
    """
    :param json_filename:
    :return:
    """
    # TODO: Handlers need other handlers, so we need to add those arguments
    # to the agent and objective above. Also add the NN handler
    # but we won't use those directly, instead use this function below
    # it will load everything in the correct order
    raise NotImplementedError


class PARAMETER_TYPE(Enum):
    """
    Class for specifying parameter types. Should match those in the C++ code:
    parameter_types.hpp
    """
    __order__ = 'BIAS RTAUS WEIGHT RANGE REFRACTORY RESISTANCE'
    BIAS = 0
    RTAUS = 1
    WEIGHT = 2
    RANGE = 3  # difference between reset value and threshold for IF models
    REFRACTORY = 4  # refractory period for IF models
    RESISTANCE = 5  # resistance for IF models


def draw_uniform_initial_guess(rng, boundary_array):
    """
    Uses a numpy RandomState to draw a uniform set of initial values between
    the specified bounds.
    :param rng: numpy RandomState object
    :param boundary_array: Nx2 np.float32 array
    :return: 1-D array of size N and dtype np.float32
    """

    initial_guess = np.zeros(len(boundary_array), dtype=np.float32)
    for iii, (low, high) in enumerate(boundary_array):
        initial_guess[iii] = rng.uniform(low, high)

    return initial_guess


def boundary_array_for_parameter_layout(parameter_layout, type_bounds):
    """
    Creates a np array with the bounds for a given parameter layout.
    :param parameter_layout: 1D array with # parameter elements coded as PARAMETER_TYPE (int)
    :param type_bounds: key is PARAMETER_TYPE, value is (low, high)
    :return: a Nx2 np.float32 array
    """
    if coverage_is_complete(parameter_layout, type_bounds):
        boundaries = np.zeros((len(parameter_layout), 2), dtype=np.float32)
        for par_type, bounds in type_bounds.items():
            type_indexes = np.where(parameter_layout == par_type.value)
            boundaries[type_indexes,0] = bounds[0]
            boundaries[type_indexes,1] = bounds[1]
        return boundaries

    else:
        raise AttributeError("Error: type_bounds does not fully cover the " +
                             "types of parameters in layout or invalid type.")


def coverage_is_complete(parameter_layout, type_bounds):
    """
    Returns True if all types in layout are present in type_bounds

    Returns False if type is not a valid PARAMETER_TYPE or if type_bounds
    doesn't contain all the necessary PARAMETER_TYPEs for the layout.
    """

    for parameter in parameter_layout:
        try: # Raises value error if conversion doesn't go through
            par = PARAMETER_TYPE(parameter)
        except:
            return False

        if par not in type_bounds:
            return False

    return True


class ACTIVATOR_TYPE(Enum):
    """
    Class for distinguishing the various types of neural activators
    Should keep in sync with ACTIVATOR_TYPE in activator.hpp
    """
    BASE=0
    IDENTITY=1
    CTRNN=2
    CONV_CTRNN=3
    IAF=4
    CONV_IAF=5


class INTEGRATOR_TYPE(Enum):
    """
    Class for distinguishing the various types of neural integrators
    Should keep in sync with INTEGRATOR_TYPE in integrator.hpp
    """
    BASE=0
    NONE=1
    ALL2ALL=2
    CONV=3
    RECURRENT=4
    RESERVOIR=5


ACTMAP = { ACTIVATOR_TYPE.IAF : ACTIVATOR_TYPE.CONV_IAF,
           ACTIVATOR_TYPE.CTRNN : ACTIVATOR_TYPE.CONV_CTRNN }


class NervousSystem:
    """
    Builds a nervous system. A single activator is chosen for the whole network.
    input_shape determines the dimensions of the input the network will receive.
    num_outputs will determine motor_layer size.

    Both input and output layers will be generated automatically.
    nn_parameters should contain parameters for the internal layers of the
    network. It should be a list of dictionaries.

    Activator type and arguments should be the base type and args. For example,
    act_type = ACTIVATION_TYPE.CTRNN will need act_args = tuple(float(step_size)).
    Shape arguments and activation types for CONV layers will be added 
    automatically.

    Activator types:
        ACTIVATION_TYPE.CTRNN: (float(step_size),)
        ACTIVATION_TYPE.IAF: (float(step_size), float(peak), float(reset))

    Convolutional layers should have a dictionary with the following keys:
        'layer_type' = "conv"
        'filter_shape' = 2-element list/array with filter dimensions
        'num_filters'
        'stride'

    Reservoir layers have an untrained set of connections determined the
    input graph. The back connections are determined by the input graph.
    Both must specify weights. Should have a dictionary with the following keys:
        'layer_type' = "reservoir"
        'num_input_graph_nodes' = N
        'input_graph' = Nx2, dtype=np.uint64 bipartite edge graph
        'input_weights' = N element, dtype=np.float32
        'num_internal_nodes' = M
        'internal_graph' = Mx2, dtype=np.uint64 edge array
        'internal_weights' = M element, dtype=np.float32 array

    Recurrent layers define the connections, but not the weights of the internal
    and back connections. All connections are trained.
    Should have a dictionary with the following keys:
        'layer_type' = "recurrent"
        'num_input_graph_nodes' = N
        'input_graph' = Nx2, dtype=np.uint64 bipartite edge graph
        'num_internal_nodes' = M
        'internal_graph' = Mx2, dtype=np.uint64 edge array

    conv_recurrent layers have a convolutional input into the layer, but also
    have internal connections determined in the same way as the recurrent layer.
    Should have a dictionary with the following keys:
        'layer_type' = "conv_recurrent"
        'filter_shape'
        'num_filters'
        'stride'
        'num_internal_nodes'
        'internal_graph'

    conv_reservoir layers have a convolutional input into the layer, but also
    have internal connections determined in the same way as the reservoir layer.
    Should have a dictionary with the following keys:
        'layer_type' = "conv_reservoir"
        'filter_shape'
        'num_filters'
        'stride'
        'num_internal_nodes'
        'internal_graph'
        'internal_weights'

    a2a_recurrent layers have full input from the previous layer, but internal
    connections are determined in the same way as the recurrent layer.
    Should have a dictionary with the following keys:
        'layer_type' = "a2a_recurrent"
        'num_internal_nodes'
        'internal_graph'

    a2a_a2a layer has fully connected back and internal connections.
    Should have a dictionary with the following keys:
        'layer_type' = "a2a_a2a"
        'num_internal_nodes'

    a2a_ff layer has fully connected back connections and no internal connections.
    It is the same as a standard feed-forward layer.
    Should have a dictionary with the following keys:
        'layer_type' = "a2a_ff"
        'num_internal_nodes'

    conv_recurrent layers allow the construction of layers that have a conv back
    connection, but recurrent internal connection.

    Valid back_integrator types:
        INTEGRATOR_TYPE.CONV : args(filter-shape, # filters, stride)

    In the bipartite graph, all tail edges will be interpreted as belonging
    to the corresponding index state of the previous layer, while all heads will
    be interpreted as index states of the current layer.

    The internal graph, all heads and tails will be interpreted as belongining
    to the current layer.

    All node IDs should start at 0 and correspond to the state index.

    No parameters are saved, as this is a waste of space. All layers created and
    necessary arguments taken are copied and ownership is transfered to the
    NN.

    Reference:

        back_connections:
            INTEGRATOR_TYPE.RECURRENT - (#nodes, edge_list)[bipartite]
            INTEGRATOR_TYPE.RESERVOIR - (#nodes, edge_list), (weights)
            INTEGRATOR_TYPE.ALL2ALL

        self_connections:
            INTEGRATOR_TYPE.RECURRENT - (#nodes, edge_list)[graph]
            INTEGRATOR_TYPE.RESERVOIR - (#nodes, edge_list), (weights)
            INTEGRATOR_TYPE.ALL2ALL

        activator:
            ACTIVATOR_TYPE.CTRNN - (step_size)
    """

    def __init__(self, input_shape, num_outputs, nn_parameters,
                 act_type, act_args):
        """
        input_shape - shape of input into the NN (should be 3D) 1st dim is
                      channels, second is height, then width. Will be cast
                      as dtype np.uint64 automatically
        num_outputs - the size of the motor layer
        nn_parameters - list of dictionaries. One for each layer.
        Input layers are added by the NervousSystem itself by default.

        act_type = general ACTIVATION_TYPE for model
        act_args = general arguments for model

        CONV layers usually have additional arguments, like shape, for 
        parameter sharing of act_args. They also have their own ACTIVATION_TYPE.
        These are automatically added to the CONV layers.
        """
        input_shape = np.array(input_shape, dtype=np.uint64)
        layers = []
        # interpreted shapes are for some back integrators which need
        # to know how to interpret the layer for convolution
        interpreted_shapes, layer_shapes = self._configure_layer_shapes(
                                            input_shape, nn_parameters)
        layer_act_types, layer_act_args = self._configure_layer_activations(
                                            layer_shapes, interpreted_shapes,
                                            nn_parameters,
                                            act_type, act_args)

        for i, layer_pars in enumerate(nn_parameters):
            if layer_pars['layer_type'] == "conv":
                layers.append(self._create_conv_layer(
                    interpreted_shapes[i], 
                    interpreted_shapes[i+1],
                    layer_pars['filter_shape'], 
                    layer_pars['stride'], 
                    layer_act_types[i], 
                    layer_act_args[i]))

            elif layer_pars['layer_type'] == "recurrent":
                layers.append(self._create_recurrent_layer(
                    layer_pars['num_input_graph_nodes'], 
                    layer_pars['input_graph'],
                    layer_pars['num_internal_nodes'],
                    layer_pars['internal_graph'], 
                    layer_act_types[i], 
                    layer_act_args[i],
                    layer_shapes[i+1]))

            elif layer_pars['layer_type'] == "reservoir":
                layers.append(self._create_reservoir_layer(
                    layer_pars['num_input_graph_nodes'], 
                    layer_pars['input_graph'],
                    layer_pars['input_weights'], 
                    layer_pars['num_internal_nodes'],
                    layer_pars['internal_graph'],
                    layer_pars['internal_weights'], 
                    layer_act_types[i], 
                    layer_act_args[i],
                    layer_shapes[i+1]))

            elif layer_pars['layer_type'] == "conv_recurrent":
                layers.append(self._create_conv_recurrent_layer(
                    interpreted_shapes[i],
                    interpreted_shapes[i+1],
                    layer_pars['filter_shape'],
                    layer_pars['stride'],
                    layer_pars['num_internal_nodes'],
                    layer_pars['internal_graph'], 
                    layer_act_types[i], 
                    layer_act_args[i],
                    layer_shapes[i+1]))

            elif layer_pars['layer_type'] == "conv_reservoir":
                layers.append(self._create_conv_reservoir_layer(
                    interpreted_shapes[i],
                    interpreted_shapes[i+1],
                    layer_pars['filter_shape'],
                    layer_pars['stride'],
                    layer_pars['num_internal_nodes'],
                    layer_pars['internal_graph'],
                    layer_pars['internal_weights'],
                    layer_act_types[i], 
                    layer_act_args[i],
                    layer_shapes[i+1]))

            elif layer_pars['layer_type'] == 'a2a_recurrent':
                layers.append(self._create_a2a_recurrent_layer(
                    layer_shapes[i],
                    layer_pars['num_internal_nodes'],
                    layer_pars['internal_graph'],
                    layer_act_types[i], 
                    layer_act_args[i],
                    layer_shapes[i+1]))

            elif layer_pars['layer_type'] == 'a2a_a2a':
                layers.append(self._create_a2a_a2a_layer(
                    layer_shapes[i],
                    layer_pars['num_internal_nodes'],
                    layer_act_types[i], 
                    layer_act_args[i],
                    layer_shapes[i+1]))

            elif layer_pars['layer_type'] == 'a2a_ff':
                layers.append(self._create_a2a_ff_layer(
                    layer_shapes[i],
                    layer_pars['num_internal_nodes'],
                    layer_act_types[i],
                    layer_act_args[i],
                    layer_shapes[i+1]))
            else:
                raise NotImplementedError

        # Build motor later
        prev_layer_size = int(np.prod(layer_shapes[-1]))
        layers.append(self._create_motor_layer(num_outputs,
                                               prev_layer_size,
                                               act_type.value,
                                               (num_outputs, *act_args)))

        # Generate NN
        self.neural_network = nn_generator.CreateNervousSystem(input_shape,
            tuple(layers))

    def _configure_layer_activations(self, layer_shapes, interpreted_shapes,
                                     nn_parameters, act_type, act_args):
        """
        outputs the necessary tuples for layer activations of both conv and 
        non-conv layers
        layer_shapes includes input layer, so i+1 is synced with nn_params
        no nn_params are included for input layer
        """

        layer_act_types = []
        layer_act_args = []
        for i, layer_pars in enumerate(nn_parameters):
            if layer_pars['layer_type'] == 'conv':
                layer_act_types.append(ACTMAP[act_type].value)
                layer_act_args.append((interpreted_shapes[i+1], *act_args))
            else:
                layer_act_types.append(act_type.value)
                layer_act_args.append((int(np.prod(layer_shapes[i+1])), *act_args))

        return layer_act_types, layer_act_args

    def _configure_layer_shapes(self, input_shape, nn_parameters):
        """
        For a given activation type it sets an internal attribute that
        can be used by the other layer creation functions

        for conv layers the interpreted_shape will take into account the
        'num_filters' parameter
        """

        interpreted_shapes = [input_shape]
        layer_shapes = [input_shape]
        for i, layer_pars in enumerate(nn_parameters):
            if 'conv' in layer_pars['layer_type']:
                interpreted_shapes.append(calc_conv_layer_shape(interpreted_shapes[i], 
                                layer_pars['num_filters'], layer_pars['stride']))
                if 'num_internal_nodes' in layer_pars:
                    interpreted_size = np.prod(interpreted_shapes[-1])
                    if layer_pars['num_internal_nodes'] < interpreted_size:
                        print("Warning: Layer defined by:", layer_pars,
                              "requires more neurons to match interpreted shape"
                              " setting to interpreted size."
                              " Integration may result in disconnected"
                              " components.")
                    layer_shapes.append(np.array([
                        max(layer_pars['num_internal_nodes'], 
                            interpreted_size)],
                        dtype=np.uint64))
                else:
                    layer_shapes.append(np.array([np.prod(interpreted_shapes[-1])],
                                    dtype=np.uint64))
            else:
                interpreted_shapes.append(np.array([layer_pars['num_internal_nodes']],
                                    dtype=np.uint64))
                layer_shapes.append(np.array([layer_pars['num_internal_nodes']],
                                    dtype=np.uint64))

        return interpreted_shapes, layer_shapes

    def _create_a2a_recurrent_layer(self, prev_layer_shape, num_internal_nodes, 
                             internal_edge_array, act_type, act_args,
                             layer_shape):
        """
        Creates a recurrent layer with internal connections defined by
        the input graph and all-to-all connections with the previous layer.
        Restructures input parameters into the correct format for the
        C++ function call, then calls the CreateLayer function.
        :param prev_layer_shape: shape of the previous layer
        :param num_internal_nodes: number of neurons in layer
        :param internal_edge_array: array for self-connections
        :param act_type: ACTIVATOR_TYPE
        :param act_args: arguments for that ACTIVATOR_TYPE
        :param layer_shape: shape of the layer
        :return: python capsule with pointer to the layer
        """
        back_type = INTEGRATOR_TYPE.ALL2ALL.value
        back_args = (int(num_internal_nodes),
                     int(np.prod(prev_layer_shape)))
        self_type = INTEGRATOR_TYPE.RECURRENT.value
        self_args = (int(num_internal_nodes),
                     internal_edge_array)
        assert(act_args[0] == num_internal_nodes)
        return layer_generator.CreateLayer(back_type, back_args, self_type,
            self_args, act_type, act_args, layer_shape)

    def _create_a2a_a2a_layer(self, prev_layer_shape, num_internal_nodes, 
                            act_type, act_args, layer_shape):
        """
        Creates a layer with all-to-all internal and back connections.
        Restructures input parameters into the correct format for the
        C++ function call, then calls the function.
        :param prev_layer_shape: shape of the previous layer
        :param num_internal_nodes: number of neurons in layer
        :param act_type: ACTIVATOR_TYPE
        :param act_args: arguments for that ACTIVATOR_TYPE
        :param layer_shape: shape of the layer
        :return: python capsule with pointer to the layer
        """
        back_type = INTEGRATOR_TYPE.ALL2ALL.value
        back_args = (int(num_internal_nodes),
                     int(np.prod(prev_layer_shape)))
        self_type = INTEGRATOR_TYPE.ALL2ALL.value
        self_args = (int(num_internal_nodes), 
                     int(num_internal_nodes))
        assert(act_args[0] == num_internal_nodes)
        return layer_generator.CreateLayer(back_type, back_args, self_type,
            self_args, act_type, act_args, layer_shape)

    def _create_a2a_ff_layer(self, prev_layer_shape, num_internal_nodes,
                             act_type, act_args, layer_shape):
        """
        Creates a layer with all-to-all back connections, but not internal
        connections.
        Restructures input parameters into the correct format for the
        C++ function call, then calls the function.
        :param prev_layer_shape: shape of the previous layer
        :param num_internal_nodes: number of neurons in layer
        :param act_type: ACTIVATOR_TYPE
        :param act_args: arguments for that ACTIVATOR_TYPE
        :param layer_shape: shape of the layer
        :return: python capsule with pointer to the layer
        """
        back_type = INTEGRATOR_TYPE.ALL2ALL.value
        back_args = (int(num_internal_nodes),
                     int(np.prod(prev_layer_shape)))
        self_type = INTEGRATOR_TYPE.NONE.value
        self_args = tuple()
        assert(act_args[0] == num_internal_nodes)
        return layer_generator.CreateLayer(back_type, back_args, self_type,
                                self_args, act_type, act_args, layer_shape)

    def _create_conv_recurrent_layer(self, prev_layer_shape, interpreted_shape,
            filter_shape, stride, num_internal_nodes,
            internal_edge_array, act_type, act_args, layer_shape):
        """
        Creates a layer with convolutional back connections and self-connections
        defined by the input graph.
        Restructures input parameters into the correct format for the
        C++ function call, then calls the CreateLayer function.
        :param prev_layer_shape: shape of the previous layer
        :param interpreted_shape: shape the convolution will take
        :param filter_shape: shape of the filter
        :param stride: stride for the convolution
        :param num_internal_nodes: number of neurons in layer
        :param internal_edge_array: array for self-connections
        :param act_type: ACTIVATOR_TYPE
        :param act_args: arguments for that ACTIVATOR_TYPE
        :param layer_shape: shape of the layer
        :return: python capsule with pointer to the layer
        """
        back_type = INTEGRATOR_TYPE.CONV.value
        back_args = (np.array([prev_layer_shape[0]] + list(filter_shape), dtype=np.uint64), 
                     interpreted_shape, #layer_shape funct outputs dtype=np.uint64
                     np.array(prev_layer_shape, dtype=np.uint64),
                     int(stride))
        
        self_type = INTEGRATOR_TYPE.RECURRENT.value
        self_args = (int(num_internal_nodes),
                     internal_edge_array)

        return layer_generator.CreateLayer(back_type, back_args, self_type,
            self_args, act_type, act_args, layer_shape)

    def _create_conv_reservoir_layer(self, prev_layer_shape, interpreted_shape,
            filter_shape, stride, num_internal_nodes,
            internal_edge_array, internal_weight_array, act_type, act_args,
            layer_shape):
        """
        Creates a layer with convolutional back connections and a randomly
        connected and weight internal connections.
        Restructures input parameters into the correct format for the
        C++ function call, then calls the CreateLayer function.
        Restructures input parameters into the correct format for the
        C++ function call, then calls the CreateLayer function.
        :param prev_layer_shape: shape of the previous layer
        :param interpreted_shape: shape the convolution will take
        :param filter_shape: shape of the filter
        :param stride: stride for the convolution
        :param num_internal_nodes: number of neurons in layer
        :param internal_edge_array: array for self-connections
        :param internal_weight_array: weights for the graph
        :param act_type: ACTIVATOR_TYPE
        :param act_args: arguments for that ACTIVATOR_TYPE
        :param layer_shape: shape of the layer
        :return: python capsule with pointer to the layer
        """

        back_type = INTEGRATOR_TYPE.CONV.value
        back_args = (np.array([prev_layer_shape[0]] + list(filter_shape), dtype=np.uint64), 
                     interpreted_shape, # layer_shape funct outputs dtype=np.uint64
                     np.array(prev_layer_shape, dtype=np.uint64),
                     int(stride))
        self_type = INTEGRATOR_TYPE.RESERVOIR.value
        self_args = (int(num_internal_nodes),
                     internal_edge_array,
                     internal_weight_array)

        return layer_generator.CreateLayer(back_type, back_args, self_type,
            self_args, act_type, act_args, layer_shape)

    def _create_recurrent_layer(self, num_bipartite_nodes,
            bipartite_input_edge_array, num_internal_nodes,
            internal_edge_array, act_type, act_args, layer_shape):
        """
        Creates a recurrent layer with graphs specifying back and self
        connections.
        Restructures input parameters into the correct format for the
        C++ function call, then calls the CreateLayer function.
        :param num_bipartite_nodes: number of nodes in bipartite graph
        :param bipartite_input_edge_array: array for back-connections (Nx2)
            dtype=np.uint64
        :param num_internal_nodes: number of neurons in layer
        :param internal_edge_array: array for self-connections (Nx2)
            dtype=np.uint64
        :param act_type: ACTIVATOR_TYPE
        :param act_args: arguments for that ACTIVATOR_TYPE
        :param layer_shape: shape of the layer
        :return: python capsule with pointer to the layer
        """
        back_type = INTEGRATOR_TYPE.RECURRENT.value
        back_args = (int(num_bipartite_nodes),
                    bipartite_input_edge_array)
        self_type = INTEGRATOR_TYPE.RECURRENT.value
        self_args = (int(num_internal_nodes),
                    internal_edge_array)
        assert(act_args[0] == num_internal_nodes)
        return layer_generator.CreateLayer(back_type,
            back_args, self_type, self_args, act_type, act_args, layer_shape)

    def _create_reservoir_layer(self, num_bipartite_nodes,
            bipartite_input_edge_array, input_weights, num_internal_nodes,
            internal_edge_array, internal_weight_array, act_type, act_args,
            layer_shape):
        """
        Layer with internal and back connection and weights specified by
        the input graphs.
        Restructures input parameters into the correct format for the
        C++ function call, then calls the CreateLayer function.
        :param num_bipartite_nodes: number of nodes in bigraph
        :param bipartite_input_edge_array: Nx2 dtype=np.uint64 graph
        :param input_weights: N dtype=np.float32 array
        :param num_internal_nodes: number of neurons in layer
        :param internal_edge_array: array for self-connections Nx2 np.uint64
        :param internal_weight_array: weights for graph N dtype=np.float32
        :param act_type: ACTIVATOR_TYPE
        :param act_args: arguments for that ACTIVATOR_TYPE
        :param layer_shape: shape of the layer
        :return: python capsule with pointer to the layer
        """
        back_type = INTEGRATOR_TYPE.RESERVOIR.value
        back_args = (int(num_bipartite_nodes),
                     bipartite_input_edge_array,
                     input_weights)
        self_type = INTEGRATOR_TYPE.RESERVOIR.value
        self_args = (int(num_internal_nodes),
                     internal_edge_array,
                     internal_weight_array)
        assert(act_args[0] == num_internal_nodes)
        return layer_generator.CreateLayer(back_type,
            back_args, self_type, self_args, act_type, act_args, layer_shape)

    def _create_conv_layer(self, prev_layer_shape, interpreted_shape,
                           filter_shape, stride, act_type, act_args):
        """
        Creates a layer with convolutional back connections and no self
        connections.
        act_args needs to be in the correct tuple format with properly
        typed numpy arrays for any arguments
        filter_shape = 2 element array/list with filter dimenstions
        Final shape, which includes depth, depends on shape of prev layer
        Restructures input parameters into the correct format for the
        C++ function call, then calls the CreateLayer function.
        :param prev_layer_shape: shape of the previous layer
        :param interpreted_shape: shape the convolution will take
        :param filter_shape: shape of the filter
        :param stride: stride for the convolution
        :param act_type: ACTIVATOR_TYPE
        :param act_args: arguments for that ACTIVATOR_TYPE
        :return: python capsule with pointer to the layer
        """

        back_type = INTEGRATOR_TYPE.CONV.value
        # Appropriate depth is added to filter shape to build the # 3-element 1D array
        back_args = (np.array([prev_layer_shape[0]] + list(filter_shape), dtype=np.uint64), 
                     interpreted_shape, # layer_shape funct outputs dtype=np.uint64
                     np.array(prev_layer_shape, dtype=np.uint64),
                     int(stride))
        self_type = INTEGRATOR_TYPE.NONE.value
        self_args = tuple()
        return layer_generator.CreateLayer(back_type,
            back_args, self_type, self_args, act_type, act_args, interpreted_shape)

    def _create_motor_layer(self, num_outputs, size_of_prev_layer, act_type, act_args):
        """
        Generates a motor layer for the neural network, which will represent
        the output of the network. It is fully connected to whatever layer
        precedes it.
        act_args needs to be in the correct tuple format for the activator
        """

        assert(act_args[0] == num_outputs)
        return layer_generator.CreateMotorLayer(
            int(num_outputs), int(size_of_prev_layer), act_type, act_args)

    def get_parameter_count(self):
        """
        Returns the number of parameters needed to configure the NN
        """

        return nn_handler.GetParameterCount(self.neural_network)

    def num_layers(self):
        """
        Returns the number of parameters needed to configure the NN
        """

        return nn_handler.GetSize(self.neural_network)
        
    def parameter_layout(self):
        """
        Returns an np_int array with PARAMETER_TYPE codes
        """

        return nn_handler.GetParameterLayout(self.neural_network)


def calc_conv_layer_shape(prev_layer_shape, num_filters, stride):
    """
    Determines layer shape given previous layer shape and the number of chosen
    filters and stride for the next layer.
    """

    return np.array([num_filters] + calc_image_dimensions(
                    prev_layer_shape[1:], stride), dtype=np.uint64)


def calc_image_dimensions(prev_image_dimensions, stride):
    """
    Determine new layer window based on previous window and stride.
    This is not the same as shape, as the first dimension (depth) is
    excluded, as that is determined by the # of filters.
    """

    return [ calc_num_pixels(image_dimension, stride) 
                for image_dimension in prev_image_dimensions ]


def calc_num_pixels(num_pixels, stride):
    """
    Converts the current number of pixels to the number there will be given
    a specific stride.
    """

    return 1 + (num_pixels - 1) // stride


class SimpleCTRNN(NervousSystem):
    """
    A CTRNN with all-2-all input to network connections and all-2-all network
    to motor layer connections
    """
    def __init__(self, input_shape, num_outputs, num_neurons, step_size):
        """
        :param input_shape: shape of input into the NN (should be 3D) 1st dim is
            channels, second is height, then width. Will be cast
            as dtype np.uint64 automatically.
        :param num_outputs: number of neural outputs, for ALE this will be the
            # controller inputs
        :param num_neurons: size of the network
        :param step_size: integration step size during activation
        """
        nn_parameters = [{
            'layer_type' : "a2a_a2a",
            'num_internal_nodes': num_neurons}]
        act_type = ACTIVATOR_TYPE.CTRNN
        act_args = (float(step_size),)
        super().__init__(input_shape, num_outputs, nn_parameters, act_type, act_args)


if __name__ == '__main__':
    """
    example
    """
    pass