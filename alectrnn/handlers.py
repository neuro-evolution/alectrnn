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
    roms_list = resource_listdir("alectrnn", "roms")
    rom_path_list = [ resource_filename("alectrnn", "roms/" + rom) 
                        for rom in roms_list if ".bin" in rom ]
    rom_name_list = [ rom[:-4] for rom in roms_list if ".bin" in rom ]
    return { rom_name_list[i] : rom_path_list[i] 
            for i in range(len(rom_path_list)) }


class Handler:

    def __init__(self, handle_type, handle_parameters=None):

        if handle_parameters is None:
            handle_parameters = {}

        self.handle_type = handle_type
        self.handle_parameters = handle_parameters
        self.handle = None

    def create(self):
        raise NotImplementedError

    def get(self):
        return self.handle


class ObjectiveHandler(Handler):

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
        self.ale = ale
        self.agent = agent

    def create(self):
        if self.handle_type == "totalcost":
            self.handle = partial(objective.TotalCostObjective, 
                                  ale=self.ale, agent=self.agent)


class AgentHandler(Handler):

    def __init__(self, ale, agent_type, agent_parameters=None):
        """
        Agent parameters:
          agent_type - "ctrnn"
          agent_parameters - dictionary of keyword arguments for the agent

            For "ctrnn": (num_neurons, num_sensor_neurons,
                          input_screen_width, input_screen_height,
                          use_color, step_size)
        """
        if agent_parameters is None:
            agent_parameters = {}

        super().__init__(agent_type, agent_parameters)
        self.ale = ale

    def create(self):
        # Create Agent handle
        if self.handle_type == "ctrnn":
            self.handle = agent_generator.CreateCtrnnAgent(self.ale, 
                                                        **self.handle_parameters)
        elif self.handle_type == "nervous_system":
            self.handle = agent_generator.CreateNervousSystemAgent(self.ale, 
                                                        **self.handle_parameters)
        else:
            sys.exit("No agent by that name is implemented")


class NervousSystemAgentHandler(AgentHandler):

    def __init__(self, ale, nervous_system, update_rate, logging=False):
        self._logging = logging
        super().__init__(ale, "nervous_system", {'nervous_system': nervous_system, 
                                                 'update_rate': update_rate,
                                                 'logging': int(logging)})

    def layer_history(self, layer_index):
        """
        Returns a numpy array with dimensions equal to the layer dimensions
        and # elements = # states in that layer.
        It will be of dtype=np.float32
        """
        if self._logging:
            return agent_handler.GetLayerHistory(self.handle, layer_index)
        else:
            raise AssertionError("Error: Logging not active, no history table")


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

    def create(self):
        # Create ALE handle
        self.handle = ale_generator.CreatALE(
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
        return ale_handler.NumOutputs(ale=self.handle)

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
    return AgentHandler()

######### Handlers need other handlers, so we need to add those arguments
####### to the agent and objective above. Also add the NN handler
##### but we won't use those directly, instead use this function below
#### it will load everything in the correct order
def load_game_configuration(json_filename):
    """

    :param json_filename:
    :return:
    """
    pass

# Should keep in sync with PARAMETER_TYPE in parameter_types.hpp
class PARAMETER_TYPE(Enum):
    BIAS=0
    RTAUS=1
    WEIGHT=2


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


# Should keep in sync with ACTIVATOR_TYPE in activator.hpp
class ACTIVATOR_TYPE(Enum):
    BASE=0
    IDENTITY=1
    CTRNN=2
    CONV_CTRNN=3
    IAF=4
    CONV_IAF=5


# Should keep in sync with INTEGRATOR_TYPE in integrator.hpp
class INTEGRATOR_TYPE(Enum):
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
                    layer_act_args[i],
                    layer_shapes[i+1]))

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
        All graphs should be Nx2 with dtype=np.uint64
        act_args should be in the proper tuple format for inputs into the
        activator function
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
        All graphs should be Nx2 with dtype=np.uint64
        All weights should be Nx1 with dtype=np.float32
        act_args should be in the proper tuple format for inputs into the
        activator function
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
                           filter_shape, stride, act_type, act_args, layer_shape):
        """
        act_args needs to be in the correct tuple format with properly
        typed numpy arrays for any arguments

        filter_shape = 2 element array/list with filter dimenstions
        Final shape, which includes depth, depends on shape of prev layer
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