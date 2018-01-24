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
import alectrnn.agent_handler
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
            self.handle = partial(objective.TotalCostObjective, 
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
            self.handle = agent_generator.CreateCtrnnAgent(self.ale, 
                                                        **self.handle_parameters)
        elif self.handle_type == "nervous_system":
            self.handle = agent_generator.CreateNervousSystemAgent(self.ale, 
                                                        **self.handle_parameters)
        else:
            sys.exit("No agent by that name is implemented")

class NervousSystemAgentHandler(AgentHandler):

    def __init__(self, ale, nervous_system, update_rate):
        super(ale, "nervous_system", {'nervous_system': nervous_system, 
                                 'update_rate': update_rate})

    def create(self):
        self.handle = agent_generator.CreateNervousSystemAgent(self.ale,
                                                    **self.handle_parameters)
    def layer_history(self, layer_index):
        """
        Returns a numpy array with dimensions equal to the layer dimensions
        and # elements = # states in that layer.
        It will be of dtype=float32
        """
        return agent_handler.GetLayerHistory(self.handle, layer_index)

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
    CONV_IAF=5

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
    act_type = ACTIVATION_TYPE.CTRNN will need act_args = tuple(int(step_size)).
    Shape arguments and activation types for CONV layers will be added 
    automatically.

    Convolutional layers should have a dictionary with the following keys:
        'layer_type' = "conv"
        'filter_shape' = 2-element list/array with filter dimensions
        'num_filters'
        'stride'

    Reservoir layers should have a dictionary with the following keys:
        'layer_type' = "reservoir"
        'num_input_graph_nodes' = N
        'input_graph' = Nx2, dtype=uint64 bipartite edge graph
        'input_weights' = Nx1, dtype=float32
        'num_internal_nodes' = M
        'internal_graph' = Mx2, dtype=uint64 edge array
        'internal_weights' = Mx1, dtype=float32 array

    Recurrent layers should have a dictionary with the following keys:
        'layer_type' = "recurrent"
        'num_input_graph_nodes' = N
        'input_graph' = Nx2, dtype=uint64 bipartite edge graph
        'num_internal_nodes' = M
        'internal_graph' = Mx2, dtype=uint64 edge array

    conv_recurrent layers should have a dictionary with the following keys:
        'layer_type' = "conv_recurrent"
        'filter_shape'
        'num_filters'
        'stride'
        'num_internal_nodes'
        'internal_graph'

    conv_reservoir layers should have a dictionary with the following keys:
        'layer_type' = "conv_reservoir"
        'filter_shape'
        'num_filters'
        'stride'
        'num_internal_nodes'
        'internal_graph'
        'internal_weights'

    a2a_recurrent layers should have a dictionary with the following keys:
        'layer_type' = "a2a_recurrent"
        'num_internal_nodes'
        'internal_graph'

    a2a_a2a layer should have a dictionary with the following keys:
        'layer_type' = "a2a_a2a"
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
        nn_parameters - list of dictionaries. One for each layer.
        Input layers are added by the NervousSystem itself by default.

        act_type = general ACTIVATION_TYPE for model
        act_args = general arguments for model

        CONV layers usually have additional arguments, like shape, for 
        parameter sharing of act_args. They also have their own ACTIVATION_TYPE.
        These are automatically added to the CONV layers.
        """
        input_shape = np.array(input_shape, dtype=uint64)
        layers = []
        layer_shapes = self.configure_layer_shapes(input_shape, nn_parameters)
        layer_act_types, layer_act_args = self.configure_layer_activations(
                                            layer_shapes, nn_parameters,
                                            act_type, act_args)

        for i, layer_pars in enumerate(nn_parameters):
            if layer_pars['layer_type'] == "conv":
                layers.append(self.create_conv_layer(
                    layer_shapes[i], 
                    layer_pars['num_filters'], 
                    layer_pars['filter_shape'], 
                    layer_pars['stride'], 
                    layer_act_types[i], 
                    layer_act_args[i]))

            elif layer_pars['layer_type'] == "recurrent":
                layers.append(self.create_recurrent_layer(
                    layer_pars['num_input_graph_nodes'], 
                    layer_pars['input_graph'],
                    layer_pars['num_internal_nodes'],
                    layer_pars['internal_graph'], 
                    layer_act_types[i], 
                    layer_act_args[i]))

            elif layer_pars['layer_type'] == "reservor":
                layers.append(self.create_reservoir_layer(
                    layer_pars['num_input_graph_nodes'], 
                    layer_pars['input_graph'],
                    layer_pars['input_weights'], 
                    layer_pars['num_internal_nodes'],
                    layer_pars['internal_graph'],
                    layer_pars['internal_weights'], 
                    layer_act_types[i], 
                    layer_act_args[i]))

            elif layer_pars['layer_type'] == "conv_recurrent":
                layers.append(self.create_conv_recurrent_layer(
                    layer_shapes[i],
                    layer_pars['num_filters'],
                    layer_pars['filter_shape'],
                    layer_pars['stride'],
                    layer_pars['num_internal_nodes'],
                    layer_pars['internal_graph']), 
                    layer_act_types[i], 
                    layer_act_args[i])

            elif layer_pars['layer_type'] == "conv_reservoir":
                layers.append(self.create_conv_reservoir_layer(
                    layer_shapes[i],
                    layer_pars['num_filters'],
                    layer_pars['filter_shape'],
                    layer_pars['stride'],
                    layer_pars['num_internal_nodes'],
                    layer_pars['internal_graph'],
                    layer_pars['internal_weights'],
                    layer_act_types[i], 
                    layer_act_args[i]))

            elif layer_pars['layer_type'] == 'a2a_recurrent':
                layers.append(self.create_a2a_recurrent_layer(
                    layer_shapes[i],
                    layer_pars['num_internal_nodes'],
                    layer_pars['internal_graph'],
                    layer_act_types[i], 
                    layer_act_args[i]))

            elif layer_pars['layer_type'] == 'a2a_a2a':
                layers.append(self.create_a2a_a2a_layer(
                    layer_shapes[i],
                    layer_pars['num_internal_nodes'],
                    layer_act_types[i], 
                    layer_act_args[i]))

        # Build motor later
        prev_layer_size = float(np.cumprod(prev_layer_shape))
        layers.append(create_motor_layer(prev_layer_size, 
            num_outputs, act_type, act_args))

        # Generate NN
        self.neural_network = nn_generator.CreateNervousSystem(input_shape,
            tuple(layers))

    def configure_layer_activations(self, layer_shapes, nn_parameters, 
                                    act_type, act_args):
        """
        outputs the necessary tuples for layer activations of both conv and 
        non-conv layers
        """

        layer_act_types = []
        layer_act_args = []
        for i, layer_pars in enumerate(nn_parameters):
            if 'conv' in layer_pars['layer_type']:
                layer_act_types.append(ACTMAP[act_type])
                layer_act_args.append((layer_shapes[i+1], *act_args))
            else:
                layer_act_types.append(act_type)
                layer_act_args.append(act_args)

        return layer_act_types, layer_act_args

    def configure_layer_shapes(self, input_shape, nn_parameters):
        """
        For a given activation type it sets an internal attribute that
        can be used by the other layer creation functions
        """

        layer_shapes = [input_shape]
        for i, layer_pars in enumerate(nn_parameters):
            if 'conv' in layer_pars['layer_type']:
                layer_shapes.append(calc_conv_layer_shape(layer_shapes[i], 
                layer_pars['num_filters'], layer_pars['stride']))
            else:
                layer_shapes.append(np.array([layer_pars['num_internal_nodes']],
                                    dtype=uint64))

        return layer_shapes

    def create_a2a_recurrent_layer(self, prev_layer_shape, num_internal_nodes, 
                             internal_edge_array, act_type, act_args):
        back_type = INTEGRATOR_TYPE.ALL2ALL
        back_args = (np.cumprod(prev_layer_shape, dtype=float32), 
                    int(num_internal_nodes))
        self_type = INTEGRATOR_TYPE.RECURRENT
        self_args = (int(num_internal_nodes),
                    internal_edge_array)
        return layer_generator.CreateLayer(back_type, back_args, self_type,
            self_args, act_type, act_args)

    def create_a2a_a2a_layer(self, prev_layer_shape, num_internal_nodes, 
                            act_type, act_args):
        back_type = INTEGRATOR_TYPE.ALL2ALL
        back_args = (np.cumprod(prev_layer_shape, dtype=float32), 
                    int(num_internal_nodes))
        self_type = INTEGRATOR_TYPE.ALL2ALL
        self_args = (int(num_internal_nodes), 
                    int(num_internal_nodes))
        return layer_generator.CreateLayer(back_type, back_args, self_type,
            self_args, act_type, act_args)

    def create_conv_recurrent_layer(self, prev_layer_shape, num_filters, 
            filter_shape, stride, num_internal_nodes, internal_edge_array, 
            act_type, act_args):

        shape = calc_conv_layer_shape(prev_layer_shape, num_filters, stride)
        back_type = INTEGRATOR_TYPE.CONV
        back_args = (np.array([prev_layer_shape[0]] + list(filter_shape), dtype=uint64), 
                    shape, #layer_shape funct outputs dtype=uint64
                    np.array(prev_layer_shape, dtype=uint64,
                    int(stride)))
        
        self_type = INTEGRATOR_TYPE.RECURRENT
        self_args = (int(num_internal_nodes),
                    internal_edge_array)

        return layer_generator.CreateLayer(back_type, back_args, self_type,
            self_args, act_type, act_args)

    def create_conv_reservoir_layer(self, prev_layer_shape, num_filters, 
            filter_shape, stride, num_internal_nodes, internal_edge_array,
            internal_weight_array, act_type, act_args):

        shape = calc_conv_layer_shape(prev_layer_shape, num_filters, stride)
        back_type = INTEGRATOR_TYPE.CONV
        back_args = (np.array([prev_layer_shape[0]] + list(filter_shape), dtype=uint64), 
                    shape, #layer_shape funct outputs dtype=uint64
                    np.array(prev_layer_shape, dtype=uint64,
                    int(stride)))
        self_type = INTEGRATOR_TYPE.RESERVOIR
        self_args = (int(num_internal_nodes),
                    internal_edge_array,
                    internal_weight_array)

        return layer_generator.CreateLayer(back_type, back_args, self_type,
            self_args, act_type, act_args)

    def create_recurrent_layer(self, num_bipartite_nodes,
            bipartite_input_edge_array, num_internal_nodes,
            internal_edge_array, act_type, act_args):
        """
        All graphs should be Nx2 with dtype=uint64
        act_args should be in the proper tuple format for inputs into the
        activator function
        """
        back_type = INTEGRATOR_TYPE.RECURRENT
        back_args = (int(num_bipartite_nodes),
                    bipartite_input_edge_array)
        self_type = INTEGRATOR_TYPE.RECURRENT
        self_args = (int(num_internal_nodes),
                    internal_edge_array)
        return layer_generator.CreateLayer(back_type,
            back_args, self_type, self_args, act_type, act_args)

    def create_reservoir_layer(self, num_bipartite_nodes,
            bipartite_input_edge_array, input_weights, num_internal_nodes,
            internal_edge_array, internal_weight_array, act_type, act_args):
        """
        All graphs should be Nx2 with dtype=uint64
        All weights should be Nx1 with dtype=float32
        act_args should be in the proper tuple format for inputs into the
        activator function
        """
        back_type = INTEGRATOR_TYPE.RESERVOIR
        back_args = (int(num_bipartite_nodes),
                    bipartite_input_edge_array, 
                    input_weights)
        self_type = INTEGRATOR_TYPE.RESERVOIR
        self_args = (int(num_internal_nodes),
                    internal_edge_array,
                    internal_weight_array)
        return layer_generator.CreateLayer(back_type,
            back_args, self_type, self_args, act_type, act_args)

    def create_conv_layer(self, prev_layer_shape, num_filters, filter_shape, 
            stride, act_type, act_args):
        """
        act_args needs to be in the correct tuple format with properly
        typed numpy arrays for any arguments

        filter_shape = 2 element array/list with filter dimenstions
        Final shape, which includes depth, depends on shape of prev layer
        """
        shape = calc_conv_layer_shape(prev_layer_shape, num_filters, stride)

        back_type = INTEGRATOR_TYPE.CONV
        # Appropriate depth is added to filter shape to build the # 3-element 1D array
        back_args = (np.array([prev_layer_shape[0]] + list(filter_shape), dtype=uint64), 
                    shape, #layer_shape funct outputs dtype=uint64
                    np.array(prev_layer_shape, dtype=uint64,
                    int(stride)))
        self_type = INTEGRATOR_TYPE.NONE
        self_args = tuple()
        return layer_generator.CreateLayer(back_type,
            back_args, self_type, self_args, act_type, act_args)

    def create_motor_layer(self, size_of_prev_layer, num_outputs, act_type, act_args):
        """
        act_args needs to be in the correct tuple format for the activator
        """

        return layer_generator.CreateMotorLayer(
            int(size_of_prev_layer), int(num_outputs), act_type, act_args)

def calc_conv_layer_shape(prev_layer_shape, num_filters, stride):
    """
    Determines layer shape given previous layer shape and the number of chosen
    filters and stride for the next layer.
    """

    return np.array([num_filters] + calc_image_dimensions(
                    image_dimensions[1:], stride), dtype=np.uint64)

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

    def __init__(self, input_shape, num_outputs, num_neurons, step_size):
        nn_parameters = [{
            'layer_type' : "a2a_a2a"
            'num_internal_nodes': num_neurons}]
        act_type = ACTIVATOR_TYPE.CTRNN
        act_args = (int(step_size),)
        super(input_shape, num_outputs, nn_parameters, act_type, act_args)

if __name__ == '__main__':
    """
    example
    """
    pass