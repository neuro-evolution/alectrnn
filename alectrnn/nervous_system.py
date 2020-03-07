"""
Contains the NervousSystem class and associated subclasses, helper functions,
and enums.
"""

from enum import Enum
import numpy as np
from alectrnn import layer_generator
from alectrnn import nn_generator
from alectrnn import nn_handler


class PARAMETER_TYPE(Enum):
    """
    Class for specifying parameter types. Should match those in the C++ code:
    parameter_types.hpp
    """
    __order__ = 'BIAS RTAUS WEIGHT RANGE REFRACTORY RESISTANCE GAIN DECAY'
    BIAS = 0  # bias or input bias
    RTAUS = 1  # inverse of tau, time-constant
    WEIGHT = 2
    RANGE = 3  # difference between reset value and threshold for IF models
    REFRACTORY = 4  # refractory period for IF models
    RESISTANCE = 5  # resistance for IF models
    GAIN = 6  # input gain parameters
    DECAY = 7  # 0-1 how quickly state decays.


def draw_uniform_initial_guess(boundary_array, rng):
    """
    Uses a numpy RandomState to draw a uniform set of initial values between
    the specified bounds.
    :param boundary_array: Nx2 np.float32 array
    :param rng: numpy RandomState object
    :return: 1-D array of size N and dtype np.float32
    """

    initial_guess = np.zeros(len(boundary_array), dtype=np.float32)
    for iii, (low, high) in enumerate(boundary_array):
        initial_guess[iii] = rng.uniform(low, high)

    return initial_guess


def normalized_weight_bound(nervous_system, norm_type='sqrt'):
    """
    Uses the degree of all the nodes in the neural network to determine
    the normalized weight for each link, which is 1/sqrt(N), where N is the
    number of pre-synaptic connections.

    :param nervous_system: A NervousSystem instance
    :param norm_type: 'sqrt' for 1/sqrt(N) and 'norm' for 1/N
    :return: a numpy float32 that is the size of the # parameters
    """

    normalization_factors = nn_handler.GetWeightNormalizationFactors(nervous_system.neural_network)
    if norm_type == 'sqrt':
        np.sqrt(normalization_factors, out=normalization_factors)
    elif norm_type == 'norm':
        pass
    else:
        raise AssertionError("Unsupported normalization type")

    np.divide(1., normalization_factors,
              where=normalization_factors != 0.0,
              out=normalization_factors)

    return normalization_factors


def draw_initial_guess(type_bounds, nervous_system, rng, normalized_weights=True,
                       norm_type='sqrt'):
    """
    :param type_bounds: key is PARAMETER_TYPE, value is (low, high)
    :param nervous_system: an instance of NervousSystem
    :param rng: an instance of numpy RandomState
    :param normalized_weights: if True, divides weights by 1/sqrt(N), where N
        is the number of pre-synaptic neurons (the neuron at the tail of the
        weight). If false, uses bounds defined in parameter type.
    :param norm_type: 'sqrt' for 1/sqrt(N) and 'norm' for 1/N
    :return: a 1D numpy float 32 array representing the initial guess
    """

    if normalized_weights:
        # Will be set later on a parameter by parameter basis
        type_bounds[PARAMETER_TYPE.WEIGHT] = (0.0, 0.0)

    parameter_layout = nervous_system.parameter_layout()
    boundary_array = boundary_array_for_parameter_layout(parameter_layout,
                                                         type_bounds)

    # adjust boundary_array with weight initial values
    if normalized_weights:
        weight_bound = normalized_weight_bound(nervous_system, norm_type)

        # set boundary array
        boundary_array[:, 0] -= weight_bound
        boundary_array[:, 1] += weight_bound

    return draw_uniform_initial_guess(boundary_array, rng)


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
            boundaries[type_indexes, 0] = bounds[0]
            boundaries[type_indexes, 1] = bounds[1]
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
        try:  # Raises value error if conversion doesn't go through
            par = PARAMETER_TYPE(parameter)
        except ValueError:
            return False

        if par not in type_bounds:
            return False

    return True


class ACTIVATOR_TYPE(Enum):
    """
    Class for distinguishing the various types of neural activators
    Should keep in sync with ACTIVATOR_TYPE in activator.hpp
    """
    BASE = 0
    IDENTITY = 1
    CTRNN = 2
    CONV_CTRNN = 3
    IAF = 4
    CONV_IAF = 5
    SOFT_MAX = 6
    RESERVOIR_CTRNN = 7
    RESERVOIR_IAF = 8
    SIGMOID = 9
    TANH = 10
    RELU = 11
    BOUNDED_RELU = 12
    NOISY_RELU = 13
    NOISY_SIGMOID = 14


class INTEGRATOR_TYPE(Enum):
    """
    Class for distinguishing the various types of neural integrators
    Should keep in sync with INTEGRATOR_TYPE in integrator.hpp
    """
    BASE = 0
    NONE = 1
    ALL2ALL = 2
    CONV = 3
    RECURRENT = 4
    RESERVOIR = 5
    RESERVOIR_HYBRID = 6
    TRUNCATED_RECURRENT = 7
    CONV_EIGEN = 8
    ALL2ALL_EIGEN=9
    RECURRENT_EIGEN = 10
    RESERVOIR_EIGEN = 11
    REWARD_MODULATED = 12
    REWARD_MODULATED_ALL2ALL = 13
    REWARD_MODULATED_RECURRENT = 14
    REWARD_MODULATED_CONV = 15


ACTMAP = {ACTIVATOR_TYPE.IAF: ACTIVATOR_TYPE.CONV_IAF,
          ACTIVATOR_TYPE.CTRNN: ACTIVATOR_TYPE.CONV_CTRNN}

RESERVOIR_ACTMAP = {ACTIVATOR_TYPE.IAF: ACTIVATOR_TYPE.RESERVOIR_IAF,
                    ACTIVATOR_TYPE.CTRNN: ACTIVATOR_TYPE.RESERVOIR_CTRNN}


class NervousSystem:
    """
    Builds a nervous system. A single activator is chosen for the whole network.

    input_shape determines the dimensions of the input the network will receive.
    num_outputs will determine motor_layer size. The first dimension is the
    number of temporal channels, e.g. how many instances of time are maintained
    in the input. The following two dimensions are H and W respectively, with
    W being the major axis dimension.

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
        ACTIVATION_TYPE.TANH: ()
        ACTIVATION_TYPE.SIGMOID: (float(saturation_point),)
        ACTIVATION_TYPE.RELU: ()
        ACTIVATION_TYPE.BOUNDED_RELU: (bound,)

    Motor layer is the last layer of the network and should have a dictionary
    with the following keys:
        'layer_type' = "motor"
        'motor_type' = ('standard' | 'softmax' | 'eigen')

    A standard motor layer takes on the same activation types and arguments as
    the other layers. The softmax layer has no memory and implements the softmax
    function. It takes an additional argument specified as:
        'temperature' = float, low temp means greedy, high temp means exploration,
            defaults to 1.

    Convolutional layers should have a dictionary with the following keys:
        'layer_type' = "conv"
        'filter_shape' = 2-element list/array with filter dimensions
        'num_filters'
        'stride'

    Eigen Convolutional layers uses eigen integrators:
        'layer_type' = "eigen_conv"
        'filter_shape' = 2-element list/array with filter dimensions
        'num_filters'
        'stride'

    Reservoir layers have an untrained set of connections determined the
    input graph. The back connections are determined by the input graph.
    Both must specify weights. Should have a dictionary with the following keys:
        'layer_type' = "reservoir"
        'input_graph' = E1x2, dtype=np.uint64 bipartite edge graph
        'input_weights' = N element, dtype=np.float32
        'num_internal_nodes' = M
        'internal_graph' = E2x2, dtype=np.uint64 edge array
        'internal_weights' = M element, dtype=np.float32 array
        'neuron_parameters' = a tuple of parameters for the neurons

    Recurrent layers define the connections, but not the weights of the internal
    and back connections. All connections are trained.
    Should have a dictionary with the following keys:
        'layer_type' = "recurrent"
        'input_graph' = E1x2, dtype=np.uint64 bipartite edge graph
        'num_internal_nodes' = M
        'internal_graph' = E2x2, dtype=np.uint64 edge array

    Eigen Recurrent layers use eigen integrators.
    Should have a dictionary with the following keys:
        'layer_type' = "eigen_recurrent"
        'input_graph' = E1x2, dtype=np.uint64 bipartite edge graph
        'num_internal_nodes' = M
        'internal_graph' = E2x2, dtype=np.uint64 edge array

    Truncated Recurrent layers are the same as recurrent layers, except with an
    additional parameter that specifies a weight threshold. When the weights
    are below the thresholds magnitude, the connection is considered non-existent.
    Should have a dictionary with the following keys:
        'layer_type' = "truncated_recurrent"
        'input_graph' = E1x2, dtype=np.uint64 bipartite edge graph
        'num_internal_nodes' = M
        'internal_graph' = E2x2, dtype=np.uint64 edge array
        'weight_threshold' = float dtype=np.float32 (>=0.0)

    Trained-input Reservoir layers have trained recurrent connections to the
    previous layer, but the reservoir is untrained.
        'layer_type' = "trained_input_reservoir"
        'input_graph' = E1x2, dtype=np.uint64 bipartite edge graph
        'num_internal_nodes' = M
        'internal_graph' = E2x2, dtype=np.uint64 edge array
        'internal_weights' = M element, dtype=np.float32 array
        'neuron_parameters' = a tuple of parameters for the neurons

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
        'neuron_parameters'

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

    eigen_a2a_ff layer has fully connected back connections and no internal connections.
    It is the same as a standard feed-forward layer. Used Eigen integrators
    Should have a dictionary with the following keys:
        'layer_type' = "a2a_ff"
        'num_internal_nodes'

    conv_recurrent layers allow the construction of layers that have a conv back
    connection, but recurrent internal connection.

    Valid back_integrator types:
        INTEGRATOR_TYPE.CONV : args(filter-shape, # filters, stride)

    :note: In the connector graph, all tail edges will be interpreted as belonging
        to the corresponding index state of the previous layer, while all heads will
        be interpreted as index states of the current layer.

    :note: For the internal graph, all heads and tails will be interpreted as
        belonging to the current layer.

    :note: All node IDs for networks should start at 0 and correspond to the
        state index.

    :note: For input graphs, the order of the numpy array should be row-major
        with shape Ex2 with major axis as the edge with (tail, head) so that
        X[edge#][0]=tail, X[edge#][1]=head

    TODO: Support even filter shapes
    """

    def __init__(self, input_shape, num_outputs, nn_parameters,
                 act_type, act_args, verbose=False):
        """
        input_shape - shape of input into the NN (should be 3D) 1st dim is
                      channels, second is height, then width. Will be cast
                      as dtype np.uint64 automatically
        num_outputs - the size of the motor layer
        nn_parameters - list of dictionaries. One for each layer.
        Input layers are added by the NervousSystem itself by default.

        act_type = general ACTIVATION_TYPE for model
        act_args = general arguments for model

        :param motor_type: specifies the type of motor layer created.
            defaults to 'motor', which adopts network activators
            'softmax' is a memoryless motor which assign probabilities to outputs

        CONV layers usually have additional arguments, like shape, for
        parameter sharing of act_args. They also have their own ACTIVATION_TYPE.
        These are automatically added to the CONV layers.
        """
        self.verbose = verbose
        self.num_outputs = num_outputs
        input_shape = np.array(input_shape, dtype=np.uint64)
        layers = []
        # interpreted shapes are for some back integrators which need
        # to know how to interpret the layer for convolution
        interpreted_shapes, layer_shapes = self._configure_layer_shapes(
            input_shape, nn_parameters)

        layer_act_types, layer_act_args = ActivationAPIMap.layer_config[act_type.value](
            layer_shapes, interpreted_shapes, nn_parameters, act_type, act_args)

        # Build layers
        for i, layer_pars in enumerate(nn_parameters):
            if self.verbose:
                print("Building layer...", i, "with pars", nn_parameters[i])

            if layer_pars['layer_type'] == "conv":
                layers.append(self._create_conv_layer(
                    interpreted_shapes[i],
                    interpreted_shapes[i+1],
                    layer_pars['filter_shape'],
                    layer_pars['stride'],
                    layer_act_types[i],
                    layer_act_args[i]))

            elif layer_pars['layer_type'] == "eigen_conv":
                layers.append(self._create_eigen_conv_layer(
                    interpreted_shapes[i],
                    interpreted_shapes[i+1],
                    layer_pars['filter_shape'],
                    layer_pars['stride'],
                    layer_act_types[i],
                    layer_act_args[i]))

            elif layer_pars['layer_type'] == "rm_conv":
                layers.append(self._create_rm_conv_layer(
                    interpreted_shapes[i],
                    interpreted_shapes[i+1],
                    layer_pars['filter_shape'],
                    layer_pars['stride'],
                    layer_pars['reward_smoothing_factor'],
                    layer_pars['activation_smoothing_factor'],
                    layer_pars['learning_rate'],
                    layer_act_types[i],
                    layer_act_args[i]))

            elif layer_pars['layer_type'] == "nrm_conv":
                layers.append(self._create_nrm_conv_layer(
                    interpreted_shapes[i],
                    interpreted_shapes[i+1],
                    layer_pars['filter_shape'],
                    layer_pars['stride'],
                    layer_pars['reward_smoothing_factor'],
                    layer_pars['activation_smoothing_factor'],
                    layer_pars['standard_deviation'],
                    layer_pars['seed'],
                    layer_pars['learning_rate'],
                    layer_act_types[i],
                    layer_act_args[i]))

            elif layer_pars['layer_type'] == "recurrent":
                layers.append(self._create_recurrent_layer(
                    layer_pars['input_graph'],
                    layer_pars['num_internal_nodes'],
                    layer_pars['internal_graph'],
                    layer_act_types[i],
                    layer_act_args[i],
                    layer_shapes[i+1]))

            elif layer_pars['layer_type'] == "eigen_recurrent":
                layers.append(self._create_eigen_recurrent_layer(
                    layer_pars['input_graph'],
                    layer_pars['num_internal_nodes'],
                    layer_pars['internal_graph'],
                    layer_act_types[i],
                    layer_act_args[i],
                    layer_shapes[i+1],
                    layer_shapes[i]))

            elif layer_pars['layer_type'] == "truncated_recurrent":
                layers.append(self._create_truncated_recurrent_layer(
                    layer_pars['input_graph'],
                    layer_pars['num_internal_nodes'],
                    layer_pars['internal_graph'],
                    layer_act_types[i],
                    layer_act_args[i],
                    layer_shapes[i+1],
                    layer_pars['weight_threshold']))

            elif layer_pars['layer_type'] == "reservoir":
                layers.append(self._create_reservoir_layer(
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

            elif layer_pars['layer_type'] == 'rm_a2a_ff':
                layers.append(self._create_rm_a2a_ff_layer(
                    layer_shapes[i],
                    layer_pars['num_internal_nodes'],
                    layer_pars['reward_smoothing_factor'],
                    layer_pars['activation_smoothing_factor'],
                    layer_pars['learning_rate'],
                    layer_act_types[i],
                    layer_act_args[i],
                    layer_shapes[i+1]))

            elif layer_pars['layer_type'] == 'nrm_a2a_ff':
                layers.append(self._create_nrm_a2a_ff_layer(
                    layer_shapes[i],
                    layer_pars['num_internal_nodes'],
                    layer_pars['reward_smoothing_factor'],
                    layer_pars['activation_smoothing_factor'],
                    layer_pars['standard_deviation'],
                    layer_pars['seed'],
                    layer_pars['learning_rate'],
                    layer_act_types[i],
                    layer_act_args[i],
                    layer_shapes[i+1]))

            elif layer_pars['layer_type'] == 'eigen_a2a_ff':
                layers.append(self._create_eigen_a2a_ff_layer(
                    layer_shapes[i],
                    layer_pars['num_internal_nodes'],
                    layer_act_types[i],
                    layer_act_args[i],
                    layer_shapes[i+1]))

            elif layer_pars['layer_type'] == 'trained_input_reservoir':
                layers.append(self._create_trained_input_reservoir_layer(
                    layer_pars['input_graph'],
                    layer_pars['num_internal_nodes'],
                    layer_pars['internal_graph'],
                    layer_pars['internal_weights'],
                    layer_act_types[i],
                    layer_act_args[i],
                    layer_shapes[i+1]))

            elif layer_pars['layer_type'] == 'motor':
                if layer_pars['motor_type'].lower() == 'standard':
                    layers.append(self._create_motor_layer(
                        layer_shapes[i+1],
                        layer_shapes[i],
                        layer_act_types[i],
                        layer_act_args[i],
                    ))
                elif layer_pars['motor_type'].lower() == 'eigen':
                    layers.append(self._create_eigen_motor_layer(
                                  layer_shapes[i+1],
                                  layer_shapes[i],
                                  layer_act_types[i],
                                  layer_act_args[i],
                                  ))
                elif layer_pars['motor_type'].lower() == 'softmax':
                    layers.append(self._create_softmax_motor_layer(
                        layer_shapes[i+1],
                        layer_shapes[i],
                        layer_pars['temperature']
                    ))
                elif layer_pars['motor_type'].lower() == 'rm':
                    layers.append(self._create_rm_motor_layer(
                        layer_shapes[i+1],
                        layer_shapes[i],
                        layer_pars['reward_smoothing_factor'],
                        layer_pars['activation_smoothing_factor'],
                        layer_pars['learning_rate'],
                        layer_act_types[i],
                        layer_act_args[i],
                    ))
                elif layer_pars['motor_type'].lower() == 'nrm':
                    layers.append(self._create_nrm_motor_layer(
                        layer_shapes[i+1],
                        layer_shapes[i],
                        layer_pars['reward_smoothing_factor'],
                        layer_pars['activation_smoothing_factor'],
                        layer_pars['standard_deviation'],
                        layer_pars['seed'],
                        layer_pars['learning_rate'],
                        layer_act_types[i],
                        layer_act_args[i],
                    ))
            else:
                raise NotImplementedError("Doesn't support "
                                          + layer_pars['layer_type'])

        # Generate NN
        self.neural_network = nn_generator.CreateNervousSystem(input_shape,
                                                               tuple(layers))
        self.layer_shapes = layer_shapes
        self.interpreted_shapes = interpreted_shapes
        self.nn_parameters = nn_parameters

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
                                                                layer_pars['num_filters'],
                                                                layer_pars['stride']))
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
            elif 'motor' in layer_pars['layer_type']:
                interpreted_shapes.append(np.array([self.num_outputs], dtype=np.uint64))
                layer_shapes.append(np.array([self.num_outputs], dtype=np.uint64))

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
        self_args = (internal_edge_array,)
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

    def _create_eigen_a2a_ff_layer(self, prev_layer_shape, num_internal_nodes,
                                   act_type, act_args, layer_shape):
        """
        Creates a layer with all-to-all back connections, but not internal
        connections. Uses Eigen integrators.
        Restructures input parameters into the correct format for the
        C++ function call, then calls the function.
        :param prev_layer_shape: shape of the previous layer
        :param num_internal_nodes: number of neurons in layer
        :param act_type: ACTIVATOR_TYPE
        :param act_args: arguments for that ACTIVATOR_TYPE
        :param layer_shape: shape of the layer
        :return: python capsule with pointer to the layer
        """
        back_type = INTEGRATOR_TYPE.ALL2ALL_EIGEN.value
        back_args = (int(num_internal_nodes),
                     int(np.prod(prev_layer_shape)))
        self_type = INTEGRATOR_TYPE.NONE.value
        self_args = tuple()
        assert(act_args[0] == num_internal_nodes)
        return layer_generator.CreateLayer(back_type, back_args, self_type,
                                           self_args, act_type, act_args, layer_shape)

    def _create_rm_a2a_ff_layer(self, prev_layer_shape, num_internal_nodes,
                                   reward_smoothing_factor,
                                   activation_smoothing_factor,
                                   learning_rate,
                                   act_type, act_args, layer_shape):
        """
        Creates a layer with all-to-all back connections, but not internal
        connections. It uses reward modulation to update weights online.
        Uses Eigen integrators.
        Restructures input parameters into the correct format for the
        C++ function call, then calls the function.
        :param prev_layer_shape: shape of the previous layer
        :param num_internal_nodes: number of neurons in layer
        :param reward_smoothing_factor: memory time constant for exponential averaging.
        :param activation_smoothing_factor: memory time constant for exponential averaging.
        :param learning_rate: factor that controls rate of weight change.
        :param act_type: ACTIVATOR_TYPE
        :param act_args: arguments for that ACTIVATOR_TYPE
        :param layer_shape: shape of the layer
        :return: python capsule with pointer to the layer
        """
        back_type = INTEGRATOR_TYPE.REWARD_MODULATED_ALL2ALL.value
        back_args = (int(num_internal_nodes),
                     int(np.prod(prev_layer_shape)),
                     float(learning_rate))
        self_type = INTEGRATOR_TYPE.NONE.value
        self_args = tuple()
        assert(act_args[0] == num_internal_nodes)
        return layer_generator.CreateRewardModulatedLayer(back_type, back_args,
                                                          self_type, self_args,
                                                          act_type, act_args,
                                                          layer_shape,
                                                          reward_smoothing_factor,
                                                          activation_smoothing_factor)

    def _create_nrm_a2a_ff_layer(self, prev_layer_shape, num_internal_nodes,
                                reward_smoothing_factor,
                                activation_smoothing_factor,
                                standard_deviation,
                                seed,
                                learning_rate,
                                act_type, act_args, layer_shape):
        """
        Creates a layer with all-to-all back connections, but not internal
        connections. It uses reward modulation to update weights online.
        Uses Eigen integrators.
        Restructures input parameters into the correct format for the
        C++ function call, then calls the function.
        :param prev_layer_shape: shape of the previous layer
        :param num_internal_nodes: number of neurons in layer
        :param reward_smoothing_factor: memory time constant for exponential averaging.
        :param activation_smoothing_factor: memory time constant for exponential averaging.
        :param standard_deviation: strength of noise
        :param seed: for rng
        :param learning_rate: factor that controls rate of weight change.
        :param act_type: ACTIVATOR_TYPE
        :param act_args: arguments for that ACTIVATOR_TYPE
        :param layer_shape: shape of the layer
        :return: python capsule with pointer to the layer
        """
        back_type = INTEGRATOR_TYPE.REWARD_MODULATED_ALL2ALL.value
        back_args = (int(num_internal_nodes),
                     int(np.prod(prev_layer_shape)),
                     float(learning_rate))
        self_type = INTEGRATOR_TYPE.NONE.value
        self_args = tuple()
        assert(act_args[0] == num_internal_nodes)
        return layer_generator.CreateNoisyRewardModulatedLayer(back_type, back_args,
                                                          self_type, self_args,
                                                          act_type, act_args,
                                                          layer_shape,
                                                          reward_smoothing_factor,
                                                          activation_smoothing_factor,
                                                               standard_deviation,
                                                               seed)

    def _create_conv_recurrent_layer(self, prev_layer_shape, interpreted_shape,
                                     filter_shape, stride,
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
        self_args = (internal_edge_array,)

        return layer_generator.CreateLayer(back_type, back_args, self_type,
                                           self_args, act_type, act_args, layer_shape)

    def _create_conv_reservoir_layer(self, prev_layer_shape, interpreted_shape,
                                     filter_shape, stride,
                                     internal_edge_array, internal_weight_array,
                                     act_type, act_args,
                                     layer_shape):
        """
        Creates a layer with convolutional back connections and a randomly
        connected and weight internal connections.
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
        self_args = (internal_edge_array,
                     internal_weight_array)

        return layer_generator.CreateLayer(back_type, back_args, self_type,
                                           self_args, act_type, act_args, layer_shape)

    def _create_recurrent_layer(self, bipartite_input_edge_array,
                                num_internal_nodes, internal_edge_array,
                                act_type, act_args, layer_shape):
        """
        Creates a recurrent layer with graphs specifying back and self
        connections.
        Restructures input parameters into the correct format for the
        C++ function call, then calls the CreateLayer function.
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
        back_args = (bipartite_input_edge_array,)
        self_type = INTEGRATOR_TYPE.RECURRENT.value
        self_args = (internal_edge_array,)
        assert(act_args[0] == num_internal_nodes)
        return layer_generator.CreateLayer(back_type,
                                           back_args, self_type, self_args,
                                           act_type, act_args, layer_shape)

    def _create_eigen_recurrent_layer(self, bipartite_input_edge_array,
                                      num_internal_nodes, internal_edge_array,
                                      act_type, act_args, layer_shape,
                                      prev_layer_shape):
        """
        Creates a eigen recurrent layer with graphs specifying back and self
        connections. Uses Eigen integrators
        Restructures input parameters into the correct format for the
        C++ function call, then calls the CreateLayer function.
        :param bipartite_input_edge_array: array for back-connections (Nx2)
            dtype=np.uint64
        :param num_internal_nodes: number of neurons in layer
        :param internal_edge_array: array for self-connections (Nx2)
            dtype=np.uint64
        :param act_type: ACTIVATOR_TYPE
        :param act_args: arguments for that ACTIVATOR_TYPE
        :param layer_shape: shape of the layer
        :param prev_layer_shape: shape of last layer
        :return: python capsule with pointer to the layer
        """
        back_type = INTEGRATOR_TYPE.RECURRENT_EIGEN.value
        back_args = (bipartite_input_edge_array, num_internal_nodes,
                     int(np.prod(prev_layer_shape)))
        self_type = INTEGRATOR_TYPE.RECURRENT_EIGEN.value
        self_args = (internal_edge_array, num_internal_nodes, num_internal_nodes)
        assert(act_args[0] == num_internal_nodes)
        return layer_generator.CreateLayer(back_type,
                                           back_args, self_type, self_args,
                                           act_type, act_args, layer_shape)

    def _create_truncated_recurrent_layer(self, bipartite_input_edge_array,
                                          num_internal_nodes,
                                          internal_edge_array, act_type,
                                          act_args, layer_shape,
                                          weight_threshold):
        """
        Creates a recurrent layer with graphs specifying back and self
        connections.
        Restructures input parameters into the correct format for the
        C++ function call, then calls the CreateLayer function.
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
        back_type = INTEGRATOR_TYPE.TRUNCATED_RECURRENT.value
        back_args = (bipartite_input_edge_array, weight_threshold)
        self_type = INTEGRATOR_TYPE.TRUNCATED_RECURRENT.value
        self_args = (internal_edge_array, weight_threshold)
        assert(act_args[0] == num_internal_nodes)
        return layer_generator.CreateLayer(back_type, back_args, self_type,
                                           self_args, act_type, act_args,
                                           layer_shape)

    def _create_reservoir_layer(self, bipartite_input_edge_array, input_weights,
                                num_internal_nodes, internal_edge_array,
                                internal_weight_array, act_type, act_args,
                                layer_shape):
        """
        Layer with internal and back connection and weights specified by
        the input graphs.
        Restructures input parameters into the correct format for the
        C++ function call, then calls the CreateLayer function.
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
        back_args = (bipartite_input_edge_array,
                     input_weights)
        self_type = INTEGRATOR_TYPE.RESERVOIR.value
        self_args = (internal_edge_array,
                     internal_weight_array)
        assert(act_args[0] == num_internal_nodes)
        return layer_generator.CreateLayer(back_type,
                                           back_args, self_type, self_args,
                                           act_type, act_args,
                                           layer_shape)

    def _create_trained_input_reservoir_layer(self, bipartite_input_edge_array,
                                              num_internal_nodes,
                                              internal_edge_array,
                                              internal_weight_array, act_type,
                                              act_args, layer_shape):
        """
        Layer with internal connection and weights specified by
        the input graphs. Input weights are trained.
        Restructures input parameters into the correct format for the
        C++ function call, then calls the CreateLayer function.
        :param bipartite_input_edge_array: Nx2 dtype=np.uint64 graph
        :param num_internal_nodes: number of neurons in layer
        :param internal_edge_array: array for self-connections Nx2 np.uint64
        :param internal_weight_array: weights for graph N dtype=np.float32
        :param act_type: ACTIVATOR_TYPE
        :param act_args: arguments for that ACTIVATOR_TYPE
        :param layer_shape: shape of the layer
        :return: python capsule with pointer to the layer
        """
        back_type = INTEGRATOR_TYPE.RECURRENT.value
        back_args = (bipartite_input_edge_array,)
        self_type = INTEGRATOR_TYPE.RESERVOIR.value
        self_args = (internal_edge_array,
                     internal_weight_array)
        assert(act_args[0] == num_internal_nodes)
        return layer_generator.CreateLayer(back_type,
                                           back_args, self_type, self_args,
                                           act_type, act_args,
                                           layer_shape)

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
                     interpreted_shape,  # layer_shape funct outputs dtype=np.uint64
                     np.array(prev_layer_shape, dtype=np.uint64),
                     int(stride))
        self_type = INTEGRATOR_TYPE.NONE.value
        self_args = tuple()
        return layer_generator.CreateLayer(back_type,
                                           back_args, self_type, self_args,
                                           act_type, act_args, interpreted_shape)

    def _create_eigen_conv_layer(self, prev_layer_shape, interpreted_shape,
                                 filter_shape, stride, act_type, act_args):
        """
        Creates a layer with convolutional back connections and no self
        connections. Uses Eigen integrators
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

        back_type = INTEGRATOR_TYPE.CONV_EIGEN.value
        # Appropriate depth is added to filter shape to build the # 3-element 1D array
        back_args = (np.array([prev_layer_shape[0]] + list(filter_shape), dtype=np.uint64),
                     interpreted_shape,  # layer_shape funct outputs dtype=np.uint64
                     np.array(prev_layer_shape, dtype=np.uint64),
                     int(stride))
        self_type = INTEGRATOR_TYPE.NONE.value
        self_args = tuple()
        return layer_generator.CreateLayer(back_type,
                                           back_args, self_type, self_args,
                                           act_type, act_args, interpreted_shape)

    def _create_motor_layer(self, num_outputs, prev_layer_shape, act_type, act_args):
        """
        Generates a motor layer for the neural network, which will represent
        the output of the network. It is fully connected to whatever layer
        precedes it.
        act_args needs to be in the correct tuple format for the activator
        """

        size_of_prev_layer = int(np.prod(prev_layer_shape))
        assert(act_args[0] == num_outputs)
        return layer_generator.CreateMotorLayer(
            int(num_outputs), size_of_prev_layer, act_type, act_args)

    def _create_eigen_motor_layer(self, num_outputs, prev_layer_shape, act_type,
                                  act_args):
        """
        Generates an eigen motor layer for the neural network, which will represent
        the output of the network. It is fully connected to whatever layer
        precedes it.
        act_args needs to be in the correct tuple format for the activator
        """

        size_of_prev_layer = int(np.prod(prev_layer_shape))
        assert(act_args[0] == num_outputs)
        return layer_generator.CreateEigenMotorLayer(
            int(num_outputs), size_of_prev_layer, act_type, act_args)

    def _create_rm_motor_layer(self, num_outputs, prev_layer_shape,
                               reward_smoothing_factor,
                               activation_smoothing_factor,
                               learning_rate,
                               act_type,
                               act_args):
        """
        Generates an reward mod motor layer for the neural network, which will
        represent the output of the network. It is fully connected to whatever layer
        precedes it.
        act_args needs to be in the correct tuple format for the activator
        """

        size_of_prev_layer = int(np.prod(prev_layer_shape))
        assert(act_args[0] == num_outputs)
        return layer_generator.CreateRewardModulatedMotorLayer(
            int(num_outputs), size_of_prev_layer, float(reward_smoothing_factor),
            float(activation_smoothing_factor), float(learning_rate),
            act_type, act_args)

    def _create_nrm_motor_layer(self, num_outputs, prev_layer_shape,
                               reward_smoothing_factor,
                               activation_smoothing_factor,
                                standard_deviation,
                                seed,
                               learning_rate,
                               act_type,
                               act_args):
        """
        Generates an reward mod motor layer for the neural network, which will
        represent the output of the network. It is fully connected to whatever layer
        precedes it.
        act_args needs to be in the correct tuple format for the activator
        """

        size_of_prev_layer = int(np.prod(prev_layer_shape))
        assert(act_args[0] == num_outputs)
        return layer_generator.CreateNoisyRewardModulatedMotorLayer(
            int(num_outputs), size_of_prev_layer, float(reward_smoothing_factor),
            float(activation_smoothing_factor), float(standard_deviation),
            int(seed), float(learning_rate),
            act_type, act_args)

    def _create_softmax_motor_layer(self, num_outputs, prev_layer_shape, temperature=1.0):
        """
        :param num_outputs: number of outputs expected by the application
        :param prev_layer_shape: shape of previous layer
        :param temperature: adjusts probablities, lower temperature -> greedy
            selection, high temperature -> exploration.
        :return: A softmax motor capsule
        """

        size_of_prev_layer = int(np.prod(prev_layer_shape))
        return layer_generator.CreateSoftMaxMotorLayer(int(num_outputs),
                                                       size_of_prev_layer,
                                                       float(temperature))

    def _create_rm_conv_layer(self, prev_layer_shape, interpreted_shape,
                              filter_shape, stride, reward_smoothing_factor,
                              activation_smoothing_factor,
                              learning_rate, act_type, act_args):
        """
        Creates a layer with convolutional back connections and no self
        connections with reward modulation on its weights. Uses Eigen integrators
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
        :param reward_smoothing_factor: memory time constant for exponential averaging.
        :param activation_smoothing_factor: memory time constant for exponential averaging.
        :param learning_rate: factor that controls rate of weight change.
        :param act_type: ACTIVATOR_TYPE
        :param act_args: arguments for that ACTIVATOR_TYPE
        :return: python capsule with pointer to the layer
        """

        back_type = INTEGRATOR_TYPE.REWARD_MODULATED_CONV.value
        # Appropriate depth is added to filter shape to build the # 3-element 1D array
        back_args = (np.array([prev_layer_shape[0]] + list(filter_shape), dtype=np.uint64),
                     interpreted_shape,  # layer_shape funct outputs dtype=np.uint64
                     np.array(prev_layer_shape, dtype=np.uint64),
                     int(stride),
                     float(learning_rate))
        self_type = INTEGRATOR_TYPE.NONE.value
        self_args = tuple()
        return layer_generator.CreateRewardModulatedLayer(back_type,
                                                          back_args, self_type,
                                                          self_args,
                                                          act_type, act_args,
                                                          interpreted_shape,
                                                          reward_smoothing_factor,
                                                          activation_smoothing_factor)

    def _create_nrm_conv_layer(self, prev_layer_shape, interpreted_shape,
                              filter_shape, stride, reward_smoothing_factor,
                              activation_smoothing_factor,
                              standard_deviation, seed,
                              learning_rate, act_type, act_args):
        """
        Creates a layer with convolutional back connections and no self
        connections with reward modulation on its weights. Uses Eigen integrators
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
        :param reward_smoothing_factor: memory time constant for exponential averaging.
        :param activation_smoothing_factor: memory time constant for exponential averaging.
        :param standard_deviation: strength of noise.
        :param seed: for rng.
        :param learning_rate: factor that controls rate of weight change.
        :param act_type: ACTIVATOR_TYPE
        :param act_args: arguments for that ACTIVATOR_TYPE
        :return: python capsule with pointer to the layer
        """

        back_type = INTEGRATOR_TYPE.REWARD_MODULATED_CONV.value
        # Appropriate depth is added to filter shape to build the # 3-element 1D array
        back_args = (np.array([prev_layer_shape[0]] + list(filter_shape), dtype=np.uint64),
                     interpreted_shape,  # layer_shape funct outputs dtype=np.uint64
                     np.array(prev_layer_shape, dtype=np.uint64),
                     int(stride),
                     float(learning_rate))
        self_type = INTEGRATOR_TYPE.NONE.value
        self_args = tuple()
        return layer_generator.CreateNoisyRewardModulatedLayer(back_type,
                                                          back_args, self_type,
                                                          self_args,
                                                          act_type, act_args,
                                                          interpreted_shape,
                                                          reward_smoothing_factor,
                                                          activation_smoothing_factor,
                                                          standard_deviation,
                                                          seed)

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

    def parameter_layer_indices(self):
        """
        Returns an np_int array with layer indices corresponding to the parameters.
        """
        return nn_handler.GetParameterLayerIndices(self.neural_network)

    def run_neural_network(self, inputs, parameters):
        """
        Evaluates the NN on a TxI matrix where T is the number of time-steps and
        is the first dimension of the matrix and I is the number of inputs into
        the NN (the dimensions should be squashed). Returns a tuple of state
        arrays for each layer. The parameters will be used to configure the
        neural network.
        """
        return nn_handler.RunNeuralNetwork(self.neural_network, inputs, parameters)


def configure_layer_activations(layer_shapes, interpreted_shapes,
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
        l_act_args = act_args

        # If the user gave separate parameters for each layer, then
        # use those, else assign same parameters for each layer
        if len(act_args) != 0:
            if isinstance(tuple, act_args[0]):
                l_act_args = act_args[i]

        # Only applies to conv layer
        # conv_recurrent and conv_reservoir don't use activator parameter sharing
        if 'conv' in layer_pars['layer_type']:
            layer_act_types.append(act_type.value)
            layer_act_args.append((interpreted_shapes[i+1], True, *l_act_args))
        elif 'reservoir' in layer_pars['layer_type']:
            layer_act_types.append(act_type.value)
            layer_act_args.append((layer_shapes[i+1], False, *l_act_args,
                                  *layer_pars['neuron_parameters']))
        else:
            layer_act_types.append(act_type.value)
            layer_act_args.append((layer_shapes[i+1], False, *l_act_args))

    return layer_act_types, layer_act_args


def configure_oldstyle_layer_activations(layer_shapes, interpreted_shapes,
                                         nn_parameters, act_type, act_args):
    """
    USED FOR OLD API

    outputs the necessary tuples for layer activations of both conv and
    non-conv layers
    layer_shapes includes input layer, so i+1 is synced with nn_params
    no nn_params are included for input layer
    """

    layer_act_types = []
    layer_act_args = []
    for i, layer_pars in enumerate(nn_parameters):
        # Only applies to conv layer
        # conv_recurrent and conv_reservoir don't use activator parameter sharing
        if 'conv' in layer_pars['layer_type']:
            layer_act_types.append(ACTMAP[act_type].value)
            layer_act_args.append((interpreted_shapes[i+1], *act_args))
        elif 'reservoir' in layer_pars['layer_type']:
            layer_act_types.append(RESERVOIR_ACTMAP[act_type].value)
            layer_act_args.append((int(np.prod(layer_shapes[i+1])), *act_args,
                                   *layer_pars['neuron_parameters']))
        else:
            layer_act_types.append(act_type.value)
            layer_act_args.append((int(np.prod(layer_shapes[i+1])), *act_args))

    return layer_act_types, layer_act_args


class ActivationAPIMap:

    layer_config = {ACTIVATOR_TYPE.IDENTITY.value: configure_oldstyle_layer_activations,
                    ACTIVATOR_TYPE.CTRNN.value: configure_oldstyle_layer_activations,
                    ACTIVATOR_TYPE.CONV_CTRNN.value: configure_oldstyle_layer_activations,
                    ACTIVATOR_TYPE.IAF.value: configure_oldstyle_layer_activations,
                    ACTIVATOR_TYPE.CONV_IAF.value: configure_oldstyle_layer_activations,
                    ACTIVATOR_TYPE.SOFT_MAX.value: configure_oldstyle_layer_activations,
                    ACTIVATOR_TYPE.RESERVOIR_CTRNN.value: configure_oldstyle_layer_activations,
                    ACTIVATOR_TYPE.RESERVOIR_IAF.value: configure_oldstyle_layer_activations,
                    ACTIVATOR_TYPE.SIGMOID.value: configure_layer_activations,
                    ACTIVATOR_TYPE.TANH.value: configure_layer_activations,
                    ACTIVATOR_TYPE.RELU.value: configure_layer_activations,
                    ACTIVATOR_TYPE.BOUNDED_RELU.value: configure_layer_activations,
                    ACTIVATOR_TYPE.NOISY_RELU.value: configure_layer_activations,
                    ACTIVATOR_TYPE.NOISY_SIGMOID.value: configure_layer_activations}


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

    return [calc_num_pixels(image_dimension, stride)
            for image_dimension in prev_image_dimensions]


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


class StandardAtariCNN(NervousSystem):
    """
    A 5-layer relu CNN based closely off of previous Atari playing CNN's.
    It uses the legal action set size for the motor layer so that it works
    with the shared motor agent (use this only with shared motor agents).
    """
    def __init__(self, *args, **kwargs):
        input_shape = [4, 88, 88]
        num_outputs = 18
        nn_parameters = [{
            'layer_type': "eigen_conv",
            'filter_shape': [7, 7],
            'num_filters': 32,
            'stride': 4},
            {'layer_type': "eigen_conv",
             'filter_shape': [3, 3],
             'num_filters': 64,
             'stride': 2
             },
            {'layer_type': "eigen_conv",
             'filter_shape': [3, 3],
             'num_filters': 64,
             'stride': 1
             },
            {'layer_type': "eigen_a2a_ff",
             'num_internal_nodes': 512,
             },
            {'layer_type': 'motor',
             'motor_type': 'eigen'}]
        act_type = ACTIVATOR_TYPE.RELU
        act_args = ()
        super().__init__(input_shape, num_outputs, nn_parameters, act_type, act_args)


# class RewardModulatedAtariCNN(NervousSystem):
#     """
#     A 5-layer relu CNN based closely off of previous Atari playing CNN's.
#     It uses the legal action set size for the motor layer so that it works
#     with the shared motor agent (use this only with shared motor agents).
#     It employs reward modulation which adjusted weights online.
#     """
#     def __init__(self, reward_smoothing_factor,
#                  activation_smoothing_factor,
#                  learning_rate, noise_strength, seeds):
#         """
#         Initialize RewardModulatedAtariCNN
#         :param reward_smoothing_factor: float, time constant of exp avg.
#         :param activation_smoothing_factor: float, time constant of exp avg.
#         :param learning_rate: float, determines size of weight change in hebb update.
#         :param noise_strength: float, standard deviation of noise.
#         :param seeds: list of ints, seeds for activator noise RNGs.
#         """
#         input_shape = [4, 88, 88]
#         num_outputs = 18
#         nn_parameters = [{
#             'layer_type': "rm_conv",
#             'filter_shape': [7, 7],
#             'num_filters': 32,
#             'stride': 4,
#             'reward_smoothing_factor': reward_smoothing_factor,
#             'activation_smoothing_factor': activation_smoothing_factor,
#             'learning_rate': learning_rate},
#             {'layer_type': "rm_conv",
#              'filter_shape': [3, 3],
#              'num_filters': 64,
#              'stride': 2,
#              'reward_smoothing_factor': reward_smoothing_factor,
#              'activation_smoothing_factor': activation_smoothing_factor,
#              'learning_rate': learning_rate
#              },
#             {'layer_type': "rm_conv",
#              'filter_shape': [3, 3],
#              'num_filters': 64,
#              'stride': 1,
#              'reward_smoothing_factor': reward_smoothing_factor,
#              'activation_smoothing_factor': activation_smoothing_factor,
#              'learning_rate': learning_rate
#              },
#             {'layer_type': "rm_a2a_ff",
#              'num_internal_nodes': 512,
#              'reward_smoothing_factor': reward_smoothing_factor,
#              'activation_smoothing_factor': activation_smoothing_factor,
#              'learning_rate': learning_rate
#              },
#             {'layer_type': 'motor',
#              'motor_type': 'rm',
#              'reward_smoothing_factor': reward_smoothing_factor,
#              'activation_smoothing_factor': activation_smoothing_factor,
#              'learning_rate': learning_rate}]
#         act_type = ACTIVATOR_TYPE.NOISY_RELU
#
#         act_args = [(noise_strength, seeds[i]) for i in range(5)]  # 5 is num layers
#         super().__init__(input_shape, num_outputs, nn_parameters, act_type, act_args)


class NoisyRewardModulatedAtariCNN(NervousSystem):
    """
    A 5-layer relu CNN based closely off of previous Atari playing CNN's.
    It uses the legal action set size for the motor layer so that it works
    with the shared motor agent (use this only with shared motor agents).
    It employs reward modulation which adjusted weights online.
    """
    def __init__(self, reward_smoothing_factor,
                 activation_smoothing_factor,
                 learning_rate, noise_strength, seeds, *args, **kwargs):
        """
        Initialize RewardModulatedAtariCNN
        :param reward_smoothing_factor: float, time constant of exp avg.
        :param activation_smoothing_factor: float, time constant of exp avg.
        :param learning_rate: float, determines size of weight change in hebb update.
        :param noise_strength: float, standard deviation of noise.
        :param seeds: list of ints, seeds for activator noise RNGs.
        """
        input_shape = [4, 88, 88]
        num_outputs = 18
        nn_parameters = [{
            'layer_type': "nrm_conv",
            'filter_shape': [7, 7],
            'num_filters': 32,
            'stride': 4,
            'reward_smoothing_factor': reward_smoothing_factor,
            'activation_smoothing_factor': activation_smoothing_factor,
            'standard_deviation': noise_strength,
            'seed': seeds[0],
            'learning_rate': learning_rate},
            {'layer_type': "nrm_conv",
             'filter_shape': [3, 3],
             'num_filters': 64,
             'stride': 2,
             'reward_smoothing_factor': reward_smoothing_factor,
             'activation_smoothing_factor': activation_smoothing_factor,
             'standard_deviation': noise_strength,
             'seed': seeds[1],
             'learning_rate': learning_rate
             },
            {'layer_type': "nrm_conv",
             'filter_shape': [3, 3],
             'num_filters': 64,
             'stride': 1,
             'reward_smoothing_factor': reward_smoothing_factor,
             'activation_smoothing_factor': activation_smoothing_factor,
             'standard_deviation': noise_strength,
             'seed': seeds[2],
             'learning_rate': learning_rate
             },
            {'layer_type': "nrm_a2a_ff",
             'num_internal_nodes': 512,
             'reward_smoothing_factor': reward_smoothing_factor,
             'activation_smoothing_factor': activation_smoothing_factor,
             'standard_deviation': noise_strength,
             'seed': seeds[3],
             'learning_rate': learning_rate
             },
            {'layer_type': 'motor',
             'motor_type': 'nrm',
             'reward_smoothing_factor': reward_smoothing_factor,
             'activation_smoothing_factor': activation_smoothing_factor,
             'standard_deviation': noise_strength,
             'seed': seeds[4],
             'learning_rate': learning_rate}]
        act_type = ACTIVATOR_TYPE.RELU
        act_args = ()
        super().__init__(input_shape, num_outputs, nn_parameters, act_type, act_args)
