from alectrnn import handlers
from alectrnn import nervous_system as ns
from abc import ABC, abstractmethod
from alectrnn import multitask
import copy


class ALEExperimentBase(ABC):
    """
    Defines functions that all experiment classes should have
    """

    def __init__(self, ale_parameters, nervous_system_class,
                 nervous_system_class_parameters, agent_class,
                 agent_class_parameters, objective_parameters,
                 script_prefix="test"):
        """
        Initializes the experiment.

        :param ale_parameters: dictionary of parameters for the ALE handler
        :param nervous_system_class: the class instance of the desired neural
            network class you wish to use
        :param nervous_system_class_parameters: parameters for that class
            instance
        :param agent_class: chosen agent class
            options: SoftMaxAgentHandler, NervousSystemAgentHandler,
                    SharedMotorAgentHandler
            Any class is fine so long as it inherits the LoggingAndHistoryMixin
        :param agent_class_parameters: dictionary of parameters for
            agent class.
        :param objective_parameters: parameters for the objective handler
            If obj_type == 'totalcost': No parameters needed
            if obj_type == 's&cc': the following keys are required:
                :param cc_scale: adjusts contribution of connections to cost

        :param script_prefix: string that will be pre-pended to output files

        :note: parameters should leave out the handlers as these will be taken
        care of by the experiment. See handlers docs for list of available
        handlers and necessary parameters (can use help(class) in python). make
        sure to import handlers. Also, the # of outputs will automatically be
        added to the nervous_system_class_parameters from game specifications.
        The nervous_system will also automatically be added for agent
        parameters.

        :note: rom name and # of parameters are added to the end of the script
        name automatically.

        :note: parameters passed in are modified.
        """

        # Save parameters
        self.nervous_system_class = nervous_system_class
        self.agent_class = agent_class
        self.ale_parameters = ale_parameters
        self.nervous_system_class_parameters = nervous_system_class_parameters
        self.agent_class_parameters = agent_class_parameters
        self.objective_parameters = objective_parameters
        self.script_prefix = script_prefix

    def check_configuration_conflicts(self):
        """
        Some configurations maynot be compatible with others, so this function
        houses a check for such misc issues
        :return: 1, else raise error
        """

        # Check to make sure softmax agents get softmax motor layer
        if (self.agent_class == handlers.SoftMaxAgentHandler) and (
                self.nervous_system_class_parameters['nn_parameters'][-1]['motor_type'].lower()
                != 'softmax'):

            raise AssertionError("SoftMaxAgents must have a softmax motor layer.")

    def construct_nervous_system(self, nervous_system_class,
                                 nervous_system_parameters, agent_class,
                                 ale_handle):
        """
        :param nervous_system_class: class used to generate the nervous system
        :param nervous_system_parameters: parameters for the class
        :param agent_class: class used for generating the agent
        :param ale_handle: a built ale handler object
        :return: a NervousSystem object
        """

        if isinstance(agent_class, handlers.SharedMotorAgentHandler)\
                or isinstance(agent_class, handlers.FeedbackAgentHandler):
            # SharedMotorAgents require the legal rather than minimal action
            # set size to be used for the motor layer
            nervous_system_parameters['num_outputs'] = \
                ale_handle.legal_action_set_size()
        else:
            nervous_system_parameters['num_outputs'] = \
                ale_handle.action_set_size()

        return nervous_system_class(**nervous_system_parameters)

    def construct_ale_handle(self, ale_parameters):
        """
        :param ale_parameters: dictionary of ale parameters
        :return: an operational handle
        """
        ale_handle = handlers.ALEHandler(**ale_parameters)
        ale_handle.create()
        return ale_handle

    def construct_agent_handle(self, agent_class, partial_class_parameters,
                               nervous_system, ale_handle):
        """
        Builds and initializes the agent handle and creates the c++-object
        :param agent_class: class used for making the agent
        :param partial_class_parameters: dictionary that includes everything
            but the nervous system, since that will be built locally
        :param nervous_system: the built nervous system
        :param ale_handle: an ale_handle
        :return: an agent handler object
        """
        partial_class_parameters['nervous_system'] = nervous_system.neural_network
        agent_handle = agent_class(ale_handle.handle, **partial_class_parameters)
        agent_handle.create()
        return agent_handle

    def construct_objective_handle(self, objective_parameters, ale_handle,
                                   agent_handle):
        """
        :param objective_parameters: dictionary of parameters for the objective
        :param ale_handle: a built ale handler object
        :param agent_handle: a built agent handler object
        :return: an objective handler object
        """
        obj_handle = handlers.ObjectiveHandler(ale_handle.handle,
                                               agent_handle.handle,
                                               **objective_parameters)
        obj_handle.create()
        return obj_handle

    def parameter_layout(self):
        """
        :return: The parameters lay out as a numpy array valued by
            handlers.PARAMETER_TYPE values
        """
        return self._nervous_system.parameter_layout()

    def parameter_layer_indices(self):
        """
        :return: The layer indices corresponding to each parameter.
        """
        return self._nervous_system.parameter_layer_indices()

    def get_parameter_count(self):
        """
        :return: The number of parameters in the neural network
        """
        return self._nervous_system.get_parameter_count()

    def print_layer_shapes(self):
        """
        Prints the shapes calculated for each layer
        :note: not to be confused with interpreted shapes
        """

        for i, shape in enumerate(self._nervous_system.layer_shapes):
            print("Layer", i, " with shape", shape)

    def num_layers(self):
        """
        :return: The number of layers in the neural network
        """
        return self._nervous_system.num_layers()

    def draw_initial_guess(self, type_bounds, rng, normalized_weights=True,
                           norm_type='sqrt'):
        """
        :param type_bounds: a dictionary with low/high bounds for each type.
            If weights are set too 'norm', then they are initialized with a
            normalization procedure.
        :param rng: a seeded numpy RandomState
        :param normalized_weights: True (ignores WEIGHTS bounds and chooses
            weights between [-1/N, 1/N], where N==# of pre-synaptic connections)
        :param norm_type: 'sqrt' for 1/sqrt(N) (default)
                          'norm' for 1/N
        :return: a 1D numpy float 32 array
        """

        return ns.draw_initial_guess(type_bounds,
                                     self._nervous_system,
                                     rng, normalized_weights, norm_type)

    def draw_layerwise_initial_guess(self, layer_bounds, rng):
        """
        :param layer_bounds: List[Dict[key is PARAMETER_TYPE, value is (low, high)]]
        :param rng: a seeded numpy RandomState
        :return: a 1D numpy float 32 array
        """
        return ns.layerwise_initial_guess(layer_bounds,
                                          self._nervous_system,
                                          rng)

    @property
    @abstractmethod
    def objective_function(self):
        """
        :return: returns the objective function
        """
        pass


class ALEExperiment(ALEExperimentBase):
    """
    This class is for creating and managing an ALE experiment. The user gives
    parameters for the game, objective, and neural network. The class setups
    the necessary handlers and the user can then access the objective function
    for use in optimization
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Construct handles (ALE)
        self._ale_handle = self.construct_ale_handle(self.ale_parameters)

        # Construct handle (Nervous system)
        self._nervous_system = self.construct_nervous_system(self.nervous_system_class,
                                                             self.nervous_system_class_parameters,
                                                             self.agent_class,
                                                             self._ale_handle)

        # Construct handle (Agent)
        self._agent_handle = self.construct_agent_handle(self.agent_class,
                                                         self.agent_class_parameters,
                                                         self._nervous_system,
                                                         self._ale_handle)

        # Construct handle (objective)
        self._obj_handle = self.construct_objective_handle(self.objective_parameters,
                                                           self._ale_handle,
                                                           self._agent_handle)

        self.script_prefix = self.script_prefix + "_" \
                             + self._ale_handle.rom + "_npar" \
                             + str(self.get_parameter_count())

        self.check_configuration_conflicts()

    def screen_history(self):
        """
        :return: numpy array of the screen state for each time-step
        """
        return self._agent_handle.screen_history()

    def layer_history(self, layer_index):
        """
        :param layer_index: The index of the layer you want to get the history of
        :return: a numpy array of the state for each time-step
        """
        return self._agent_handle.layer_history(layer_index)

    def set_objective_function(self, objective_parameters):
        """
        Re-makes the objective handle based on given objective parameters
        :param objective_parameters: dictionary with objective parameters and
            objective type
        :return: None
        """
        self._obj_handle = handlers.ObjectiveHandler(self._ale_handle.handle,
                                                     self._agent_handle.handle,
                                                     **objective_parameters)
        self._obj_handle.create()

    def set_game_parameters(self, **kwargs):
        """
        :param kwargs: any keyword arguments used by the ale environment
        :return: None
        """

        self._ale_handle.set_parameters(**kwargs)
        self._update_game_environment()

    def set_game_seed(self, seed):
        """
        Resets the games seed. This remakes the ale, agent and objective
        handles which depend upon setting from the game.
        :param seed: a 32-bit integer
        :return: None
        """
        self._ale_handle.seed(seed)
        self._update_game_environment()

    def set_logging(self, is_logging):
        """
        Changes the logging state of the experiment. Requires remaking the agent
        and objective handles.
        :param is_logging: True/False
        :return: None
        """

        self._agent_handle.logging = is_logging
        self._obj_handle.agent = self._agent_handle.handle

    def _update_game_environment(self):
        """
        updates the necessary handles when the game environment is updated
        """
        self._agent_handle.ale = self._ale_handle.handle
        self._obj_handle.agent = self._agent_handle.handle
        self._obj_handle.ale = self._ale_handle.handle

    @property
    def objective_function(self):
        return self._obj_handle.handle


class ALERandomRomExperiment(ALEExperimentBase):
    """
    An ALE experiment
    """
    def __init__(self, roms, cost_normalizer, seed, **kwargs):
        """
        :param roms: a list of roms to choose from randomly
        :param cost_normalizer: a CostNormalizer
        :param seed: seed for drawing random roms to play
        :param kwargs: ALEExperimentBase arguments, except for agent class
            which must be a shared motor handler

        *ale parameters don't have to include rom (it will be overwritten)
        """
        kwargs['agent_class'] = handlers.SharedMotorAgentHandler
        super().__init__(**kwargs)

        # Construct handles (ALE)
        self.ale_parameters['rom'] = 'seaquest'  # give dummy rom for initialization
        self._ale_handle = self.construct_ale_handle(self.ale_parameters)

        # Construct handle (Nervous system)
        self._nervous_system = self.construct_nervous_system(self.nervous_system_class,
                                                             self.nervous_system_class_parameters,
                                                             self.agent_class,
                                                             self._ale_handle)

        # Construct handle (Agent)
        self._agent_handle = self.construct_agent_handle(self.agent_class,
                                                         self.agent_class_parameters,
                                                         self._nervous_system,
                                                         self._ale_handle)

        # Construct handle (objective)
        self._obj_handle = self.construct_objective_handle(self.objective_parameters,
                                                           self._ale_handle,
                                                           self._agent_handle)

        self.script_prefix = self.script_prefix + "_" \
                             + "randrom" + "_npar" \
                             + str(self.get_parameter_count())

        self.check_configuration_conflicts()

        self.roms = roms
        self.cost_normalizer = cost_normalizer
        self.seed = seed
        self._objective = multitask.RandomRomObjective(self.roms,
                                                       self._ale_handle,
                                                       self._agent_handle,
                                                       self._obj_handle,
                                                       self.cost_normalizer,
                                                       self.seed)

    @property
    def objective_function(self):
        return self._objective


class ALEMultiRomExperiment(ALEExperimentBase):
    """
    An ALE experiment that runs several ROMs for each objective call.
    The sum of the performance on all tasks is returned.
    """
    def __init__(self, roms, cost_normalizer, **kwargs):
        """
        :param roms: a list of roms to run when the objective is called
        :param cost_normalizer: a CostNormalizer
        :param kwargs: ALEExperimentBase arguments, except agent class, which
            must be a shared motor handler

        *ale parameters don't have to include rom (it will be overwritten)
        *agents are built from a specific rom, so we need 1 agent and 1 objective
        for each rom. The nervous system can be shared.
        """
        kwargs['agent_class'] = handlers.SharedMotorAgentHandler
        super().__init__(**kwargs)

        # Construct handles (ALE)
        self._ale_handlers = []
        for rom in roms:
            ale_parameters = copy.copy(self.ale_parameters)
            ale_parameters['rom'] = rom
            self._ale_handlers.append(self.construct_ale_handle(ale_parameters))

        # Construct handle (Nervous system)
        self._nervous_system = self.construct_nervous_system(self.nervous_system_class,
                                                             self.nervous_system_class_parameters,
                                                             self.agent_class,
                                                             self._ale_handlers[0])

        # Construct handles (Agent)
        self._agent_handlers = []
        for ale_handler in self._ale_handlers:
            self._agent_handlers.append(self.construct_agent_handle(
                self.agent_class,
                self.agent_class_parameters,
                self._nervous_system,
                ale_handler))

        # Construct handles (objective)
        self._objective_handlers = []
        for i in range(len(roms)):
            self._objective_handlers.append(self.construct_objective_handle(
                self.objective_parameters,
                self._ale_handlers[i],
                self._agent_handlers[i]))

        self.script_prefix = self.script_prefix + "_" \
                             + "multirom" + "_npar" \
                             + str(self.get_parameter_count())

        self.check_configuration_conflicts()

        self.roms = roms
        self.rom_objective_map = {self.roms[i]: self._objective_handlers[i]
                                  for i in range(len(self.roms))}
        self.cost_normalizer = cost_normalizer
        self._objective = self._initialize_objective()

    @abstractmethod
    def _initialize_objective(self):
        """
        Called on construction
        :return: an objective object
        """
        pass

    @property
    def objective_function(self):
        return self._objective


class ALEMultiRomMeanExperiment(ALEMultiRomExperiment):
    """
    An ALE experiment that runs several ROMs for each objective call.
    The mean of the performance on all tasks is returned.
    """

    def _initialize_objective(self):
        return multitask.MultiRomMeanObjective(self.rom_objective_map,
                                               self.cost_normalizer)


class ALEMultiRomProductExperiment(ALEMultiRomExperiment):
    """
    An ALE experiment that runs several ROMs for each objective call.
    The product of the performance on all tasks is returned.
    """

    def _initialize_objective(self):
        return multitask.MultiRomMultiplicativeObjective(
            self.rom_objective_map, self.cost_normalizer)
