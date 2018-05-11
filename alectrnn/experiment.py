from alectrnn import handlers


class ALEExperiment:
    """
    This class is for creating and managing an ALE experiment. The user gives
    parameters for the game, objective, and neural network. The class setups
    the necessary handlers and the user can then access the objective function
    for use in optimization
    """

    def __init__(self, ale_parameters, nervous_system_class,
                 nervous_system_class_parameters, agent_class_parameters,
                 objective_parameters, script_prefix="test"):
        """
        Initializes the experiment.

        :param ale_parameters: dictionary of parameters for the ALE handler
        :param nervous_system_class: the class instance of the desired neural
            network class you wish to use
        :param nervous_system_class_parameters: parameters for that class
            instance
        :param agent_class_parameters: dictionary of parameters for
            NervousSystemAgentHandler class.
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
        """

        # Save parameters
        self.ale_parameters = ale_parameters
        self.nervous_system_class_parameters = nervous_system_class_parameters
        self.agent_class_parameters = agent_class_parameters
        self.objective_parameters = objective_parameters

        # Construct handles (ALE)
        self._ale_handle = handlers.ALEHandler(**ale_parameters)
        self._ale_handle.create()

        # Construct handle (Nervous system)
        self.nervous_system_class_parameters['num_outputs'] = \
            self._ale_handle.action_set_size()

        self._nervous_system = nervous_system_class(**self.nervous_system_class_parameters)

        # Construct handle (Agent)
        self.agent_class_parameters['nervous_system'] = self._nervous_system.neural_network
        self._agent_handle = handlers.NervousSystemAgentHandler(self._ale_handle.handle,
                                                                **self.agent_class_parameters)
        self._agent_handle.create()

        # Construct handle (objective)
        self._obj_handle = handlers.ObjectiveHandler(self._ale_handle.handle,
                                                     self._agent_handle.handle,
                                                     **self.objective_parameters)
        self._obj_handle.create()

        self.script_prefix = script_prefix + "_" \
                             + self._ale_handle.rom + "_npar" \
                             + str(self.get_parameter_count())

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

    def get_parameter_count(self):
        """
        :return: The number of parameters in the neural network
        """
        return self._nervous_system.get_parameter_count()

    def parameter_layout(self):
        """
        :return: The parameters lay out as a numpy array valued by
            handlers.PARAMETER_TYPE values
        """
        return self._nervous_system.parameter_layout()

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

    @property
    def objective_function(self):
        return self._obj_handle.handle


if __name__ == '__main__':
    pass
