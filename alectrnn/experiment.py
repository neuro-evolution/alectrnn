from alectrnn import handlers


class ALEExperiment:
    """
    This class is for creating and managing an ALE experiment. The user gives
    parameters for the game, objective, and neural network. The class setups
    the necessary handlers and the user can then access the objective function
    for use in
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
        :param agent_class: the class instance of the desired agent handler
        :param agent_class_parameters: dictionary of parameters for said class
        :param objective_parameters: parameters for the objective handler
        :param script_prefix: string that will be pre-pended to output files

        :note: parameters should leave out the handlers as these will be taken
        care of by the experiment. See handlers docs for list of available
        handlers and necessary parameters (can use help(class) in python). make
        sure to import handlers

        :note: rom name and # of parameters are added to the end of the script
        name automatically.
        """

        self._ale_handle = handlers.ALEHandler(**ale_parameters)
        self._ale_handle.create()

        self._nn_handle = nervous_system_class(**nervous_system_class_parameters)
        self._nn_handle.create()

        self._agent_handle = agent_class(self._ale_handle, self._nn_handle,
                                         **agent_class_parameters)
        self._agent_handle.create()

        self._obj_handle = handlers.ObjectiveHandler(self._ale_handle,
                                                     self._agent_handle,
                                                     **objective_parameters)
        self._obj_handle.create()

        self.script_prefix = script_prefix + "_" \
                             + self._ale_handle.parameters['rom'] + "_npar" \
                             + str(self.get_parameter_count())

    def get_parameter_count(self):
        """
        :return: The number of parameters in the neural network
        """
        return self._nn_handle.get_parameter_count()

    def parameter_layout(self):
        """
        :return: The parameters lay out as a numpy array valued by
            handlers.PARAMETER_TYPE values
        """
        return self._nn_handle.parameter_layout()

    def set_game_seed(self, seed):
        """
        Resets the games seed. This remakes the ale, agent and objective
        handles which depend upon setting from the game.
        :param seed: a 32-bit integer
        :return: None
        """
        self._ale_handle.seed(seed)
        self._agent_handle.ale = self._ale_handle.handle
        self._obj_handle.create()

    def set_logging(self, is_logging):
        """
        Changes the logging state of the experiment. Requires remaking the agent
        and objective handles.
        :param is_logging: True/False
        :return: None
        """

        self._agent_handle.logging = is_logging
        self._obj_handle.agent = self._agent_handle.handle
        self._obj_handle.create()

    def layer_history(self, layer_index):
        """
        :param layer_index: The index of the layer you want to get the history of
        :return: a numpy array of the state for each time-step
        """
        if isinstance(self._agent_handle, handlers.NervousSystemAgentHandler):
            return self._agent_handle.layer_history(layer_index)
        else:
            raise NotImplementedError("Layer history only available for"
                                      " handlers with logging")

    def num_layers(self):
        """
        :return: The number of layers in the neural network
        """
        return self._nn_handle.num_layers()


if __name__ == '__main__':
    pass
