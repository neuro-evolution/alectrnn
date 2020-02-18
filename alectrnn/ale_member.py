from copy import deepcopy
from alectrnn.multitask import *
import asyncevo


def ale_fitness_function(member):
    return -member.experiment.objective_function(member.parameters)


class AleMember(asyncevo.Member):
    """
    Constructs an experiment environment inside a member so that it will be
    initialized in each process.
    """
    def __init__(self, parameter_batch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parameters = deepcopy(parameter_batch)
        # initialize experiment
        for ref, cost in \
                self._parameters['cost_normalization_parameters']['costs'].items():
            self._parameters['normalizer'].internal_log[ref] = cost

        self._experiment = self._parameters['experiment'](
            self._parameters['experiment_parameters']['roms'],
            CostNormalizer(self._parameters['normalizer']),
            ale_parameters=self._parameters['ale_parameters'],
            nervous_system_class=self._parameters['nervous_system_class'],
            nervous_system_class_parameters=self._parameters['nervous_system_parameters'],
            agent_class_parameters=self._parameters['agent_parameters'],
            objective_parameters=self._parameters['objective_parameters']
        )

    @property
    def experiment(self):
        return self._experiment

    @experiment.setter
    def experiment(self, value):
        raise NotImplementedError
