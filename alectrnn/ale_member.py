from copy import deepcopy
import numpy as np
from alectrnn.multitask import *
import asyncevo
from alectrnn.nervous_system import PARAMETER_TYPE


def ale_fitness_function(member):
    return -member.experiment.objective_function(member.parameters)


class ALEAddon:
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


class AleMember(ALEAddon, asyncevo.Member):
    pass


class SpikeMember(AleMember):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._abs_x = np.copy(self._x)

    @property
    def parameters(self):
        return np.fabs(self._x, out=self._abs_x)

    @parameters.setter
    def parameters(self, value):
        raise NotImplementedError


class NormalizedSpikeMember(AleMember):
    def __init__(self, weight_scale: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight_mask = self._experiment.parameter_layout() \
            == PARAMETER_TYPE.WEIGHT.value
        self._weight_scale = weight_scale
        self._norm_x = np.copy(self._x)

    @property
    def parameters(self):
        np.fabs(self._x, out=self._norm_x)
        weight_sum = np.sum(self._norm_x, where=self._weight_mask)
        np.divide(self._norm_x, weight_sum / self._weight_scale,
                  where=self._weight_mask, out=self._norm_x)
        return self._norm_x

    @parameters.setter
    def parameters(self, value):
        raise NotImplementedError


class AleCSAMember(ALEAddon, asyncevo.CSAMember):
    pass


class AleDaignosticCSAMember(ALEAddon, asyncevo.DiagnosticCSAMember):
    pass
