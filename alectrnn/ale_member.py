from copy import deepcopy
import numpy as np
from alectrnn.multitask import *
import asyncevo
from typing import List
from typing import Dict
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
            agent_class=self._parameters['agent_class'],
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


class OperatorMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._x_mod = np.copy(self._x)
    
    def _modify_parameters(self, x):
        return x

    @property
    def parameters(self):
        return self._modify_parameters(self._x)

    @parameters.setter
    def parameters(self, value):
        raise NotImplementedError
    

class AbsMixin(OperatorMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _modify_parameters(self, x):
        return super()._modify_parameters(np.fabs(x, out=self._x_mod))


class SpikeMember(AbsMixin, AleMember):
    pass


class RescalingMixin(OperatorMixin):
    def __init__(self, parameter_rescalings: Dict[PARAMETER_TYPE, List[float]],
                 *args, **kwargs):
        """
        Excluding the input layer.
        :param parameter_rescalings: a dictionary keyed by parameter type and
        valued by a list of scalars that will be used to rescale the parameters.
        :param args: arguments for SpikeMember
        :param kwargs: keyword arguments for SpikeMember
        """
        super().__init__(*args, **kwargs)
        self._parameter_rescalings = parameter_rescalings
        self._parameter_layout = self._experiment.parameter_layout()
        self._layer_indices = self._experiment.parameter_layer_indices()

        self._scalings = np.zeros(len(self._parameter_layout), np.float32)
        for i in range(len(self._scalings)):
            # the first layer is inputs and has no parameters so -1 from indices
            self._scalings[i] = self._parameter_rescalings[
                PARAMETER_TYPE(self._parameter_layout[i])][
                self._layer_indices[i] - 1]

    def _modify_parameters(self, x):
        return super()._modify_parameters(np.multiply(x, self._scalings,
                                                      out=self._x_mod))


class RescaledSpikeMember(RescalingMixin, SpikeMember):
    pass


class NormalizationMixin(OperatorMixin):
    def __init__(self, weight_scale: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight_mask = self._experiment.parameter_layout() \
                            == PARAMETER_TYPE.WEIGHT.value
        self._weight_scale = weight_scale

    def _modify_parameters(self, x):
        weight_sum = np.sum(x, where=self._weight_mask)
        return super()._modify_parameters(
            np.divide(x, weight_sum / self._weight_scale,
                      where=self._weight_mask, out=self._x_mod))


class NormalizedSpikeMember(AbsMixin, NormalizationMixin, AleMember):
    pass


class RescaledNormalizeSpikeMember(AbsMixin, RescalingMixin, NormalizationMixin, AleMember):
    pass


class AleCSAMember(ALEAddon, asyncevo.CSAMember):
    pass


class AleDaignosticCSAMember(ALEAddon, asyncevo.DiagnosticCSAMember):
    pass
