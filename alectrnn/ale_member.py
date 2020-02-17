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

        # initialize experiment
        for ref, cost in \
                parameter_batch['cost_normalization_parameters']['costs'].items():
            parameter_batch['normalizer'].internal_log[ref] = cost

        self._experiment = parameter_batch['experiment'](
            parameter_batch['experiment_parameters']['roms'],
            CostNormalizer(parameter_batch['normalizer']),
            ale_parameters=parameter_batch['ale_parameters'],
            nervous_system_class=parameter_batch['nervous_system_class'],
            nervous_system_class_parameters=parameter_batch['nervous_system_parameters'],
            agent_class_parameters=parameter_batch['agent_parameters'],
            objective_parameters=parameter_batch['objective_parameters']
        )

    @property
    def experiment(self):
        return self._experiment

    @experiment.setter
    def experiment(self, value):
        raise NotImplementedError
