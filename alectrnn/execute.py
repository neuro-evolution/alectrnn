from alectrnn import batching
from alectrnn.multitask import *
import numpy as np
from evostrat import *


def execute_experiment(parameter_batch, index, batch_id):
    for ref, cost in \
            parameter_batch['cost_normalization_parameters']['costs'].items():
        parameter_batch['normalizer'].internal_log[ref] = cost

    experiment = parameter_batch['experiment'](
        parameter_batch['experiment_parameters']['roms'],
        CostNormalizer(parameter_batch['normalizer']),
        ale_parameters=parameter_batch['ale_parameters'],
        nervous_system_class=parameter_batch['nervous_system_class'],
        nervous_system_class_parameters=parameter_batch['nervous_system_parameters'],
        agent_class_parameters=parameter_batch['agent_parameters'],
        objective_parameters=parameter_batch['objective_parameters']
    )

    initial_state_rng = np.random.RandomState(
        parameter_batch['training_parameters']['seed'])
    guess_bounds = parameter_batch['training_parameters']['bounds']
    initial_guess = experiment.draw_initial_guess(guess_bounds,
                                                  initial_state_rng,
                                                  normalized_weights=False)
    ga = parameter_batch['trainer'](
        initial_guess=initial_guess,
        objective=experiment.objective_function,
        **parameter_batch['training_parameters']['trainer_args'])
    ga(parameter_batch['training_parameters']['run_args'],
       save=True, save_filename=batch_id + "_" + str(index) + ".ga")
