from alectrnn.multitask import *
from alectrnn.ale_member import AleMember
from alectrnn.ale_member import ale_fitness_function
import numpy as np


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
    ga(**parameter_batch['training_parameters']['run_args'],
       save=True, save_filename=batch_id + "_" + str(index) + ".ga")


def execute_async_experiment(parameter_batch, index, batch_id):
    # initialize schedule
    from asyncevo.initialize import mpi_scheduler

    # initialize experiment
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
        initial_state=initial_guess,
        scheduler=mpi_scheduler,
        member_type=AleMember,
        member_type_kwargs=parameter_batch,
        save_filename=batch_id + "_" + str(index) + ".ga",
        **parameter_batch['training_parameters']['trainer_args'])
    ga.run(ale_fitness_function,
           **parameter_batch['training_parameters']['run_args'],
           take_member=True)
