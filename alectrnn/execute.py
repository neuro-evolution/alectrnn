from copy import deepcopy
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
    from asyncevo import Scheduler
    with Scheduler(parameter_batch['storage_parameters'].get('initialization_args', {}),
                   parameter_batch['storage_parameters'].get('client_args', {})) \
                    as mpi_scheduler:

        # initialize experiment in order to generate initial state
        working_parameters = deepcopy(parameter_batch)
        for ref, cost in \
                working_parameters['cost_normalization_parameters']['costs'].items():
            working_parameters['normalizer'].internal_log[ref] = cost

        experiment = working_parameters['experiment'](
            working_parameters['experiment_parameters']['roms'],
            CostNormalizer(working_parameters['normalizer']),
            ale_parameters=working_parameters['ale_parameters'],
            nervous_system_class=working_parameters['nervous_system_class'],
            nervous_system_class_parameters=working_parameters['nervous_system_parameters'],
            agent_class_parameters=working_parameters['agent_parameters'],
            objective_parameters=working_parameters['objective_parameters']
        )

        initial_state_rng = np.random.RandomState(
            working_parameters['training_parameters']['seed'])
        guess_bounds = working_parameters['training_parameters']['bounds']
        initial_guess = experiment.draw_layerwise_initial_guess(guess_bounds,
                                                                initial_state_rng)
        del experiment
        del working_parameters

        ga = parameter_batch['trainer'](
            initial_state=initial_guess,
            scheduler=mpi_scheduler,
            member_type=AleMember,
            member_type_kwargs={'parameter_batch': parameter_batch},
            save_filename=batch_id + "_" + str(index) + ".ga",
            **parameter_batch['training_parameters']['trainer_args'])
        ga.run(ale_fitness_function,
               **parameter_batch['training_parameters']['run_args'],
               take_member=True)


def execute_async_batch(batch):
    from asyncevo import Scheduler
    # assumes scheduler parameters are the same throughout
    with Scheduler(batch[0]['storage_parameters'].get('initialization_args', {}),
                   batch[0]['storage_parameters'].get('client_args', {})) \
                    as mpi_scheduler:

        for index, parameter_batch in enumerate(batch):
            # initialize experiment in order to generate initial state
            working_parameters = deepcopy(parameter_batch)
            for ref, cost in \
                    working_parameters['cost_normalization_parameters']['costs'].items():
                working_parameters['normalizer'].internal_log[ref] = cost

            experiment = working_parameters['experiment'](
                working_parameters['experiment_parameters']['roms'],
                CostNormalizer(working_parameters['normalizer']),
                ale_parameters=working_parameters['ale_parameters'],
                nervous_system_class=working_parameters['nervous_system_class'],
                nervous_system_class_parameters=working_parameters['nervous_system_parameters'],
                agent_class_parameters=working_parameters['agent_parameters'],
                objective_parameters=working_parameters['objective_parameters']
            )

            initial_state_rng = np.random.RandomState(
                working_parameters['training_parameters']['seed'])
            guess_bounds = working_parameters['training_parameters']['bounds']
            initial_guess = experiment.draw_layerwise_initial_guess(guess_bounds,
                                                                    initial_state_rng)
            del experiment
            del working_parameters

            ga = parameter_batch['trainer'](
                initial_state=initial_guess,
                scheduler=mpi_scheduler,
                member_type=AleMember,
                member_type_kwargs={'parameter_batch': parameter_batch},
                save_filename=batch_id + "_" + str(index) + ".ga",
                **parameter_batch['training_parameters']['trainer_args'])
            ga.run(ale_fitness_function,
                   **parameter_batch['training_parameters']['run_args'],
                   take_member=True)
