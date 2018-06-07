"""
This is an example script that shows how you can run ALECTRNN using evolutionary
strategies. Running the script requires the evostrat package from github:
https://github.com/Nathaniel-Rodriguez/evostrat

It also requires a working installation of mpi4py, with some accompanying
MPI software like MPICH or OpenMPI. Only the evolutionary run section should
be run in MPI mode. The sections following that are just for analysis and should
be run without mpi.

To execute the script for optimization use:

mpiexec -n 240 python alectrnn_experiment_template.py

To execute the script for analysis use:

python alectrnn_experiment_template.py

"""

from evostrat import SusRandNumTableGA
from alectrnn import nervous_system as ns
from alectrnn.handlers import NervousSystemAgentHandler
from alectrnn.experiment import ALEExperiment
import numpy as np
import cProfile


################################################################################
# Experiment parameters (leave uncommented)
################################################################################
# Create ALE environment
ale_parameters = {'rom': "atlantis",
                  'seed': 12,
                  'color_avg': True,
                  'max_num_frames': 20000,
                  'max_num_episodes': 1,
                  'max_num_frames_per_episode': 20000,
                  'frame_skip': 4,
                  'use_environment_distribution': True,
                  'system_reset_steps': 4,
                  'num_random_environments': 30}

nervous_system_parameters = {'input_shape': [1, 88, 88],
                             'nn_parameters': [{
                                'layer_type': "conv",
                                'filter_shape': [8, 8],
                                'num_filters': 32,
                                'stride': 4},
                               {'layer_type': "conv",
                                'filter_shape': [4, 4],
                                'num_filters': 64,
                                'stride': 2
                                },
                               {'layer_type': "conv",
                                'filter_shape': [3, 3],
                                'num_filters': 64,
                                'stride': 1
                                },
                               {'layer_type': "a2a_ff",
                                'num_internal_nodes': 512
                                },
                               {'layer_type': 'motor',
                                'motor_type': 'standard'}],
                             'act_type': ns.ACTIVATOR_TYPE.CTRNN,
                             'act_args': (1.0,)}

agent_parameters = {'update_rate': 1,
                    'logging': False}

objective_parameters = {'obj_type': 'totalcost',
                        'obj_parameters': None}

ale_experiment = ALEExperiment(ale_parameters=ale_parameters,
                               nervous_system_class=ns.NervousSystem,
                               nervous_system_class_parameters=nervous_system_parameters,
                               agent_class=NervousSystemAgentHandler,
                               agent_class_parameters=agent_parameters,
                               objective_parameters=objective_parameters,
                               script_prefix='example')
par_layout = ale_experiment.parameter_layout()

################################################################################
# Initiate evolutionary run (requires Experiment parameters)
################################################################################
# Start profiler
alectrnn_profiler = cProfile.Profile()
alectrnn_profiler.enable()

initial_state_rng = np.random.RandomState(5)
guess_bounds = {ns.PARAMETER_TYPE.RTAUS: (0.5, 1.5),
                ns.PARAMETER_TYPE.RANGE: (1.0, 1.0),
                ns.PARAMETER_TYPE.REFRACTORY: (0.3, 0.3),
                ns.PARAMETER_TYPE.RESISTANCE: (1.0, 1.0),
                ns.PARAMETER_TYPE.WEIGHT: (-0.1, 0.1)}
initial_guess = ale_experiment.draw_initial_guess(guess_bounds,
                                                  initial_state_rng,
                                                  normalized_weights=False)
ga = SusRandNumTableGA(initial_guess=initial_guess,
                       sigma=0.1,
                       num_elite=5,
                       objective=ale_experiment.objective_function,
                       seed=7,
                       verbose=True,
                       rand_num_table_size=20000000)

# Run evolution
ga(100, save=True, save_every=10, save_filename=ale_experiment.script_prefix + ".ga")

# Record results
alectrnn_profiler.disable()
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
if rank == 0:
    alectrnn_profiler.print_stats(sort='cumtime')

################################################################################
# Load and continue evolution (requires Experiment parameters) (optional)
################################################################################
# ga = SusRandNumTableGA.load(ale_experiment.script_prefix + ".ga")
# ale_experiment.set_game_parameters(seed=15, max_num_frames=10000,
#                                    max_num_frames_per_episode=10000)
# ga(50, ale_experiment.objective_function, save=True,
#    save_filename=ale_experiment.script_prefix + ".ga")

################################################################################
# Load and plot results (requires Experiment parameters)
################################################################################
# ga = SusRandNumTableGA.load(ale_experiment.script_prefix + ".ga")
# ga.plot_cost_over_time(ale_experiment.script_prefix, savefile=True, logy=False)

################################################################################
# Evaluate
################################################################################
# # In the google paper, they ran for 5min (18000 frames) for 30 games
# frames = 18000
# trials = 2
# max_episodes = 10
# new_seed = 32535742
#
# # get best
# ga = SusRandNumTableGA.load(ale_experiment.script_prefix + ".ga")
# costs = []
# # run for number of trials
# ale_experiment.set_objective_function({'obj_type': 'totalcost'})
# for i in range(trials):
#     ale_experiment.set_game_parameters(seed=new_seed+i, max_num_frames=frames,
#                                        max_num_frames_per_episode=frames,
#                                        max_num_episodes=max_episodes)
#     costs.append(ale_experiment.objective_function(ga.best))
#     print("trial", i, "complete")
#
# print("costs: ", costs)
# print("mean cost: ", np.mean(costs))
# print("total cost: ", np.sum(costs))

################################################################################
# MPI - Evaluate
################################################################################
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()
#
# # In the google paper, they ran for 5min (18000 frames) for 30 games
# frames = 18000
# max_episodes = 2
# trials = size
# new_seed = 32535742
# seeds = [new_seed + i for i in range(size)]
#
# # get best
# ga = SusRandNumTableGA.load(ale_experiment.script_prefix + ".ga")
# local_cost = np.empty(1, dtype=np.float32)
# ale_experiment.set_objective_function({'obj_type': 'totalcost'})
# ale_experiment.set_game_parameters(seed=seeds[rank], max_num_frames=frames,
#                                    max_num_frames_per_episode=frames,
#                                    max_num_episodes=max_episodes)
# local_cost[0] = ale_experiment.objective_function(ga.best)
# all_costs = np.empty(size, dtype=np.float32)
# comm.Allgather([local_cost, MPI.FLOAT],
#                [all_costs, MPI.FLOAT])
#
# if rank == 0:
#     print("costs: ", all_costs)
#     print("mean cost: ", np.mean(all_costs))
#     print("total cost: ", np.sum(all_costs))

################################################################################
# Record neural activity
################################################################################
# from alectrnn import analysis_tools
# # get best
# ga = SusRandNumTableGA.load(ale_experiment.script_prefix + ".ga")
# ale_experiment.set_logging(True)
# ale_experiment.set_game_parameters(seed=20, max_num_frames=6000,
#                                    max_num_episodes=1,
#                                    max_num_frames_per_episode=6000)
#
# print("running obj")
# print("cost: ", ale_experiment.objective_function(ga.best))
#
# # Print screen history
# screen_history = ale_experiment.screen_history()
# analysis_tools.animate_screen(screen_history, ale_experiment.script_prefix)

# # Get neural system history
# for layer_index in range(ale_experiment.num_layers()):
#     print("layer...", layer_index)
#     history = ale_experiment.layer_history(layer_index)
#     print("\t has shape: ", history.shape)
#     if layer_index == 0:
#         # analysis_tools.animate_input(history, (88, 88), ale_experiment.script_prefix)
#         pass
#     elif layer_index == (ale_experiment.num_layers() - 1):
#         analysis_tools.plot_output(history, ale_experiment.script_prefix)
#     else:
#         analysis_tools.plot_internal_state_distribution(history, layer_index,
#                                                         ale_experiment.script_prefix)
#
# # Plot trajectories of specific neurons
# layer_index = ale_experiment.num_layers()-1
# history = ale_experiment.layer_history(layer_index)
# analysis_tools.plot_internal_spike_train(history, index=layer_index,
#                                    neuron_ids=[1, 2, 3, 4],
#                                    prefix=ale_experiment.script_prefix)

################################################################################
# Plot parameter distributions
################################################################################
# from alectrnn import analysis_tools
# ga = SusRandNumTableGA.load(ale_experiment.script_prefix + ".ga")
# analysis_tools.plot_parameter_distributions(ga.best, par_layout, marker='None',
#                                             prefix=ale_experiment.script_prefix)
