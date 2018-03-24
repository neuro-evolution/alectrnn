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

from evostrat import BoundedRandNumTableES
from alectrnn import handlers
import numpy as np
import cProfile


################################################################################
# Experiment parameters (leave uncommented)
################################################################################
# Create ALE environment
rom = "atlantis"
ale_handle = handlers.ALEHandler(rom=rom, ale_seed=20, color_avg=True,
                                 max_num_frames=5000, max_num_episodes=2,
                                 max_num_frames_per_episode=5000)
ale_handle.create()

# Create NN
nn_handle = handlers.NervousSystem(input_shape=[1, 88, 88],
                                   num_outputs=ale_handle.action_set_size(),
                                   nn_parameters=[
                                       {'layer_type': "conv",
                                        'filter_shape': [8,8],
                                        'num_filters': 32,
                                        'stride': 4
                                        },
                                       {'layer_type': "conv",
                                        'filter_shape': [4,4],
                                        'num_filters': 64,
                                        'stride': 2
                                       },
                                       {'layer_type': "conv",
                                        'filter_shape': [3,3],
                                        'num_filters': 64,
                                        'stride': 1
                                       },
                                       {'layer_type': "a2a_ff",
                                        'num_internal_nodes': 512}
                                   ],
                                   act_type=handlers.ACTIVATOR_TYPE.CTRNN,
                                   act_args=(1.0,))

# Create Agent
agent_handle = handlers.NervousSystemAgentHandler(ale=ale_handle.handle,
                                                  nervous_system=nn_handle.neural_network,
                                                  update_rate=1,
                                                  logging=False)
agent_handle.create()

# Create objective
obj_handle = handlers.ObjectiveHandler(ale=ale_handle.handle,
                                       agent=agent_handle.handle,
                                       obj_type='totalcost',
                                       obj_parameters={})
obj_handle.create()

# Name save file
num_pars = nn_handle.get_parameter_count()
script_name = rom + "_npar" + str(num_pars) + "_pop240_google"

################################################################################
# Initiate evolutionary run (requires Experiment parameters)
################################################################################
# Define bounds
num_pars = nn_handle.get_parameter_count()
par_layout = nn_handle.parameter_layout()
type_bounds = {handlers.PARAMETER_TYPE.BIAS: (-100.0, 100.0),
               handlers.PARAMETER_TYPE.RTAUS: (0.0001, 100.0),
               handlers.PARAMETER_TYPE.WEIGHT: (-100.0, 100.0)}
bounds = handlers.boundary_array_for_parameter_layout(par_layout, type_bounds)

# Initial guess
rng = np.random.RandomState(1)
guess_bounds = {handlers.PARAMETER_TYPE.BIAS: (-10.0, 10.0),
                handlers.PARAMETER_TYPE.RTAUS: (0.001, 10.0),
                handlers.PARAMETER_TYPE.WEIGHT: (-10.0, 10.0)}
guess_range = handlers.boundary_array_for_parameter_layout(par_layout,
                                                           guess_bounds)
initial_guess = handlers.draw_uniform_initial_guess(rng, guess_range)
del guess_range
del par_layout

# Start profiler
alectrnn_profiler = cProfile.Profile()
alectrnn_profiler.enable()

es = BoundedRandNumTableES(xo=initial_guess,
                           step_size=1.0,
                           bounds=bounds,
                           objective=obj_handle.handle,
                           seed=6,
                           verbose=True,
                           rand_num_table_size=20000000)
del initial_guess

# Run evolution
es(50)
es.save(script_name + ".es")

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
# es = BoundedRandNumTableES.load(script_name + ".es")
# es(50, obj_handle.handle)
# es.save(script_name + ".es")

################################################################################
# Load and plot results (requires Experiment parameters)
################################################################################
# es = BoundedRandNumTableES.load(script_name + ".es")
# es.plot_cost_over_time(script_name, savefile=True, logy=False)

################################################################################
# Evaluate
################################################################################
frames = 18000
trials = 30
new_seed = 32535743
ale_evaluation_handle = ale_handle

################################################################################
# Record neural activity
################################################################################
