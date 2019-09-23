import numpy as np
from alectrnn import handlers
from random import Random
import copy
from alectrnn.analysis_tools import animate_screen


class ALEExperimentManager:
    """
    A wrapper meant to consolidate a lot of duplicate code.

    Can plot results.
    Requires mpi4py if using MPI evaluation.
    """

    def __init__(self, ale_experiment, optimization_class, file_extension=None):
        """
        :param ale_experiment: A completed ALE experimental run
        :param optimization_class: The class used to run the experiment
        :param file_extension: extension of the saved optimization run.
            Default: .opt
        """

        if file_extension is None:
            self.file_extension = ".opt"
        else:
            self.file_extension = file_extension

        self.ale_experiment = ale_experiment
        self.opt_object = optimization_class.load(self.ale_experiment.script_prefix
                                                  + self.file_extension)

    def plot_cost_over_time(self, **kwargs):
        """
        Assumes optimization class has a plot_cost_over_time method for plotting.
        Assumes a signature of f(prefix, ..., savefile, ...)
        :param kwargs: any addition arguments for the plotting function.
        :return: None
        """

        self.opt_object.plot_cost_over_time(self.ale_experiment.script_prefix,
                                            savefile=True,
                                            **kwargs)

    def run_single_game(self, **kwargs):
        """
        Evaluates a single game.
        :param rom: game name
        :param seed: seed for generating seeds for the games
        :param kwargs: other arguments for atari game
        :return: history
        """
        game_parameters = copy.copy(self.ale_experiment.ale_parameters)
        for key, value in kwargs.items():
            game_parameters[key] = value

        ale_handle = self.ale_experiment.construct_ale_handle(game_parameters)
        agent_handle = self.ale_experiment.construct_agent_handle(
            self.ale_experiment.agent_class,
            self.ale_experiment.agent_class_parameters,
            self.ale_experiment._nervous_system,
            ale_handle)
        agent_handle.logging = True
        obj_handle = handlers.ObjectiveHandler(ale_handle.handle,
                                               agent_handle.handle,
                                               'totalcost')
        obj_handle.create()
        parameters = kwargs.get("parameters", None)
        if parameters is None:
            print("cost: ", obj_handle.handle(self.opt_object.best))
        else:
            print("cost: ", obj_handle.handle(parameters))
        return [agent_handle.layer_history(layer)
                for layer in range(self.ale_experiment.num_layers())]

    def evaluate_single_game_cost(self, rom, seed, **kwargs):
        """
        Prints results. Runs 1 game for each rank.
        :param rom: game name
        :param seed: seed for generating seeds for the games
        :param kwargs: other arguments for atari game
        :return: None
        """

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        max_num_frames = kwargs.get('max_num_frames', 18000)
        max_num_frames_per_episode = kwargs.get('max_num_frames_per_episode',
                                                18000)
        max_num_episodes = kwargs.get('max_num_episodes', 1)
        rng = Random(seed)
        seed = [rng.randint(1, 2000000) for i in range(size)]

        game_parameters = copy.copy(self.ale_experiment.ale_parameters)
        game_parameters['rom'] = rom
        game_parameters['seed'] = seed[rank]
        game_parameters['max_num_frames'] = max_num_frames
        game_parameters['max_num_frames_per_episode'] = max_num_frames_per_episode
        game_parameters['max_num_episodes'] = max_num_episodes
        ale_handle = self.ale_experiment.construct_ale_handle(game_parameters)
        agent_handle = self.ale_experiment.construct_agent_handle(
            self.ale_experiment.agent_class,
            self.ale_experiment.agent_class_parameters,
            self.ale_experiment._nervous_system,
            ale_handle)
        obj_handle = handlers.ObjectiveHandler(ale_handle.handle,
                                               agent_handle.handle,
                                               'totalcost')
        obj_handle.create()

        local_cost = np.empty(1, dtype=np.float32)
        local_cost[0] = obj_handle.handle(self.opt_object.best)
        all_costs = np.empty(size, dtype=np.float32)
        comm.Allgather([local_cost, MPI.FLOAT],
                       [all_costs, MPI.FLOAT])
        if rank == 0:
            print("rom: ", rom)
            print("costs: ", all_costs)
            print("mean cost: ", np.mean(all_costs))
            print("total cost: ", np.sum(all_costs))

    def record_single_game(self, **kwargs):
        """
        :param kwargs: parameters necessary for ALE rom
        :return: None, saves game animation to file
        """

        game_parameters = copy.copy(self.ale_experiment.ale_parameters)
        for key, value in kwargs.items():
            game_parameters[key] = value

        ale_handle = self.ale_experiment.construct_ale_handle(game_parameters)
        agent_handle = self.ale_experiment.construct_agent_handle(
            self.ale_experiment.agent_class,
            self.ale_experiment.agent_class_parameters,
            self.ale_experiment._nervous_system,
            ale_handle)
        agent_handle.logging = True
        obj_handle = handlers.ObjectiveHandler(ale_handle.handle,
                                               agent_handle.handle,
                                               'totalcost')
        obj_handle.create()
        print("cost: ", obj_handle.handle(self.opt_object.best))
        screen_history = agent_handle.screen_history()
        animate_screen(screen_history, self.ale_experiment.script_prefix)
