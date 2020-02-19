import argparse
from pathlib import Path
from alectrnn.batching import *
from alectrnn.consolidate import *
from alectrnn.execute import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("function")
    parser.add_argument("batch_id")
    parser.add_argument("index", nargs='?', default=-1, type=int)
    args = parser.parse_args()

    batch = load(Path.cwd().joinpath(args.batch_id + ".batch"))
    if args.function == "execute":
        # assuming cwd for now
        execute_experiment(batch['batch'][args.index],
                           args.index, args.batch_id)

    elif args.function == "async_execute":
        # assuming cwd for now
        execute_async_experiment(batch['batch'][args.index],
                                 args.index, args.batch_id)

    elif args.function == "consolidate":
        # assuming cwd for now
        consolidate_experiment(len(batch['batch']), args.batch_id,
                               Path.cwd(), Path.cwd())

    elif args.function == "info":
        parameter_batch = batch['batch'][0]
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
        print("Number trials:", len(batch['batch']))
        print("Parameter count:", experiment.get_parameter_count())
