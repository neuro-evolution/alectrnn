from pathlib import Path
from alectrnn.batching import *


def consolidate_experiment(num_trials: int, batch_id: str,
                           search_dir: Path, outdir: Path):
    results = []
    for i in range(num_trials):
        filename = batch_id + "_" + str(i) + ".ga"
        try:
            file_path = search_dir.joinpath(filename)
            results.append(load(file_path))

        except (OSError, FileNotFoundError):
            print("Couldn't find file:", str(file_path))
            results.append(None)

    out_path = outdir.joinpath(batch_id + ".out")
    save(results, out_path)


def load_experiment(batch_id: str, batch_path: Path=None):
    """
    :param batch_id: name of batch
    :param batch_path: defaults to cwd
    :return: results from experiments, batch, and AleExperiments. Each are
        returned as a list where elements correspond by index.
    """
    if batch_path is None:
        batch = load(Path.cwd().joinpath(args.batch_id + ".batch"))
    else:
        batch = load(batch_path.joinpath(args.batch_id + ".batch"))
    results = []
    for i in range(len(batch)):
        filename = batch_id + "_" + str(i) + ".ga"
        try:
            file_path = search_dir.joinpath(filename)
            results.append(load(file_path))

        except (OSError, FileNotFoundError):
            print("Couldn't find file:", str(file_path))
            results.append(None)

    experiments = []
    for index, parameter_batch in enumerate(batch):
        working_parameters = deepcopy(parameter_batch)
        for ref, cost in \
                working_parameters['cost_normalization_parameters']['costs'].items():
            working_parameters['normalizer'].internal_log[ref] = cost

        experiments.append(working_parameters['experiment'](
            working_parameters['experiment_parameters']['roms'],
            CostNormalizer(working_parameters['normalizer']),
            ale_parameters=working_parameters['ale_parameters'],
            nervous_system_class=working_parameters['nervous_system_class'],
            nervous_system_class_parameters=working_parameters['nervous_system_parameters'],
            agent_class_parameters=working_parameters['agent_parameters'],
            objective_parameters=working_parameters['objective_parameters']
        ))

    return results, batch, experiments
