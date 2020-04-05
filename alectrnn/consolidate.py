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
    :return: results from experiments, and batch.
    """
    if batch_path is None:
        batch = load(Path.cwd().joinpath(batch_id + ".batch"))
    else:
        batch = load(batch_path.joinpath(batch_id + ".batch"))
    results = []
    for i in range(len(batch['batch'])):
        filename = batch['id'] + "_" + str(i) + ".ga"
        try:
            file_path = search_dir.joinpath(filename)
            results.append(load(file_path))

        except (OSError, FileNotFoundError):
            print("Couldn't find file:", str(file_path))
            results.append(None)

    return results, batch
