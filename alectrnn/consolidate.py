from pathlib import Path
from alectrnn.batching import *


def consolidate_experiment(batch_id: str,
                           search_dir: Path = Path.cwd(),
                           outdir: Path = Path.cwd(),
                           batch_path: Path = Path.cwd()):
    batch = load(batch_path.joinpath(batch_id + ".batch"))
    results = []
    for i in range(len(batch['batch'])):
        filename = batch_id + "_" + str(i) + ".ga"
        try:
            file_path = search_dir.joinpath(filename)
            results.append(load(file_path))
            file_path.unlink()

        except (OSError, FileNotFoundError):
            print("Couldn't find file:", str(file_path))
            results.append(None)

    out_path = outdir.joinpath(batch_id + ".results")
    save(results, out_path)


def load_experiment(batch_id: str, batch_path: Path=None,
                    results_path: Path = Path.cwd()):
    """
    :param batch_id: name of batch
    :param batch_path: defaults to cwd (dir for .batch file)
    :param results_path: defaults to cwd (dir for .results file)
    :return: results from experiments, and batch.
    """
    if batch_path is None:
        batch_path = Path.cwd()

    batch = load(batch_path.joinpath(batch_id + ".batch"))
    results = load(results_path.joinpath(batch_id + '.results'))

    return results, batch
