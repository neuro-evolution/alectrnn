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

        except:
            pass

    out_path = outdir.joinpath(batch_id + ".out")
    save(results, out_path)
