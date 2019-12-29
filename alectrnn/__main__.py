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

    elif args.function == "consolidate":
        # assuming cwd for now
        consolidate_experiment(len(batch['batch']), args.batch_id,
                               Path.cwd(), Path.cwd())
