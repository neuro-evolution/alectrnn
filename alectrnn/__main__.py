import argparse
from pathlib import Path
from alectrnn.batching import *
from alectrnn.consolidate import *
from alectrnn.execute import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("function")
    parser.add_argument("batch_id")
    parser.add_argument("index", default=-1, type=int)
    args = parser.parse_args()

    if args.function == "execute":
        

    elif args.function == "consolidate":
        pass
