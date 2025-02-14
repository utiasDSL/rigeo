import argparse
import numpy as np

import IPython

parser = argparse.ArgumentParser()
parser.add_argument("file", help="NPZ file containing the data.")
args = parser.parse_args()

data = np.load(args.file)


IPython.embed()
