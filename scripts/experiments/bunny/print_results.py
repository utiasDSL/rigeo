#!/usr/bin/env python3
import argparse
import pickle

import numpy as np


MEAN_FUNC = np.median


def main():
    np.set_printoptions(suppress=True, precision=6)

    parser = argparse.ArgumentParser()
    parser.add_argument("pklfile", help="Pickle file to load the data from.")
    args = parser.parse_args()

    with open(args.pklfile, "rb") as f:
        results = pickle.load(f)

    print(f"Solve time = {MEAN_FUNC(results['solve_times'])}")
    print(f"Iterations = {MEAN_FUNC(results['num_iters'])}")


main()
