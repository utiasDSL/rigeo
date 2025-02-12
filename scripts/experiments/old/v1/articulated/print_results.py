#!/usr/bin/env python3
import argparse
import pickle

import numpy as np


MEAN_FUNC = np.median


def print_medians(results, name, key):
    print(f"\n{name}")
    print(f"Nominal = {MEAN_FUNC(results['nominal'][key])}")
    print(f"Ellipsoid = {MEAN_FUNC(results['ellipsoid'][key])}")
    print(f"Ell+CoM = {MEAN_FUNC(results['ell_com'][key])}")
    print(f"Polyhedron = {MEAN_FUNC(results['polyhedron'][key])}")


def main():
    np.set_printoptions(suppress=True, precision=6)

    parser = argparse.ArgumentParser()
    parser.add_argument("pklfile", help="Pickle file to load the data from.")
    args = parser.parse_args()

    with open(args.pklfile, "rb") as f:
        results = pickle.load(f)

    print_medians(results, "Geodesic error", "riemannian_errors")
    print_medians(results, "Validation error", "validation_errors")
    print_medians(results, "Solve times", "solve_times")
    print_medians(results, "Iterations", "num_iters")
    print_medians(results, "Objective", "objective_values")
    print_medians(results, "Num feasible", "num_feasible")


main()
