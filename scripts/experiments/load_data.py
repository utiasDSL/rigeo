#!/usr/bin/env python3
"""Compare DRIP constraint data."""
import argparse
import numpy as np

import IPython

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="NPZ file containing the data.")
    args = parser.parse_args()

    data = np.load(args.file)

    # average times
    # reshape to get rid of the first solve in each run, which takes longer
    n = data["times"].shape[0]
    moment_avg_time = np.mean(data["verify_times_moment"].reshape((n, -1))[1:])
    box_avg_time = np.mean(data["verify_times_box"].reshape((n, -1))[1:])
    ell_avg_time = np.mean(data["verify_times_ell"].reshape((n, -1))[1:])

    # value comparisons
    mom_box_max = np.max(data["violations_moment"] - data["violations_box"])
    mom_box_min = np.min(data["violations_moment"] - data["violations_box"])
    assert mom_box_min >= -1e-6

    ell_box_max = np.max(data["violations_ell"] - data["violations_box"])
    ell_box_min = np.min(data["violations_ell"] - data["violations_box"])
    assert ell_box_min >= -1e-6

    IPython.embed()

main()
