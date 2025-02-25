#!/usr/bin/env python3
"""Compare DRIP constraint data."""
import argparse
import numpy as np

import IPython


DATA_FILES = [
    "drip_data/center_h60_drip_data.npz",
    "drip_data/top_h60_2025-01-20_19-43-56_drip_data.npz",
    "drip_data/robust_h60_2025-01-20_18-28-33_drip_data.npz",
]

# TODO:
# * compute average times for all
# * compute max differences mom box for all
# * compute actual violations for all


def flatten_times(times):
    # get rid of the first solve in each run (which takes longer), then flatten
    n = times.shape[0]
    return times.reshape((n, -1))[:, 1:].flatten()


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("file", help="NPZ file containing the data.")
    # args = parser.parse_args()

    # data = np.load(args.file)

    verify_times_moment = []
    verify_times_box = []
    verify_times_ell = []

    for f in DATA_FILES:
        print(f)
        data = np.load(f)

        verify_times_moment.append(flatten_times(data["verify_times_moment"]))
        verify_times_box.append(flatten_times(data["verify_times_box"]))
        verify_times_ell.append(flatten_times(data["verify_times_ell"]))

        diff_mom_box = data["violations_moment"] - data["violations_box"]
        mom_box_avg = np.mean(diff_mom_box)
        mom_box_max = np.max(diff_mom_box)
        mom_box_min = np.min(diff_mom_box)

        print("max violations")
        print(f"moment: {np.max(data['violations_moment'])}")
        print(f"box:    {np.max(data['violations_box'])}")
        print(f"ell:    {np.max(data['violations_ell'])}")
        print()

        print("diff mom box")
        print(f"avg: {mom_box_avg}")
        print(f"max: {mom_box_max}")
        print(f"min: {mom_box_min}")
        print()

    print("average times (ms)")
    print(f"moment: {1000 * np.mean(verify_times_moment)}")
    print(f"box:    {1000 * np.mean(verify_times_box)}")
    print(f"ell:    {1000 * np.mean(verify_times_ell)}")
    return


    # average times
    # reshape to get rid of the first solve in each run, which takes longer
    n = data["times"].shape[0]
    moment_avg_time = np.mean(
        data["verify_times_moment"].reshape((n, -1))[:, 1:]
    )
    box_avg_time = np.mean(data["verify_times_box"].reshape((n, -1))[:, 1:])
    ell_avg_time = np.mean(data["verify_times_ell"].reshape((n, -1))[:, 1:])

    # value comparisons
    diff_mom_box = data["violations_moment"] - data["violations_box"]
    mom_box_max = np.max(diff_mom_box)
    mom_box_min = np.min(diff_mom_box)
    # assert mom_box_min >= -1e-6

    diff_ell_box = data["violations_ell"] - data["violations_box"]
    ell_box_max = np.max(diff_ell_box)
    ell_box_min = np.min(diff_ell_box)
    # assert ell_box_min >= -1e-6

    IPython.embed()


main()
