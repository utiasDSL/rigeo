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

    moment_d2_values = data["moment_d2_values"]
    moment_d3_values = data["moment_d3_values"]
    box_values = data["box_values"]

    d3_d2_diff = moment_d3_values - moment_d2_values
    d3_box_diff = moment_d3_values - box_values
    d2_box_diff = moment_d2_values - box_values

    moment_d2_times = data["moment_d2_times"]
    moment_d3_times = data["moment_d3_times"]
    box_times = data["box_times"]

    IPython.embed()


main()
