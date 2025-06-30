#!/usr/bin/env python3
"""Compare the moment SDP constraints with specialized custom constraints."""
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="NPZ file containing the data.")
    args = parser.parse_args()

    data = np.load(args.file)

    moment_d2_values = data["moment_d2_values"]
    moment_d3_values = data["moment_d3_values"]
    custom_values = data["custom_values"]

    # the custom values should always be as least as tight as the moment
    # constraints (our proofs say so)
    d3_d2_diff = moment_d3_values - moment_d2_values
    d3_custom_diff = moment_d3_values - custom_values
    d2_custom_diff = moment_d2_values - custom_values

    print("Max difference")
    print(f"moment (d=3) - custom: {np.max(d3_custom_diff)}")
    print(f"moment (d=2) - custom: {np.max(d2_custom_diff)}")
    print()

    print("Min difference")
    print("(this should be near zero)")
    print(f"moment (d=3) - custom: {np.min(d3_custom_diff)}")
    print(f"moment (d=2) - custom: {np.min(d2_custom_diff)}")
    print()

    moment_d2_times = data["moment_d2_times"]
    moment_d3_times = data["moment_d3_times"]
    custom_times = data["custom_times"]

    print("Average compute times")
    print(f"moment d=2: {1000 * np.mean(moment_d2_times)} ms")
    print(f"moment d=3: {1000 * np.mean(moment_d3_times)} ms")
    print(f"custom:     {1000 * np.mean(custom_times)} ms")


main()
