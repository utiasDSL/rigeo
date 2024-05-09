#!/usr/bin/env python3
from pathlib import Path
import pickle
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn
import numpy as np

import IPython


FIGURE_PATH = "figures/single_body_plot2.pdf"

DATA_DIR = Path("data")

# DATA_PATHS_FULL = [
#     DATA_DIR / name
#     for name in [
#         "regress_w0.05_b0_full.pkl",
#         "regress_w0.05_b0.1_full.pkl",
#         "regress_w0.2_b0.1_full.pkl",
#         "regress_w0.3_b0.1_full.pkl",
#     ]
# ]
#
# DATA_PATHS_PLANAR = [
#     DATA_DIR / name
#     for name in [
#         "regress_w0.05_b0_planar.pkl",
#         "regress_w0.05_b0.1_planar.pkl",
#         "regress_w0.2_b0.1_planar.pkl",
#         "regress_w0.3_b0.1_planar.pkl",
#     ]
# ]

DATA_PATHS_FULL = [
    DATA_DIR / name
    for name in [
        "regress_w0.05_b0.1_full.pkl",
        "regress_w0.05_b0.1_full_box.pkl",
    ]
]

DATA_PATHS_PLANAR = [
    DATA_DIR / name
    for name in [
        "regress_w0.05_b0.1_planar.pkl",
        "regress_w0.05_b0.1_planar_box.pkl",
    ]
]


def parse_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    n = len(data["polyhedron"]["validation_errors"])
    poly_over_nom_validation_errors = []
    poly_over_ell_validation_errors = []
    for i in range(n):
        nom_err = data["nominal"]["validation_errors"][i]
        ell_err = data["ellipsoid"]["validation_errors"][i]
        poly_err = data["polyhedron"]["validation_errors"][i]
        poly_over_nom_validation_errors.append(poly_err / nom_err)
        poly_over_ell_validation_errors.append(poly_err / ell_err)

    poly_over_nom_validation_errors = np.array(poly_over_nom_validation_errors)
    poly_over_ell_validation_errors = np.array(poly_over_ell_validation_errors)

    nom_solve_times = np.array(data["nominal"]["solve_times"])
    ell_solve_times = np.array(data["ellipsoid"]["solve_times"])
    poly_solve_times = np.array(data["polyhedron"]["solve_times"])

    IPython.embed()

    return {
        "poly_over_nom_validation_errors": poly_over_nom_validation_errors,
        "poly_over_ell_validation_errors": poly_over_ell_validation_errors,
        "nom_solve_times": nom_solve_times,
        "ell_solve_times": ell_solve_times,
        "poly_solve_times": poly_solve_times,
    }


def validation_bar_data(ax, results, bar_width=0.3):
    palette = seaborn.color_palette("deep")

    label_xs = np.arange(2)  # each noise combination
    locs = (
        np.array([-0.5, 0.5]) * bar_width
    )  # each constraint type per noise combination
    methods = ["Over Nominal", "Over Ellipsoid"]

    over_nominal_medians = np.median(
        [result["poly_over_nom_validation_errors"] for result in results], axis=1
    )
    over_nominal_maxes = np.max(
        [result["poly_over_nom_validation_errors"] for result in results], axis=1
    )
    over_nominal_mins = np.min(
        [result["poly_over_nom_validation_errors"] for result in results], axis=1
    )
    nom_yerr = np.vstack(
        (
            over_nominal_medians - over_nominal_mins,
            over_nominal_maxes - over_nominal_medians,
        )
    )

    over_ellipsoid_medians = np.median(
        [result["poly_over_ell_validation_errors"] for result in results], axis=1
    )
    over_ellipsoid_maxes = np.max(
        [result["poly_over_ell_validation_errors"] for result in results], axis=1
    )
    over_ellipsoid_mins = np.min(
        [result["poly_over_ell_validation_errors"] for result in results], axis=1
    )
    ell_yerr = np.vstack(
        (
            over_ellipsoid_medians - over_ellipsoid_mins,
            over_ellipsoid_maxes - over_ellipsoid_medians,
        )
    )

    # TODO error bars
    bar1 = ax.bar(
        label_xs + locs[0], over_nominal_medians, color=palette[0], width=bar_width
    )
    ax.errorbar(
        label_xs + locs[0],
        over_nominal_medians,
        yerr=nom_yerr,
        fmt="none",
        elinewidth=1,
        capsize=3,
        ecolor="k",
    )

    bar2 = ax.bar(label_xs + locs[1], over_ellipsoid_medians, color=palette[3], width=bar_width)
    ax.errorbar(
        label_xs + locs[1],
        over_ellipsoid_medians,
        yerr=ell_yerr,
        fmt="none",
        elinewidth=1,
        capsize=3,
        ecolor="k",
    )
    return [bar1, bar2]


def hide_y_ticks(ax):
    ax.set_yticklabels([])
    ax.tick_params(axis="y", colors=(0, 0, 0, 0))


def plot_results():
    full_results = [parse_pickle(path) for path in DATA_PATHS_FULL]
    planar_results = [parse_pickle(path) for path in DATA_PATHS_PLANAR]

    mpl.use("pgf")
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.size": 6,
            "font.family": "serif",
            "font.sans-serif": "DejaVu Sans",
            "font.weight": "normal",
            "text.usetex": True,
            "legend.fontsize": 6,
            "axes.titlesize": 6,
            "axes.labelsize": 6,
            "xtick.labelsize": 6,
            "pgf.preamble": "\n".join(
                [
                    r"\usepackage[utf8]{inputenc}",
                    r"\usepackage[T1]{fontenc}",
                    r"\usepackage{siunitx}",
                    r"\usepackage{bm}",
                ]
            ),
        }
    )

    palette = seaborn.color_palette("deep")

    fig = plt.figure(figsize=(3.25, 1.75))
    label_xs = np.arange(2)
    constraint_labels = ["Hull", "Box"]

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Full")
    bars = validation_bar_data(ax1, full_results)
    ax1.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5, axis="y")
    ax1.set_xticks(label_xs, constraint_labels, rotation=0)
    ax1.tick_params(axis="x", length=0)
    ax1.set_xlim([-0.5, 1.5])
    ax1.set_ylim([0, 1.5])
    ax1.set_ylabel("Relative error")

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Planar")
    validation_bar_data(ax2, planar_results)
    ax2.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5, axis="y")
    ax2.set_xticks(label_xs, constraint_labels, rotation=0)
    ax2.tick_params(axis="x", length=0)
    ax2.set_xlim([-0.5, 1.5])
    ax2.set_ylim([0, 1.5])
    hide_y_ticks(ax2)

    # for i, method in enumerate(methods):
    #     x = label_xs + locs[i]
    #     # medians =
    #
    #     ax.errorbar(
    #         x,
    #         medians,
    #         # yerr=yerr,
    #         # fmt="none",
    #         # elinewidth=1,
    #         # capsize=1.5,
    #         # ecolor=error_colors[i],
    #     )

    labels = ["vs nom", "vs ell"]
    fig.tight_layout(pad=0.1)
    plt.subplots_adjust(bottom=0.2)
    plt.figlegend(
        bars,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.565, 0),
        ncols=len(labels),
        handlelength=1,
    )
    fig.savefig(FIGURE_PATH)
    print(f"Saved figure to {FIGURE_PATH}")


def main():
    if len(sys.argv) > 1:
        parse_pickle(sys.argv[1])
        return
    plot_results()


main()
