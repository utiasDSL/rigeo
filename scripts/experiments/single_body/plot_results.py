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

# DATA_PATHS_FULL = [
#     DATA_DIR / name
#     for name in [
#         "regress_w0.05_b0.1_full.pkl",
#         "regress_w0.05_b0.1_full_box.pkl",
#     ]
# ]

# DATA_PATHS_PLANAR = [
#     DATA_DIR / name
#     for name in [
#         "regress_w0.05_b0.1_planar.pkl",
#         "regress_w0.05_b0.1_planar_box.pkl",
#     ]
# ]

DATA_PATHS_FULL = [
    DATA_DIR / name
    for name in [
        # "noisy_velocity_data/regress_w0.05_b0_full.pkl",
        # "regress_wrench_bias0_noweight.pkl",
        "regress_wrench_bias_diff_fonly.pkl",
        # "regress_wrench_bias1.pkl",
        # "regress_wrench_bias2.pkl",
        # "regress_wrench_bias5.pkl",
    ]
]


def parse_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    # n = len(data["polyhedron"]["validation_errors"])
    # poly_over_nom_validation_errors = []
    # poly_over_ell_validation_errors = []
    # for i in range(n):
    #     nom_err = data["nominal"]["validation_errors"][i]
    #     ell_err = data["ellipsoid"]["validation_errors"][i]
    #     poly_err = data["polyhedron"]["validation_errors"][i]
    #     poly_over_nom_validation_errors.append(poly_err / nom_err)
    #     poly_over_ell_validation_errors.append(poly_err / ell_err)
    #
    # poly_over_nom_validation_errors = np.array(poly_over_nom_validation_errors)
    # poly_over_ell_validation_errors = np.array(poly_over_ell_validation_errors)

    nom_geo_err = np.array(data["nominal"]["riemannian_errors"])
    ell_geo_err = np.array(data["ellipsoid"]["riemannian_errors"])
    poly_geo_err = np.array(data["polyhedron"]["riemannian_errors"])

    nom_val_err = np.array(data["nominal"]["validation_errors"])
    ell_val_err = np.array(data["ellipsoid"]["validation_errors"])
    poly_val_err = np.array(data["polyhedron"]["validation_errors"])

    # convert to ms
    nom_solve_times = 1000 * np.array(data["nominal"]["solve_times"])
    ell_solve_times = 1000 * np.array(data["ellipsoid"]["solve_times"])
    poly_solve_times = 1000 *np.array(data["polyhedron"]["solve_times"])

    return {
        "nom_geo_err": nom_geo_err,
        "ell_geo_err": ell_geo_err,
        "poly_geo_err": poly_geo_err,
        "nom_val_err": nom_val_err,
        "ell_val_err": ell_val_err,
        "poly_val_err": poly_val_err,
        "nom_solve_times": nom_solve_times,
        "ell_solve_times": ell_solve_times,
        "poly_solve_times": poly_solve_times,
    }


def geodesic_bar_data(ax, results, bar_width=0.3):
    palette = seaborn.color_palette("deep")

    label_xs = np.arange(len(results))  # each noise combination
    locs = (
        np.array([-1, 0, 1]) * bar_width
    )  # each constraint type per noise combination
    methods = ["Over Nominal", "Over Ellipsoid"]

    nom_err_medians = np.median(
        [result["nom_geo_err"] for result in results], axis=1)
    nom_err_mins = np.min(
        [result["nom_geo_err"] for result in results], axis=1)
    nom_err_maxes = np.max(
        [result["nom_geo_err"] for result in results], axis=1)
    nom_yerr = np.vstack(
        (
            nom_err_medians - nom_err_mins,
            nom_err_maxes - nom_err_medians,
        )
    )

    poly_err_medians = np.median(
        [result["poly_geo_err"] for result in results], axis=1)
    poly_err_mins = np.min(
        [result["poly_geo_err"] for result in results], axis=1)
    poly_err_maxes = np.max(
        [result["poly_geo_err"] for result in results], axis=1)
    poly_yerr = np.vstack(
        (
            poly_err_medians - poly_err_mins,
            poly_err_maxes - poly_err_medians,
        )
    )

    ell_err_medians = np.median(
        [result["ell_geo_err"] for result in results], axis=1)
    ell_err_mins = np.min(
        [result["ell_geo_err"] for result in results], axis=1)
    ell_err_maxes = np.max(
        [result["ell_geo_err"] for result in results], axis=1)
    ell_yerr = np.vstack(
        (
            ell_err_medians - ell_err_mins,
            ell_err_maxes - ell_err_medians,
        )
    )

    bar0 = ax.bar(
        label_xs + locs[0], nom_err_medians, color=palette[2], width=bar_width
    )
    ax.errorbar(
        label_xs + locs[0],
        nom_err_medians,
        yerr=nom_yerr,
        fmt="none",
        elinewidth=1,
        capsize=3,
        ecolor="k",
    )

    bar1 = ax.bar(
        label_xs + locs[1], ell_err_medians, color=palette[0], width=bar_width
    )
    ax.errorbar(
        label_xs + locs[1],
        ell_err_medians,
        yerr=ell_yerr,
        fmt="none",
        elinewidth=1,
        capsize=3,
        ecolor="k",
    )

    bar2 = ax.bar(label_xs + locs[2], poly_err_medians, color=palette[3], width=bar_width)
    ax.errorbar(
        label_xs + locs[2],
        poly_err_medians,
        yerr=poly_yerr,
        fmt="none",
        elinewidth=1,
        capsize=3,
        ecolor="k",
    )
    return [bar0, bar1, bar2]


def validation_bar_data(ax, results, bar_width=0.3):
    palette = seaborn.color_palette("deep")

    label_xs = np.arange(len(results))  # each noise combination
    locs = (
        np.array([-1, 0, 1]) * bar_width
    )  # each constraint type per noise combination
    methods = ["Over Nominal", "Over Ellipsoid"]

    # over_nominal_medians = np.median(
    #     [result["poly_over_nom_validation_errors"] for result in results], axis=1
    # )
    # over_nominal_maxes = np.max(
    #     [result["poly_over_nom_validation_errors"] for result in results], axis=1
    # )
    # over_nominal_mins = np.min(
    #     [result["poly_over_nom_validation_errors"] for result in results], axis=1
    # )
    # nom_yerr = np.vstack(
    #     (
    #         over_nominal_medians - over_nominal_mins,
    #         over_nominal_maxes - over_nominal_medians,
    #     )
    # )
    #
    # over_ellipsoid_medians = np.median(
    #     [result["poly_over_ell_validation_errors"] for result in results], axis=1
    # )
    # over_ellipsoid_maxes = np.max(
    #     [result["poly_over_ell_validation_errors"] for result in results], axis=1
    # )
    # over_ellipsoid_mins = np.min(
    #     [result["poly_over_ell_validation_errors"] for result in results], axis=1
    # )
    # ell_yerr = np.vstack(
    #     (
    #         over_ellipsoid_medians - over_ellipsoid_mins,
    #         over_ellipsoid_maxes - over_ellipsoid_medians,
    #     )
    # )

    nom_err_medians = np.median(
        [result["nom_val_err"] for result in results], axis=1)
    nom_err_mins = np.min(
        [result["nom_val_err"] for result in results], axis=1)
    nom_err_maxes = np.max(
        [result["nom_val_err"] for result in results], axis=1)
    nom_yerr = np.vstack(
        (
            nom_err_medians - nom_err_mins,
            nom_err_maxes - nom_err_medians,
        )
    )

    poly_err_medians = np.median(
        [result["poly_val_err"] for result in results], axis=1)
    poly_err_mins = np.min(
        [result["poly_val_err"] for result in results], axis=1)
    poly_err_maxes = np.max(
        [result["poly_val_err"] for result in results], axis=1)
    poly_yerr = np.vstack(
        (
            poly_err_medians - poly_err_mins,
            poly_err_maxes - poly_err_medians,
        )
    )

    ell_err_medians = np.median(
        [result["ell_val_err"] for result in results], axis=1)
    ell_err_mins = np.min(
        [result["ell_val_err"] for result in results], axis=1)
    ell_err_maxes = np.max(
        [result["ell_val_err"] for result in results], axis=1)
    ell_yerr = np.vstack(
        (
            ell_err_medians - ell_err_mins,
            ell_err_maxes - ell_err_medians,
        )
    )

    bar0 = ax.bar(
        label_xs + locs[0], nom_err_medians, color=palette[2], width=bar_width
    )
    ax.errorbar(
        label_xs + locs[0],
        nom_err_medians,
        yerr=nom_yerr,
        fmt="none",
        elinewidth=1,
        capsize=3,
        ecolor="k",
    )

    bar1 = ax.bar(
        label_xs + locs[1], ell_err_medians, color=palette[0], width=bar_width
    )
    ax.errorbar(
        label_xs + locs[1],
        ell_err_medians,
        yerr=ell_yerr,
        fmt="none",
        elinewidth=1,
        capsize=3,
        ecolor="k",
    )

    bar2 = ax.bar(label_xs + locs[2], poly_err_medians, color=palette[3], width=bar_width)
    ax.errorbar(
        label_xs + locs[2],
        poly_err_medians,
        yerr=poly_yerr,
        fmt="none",
        elinewidth=1,
        capsize=3,
        ecolor="k",
    )
    return [bar0, bar1, bar2]


def solve_time_bar_data(ax, results, bar_width=0.3):
    palette = seaborn.color_palette("deep")

    label_xs = np.arange(len(results))  # each noise combination
    locs = (
        np.array([-1, 0, 1]) * bar_width
    )  # each constraint type per noise combination

    nom_err_medians = np.median(
        [result["nom_solve_times"] for result in results], axis=1)
    nom_err_mins = np.min(
        [result["nom_solve_times"] for result in results], axis=1)
    nom_err_maxes = np.max(
        [result["nom_solve_times"] for result in results], axis=1)
    nom_yerr = np.vstack(
        (
            nom_err_medians - nom_err_mins,
            nom_err_maxes - nom_err_medians,
        )
    )

    poly_err_medians = np.median(
        [result["poly_solve_times"] for result in results], axis=1)
    poly_err_mins = np.min(
        [result["poly_solve_times"] for result in results], axis=1)
    poly_err_maxes = np.max(
        [result["poly_solve_times"] for result in results], axis=1)
    poly_yerr = np.vstack(
        (
            poly_err_medians - poly_err_mins,
            poly_err_maxes - poly_err_medians,
        )
    )

    ell_err_medians = np.median(
        [result["ell_solve_times"] for result in results], axis=1)
    ell_err_mins = np.min(
        [result["ell_solve_times"] for result in results], axis=1)
    ell_err_maxes = np.max(
        [result["ell_solve_times"] for result in results], axis=1)
    ell_yerr = np.vstack(
        (
            ell_err_medians - ell_err_mins,
            ell_err_maxes - ell_err_medians,
        )
    )

    bar0 = ax.bar(
        label_xs + locs[0], nom_err_medians, color=palette[2], width=bar_width
    )
    ax.errorbar(
        label_xs + locs[0],
        nom_err_medians,
        yerr=nom_yerr,
        fmt="none",
        elinewidth=1,
        capsize=3,
        ecolor="k",
    )

    bar1 = ax.bar(
        label_xs + locs[1], ell_err_medians, color=palette[0], width=bar_width
    )
    ax.errorbar(
        label_xs + locs[1],
        ell_err_medians,
        yerr=ell_yerr,
        fmt="none",
        elinewidth=1,
        capsize=3,
        ecolor="k",
    )

    bar2 = ax.bar(label_xs + locs[2], poly_err_medians, color=palette[3], width=bar_width)
    ax.errorbar(
        label_xs + locs[2],
        poly_err_medians,
        yerr=poly_yerr,
        fmt="none",
        elinewidth=1,
        capsize=3,
        ecolor="k",
    )
    return [bar0, bar1, bar2]


def hide_y_ticks(ax):
    ax.set_yticklabels([])
    ax.tick_params(axis="y", colors=(0, 0, 0, 0))


def plot_results():
    full_results = [parse_pickle(path) for path in DATA_PATHS_FULL]
    # planar_results = [parse_pickle(path) for path in DATA_PATHS_PLANAR]

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

    fig = plt.figure(figsize=(5, 1.75))
    label_xs = np.arange(len(full_results))
    # constraint_labels = ["Hull", "Box"]

    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title("Geodesic error")
    bars = geodesic_bar_data(ax1, full_results)
    ax1.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5, axis="y")
    # ax1.set_xticks(label_xs, constraint_labels, rotation=0)
    ax1.tick_params(axis="x", length=0)
    # ax1.set_xlim([-0.5, 1.5])
    # ax1.set_ylim([0, 1.5])
    # ax1.set_ylabel("Relative error")

    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title("Validation RMSE")
    validation_bar_data(ax2, full_results)
    ax2.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5, axis="y")
    # ax2.set_xticks(label_xs, constraint_labels, rotation=0)
    ax2.tick_params(axis="x", length=0)
    # ax2.set_xlim([-0.5, 1.5])
    # ax2.set_ylim([0, 1.5])
    # hide_y_ticks(ax2)

    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title("Solve time [ms]")
    solve_time_bar_data(ax3, full_results)
    ax3.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5, axis="y")
    ax3.tick_params(axis="x", length=0)

    labels = ["Nominal", "Ellipsoidal", "Polyhedral"]
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
