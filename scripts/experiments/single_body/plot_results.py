#!/usr/bin/env python3
from pathlib import Path
import pickle
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn
import numpy as np

import IPython


FIGURE_PATH = "figures/single_body_plot.pdf"

DATA_DIR = Path("data")

DATA_PATHS_FULL = [
    DATA_DIR / name
    for name in [
        "regress_wrench_bias0.pkl",
        "regress_wrench_bias_f5.pkl",
    ]
]


def parse_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    nom_geo_err = np.array(data["nominal"]["riemannian_errors"])
    ell_geo_err = np.array(data["ellipsoid"]["riemannian_errors"])
    poly_geo_err = np.array(data["polyhedron"]["riemannian_errors"])

    nom_val_err = np.array(data["nominal"]["validation_errors"])
    ell_val_err = np.array(data["ellipsoid"]["validation_errors"])
    poly_val_err = np.array(data["polyhedron"]["validation_errors"])

    # convert to ms
    nom_solve_times = 1000 * np.array(data["nominal"]["solve_times"])
    ell_solve_times = 1000 * np.array(data["ellipsoid"]["solve_times"])
    poly_solve_times = 1000 * np.array(data["polyhedron"]["solve_times"])

    # convert to percentage
    n = data["num_obj"]
    nom_num_feas = 100 * data["nominal"]["num_feasible"] / n
    ell_num_feas = 100 * data["ellipsoid"]["num_feasible"] / n
    poly_num_feas = 100 * data["polyhedron"]["num_feasible"] / n

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
        "nom_num_feas": nom_num_feas,
        "ell_num_feas": ell_num_feas,
        "poly_num_feas": poly_num_feas,
    }


def make_boxplot(ax, data, positions, bar_width):
    palette_light = seaborn.color_palette("pastel")
    patch_colors = [palette_light[2], palette_light[0], palette_light[3]]
    bplot = ax.boxplot(
        data, positions=positions, widths=bar_width, patch_artist=True, whis=(0, 100)
    )
    for patch, color in zip(bplot["boxes"], patch_colors):
        patch.set_facecolor(color)
    for patch, color in zip(bplot["boxes"][len(patch_colors) :], patch_colors):
        patch.set_facecolor(color)
    for line in bplot["medians"]:
        line.set_color("gold")
    return bplot["boxes"]


def geodesic_bar_data(ax, results, bar_width=0.3):
    label_xs = np.arange(len(results))  # each noise combination
    locs = (
        np.array([-1, 0, 1]) * bar_width
    )  # each constraint type per noise combination

    data = []
    for result in results:
        data.append(result["nom_geo_err"])
        data.append(result["ell_geo_err"])
        data.append(result["poly_geo_err"])
    positions = np.concatenate((label_xs[0] + locs, label_xs[1] + locs))
    return make_boxplot(ax, data, positions, bar_width=bar_width)


def validation_bar_data(ax, results, bar_width=0.3):
    label_xs = np.arange(len(results))  # each noise combination
    locs = (
        np.array([-1, 0, 1]) * bar_width
    )  # each constraint type per noise combination

    data = []
    for result in results:
        data.append(result["nom_val_err"])
        data.append(result["ell_val_err"])
        data.append(result["poly_val_err"])
    positions = np.concatenate((label_xs[0] + locs, label_xs[1] + locs))
    return make_boxplot(ax, data, positions, bar_width=bar_width)


def solve_time_bar_data(ax, results, bar_width=0.3):
    label_xs = np.arange(len(results))  # each noise combination
    locs = (
        np.array([-1, 0, 1]) * bar_width
    )  # each constraint type per noise combination

    data = []
    for result in results:
        data.append(result["nom_solve_times"])
        data.append(result["ell_solve_times"])
        data.append(result["poly_solve_times"])
    positions = np.concatenate((label_xs[0] + locs, label_xs[1] + locs))
    return make_boxplot(ax, data, positions, bar_width=bar_width)


def feasibility_bar_data(ax, results, bar_width=0.3):
    palette = seaborn.color_palette("pastel")

    label_xs = np.arange(len(results))  # each noise combination
    locs = (
        np.array([-1, 0, 1]) * bar_width
    )  # each constraint type per noise combination

    nom_num_feas = [result["nom_num_feas"] for result in results]
    ell_num_feas = [result["ell_num_feas"] for result in results]
    poly_num_feas = [result["poly_num_feas"] for result in results]

    bar0 = ax.bar(
        label_xs + locs[0],
        nom_num_feas,
        color=palette[2],
        width=bar_width,
        edgecolor="k",
    )
    bar1 = ax.bar(
        label_xs + locs[1],
        ell_num_feas,
        color=palette[0],
        width=bar_width,
        edgecolor="k",
    )
    bar2 = ax.bar(
        label_xs + locs[2],
        poly_num_feas,
        color=palette[3],
        width=bar_width,
        edgecolor="k",
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

    fig = plt.figure(figsize=(6.5, 1.75))
    label_xs = np.arange(len(full_results))
    constraint_labels = ["1", "2"]

    ax1 = plt.subplot(1, 4, 1)
    ax1.set_title("Geodesic Error")
    bars = geodesic_bar_data(ax1, full_results)
    ax1.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5, axis="y")
    ax1.set_xticks(label_xs, constraint_labels, rotation=0)
    ax1.tick_params(axis="x", length=0)
    # ax1.set_xlim([-0.5, 1.5])
    # ax1.set_ylim([0, 1.5])
    # ax1.set_ylabel("Relative error")

    ax2 = plt.subplot(1, 4, 2)
    ax2.set_title("Validation RMSE")
    validation_bar_data(ax2, full_results)
    ax2.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5, axis="y")
    ax2.set_xticks(label_xs, constraint_labels, rotation=0)
    ax2.tick_params(axis="x", length=0)
    # ax2.set_xlim([-0.5, 1.5])
    # ax2.set_ylim([0, 1.5])
    # hide_y_ticks(ax2)

    ax3 = plt.subplot(1, 4, 3)
    ax3.set_title("Solve time [ms]")
    solve_time_bar_data(ax3, full_results)
    ax3.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5, axis="y")
    ax3.tick_params(axis="x", length=0)
    ax3.set_xticks(label_xs, constraint_labels, rotation=0)

    ax4 = plt.subplot(1, 4, 4)
    ax4.set_title("Number Realizable [%]")
    feasibility_bar_data(ax4, full_results)
    ax4.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5, axis="y")
    ax4.tick_params(axis="x", length=0)
    ax4.set_xticks(label_xs, constraint_labels, rotation=0)

    labels = ["Nominal", "Ellipsoidal", "Polyhedral"]
    fig.tight_layout(pad=0.1, w_pad=0.5)
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
