#!/usr/bin/env python3
from pathlib import Path
import pickle
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn
import numpy as np


# FIGURE_PATH = "figures/single_body_plot.pdf"
FIGURE_PATH = "/home/adam/phd/papers/rigeo/heins-wafr24/tex/current/figures/single_body_plot.pdf"


DATA_DIR = Path("data")

DATA_PATHS_FULL = [
    DATA_DIR / name
    for name in [
        "results_bias0.pkl",
        "results_bias1.pkl",
        "results_bias5.pkl",
    ]
]

CONSTRAINT_LABELS = ["Nom", "Ell", "Ell+CoM", "CHE"]
SCENARIO_LABELS = ["0", "1", "5"]

NUM_CONSTRAINT_TYPES = len(CONSTRAINT_LABELS)
NUM_SCENARIOS = len(DATA_PATHS_FULL)
assert len(SCENARIO_LABELS) == NUM_SCENARIOS

BAR_WIDTH = 0.2


def parse_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    nom_geo_err = np.array(data["nominal"]["riemannian_errors"])
    ell_geo_err = np.array(data["ellipsoid"]["riemannian_errors"])
    ell_com_geo_err = np.array(data["ell_com"]["riemannian_errors"])
    poly_geo_err = np.array(data["polyhedron"]["riemannian_errors"])

    nom_val_err = np.array(data["nominal"]["validation_errors"])
    ell_val_err = np.array(data["ellipsoid"]["validation_errors"])
    ell_com_val_err = np.array(data["ell_com"]["validation_errors"])
    poly_val_err = np.array(data["polyhedron"]["validation_errors"])

    # convert to ms
    nom_solve_times = 1000 * np.array(data["nominal"]["solve_times"])
    ell_solve_times = 1000 * np.array(data["ellipsoid"]["solve_times"])
    ell_com_solve_times = 1000 * np.array(data["ell_com"]["solve_times"])
    poly_solve_times = 1000 * np.array(data["polyhedron"]["solve_times"])

    # convert to percentage
    n = data["num_obj"]
    nom_num_feas = 100 * data["nominal"]["num_feasible"] / n
    ell_num_feas = 100 * data["ellipsoid"]["num_feasible"] / n
    ell_com_num_feas = 100 * data["ell_com"]["num_feasible"] / n
    poly_num_feas = 100 * data["polyhedron"]["num_feasible"] / n

    return {
        "nom_geo_err": nom_geo_err,
        "ell_geo_err": ell_geo_err,
        "ell_com_geo_err": ell_com_geo_err,
        "poly_geo_err": poly_geo_err,
        "nom_val_err": nom_val_err,
        "ell_val_err": ell_val_err,
        "ell_com_val_err": ell_com_val_err,
        "poly_val_err": poly_val_err,
        "nom_solve_times": nom_solve_times,
        "ell_solve_times": ell_solve_times,
        "ell_com_solve_times": ell_com_solve_times,
        "poly_solve_times": poly_solve_times,
        "nom_num_feas": nom_num_feas,
        "ell_num_feas": ell_num_feas,
        "ell_com_num_feas": ell_com_num_feas,
        "poly_num_feas": poly_num_feas,
    }


def make_boxplot(ax, data, positions, bar_width=BAR_WIDTH):
    palette_light = seaborn.color_palette("pastel")
    patch_colors = [palette_light[2], palette_light[0], palette_light[1], palette_light[3]]
    bplot = ax.boxplot(
        data, positions=positions, widths=bar_width, patch_artist=True, whis=(5, 95), showfliers=False
    )
    for i in range(NUM_SCENARIOS):
        boxes = bplot["boxes"][i * NUM_CONSTRAINT_TYPES : (i + 1) * NUM_CONSTRAINT_TYPES]
        for patch, color in zip(boxes, patch_colors):
            patch.set_facecolor(color)
    for line in bplot["medians"]:
        line.set_color("gold")
    return bplot["boxes"]


def geodesic_bar_data(ax, results, bar_width=BAR_WIDTH):
    label_xs = np.arange(NUM_SCENARIOS)
    # locs = np.array([-1, 0, 1]) * bar_width
    locs = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width
    assert len(locs) == NUM_CONSTRAINT_TYPES

    data = []
    for result in results:
        data.append(result["nom_geo_err"])
        data.append(result["ell_geo_err"])
        data.append(result["ell_com_geo_err"])
        data.append(result["poly_geo_err"])
    positions = np.concatenate([x + locs for x in label_xs])
    return make_boxplot(ax, data, positions, bar_width=bar_width)


def validation_bar_data(ax, results, bar_width=BAR_WIDTH):
    label_xs = np.arange(NUM_SCENARIOS)
    # locs = np.array([-1, 0, 1]) * bar_width
    locs = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width
    assert len(locs) == NUM_CONSTRAINT_TYPES

    data = []
    for result in results:
        data.append(result["nom_val_err"])
        data.append(result["ell_val_err"])
        data.append(result["ell_com_val_err"])
        data.append(result["poly_val_err"])
    positions = np.concatenate([x + locs for x in label_xs])
    return make_boxplot(ax, data, positions, bar_width=bar_width)


def solve_time_bar_data(ax, results, bar_width=BAR_WIDTH):
    label_xs = np.arange(NUM_SCENARIOS)
    # locs = np.array([-1, 0, 1]) * bar_width
    locs = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width
    assert len(locs) == NUM_CONSTRAINT_TYPES

    data = []
    for result in results:
        data.append(result["nom_solve_times"])
        data.append(result["ell_solve_times"])
        data.append(result["ell_com_solve_times"])
        data.append(result["poly_solve_times"])
    positions = np.concatenate([x + locs for x in label_xs])
    return make_boxplot(ax, data, positions, bar_width=bar_width)


def feasibility_bar_data(ax, results, bar_width=BAR_WIDTH):
    palette = seaborn.color_palette("pastel")

    label_xs = np.arange(NUM_SCENARIOS)
    # locs = np.array([-1, 0, 1]) * bar_width
    locs = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width
    assert len(locs) == NUM_CONSTRAINT_TYPES

    nom_num_feas = [result["nom_num_feas"] for result in results]
    ell_num_feas = [result["ell_num_feas"] for result in results]
    ell_com_num_feas = [result["ell_com_num_feas"] for result in results]
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
        ell_com_num_feas,
        color=palette[1],
        width=bar_width,
        edgecolor="k",
    )
    bar3 = ax.bar(
        label_xs + locs[3],
        poly_num_feas,
        color=palette[3],
        width=bar_width,
        edgecolor="k",
    )
    return [bar0, bar1, bar2, bar3]


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
            "font.size": 8,
            "font.family": "serif",
            "font.sans-serif": "DejaVu Sans",
            "font.weight": "normal",
            "text.usetex": True,
            "legend.fontsize": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
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

    fig = plt.figure(figsize=(6, 1.75))
    label_xs = np.arange(NUM_SCENARIOS)

    ax1 = plt.subplot(1, 4, 1)
    ax1.set_title("Geodesic Error")
    handles = geodesic_bar_data(ax1, full_results)
    ax1.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5, axis="y")
    ax1.set_xticks(label_xs, SCENARIO_LABELS, rotation=0)
    ax1.tick_params(axis="x", length=0)
    # ax1.set_xlim([-0.5, 1.5])
    # ax1.set_ylim([0, 1.5])
    # ax1.set_ylabel("Relative error")

    ax2 = plt.subplot(1, 4, 2)
    ax2.set_title("Validation RMSE")
    validation_bar_data(ax2, full_results)
    ax2.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5, axis="y")
    ax2.set_xticks(label_xs, SCENARIO_LABELS, rotation=0)
    ax2.tick_params(axis="x", length=0)
    # ax2.set_xlim([-0.5, 1.5])
    # ax2.set_ylim([0, 1.5])
    # hide_y_ticks(ax2)

    ax3 = plt.subplot(1, 4, 3)
    ax3.set_title("Solve Time [ms]")
    solve_time_bar_data(ax3, full_results)
    ax3.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5, axis="y")
    ax3.tick_params(axis="x", length=0)
    ax3.set_xticks(label_xs, SCENARIO_LABELS, rotation=0)

    ax4 = plt.subplot(1, 4, 4)
    ax4.set_title("Number Realizable [%]")
    feasibility_bar_data(ax4, full_results)
    ax4.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5, axis="y")
    ax4.tick_params(axis="x", length=0)
    ax4.set_xticks(label_xs, SCENARIO_LABELS, rotation=0)

    fig.supxlabel("Bias force $f_b$ [N]", x=0.3, y=0.02)

    fig.tight_layout(pad=0.1, w_pad=0.5)
    plt.subplots_adjust(bottom=0.2)
    plt.figlegend(
        handles,
        CONSTRAINT_LABELS,
        loc="lower center",
        bbox_to_anchor=(0.72, -0.02),
        ncols=len(CONSTRAINT_LABELS),
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
