"""Example of uniformly sampling points from the volume and surface of an ellipsoid."""
import numpy as np
import matplotlib.pyplot as plt

import rigeo as rg


def sample_volume():
    """Uniformly sample points from inside an ellipsoid."""
    N = 10000

    ell = rg.Ellipsoid(half_extents=[1, 0.5])
    points = ell.random_points(shape=N)

    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Uniformly sample volume")


def sample_surface():
    """Uniformly sample points from the surface of an ellipsoid."""
    N = 1000

    ell = rg.Ellipsoid(half_extents=[1, 0.5])
    points = ell.random_points_on(shape=N)

    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Uniformly sample surface")


if __name__ == "__main__":
    sample_volume()
    sample_surface()
    plt.show()
