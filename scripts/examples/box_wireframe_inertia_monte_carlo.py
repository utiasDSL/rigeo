"""Compare analytical box wireframe inertia with sampling-based approximation."""
import numpy as np
import rigeo as rg

# number of samples
N = 100000

HALF_EXTENTS = [1.0, 0.75, 0.5]


def main():
    np.set_printoptions(precision=6, suppress=True)
    rng = np.random.default_rng(0)

    # uniformly sample points from the edges
    box = rg.Box(half_extents=HALF_EXTENTS)
    points = box.random_points_on_edges(shape=N, rng=rng)

    # compute corresponding inertia matrix
    H_exp = points.T @ points / N
    I_exp = np.trace(H_exp) * np.eye(3) - H_exp

    # compare with predicted analytical value
    I_pred = box.wireframe_density_params(mass=1.0).I
    I_err = I_pred - I_exp

    print(f"Sampled I =\n{I_exp}")
    print(f"Analytical I =\n{I_pred}")
    print(f"Error = \n{I_err}")


main()
