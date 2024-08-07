"""Compare analytical ellipsoid shell inertia with sampling-based approximation."""
import numpy as np
import rigeo as rg

# number of samples
N = 100000

# semi-axes lengths
a = 1.0
b = 0.75
c = 0.5


def main():
    np.set_printoptions(precision=6, suppress=True)
    np.random.seed(0)

    # sample points from the ellipsoid
    ell = rg.Ellipsoid(half_extents=[a, b, c])
    points = ell.random_points_on(shape=N)

    # compute corresponding inertia matrix
    H_exp = points.T @ points / N
    I_exp = np.trace(H_exp) * np.eye(3) - H_exp

    # compare with predicted analytical value
    I_pred = ell.hollow_density_params(mass=1.0).I
    I_err = I_pred - I_exp

    print(f"Sampled I =\n{I_exp}")
    print(f"Analytical I =\n{I_pred}")
    print(f"Error = \n{I_err}")


main()
