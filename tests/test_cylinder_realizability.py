import numpy as np

import inertial_params as ip


def Q_matrices(shape):
    return [ell.Q for ell in shape.as_ellipsoidal_intersection()]


def test_cylinder_at_origin():
    np.random.seed(0)

    N = 1000  # number of trials
    n = 10  # number of point masses per trial

    cylinder = ip.Cylinder(length=1, radius=0.5)
    bounding_box = ip.Box(half_extents=[0.5, 0.5, 0.5])
    Qs = Q_matrices(cylinder)

    for i in range(N):
        for j in range(10):
            # generate points in bounding box and only take the ones in the
            # cylinder
            points = bounding_box.random_points(n)
            contained = cylinder.contains(points)
            points = points[contained, :]
            if len(points) > 0:
                break
        else:
            raise ValueError("Failed generate points in cylinder.")

        masses = np.random.random(points.shape[0])
        params = ip.InertialParameters.from_point_masses(masses=masses, points=points)
        assert np.all([np.trace(Q @ params.J) >= 0 for Q in Qs])

    points = np.array([[0, 0.5, 0.5], [0, -0.5, -0.5]])
    masses = [0.5, 0.5]
    params = ip.InertialParameters.from_point_masses(masses=masses, points=points)
    assert np.all([np.trace(Q @ params.J) >= 0 for Q in Qs])

    # fmt: off
    points = 0.5 * np.array([
        [1., 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
        [1, 0, -1], [0, 1, -1], [-1, 0, -1], [0, -1, -1]])
    # fmt: on
    masses = np.ones(8)
    params = ip.InertialParameters.from_point_masses(masses=masses, points=points)
    assert np.all([np.trace(Q @ params.J) >= 0 for Q in Qs])

    # infeasible cases
    points = 1.1 * np.array([[0, 0.5, 0.5], [0, -0.5, -0.5]])
    masses = [0.5, 0.5]
    params = ip.InertialParameters.from_point_masses(masses=masses, points=points)
    assert not np.all([np.trace(Q @ params.J) >= 0 for Q in Qs])

    points = 0.6 * np.array([
        [1., 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
        [1, 0, -1], [0, 1, -1], [-1, 0, -1], [0, -1, -1]])
    # fmt: on
    masses = np.ones(8)
    params = ip.InertialParameters.from_point_masses(masses=masses, points=points)
    assert not np.all([np.trace(Q @ params.J) >= 0 for Q in Qs])
