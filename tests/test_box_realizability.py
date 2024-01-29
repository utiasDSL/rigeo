import numpy as np

import inertial_params as ip


def Q_matrices(shape):
    return [ell.Q for ell in shape.as_ellipsoidal_intersection()]


def test_cube_at_origin():
    np.random.seed(0)

    N = 1000  # number of trials
    n = 10  # number of point masses per trial

    box = ip.Box(half_extents=[0.5, 0.5, 0.5])
    Qs = Q_matrices(box)

    for i in range(N):
        points = box.random_points(n)
        masses = np.random.random(n)
        params = ip.InertialParameters.from_point_masses(masses=masses, points=points)
        assert np.all([np.trace(Q @ params.J) >= 0 for Q in Qs])

    points = np.array([-box.half_extents, box.half_extents])
    masses = [0.5, 0.5]
    params = ip.InertialParameters.from_point_masses(masses=masses, points=points)
    assert np.all([np.trace(Q @ params.J) >= 0 for Q in Qs])

    masses = np.ones(8)
    params = ip.InertialParameters.from_point_masses(masses=masses, points=box.vertices)
    assert np.all([np.trace(Q @ params.J) >= 0 for Q in Qs])

    # infeasible case
    points = np.array([-box.half_extents, 1.1 * box.half_extents])
    masses = [0.5, 0.5]
    params = ip.InertialParameters.from_point_masses(masses=masses, points=points)
    assert not np.all([np.trace(Q @ params.J) >= 0 for Q in Qs])


def test_cuboid_offset_from_origin():
    np.random.seed(0)

    N = 1000  # number of trials
    n = 10  # number of point masses per trial

    box = ip.Box(half_extents=[1.0, 0.5, 0.1], center=[1, 1, 0])
    Qs = Q_matrices(box)

    for i in range(N):
        points = box.random_points(n)
        masses = np.random.random(n)
        params = ip.InertialParameters.from_point_masses(masses=masses, points=points)
        assert np.all([np.trace(Q @ params.J) >= 0 for Q in Qs])
