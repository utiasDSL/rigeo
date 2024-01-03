import numpy as np
import inertial_params as ip


def test_pd_distance():
    I = np.eye(4)
    assert np.isclose(ip.positive_definite_distance(I, I), 0)
