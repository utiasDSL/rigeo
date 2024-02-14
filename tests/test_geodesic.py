import numpy as np

import rigeo as rg


def test_pd_distance():
    I = np.eye(4)
    assert np.isclose(rg.positive_definite_distance(I, I), 0)
