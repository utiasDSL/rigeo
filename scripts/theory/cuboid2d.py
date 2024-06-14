"""2D example that shows that moving mass outside the shape does not
necessarily increase the variance along any direction."""
import numpy as np

import IPython

# mass at the vertices
V = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
μ = np.ones(4) / 4
S = sum([m * np.outer(v, v) for m, v in zip(μ, V)])

# mass along the axes, outside of the shape
# this results in a strictly smaller covariance S!
V2 = np.array([[1.1, 0], [0, 1.1], [-1.1, 0], [0, -1.1]])
S2 = sum([m * np.outer(v, v) for m, v in zip(μ, V2)])

IPython.embed()
