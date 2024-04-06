"""Utilities for experiments."""
import numpy as np
from scipy.integrate import solve_ivp
import wrlparser

from .shape import Box, ConvexPolyhedron
from .util import skew6, compute_evaluation_times


class WRL:
    """Parse polyhedrons from a WRL/VRML file.

    Parameters
    ----------
    data :
        The WRL data, as parsed by the wrlparser package.
    diaglen : float, non-negative
        The desired size of the bounding box, in terms of its diagonal length.
        The scene is rescaled to achieve this size.

    Attributes
    ----------
    nv : int, non-negative
        The total number of vertices in the scene.
    polys : list[rg.ConvexPolyhedron]
        The polyhedra that compose the scene.
    points : np.ndarray, shape (nv, 3)
        All of the vertices in the scene.
    """

    def __init__(self, data, diaglen=1):
        points = []
        for shape in data.nodes:
            points.append(shape.geometry.coord.point)
        points = np.vstack(points)

        # bounding box
        box = Box.from_points_to_bound(points)

        # scale to achieve desired bounding box size
        scale = diaglen / box.diaglen

        self.nv = 0
        self.polys = []

        scaled_points = []
        for shape in data.nodes:
            points = scale * np.array(shape.geometry.coord.point)
            poly = ConvexPolyhedron.from_vertices(points)
            scaled_points.append(points)
            self.nv += len(points)
            self.polys.append(poly)

        self.points = np.vstack(scaled_points)

    @classmethod
    def from_string(cls, s, **kwargs):
        """Parse from a string."""
        data = wrlparser.parse(s)
        return cls(data=data, **kwargs)

    @classmethod
    def from_file_path(cls, path, **kwargs):
        """Parse from a file."""
        with open(path) as f:
            s = f.read()
        return cls.from_string(s=s, **kwargs)

    @property
    def ns(self):
        """Number of shapes (polyhedra) in the scene."""
        return len(self.polys)

    def random_points(self, n):
        """Generate random points contained in the shapes in the scene."""
        # uniformly randomly choose which shapes to generate points in
        prob = np.ones(self.ns) / self.ns  # uniform
        num_per_poly = np.random.multinomial(n, prob)

        # generate the random points in each polyhedron
        points = []
        for num, poly in zip(num_per_poly, self.polys):
            points.append(poly.random_points(num))
        return np.vstack(points)


def generate_rigid_body_trajectory(
    params,
    duration=2 * np.pi,
    eval_step=0.1,
    planar=False,
    vel_noise_bias=0,
    vel_noise_width=0,
    wrench_noise_bias=0,
    wrench_noise_width=0,
):
    """Generate a random trajectory for a rigid body.

    Parameters
    ----------
    params : rg.InertialParameters
        The inertial parameters of the body.
    duration : float, positive
        The duration of the trajectory.
    eval_step : float, positive
        Interval at which to sample the trajectory.
    planar : bool
        Set to ``True`` to restrict the body to planar motion.

    Returns
    -------
    """
    M = params.M.copy()
    c = params.com

    def wrench(t):
        """Body-frame wrench applied to the rigid body."""
        w = np.array(
            [
                np.sin(t),
                np.sin(t + np.pi / 3),
                np.sin(t + 2 * np.pi / 3),
                np.sin(t + np.pi),
                np.sin(t + 4 * np.pi / 3),
                np.sin(t + 5 * np.pi / 3),
            ]
        )
        # scale down angular components to achieve similar magnitude motions
        # compared to linear
        w[3:] *= 0.01

        if planar:
            w[2:5] = 0
        return w

    def f(t, V):
        """Evaluate acceleration at time t given velocity V."""
        # solve Newton-Euler for acceleration
        w = wrench(t)
        A = np.linalg.solve(M, w - skew6(V) @ M @ V)
        return A

    # integrate the trajectory
    n, t_eval = compute_evaluation_times(duration=duration, step=eval_step)
    V0 = np.zeros(6)
    res = solve_ivp(fun=f, t_span=[0, duration], y0=V0, t_eval=t_eval)
    assert res.success, "IVP failed to solve!"
    assert np.allclose(res.t, t_eval)

    # extract true trajectory values
    Vs = res.y.T
    As = np.array([f(t, V) for t, V in zip(t_eval, Vs)])
    ws = np.array([wrench(t) for t in t_eval])

    # TODO is this correct?
    # if planar:
    #     # zero out non-planar components and find the wrenches that would
    #     # achieve that
    #     Vs[:, 2:5] = 0
    #     As[:, 2:5] = 0
    #     ws = np.array([M @ A + rg.skew6(V) @ M @ V for V, A, in zip(Vs, As)])

    # apply noise to velocity
    vel_noise_raw = np.random.random(size=Vs.shape) - 0.5  # mean = 0, width = 1
    vel_noise = vel_noise_width * vel_noise_raw + vel_noise_bias
    # if planar:
    #     vel_noise[:, 2:5] = 0
    Vs_noisy = Vs + vel_noise

    # compute midpoint values
    # TODO probably makes more sense to add noise to V, then numerically
    # differentiate to obtain A
    As_noisy = (Vs_noisy[1:, :] - Vs_noisy[:-1, :]) / eval_step
    # As_noisy = 0.5 * (Vs_noisy[2:, :] - Vs_noisy[:-2, :]) / eval_step
    # Vs_mid = (Vs_noisy[1:, :] + Vs_noisy[:-1, :]) / 2

    # apply noise to wrench
    # TODO planar
    w_noise_raw = np.random.random(size=ws.shape) - 0.5
    w_noise = wrench_noise_width * w_noise_raw + wrench_noise_bias
    ws_noisy = ws + w_noise

    return {
        "Vs": Vs,
        "As": As,
        "ws": ws,
        "Vs_noisy": Vs_noisy,
        "As_noisy": As_noisy,
        "ws_noisy": ws_noisy,
    }
