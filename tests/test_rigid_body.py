import numpy as np
import cvxpy as cp
from spatialmath.base import roty
import pinocchio

import rigeo as rg


def test_construction():
    box = rg.Box(half_extents=np.ones(3))

    body = rg.RigidBody(shapes=box)
    assert len(body.shapes) == 1
    assert body.params.is_same(rg.InertialParameters.zero())
    assert body.is_realizable()

    body = rg.RigidBody(shapes=[box])
    assert len(body.shapes) == 1

    ell = rg.Ellipsoid.sphere(radius=1.0)
    body = rg.RigidBody(shapes=ell)
    assert body.is_realizable()


def test_add():
    box = rg.Box(half_extents=np.ones(3))
    params = box.uniform_density_params(mass=1.0)

    body = rg.RigidBody(shapes=box, params=params)
    assert body.is_realizable()

    body2 = body + body
    assert len(body2.shapes) == 2
    assert body2.params.is_same(box.uniform_density_params(mass=2.0))


def test_can_realize():
    box1 = rg.Box(half_extents=np.ones(3))
    box2 = rg.Box(half_extents=np.ones(3), center=[2, 0, 0])

    body = rg.RigidBody(shapes=[box1, box2])
    assert body.is_realizable()

    params1 = box1.uniform_density_params(mass=1.0)
    params2 = box2.uniform_density_params(mass=1.0)

    assert body.can_realize(params1)
    assert body.can_realize(params2)
    assert body.can_realize(params1 + params2)


def test_must_realize():
    box1 = rg.Box(half_extents=np.ones(3))
    box2 = rg.Box(half_extents=np.ones(3), center=[2, 0, 0])

    body = rg.RigidBody(shapes=[box1, box2])

    J = cp.Variable((4, 4), PSD=True)
    m = J[3, 3]
    h = J[:3, 3]
    constraints = body.must_realize(J) + [m <= 1]

    objective = cp.Maximize(h[0])
    problem = cp.Problem(objective, constraints)
    problem.solve()
    assert np.isclose(objective.value, 3.0)

    # with different shape types
    box = rg.Box(half_extents=np.ones(3))
    ell = rg.Ellipsoid.sphere(radius=1.0, center=[2, 0, 0])

    body = rg.RigidBody(shapes=[box, ell])

    constraints = body.must_realize(J) + [m <= 1]
    objective = cp.Maximize(h[0])
    problem = cp.Problem(objective, constraints)
    problem.solve()
    assert np.isclose(objective.value, 3.0)


def test_transform():
    box = rg.Box(half_extents=np.ones(3))
    ell = rg.Ellipsoid.sphere(radius=1.0, center=[2, 0, 0])

    params = box.uniform_density_params(mass=1.0) + ell.uniform_density_params(mass=1.0)
    body1 = rg.RigidBody(shapes=[box, ell], params=params)

    r = np.array([0, 1, 0])
    C = roty(np.pi / 4)
    body2 = body1.transform(rotation=C, translation=r)
    for s1, s2 in zip(body1.shapes, body2.shapes):
        assert s1.transform(rotation=C, translation=r).is_same(s2)
    assert body1.params.transform(rotation=C, translation=r).is_same(body2.params)


def _pinocchio_regressor(V, A):
    # pinocchio orders the inertia matrix parameters with I_xz and I_yy swapped
    # compared to our implementation, so we have to manually correct that
    Y = pinocchio.bodyRegressor(pinocchio.Motion(V), pinocchio.Motion(A))
    Y_swapped = Y.copy()
    Y_swapped[:, 6] = Y[:, 7]
    Y_swapped[:, 7] = Y[:, 6]
    return Y_swapped


def test_regressor():
    """Test body regressor implementation."""
    np.random.seed(0)

    V = 2 * np.random.random(6) - 1
    A = 2 * np.random.random(6) - 1

    # compare to pinocchio's implementation
    Y = rg.RigidBody.regressor(V, A)
    Y_expected = _pinocchio_regressor(V, A)

    assert np.allclose(Y, Y_expected)
