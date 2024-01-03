import numpy as np
from scipy.linalg import sqrtm

import inertial_params as ip

import IPython


def check_polyhedron_realizable(vertices, params, tol=1e-8):
    V = vertices - params.com
    A, b = ip.polyhedron_span_to_face_form(V)

    # normalize so rows of A are unit vectors
    for i in range(b.shape[0]):
        a_norm = np.linalg.norm(A[i])
        A[i] /= a_norm
        b[i] /= a_norm

    for (ai, bi) in zip(A, b):
        d1 = np.max(ai @ V.T)  # = bi
        d2 = np.max(-ai @ V.T)
        aHa = ai @ params.Hc @ ai
        print(aHa - d1 * d2 * params.mass)
        if aHa > d1 * d2 * params.mass + tol:
            # IPython.embed()
            # raise ValueError()
            return False
    return True


def main():
    np.random.seed(0)

    N = 1000  # number of trials
    n = 4  # number of points per trial

    # bounding box
    half_extents = 0.5 * np.ones(3)
    box = ip.AxisAlignedBox(half_extents)

    vertices = np.array([[-1, -1, 0], [1, 1, 0]])
    masses = np.ones(2) / 2
    points = np.array([[-0.5, 0.5, 0], [0.5, -0.5, 0]])
    params = ip.RigidBody.from_point_masses(masses=masses, points=points)
    if check_polyhedron_realizable(vertices, params):
        print("passed")
    else:
        print("failed")
    return

    for i in range(N):
        masses = np.random.random(n) + 1e-8  # avoid any possibility of zero mass

        points = box.random_points(n)
        params = ip.RigidBody.from_point_masses(masses=masses, points=points)
        vertices = ip.convex_hull(points)

        # vertices = box.vertices
        # masses = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        # params = ip.RigidBody.from_point_masses(masses=masses, points=vertices)
        # params = ip.RigidBody(
        #     mass=1,
        #     h=np.zeros(3),
        #     H=ip.cuboid_inertia_matrix(mass=1, half_extents=half_extents),
        # )

        # If this case triggers, then we know the check is not necessary to
        # ensure physical realizability (i.e., there exist realizable
        # parameters for which the check is false). It may however still be
        # sufficient, in that whenever the check passes, it is guaranteed that
        # the parameters are realizable.
        if not check_polyhedron_realizable(vertices, params):
            V = vertices - params.com
            P = points - params.com
            Hc = sum([m * np.outer(p, p) for m, p in zip(masses, P)])
            A, b = ip.polyhedron_span_to_face_form(V)
            for i in range(b.shape[0]):
                a_norm = np.linalg.norm(A[i])
                A[i] /= a_norm
                b[i] /= a_norm

            ai = A[3]
            bi = b[3]
            aHa = sum([m * (ai @ p) ** 2 for m, p in zip(masses, P)])
            zHz = sum([m * p[2] ** 2 for m, p in zip(masses, P)])
            b2 = sum([m * bi**2 for m, bi in zip(masses, b)])
            print("Params not realizable!")
            IPython.embed()
            return


if __name__ == "__main__":
    main()
