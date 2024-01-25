"""Generate simulated data for random polyhedral rigid bodies."""
import argparse
import time

import numpy as np
import cvxpy as cp

import pybullet as pyb
import pybullet_data
import pyb_utils

import mobile_manipulation_central as mm
import inertial_params as ip

import IPython

RECORDING_TIMESTEP = 0.1
SIM_FREQ = 100
GRAVITY = np.array([0, 0, -9.81])

TRAIN_TEST_SPLIT = 0.5
VEL_NOISE_WIDTH = 0.1
VEL_NOISE_BIAS = 0
REGULARIZATION_COEFF = 1e-3

VISUALIZE = False


class IPIDProblem:
    """Inertial parameter identification optimization problem.

    Parameters
    ----------
    Ys : ndarray, shape (n, model.nv, nb * 10)
        Regressor matrices, one for each of the ``n`` measurements.
    τs : ndarray, shape (n, model.nv)
        Measured joint torques.
    reg_params : InertialParameters
        Nominal inertial parameters to use as a regularizer.
    reg_coeff : float
        Coefficient for the regularization term.
    """

    def __init__(self, Ys, τs, reg_params, reg_coeff=1e-3):
        self.no = τs.shape[0]  # number of measurements
        self.nb = len(reg_params)  # number of bodies

        self.A = np.vstack(Ys)
        self.b = np.concatenate(τs)

        self.θs = [cp.Variable(10) for _ in range(self.nb)]
        self.Js = [cp.Variable((4, 4), PSD=True) for _ in range(self.nb)]

        self.J0_invs = [np.linalg.inv(p.J) for p in reg_params]
        self.reg_coeff = reg_coeff

    def _solve(self, extra_constraints=None, name=None):
        # regularizer is the entropic distance proposed by (Lee et al., 2020)
        regularizer = cp.sum(
            [
                -cp.log_det(J) + cp.trace(J0_inv @ J)
                for J, J0_inv in zip(self.Js, self.J0_invs)
            ]
        )

        # psd_wrap fixes an occasional internal scipy error
        # https://github.com/cvxpy/cvxpy/issues/1421#issuecomment-865977139
        θ = cp.hstack(self.θs)
        W = cp.psd_wrap(np.eye(self.b.shape[0]))
        objective = cp.Minimize(
            0.5 / self.no * cp.quad_form(self.A @ θ - self.b, W)
            + self.reg_coeff * regularizer
        )

        constraints = []
        for i in range(self.nb):
            constraints.extend(ip.J_vec_constraint(self.Js[i], self.θs[i]))
        if extra_constraints is not None:
            constraints.extend(extra_constraints)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        assert problem.status == "optimal"
        if name is not None:
            print(f"{name} solve time = {problem.solver_stats.solve_time}")
            print(f"{name} value = {problem.value}")
        return [ip.InertialParameters.from_vector(θ.value) for θ in self.θs]
        # return ip.InertialParameters.from_pseudo_inertia_matrix(self.Jopt.value)

    def solve_nominal(self):
        return self._solve(name="nominal")

    def solve_ellipsoid(self, ellipsoids):
        extra_constraints = [
            cp.trace(ell.Q @ J) >= 0 for ell, J in zip(ellipsoids, self.Js)
        ]
        return self._solve(extra_constraints, name="ellipsoid")

    def solve_polyhedron(self, boxes):
        extra_constraints = []
        for i, box in enumerate(boxes):
            nv = box.vertices.shape[0]
            mvs = cp.Variable(nv)
            Vs = np.array([np.outer(v, v) for v in box.vertices])
            extra_constraints.extend(
                [
                    mvs >= 0,
                    cp.sum(mvs) == self.θs[i][0],
                    mvs.T @ box.vertices == self.θs[i][1:4],
                    self.Js[i][:3, :3] << cp.sum([m * V for m, V in zip(mvs, Vs)]),
                ]
            )
        sol = self._solve(extra_constraints, name="poly")
        return sol


def sinusoidal_trajectory(t):
    b = np.array([0, 2 * np.pi, 4 * np.pi]) / 3
    q = np.sin(t + b)
    v = np.cos(t + b)
    a = -np.sin(t + b)
    return q, v, a


def main():
    np.set_printoptions(suppress=True, precision=6)
    np.random.seed(0)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("outfile", help="File to save the data to.")
    # args = parser.parse_args()

    # compile the URDF
    xacro_doc = mm.XacroDoc.from_package_file(
        package_name="inertial_params",
        relative_path="urdf/threelink.urdf.xacro",
    )

    # load Pinocchio model
    model = ip.RobotModel.from_urdf_string(xacro_doc.to_urdf_string(), gravity=GRAVITY)
    # model = ip.RobotModel.from_urdf_string(xacro_doc.to_urdf_string(), gravity=np.zeros(3))

    boxes = [model.boxes[name] for name in ["link1", "link2", "link3"]]
    ellipsoids = [ip.minimum_bounding_ellipsoid(box.vertices) for box in boxes]

    # actual inertial parameter values
    params = []
    for i in range(1, 4):
        iner = model.model.inertias[i]
        mass = iner.mass
        h = iner.lever / mass
        Hc = ip.I2H(iner.inertia)
        params.append(ip.InertialParameters.translate_from_com(mass, h, Hc))
    θ = np.concatenate([p.θ for p in params])
    # θ_pin = np.concatenate([model.model.inertias[i].toDynamicParameters() for i in range(1, 4)])

    n, ts_eval = ip.compute_evaluation_times(
        duration=2 * np.pi, step=RECORDING_TIMESTEP
    )

    qs = []
    vs = []
    as_ = []
    τs = []

    # generate the trajectory
    for i in range(n):
        q, v, a = sinusoidal_trajectory(i * RECORDING_TIMESTEP)
        τ = model.compute_torques(q, v, a)

        qs.append(q)
        vs.append(v)
        as_.append(a)
        τs.append(τ)

    qs = np.array(qs)
    vs = np.array(vs)
    as_ = np.array(as_)
    τs = np.array(τs)

    # add noise
    vs_noise_raw = np.random.random(size=vs.shape) - 0.5  # mean = 0, width = 1
    vs_noisy = vs + VEL_NOISE_WIDTH * vs_noise_raw + VEL_NOISE_BIAS

    as_mid = (vs_noisy[1:, :] - vs_noisy[:-1, :]) / RECORDING_TIMESTEP
    vs_mid = (vs_noisy[1:, :] + vs_noisy[:-1, :]) / 2
    qs_mid = (qs[1:, :] - qs[:-1, :]) / 2  # TODO?

    Ys = []
    Ys_noisy = []
    for i in range(n):
        Ys.append(model.compute_joint_torque_regressor(qs[i], vs[i], as_[i]))
        if i < n - 1:
            Ys_noisy.append(
                model.compute_joint_torque_regressor(qs_mid[i], vs_mid[i], as_mid[i])
            )

    Ys = np.array(Ys)
    Ys_noisy = np.array(Ys_noisy)

    # IPython.embed()
    # return

    # partition data
    n = vs.shape[0]
    n_train = int(TRAIN_TEST_SPLIT * n)

    Ys_train = Ys_noisy[:n_train]
    τs_train = τs[:n_train]

    Ys_test = Ys[n_train:]
    τs_test = τs[n_train:]

    # solve the ID problem
    prob = IPIDProblem(
        Ys_train, τs_train, reg_params=params, reg_coeff=REGULARIZATION_COEFF
    )
    params_nom = prob.solve_nominal()
    params_poly = prob.solve_polyhedron(boxes)
    params_ell = prob.solve_ellipsoid(ellipsoids)

    # results
    riemannian_err_nom = np.sum(
        [ip.positive_definite_distance(p.J, pn.J) for p, pn in zip(params, params_nom)]
    )
    riemannian_err_ell = np.sum(
        [ip.positive_definite_distance(p.J, pn.J) for p, pn in zip(params, params_ell)]
    )
    riemannian_err_poly = np.sum(
        [ip.positive_definite_distance(p.J, pn.J) for p, pn in zip(params, params_poly)]
    )

    θ_nom = np.concatenate([p.θ for p in params_nom])
    validation_err_nom = ip.validation_rmse(Ys_test, τs_test, θ_nom)

    θ_ell = np.concatenate([p.θ for p in params_ell])
    validation_err_ell = ip.validation_rmse(Ys_test, τs_test, θ_ell)

    θ_poly = np.concatenate([p.θ for p in params_poly])
    validation_err_poly = ip.validation_rmse(Ys_test, τs_test, θ_poly)

    print("\nRiemannian error")
    print(f"nominal    = {riemannian_err_nom}")
    print(f"ellipsoid  = {riemannian_err_ell}")
    print(f"polyhedron = {riemannian_err_poly}")

    print("\nValidation error")
    print(f"nominal    = {validation_err_nom}")
    print(f"ellipsoid  = {validation_err_ell}")
    print(f"polyhedron = {validation_err_poly}")

    IPython.embed()

    if not VISUALIZE:
        return

    pyb.connect(pyb.GUI)
    pyb.setTimeStep(1.0 / SIM_FREQ)
    pyb.setGravity(*GRAVITY)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground_id = pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

    # load PyBullet model
    with xacro_doc.temp_urdf_file_path() as urdf_path:
        robot_id = pyb.loadURDF(
            urdf_path,
            [0, 0, 0],
            useFixedBase=True,
        )
    robot = pyb_utils.Robot(robot_id, tool_link_name="link3")

    # remove joint friction (only relevant for torque control)
    robot.set_joint_friction_forces([0, 0, 0])

    q0 = sinusoidal_trajectory(0)[0]
    robot.reset_joint_configuration(q0)

    Kq = np.eye(model.nq)
    Kv = np.eye(model.nv)
    dt = 1.0 / SIM_FREQ
    t = 0
    while t < 2 * np.pi:
        qd, vd, ad = sinusoidal_trajectory(t)
        q, v = robot.get_joint_states()

        # computed torque control law
        α = ad + Kq @ (qd - q) + Kv @ (vd - v)
        u = model.compute_torques(q, v, α)

        robot.command_effort(u)
        # u = Kq @ (qd - q) + vd
        # robot.command_velocity(u)

        pyb.stepSimulation()
        time.sleep(dt)

        t += dt

    IPython.embed()
    return


main()
