import numpy as np
import cvxpy as cp


# def J_vec_constraint(J, θ, eps=1e-4):
#     """Constraint to enforce consistency between J and θ representations."""
#     H = J[:3, :3]
#     I = cp.trace(H) * np.eye(3) - H
#     return [
#         J >> eps * np.eye(4),
#         J[3, 3] == θ[0],
#         J[:3, 3] == θ[1:4],
#         I[0, :3] == θ[4:7],
#         I[1, 1:3] == θ[7:9],
#         I[2, 2] == θ[9],
#     ]
