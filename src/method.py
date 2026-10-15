"""Chance-constrained NMPC with covariance-based signed-distance tightening.

Quadrotor model, hover LQR, covariance propagation, box SDF, and the
CasADi/IPOPT nonlinear program. State ordering:
x = [px py pz vx vy vz phi theta psi wx wy wz], u = [T tau_x tau_y tau_z].
"""

import casadi as ca
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.special import erfcinv

NX = 12
NU = 4


def gauss_quantile(p):
    return -np.sqrt(2.0) * erfcinv(2.0 * p)


def wrap_to_pi(x):
    return np.mod(x + np.pi, 2.0 * np.pi) - np.pi


def Rzyx_sx(phi, theta, psi):
    cphi, sphi = ca.cos(phi), ca.sin(phi)
    cth, sth = ca.cos(theta), ca.sin(theta)
    cpsi, spsi = ca.cos(psi), ca.sin(psi)

    Rz = ca.vertcat(
        ca.horzcat(cpsi, -spsi, 0),
        ca.horzcat(spsi, cpsi, 0),
        ca.horzcat(0, 0, 1),
    )
    Ry = ca.vertcat(
        ca.horzcat(cth, 0, sth),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-sth, 0, cth),
    )
    Rx = ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, cphi, -sphi),
        ca.horzcat(0, sphi, cphi),
    )
    return Rz @ Ry @ Rx


def euler_rate_matrix_sx(phi, theta):
    return ca.vertcat(
        ca.horzcat(1, ca.sin(phi) * ca.tan(theta), ca.cos(phi) * ca.tan(theta)),
        ca.horzcat(0, ca.cos(phi), -ca.sin(phi)),
        ca.horzcat(0, ca.sin(phi) / ca.cos(theta), ca.cos(phi) / ca.cos(theta)),
    )


def quad_dynamics_euler_sx(x, u, m, g, J):
    v = x[3:6]
    phi, theta, psi = x[6], x[7], x[8]
    omega = x[9:12]

    T = u[0]
    tau = u[1:4]

    R = Rzyx_sx(phi, theta, psi)
    e3 = ca.DM([0, 0, 1])

    p_dot = v
    v_dot = (T / m) * (R @ e3) - ca.DM([0, 0, g])

    J_sx = ca.SX(J)
    omega_dot = ca.inv(J_sx) @ (tau - ca.cross(omega, J_sx @ omega))

    eul_dot = euler_rate_matrix_sx(phi, theta) @ omega

    return ca.vertcat(p_dot, v_dot, eul_dot, omega_dot)


def build_dynamics(Ts, m, g, J):
    """Return RK4-discretized dynamics f_disc and its Jacobians f_A, f_B."""
    x_sym = ca.SX.sym("x", NX)
    u_sym = ca.SX.sym("u", NU)

    xdot_sym = quad_dynamics_euler_sx(x_sym, u_sym, m, g, J)
    f_ode = ca.Function("f_ode", [x_sym, u_sym], [xdot_sym])

    k1 = f_ode(x_sym, u_sym)
    k2 = f_ode(x_sym + Ts / 2 * k1, u_sym)
    k3 = f_ode(x_sym + Ts / 2 * k2, u_sym)
    k4 = f_ode(x_sym + Ts * k3, u_sym)
    x_next_sym = x_sym + Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    f_disc = ca.Function("f_disc", [x_sym, u_sym], [x_next_sym])
    f_A = ca.Function("f_A", [x_sym, u_sym], [ca.jacobian(x_next_sym, x_sym)])
    f_B = ca.Function("f_B", [x_sym, u_sym], [ca.jacobian(x_next_sym, u_sym)])
    return f_disc, f_A, f_B


def lqr_gain(Ts, m, g, J, Q, R):
    """Discrete LQR about hover. Returns K such that u = u_nom + K e."""
    Ac = np.zeros((NX, NX))
    Bc = np.zeros((NX, NU))

    Ac[0, 3] = 1
    Ac[1, 4] = 1
    Ac[2, 5] = 1

    Ac[3, 7] = g
    Ac[4, 6] = -g
    Bc[5, 0] = 1 / m

    Ac[6, 9] = 1
    Ac[7, 10] = 1
    Ac[8, 11] = 1

    Bc[9, 1] = 1 / J[0, 0]
    Bc[10, 2] = 1 / J[1, 1]
    Bc[11, 3] = 1 / J[2, 2]

    Ad = np.eye(NX) + Ts * Ac
    Bd = Ts * Bc

    P = solve_discrete_are(Ad, Bd, Q, R)
    K_dlqr = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return -K_dlqr


def propagate_covariance(f_A, f_B, Xref, Uref, K, G, W, N):
    """Sigma_{k+1} = Acl Sigma_k Acl^T + G W G^T along the reference, Sigma_0 = 0."""
    nx = Xref.shape[0]
    Sigma_seq = np.zeros((N + 1, nx, nx))
    GWG = G @ W @ G.T

    for k in range(N):
        Ak = np.array(f_A(Xref[:, k], Uref[:, k]))
        Bk = np.array(f_B(Xref[:, k], Uref[:, k]))
        Acl = Ak + Bk @ K

        S = Acl @ Sigma_seq[k] @ Acl.T + GWG
        Sigma_seq[k + 1] = 0.5 * (S + S.T)
    return Sigma_seq


def box_sdf_and_grad(p, box):
    """Signed distance and outward gradient for an axis-aligned box.

    box = [xmin, xmax, ymin, ymax, zmin, zmax].
    """
    p = np.asarray(p, dtype=float).reshape(3)
    xmin, xmax, ymin, ymax, zmin, zmax = box
    lo = np.array([xmin, ymin, zmin])
    hi = np.array([xmax, ymax, zmax])

    c = 0.5 * (lo + hi)
    h = 0.5 * (hi - lo)
    q = np.abs(p - c) - h

    outside_dist = np.linalg.norm(np.maximum(q, 0.0))
    inside_dist = min(max(q[0], q[1], q[2]), 0.0)
    d = outside_dist + inside_dist

    if outside_dist > 1e-9:
        v = p - np.minimum(np.maximum(p, lo), hi)
        grad = v / np.linalg.norm(v)
    else:
        dist_to_faces = np.array([
            abs(p[0] - xmin), abs(xmax - p[0]),
            abs(p[1] - ymin), abs(ymax - p[1]),
            abs(p[2] - zmin), abs(zmax - p[2]),
        ])
        idx = int(np.argmin(dist_to_faces))
        grad = np.zeros(3)
        grad[idx // 2] = -1.0 if idx % 2 == 0 else 1.0

    return d, grad


def solve_quadrotor_nmpc_box_sdf(
    f_disc, x0, x_goal, X_guess, U_guess,
    Sigma_seq, boxes, r_robot, beta_cc, obs_eps,
    Q, Qf, R, u_hover,
    T_min, T_max, tau_max, pos_min, pos_max, angle_max,
    N, nx=NX, nu=NU,
):
    """Solve one tightened NLP about (X_guess, U_guess).

    Returns (X_sol, U_sol, success) with X_sol of shape (nx, N+1) and
    U_sol of shape (nu, N).
    """
    X = ca.SX.sym("X", nx, N + 1)
    U = ca.SX.sym("U", nu, N)

    obj = 0
    gcon = []
    lbg = []
    ubg = []

    gcon.append(X[:, 0] - x0)
    lbg.append(np.zeros(nx))
    ubg.append(np.zeros(nx))

    for k in range(N):
        xk = X[:, k]
        uk = U[:, k]

        gcon.append(X[:, k + 1] - f_disc(xk, uk))
        lbg.append(np.zeros(nx))
        ubg.append(np.zeros(nx))

        dx = xk - x_goal
        du = uk - u_hover
        obj = obj + dx.T @ Q @ dx + du.T @ R @ du

        gcon.append(xk[0:3])
        lbg.append(pos_min)
        ubg.append(pos_max)

        gcon.append(xk[6:9])
        lbg.append(-angle_max * np.ones(3))
        ubg.append(angle_max * np.ones(3))

        p_ref = X_guess[0:3, k]
        Sig_p = Sigma_seq[k][0:3, 0:3]

        for box in boxes:
            d_ref, grad_d = box_sdf_and_grad(p_ref, box)
            sigma_sdf = np.sqrt(max(grad_d @ Sig_p @ grad_d, 0.0))
            sdf_lin = d_ref + ca.dot(ca.DM(grad_d), xk[0:3] - p_ref)

            gcon.append(sdf_lin)
            lbg.append([r_robot + beta_cc * sigma_sdf + obs_eps])
            ubg.append([np.inf])

    dxN = X[:, N] - x_goal
    obj = obj + dxN.T @ Qf @ dxN

    gcon.append(X[0:3, N])
    lbg.append(pos_min)
    ubg.append(pos_max)

    z = ca.vertcat(ca.reshape(X, nx * (N + 1), 1), ca.reshape(U, nu * N, 1))
    G_all = ca.vertcat(*gcon)
    lbg = np.concatenate([np.ravel(b) for b in lbg])
    ubg = np.concatenate([np.ravel(b) for b in ubg])

    nlp = {"x": z, "f": obj, "g": G_all}
    opts = {
        "ipopt.print_level": 0,
        "print_time": False,
        "ipopt.max_iter": 300,
        "ipopt.tol": 1e-4,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.sb": "yes",
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    lbz = -np.inf * np.ones(z.shape[0])
    ubz = np.inf * np.ones(z.shape[0])

    offset_U = nx * (N + 1)
    for k in range(N):
        i0 = offset_U + k * nu
        lbz[i0] = T_min
        ubz[i0] = T_max
        lbz[i0 + 1:i0 + 4] = -tau_max
        ubz[i0 + 1:i0 + 4] = tau_max

    z0 = np.concatenate([X_guess.flatten(order="F"), U_guess.flatten(order="F")])

    try:
        sol = solver(x0=z0, lbx=lbz, ubx=ubz, lbg=lbg, ubg=ubg)
    except RuntimeError:
        return X_guess, U_guess, False

    zsol = np.array(sol["x"]).ravel()
    X_sol = zsol[: nx * (N + 1)].reshape((nx, N + 1), order="F")
    U_sol = zsol[nx * (N + 1):].reshape((nu, N), order="F")
    return X_sol, U_sol, True
