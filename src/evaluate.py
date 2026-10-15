"""Closed-loop evaluation of the chance-constrained NMPC on the paper's three cases.

Usage:
    python src/evaluate.py --case 1
"""

import argparse
import os
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from method import (  # noqa: E402
    NU,
    NX,
    box_sdf_and_grad,
    build_dynamics,
    gauss_quantile,
    lqr_gain,
    propagate_covariance,
    solve_quadrotor_nmpc_box_sdf,
    wrap_to_pi,
)

# Box obstacles: (per row) [xmin xmax ymin ymax zmin zmax]
CASES = {
    1: dict(
        p0=[0.0, 0.0, 1.0],
        p_goal=[8.5, 8.0, 2.0],
        pos_min=[-1.0, -1.0, 0.2],
        pos_max=[10.0, 10.0, 4.0],
        boxes=[
            [2.5, 4.0, 2.0, 3.5, 0.5, 4.0],
            [5.6, 7.0, 3.4, 4.8, 0.5, 2.6],
            [3.2, 4.8, 6.1, 7.6, 0.7, 3.0],
        ],
    ),
    2: dict(
        p0=[0.0, 0.0, 2.0],
        p_goal=[9.0, 0.0, 2.0],
        pos_min=[-1.0, -1.0, 0.0],
        pos_max=[10.0, 1.0, 4.0],
        boxes=[
            [2.0, 3.5, -1.0, 1.0, 0.0, 2.0],
            [5.6, 7.0, -1.0, 1.0, 2.0, 4.0],
        ],
    ),
    3: dict(
        p0=[0.0, 0.0, 0.5],
        p_goal=[6.0, 0.0, 3.5],
        pos_min=[-1.0, -1.0, 0.0],
        pos_max=[7.0, 1.0, 4.0],
        boxes=[
            [2.5, 3.5, -1.0, 1.0, 0.0, 1.5],
            [2.5, 3.5, -1.0, 1.0, 2.5, 4.0],
            [2.5, 3.5, -1.0, -0.5, 0.0, 4.0],
            [2.5, 3.5, 0.5, 1.0, 0.0, 4.0],
        ],
    ),
}


def run(case, Ts=0.10, N=20, Tsim=80, nSCP=2, seed=2, verbose=True):
    # Model parameters
    m = 1.0
    g = 9.81
    J = np.diag([0.02, 0.02, 0.04])

    T_min = 0.2 * m * g
    T_max = 1.8 * m * g
    tau_max = np.array([0.25, 0.25, 0.15])
    angle_max = np.deg2rad(35.0)
    r_robot = 0.25

    sigma_acc = np.array([0.25, 0.25, 0.18])
    Wacc = np.diag(sigma_acc**2)

    alpha = 0.05
    beta_cc = gauss_quantile(1 - alpha)

    Q = np.diag([30, 30, 35, 4, 4, 5, 2, 2, 2, 0.2, 0.2, 0.2])
    Qf = np.diag([80, 80, 100, 8, 8, 10, 4, 4, 4, 0.5, 0.5, 0.5])
    R = np.diag([0.08, 0.5, 0.5, 0.5])

    u_hover = np.array([m * g, 0.0, 0.0, 0.0])

    K_tube = lqr_gain(Ts, m, g, J, Q, R)

    D = np.zeros((NX, 3))
    D[3:6, :] = np.eye(3)

    cfg = CASES[case]
    x0 = np.zeros(NX)
    x0[0:3] = cfg["p0"]
    x_goal = np.zeros(NX)
    x_goal[0:3] = cfg["p_goal"]
    pos_min = np.array(cfg["pos_min"])
    pos_max = np.array(cfg["pos_max"])
    boxes = np.array(cfg["boxes"])

    obs_eps = 1e-3
    rng = np.random.default_rng(seed)
    L_acc = np.linalg.cholesky(Wacc)

    f_disc, f_A, f_B = build_dynamics(Ts, m, g, J)

    x_actual = x0.copy()
    X_hist = np.zeros((NX, Tsim + 1))
    U_hist = np.zeros((NU, Tsim))
    X_hist[:, 0] = x_actual

    a = np.arange(N + 1) / N
    X_guess = np.outer(x0, 1 - a) + np.outer(x_goal, a)
    U_guess = np.tile(u_hover[:, None], (1, N))

    steps = Tsim
    for t in range(Tsim):
        x_nom0 = x_actual.copy()

        for scp in range(nSCP):
            Sigma_seq = propagate_covariance(
                f_A, f_B, X_guess, U_guess, K_tube, D, Wacc, N)

            X_sol, U_sol, success = solve_quadrotor_nmpc_box_sdf(
                f_disc, x_nom0, x_goal, X_guess, U_guess,
                Sigma_seq, boxes, r_robot, beta_cc, obs_eps,
                Q, Qf, R, u_hover,
                T_min, T_max, tau_max, pos_min, pos_max, angle_max,
                N)

            if not success:
                warnings.warn(f"NMPC failed at sim step {t + 1}, SCP iter {scp + 1}.")
                X_sol, U_sol = X_guess, U_guess
                break

            X_guess, U_guess = X_sol, U_sol

        u_nom = U_sol[:, 0]

        e = x_actual - X_sol[:, 0]
        e[6:9] = wrap_to_pi(e[6:9])

        u_applied = u_nom + K_tube @ e
        u_applied[0] = np.clip(u_applied[0], T_min, T_max)
        u_applied[1:4] = np.clip(u_applied[1:4], -tau_max, tau_max)

        x_next = np.array(f_disc(x_actual, u_applied)).ravel()

        w_acc = L_acc @ rng.standard_normal(3)
        x_next[3:6] += Ts * w_acc
        x_next[6:9] = wrap_to_pi(x_next[6:9])

        X_hist[:, t + 1] = x_next
        U_hist[:, t] = u_applied
        x_actual = x_next

        X_guess = np.hstack([X_sol[:, 1:], X_sol[:, -1:]])
        U_guess = np.hstack([U_sol[:, 1:], U_sol[:, -1:]])

        if verbose:
            dist = np.linalg.norm(x_actual[0:3] - x_goal[0:3])
            print(f"step {t + 1:3d}  |p - p_goal| = {dist:.3f} m")

        if (np.linalg.norm(x_actual[0:3] - x_goal[0:3]) < 0.25
                and np.linalg.norm(x_actual[3:6]) < 0.25):
            steps = t + 1
            print(f"Reached goal at step {steps}.")
            break

    X_hist = X_hist[:, : steps + 1]
    U_hist = U_hist[:, :steps]

    min_clearance = np.inf
    for k in range(X_hist.shape[1]):
        for box in boxes:
            d, _ = box_sdf_and_grad(X_hist[0:3, k], box)
            min_clearance = min(min_clearance, d - r_robot)
    print(f"Minimum trajectory-clearance: {min_clearance:.4f} m")

    return dict(
        X_hist=X_hist, U_hist=U_hist, x0=x0, x_goal=x_goal,
        pos_min=pos_min, pos_max=pos_max, boxes=boxes,
        min_clearance=min_clearance,
    )


def plot_box(ax, box, color=(0.85, 0.2, 0.2), alpha_val=0.35):
    xmin, xmax, ymin, ymax, zmin, zmax = box
    V = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
    ])
    F = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
         [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    poly = Poly3DCollection([V[f] for f in F], facecolors=color, alpha=alpha_val,
                            edgecolors="k", linewidths=1.0)
    ax.add_collection3d(poly)


def plot_results(res, out_dir, case):
    os.makedirs(out_dir, exist_ok=True)
    X_hist, U_hist = res["X_hist"], res["U_hist"]
    x0, x_goal = res["x0"], res["x_goal"]
    pos_min, pos_max = res["pos_min"], res["pos_max"]

    fig = plt.figure(figsize=(8, 7), facecolor="w")
    ax = fig.add_subplot(projection="3d")
    ax.plot(X_hist[0], X_hist[1], X_hist[2], "b-", linewidth=2.2)
    ax.plot([x0[0]], [x0[1]], [x0[2]], "go", markerfacecolor="g", markersize=9)
    ax.plot([x_goal[0]], [x_goal[1]], [x_goal[2]], "m*", markerfacecolor="m", markersize=14)
    for box in res["boxes"]:
        plot_box(ax, box)
    ax.set_xlim(pos_min[0], pos_max[0])
    ax.set_ylim(pos_min[1], pos_max[1])
    ax.set_zlim(pos_min[2], pos_max[2])
    ax.set_box_aspect(pos_max - pos_min)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=25, azim=35 - 90)
    traj_path = os.path.join(out_dir, f"case{case}_trajectory.png")
    fig.savefig(traj_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True, facecolor="w")
    labels = ["T", r"$\tau_x$", r"$\tau_y$", r"$\tau_z$"]
    for i, (ax, lab) in enumerate(zip(axes, labels)):
        ax.plot(U_hist[i], linewidth=1.5)
        ax.grid(True)
        ax.set_ylabel(lab)
    axes[-1].set_xlabel("step")
    inputs_path = os.path.join(out_dir, f"case{case}_inputs.png")
    fig.savefig(inputs_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {traj_path}")
    print(f"Saved {inputs_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", type=int, choices=sorted(CASES), default=1)
    parser.add_argument("--tsim", type=int, default=80, help="max closed-loop steps")
    parser.add_argument("--horizon", type=int, default=20, help="MPC horizon N")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--out", default="results", help="directory for saved plots")
    parser.add_argument("--quiet", action="store_true", help="suppress per-step output")
    args = parser.parse_args()

    res = run(args.case, N=args.horizon, Tsim=args.tsim, seed=args.seed,
              verbose=not args.quiet)
    plot_results(res, args.out, args.case)


if __name__ == "__main__":
    main()
