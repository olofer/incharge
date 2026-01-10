"""
Numerically reliable technique to "roll out" an oscillator trajectory backward in time given
its present (cos, sin) state. This is preferred when solving for the retarded time instead of
having to store an ever increasing absolute phase.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse


def rotation_matrix(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def pos_vel_acc(osc_state, rho1, rho2, omega):
    # Evaluation of (x,y), (vx, vy), (ax, ay) at a specific oscillator state (cs, sn)
    cs, sn = osc_state[0], osc_state[1]
    x = rho1 * cs
    y = rho2 * sn
    vx = -1 * omega * rho1 * sn
    vy = omega * rho2 * cs
    ax = -1 * omega * omega * x
    ay = -1 * omega * omega * y
    return x, y, vx, vy, ax, ay


def root_function(tau, c, x, y, osc_state, rho1, rho2, omega) -> float:
    # The zero crossing for this function for tau >= 0 defines the retarded time.
    # Provide tau=0 to return the distance from (x,y) to the source
    Rtau = rotation_matrix(-1 * omega * tau)
    osc_state_tau = Rtau @ osc_state
    xtau = osc_state_tau[0] * rho1
    ytau = osc_state_tau[1] * rho2
    dx, dy = x - xtau, y - ytau
    return np.sqrt(dx * dx + dy * dy) - c * tau


def bisect_bracket(
    f: callable, xa: float, xb: float, maxevals: int = 50, epx: float = 0.5e-6
):
    assert xa < xb
    fa = f(xa)
    fb = f(xb)
    assert fa * fb < 0
    evals: int = 0
    while evals < maxevals:
        if xb - xa < epx:
            break
        xc = (xa + xb) / 2
        fc = f(xc)
        evals += 1
        if fc * fa > 0:
            xa, fa = xc, fc
        else:
            xb, fb = xc, fc

    assert evals < maxevals, "Failed to converge"

    return (xa + xb) / 2, evals


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=125)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--rho1", type=float, default=1.0)
    parser.add_argument("--rho2", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.75)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--test-bisections", type=int, default=10000)
    args = parser.parse_args()

    print(args)

    num_steps = args.steps
    angular_step = 2 * np.pi / num_steps

    print("step:", angular_step, "[rad]")

    Rf = rotation_matrix(angular_step)
    Rr = rotation_matrix(-1 * angular_step)

    assert np.allclose(np.eye(2), Rf @ Rr)
    assert np.allclose(np.eye(2), Rr @ Rf)

    osc_state = np.array([1.0, 0.0])

    rad_vec = np.tile(np.nan, num_steps)
    cos_vec = np.tile(np.nan, num_steps)
    sin_vec = np.tile(np.nan, num_steps)

    # Verify (cos,sin)-extraction via "forward" rotation matrix Rf application
    for k in range(num_steps):
        rad_vec[k] = k * angular_step
        cos_vec[k] = osc_state[0]
        sin_vec[k] = osc_state[1]
        assert np.allclose(cos_vec[k], np.cos(rad_vec[k]))
        assert np.allclose(sin_vec[k], np.sin(rad_vec[k]))
        osc_state = Rf @ osc_state

    # Reverse time by applying Rr & check that the same functions are re-obtained backwards
    for k in range(num_steps):
        osc_state = Rr @ osc_state
        assert np.allclose(osc_state[0], np.cos(rad_vec[num_steps - k - 1]))
        assert np.allclose(osc_state[1], np.sin(rad_vec[num_steps - k - 1]))

    if args.show:
        plt.plot(rad_vec, cos_vec, label="cosine", linewidth=2.0)
        plt.plot(rad_vec, sin_vec, label="sine", linewidth=2.0)
        plt.xlabel("angle $\\theta$")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Velocity check
    v_lo = np.abs(args.omega) * np.min([args.rho1, args.rho2])
    v_hi = np.abs(args.omega) * np.max([args.rho1, args.rho2])
    assert args.rho1 >= 0 and args.rho2 >= 0
    vabs = np.tile(np.nan, num_steps)
    for k in range(num_steps):
        state_k = [np.cos(rad_vec[k]), np.sin(rad_vec[k])]
        x, y, vx, vy, ax, ay = pos_vel_acc(state_k, args.rho1, args.rho2, args.omega)
        vabs[k] = np.sqrt(vx * vx + vy * vy)
        assert vabs[k] >= v_lo - 1e-14
        assert vabs[k] <= v_hi + 1e-14

    if args.show:
        plt.plot(rad_vec, vabs, label="velocity $|v|$", linewidth=2.0)
        plt.axhline(y=v_hi, linestyle="--", alpha=0.5, label="upper")
        plt.axhline(y=v_lo, linestyle="-.", alpha=0.5, label="lower")
        plt.xlabel("angle $\\theta$")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    c = v_hi / args.beta
    print("Define c = v_hi / beta =", c, "(beta:", args.beta, ")")

    print("oscillator state:", osc_state)

    if args.test_bisections > 0:
        print(
            "Testing bisection algorithm for retarded time localization (%i random query points).."
            % (args.test_bisections)
        )
        total_evals: int = 0
        max_froot: float = 0.0
        for k in range(args.test_bisections):

            # Random query point; and random oscillator state
            xq, yq = 10 * np.random.randn(), 10 * np.random.randn()
            osc_state = rotation_matrix((2 * np.random.rand() - 1) * np.pi) @ osc_state

            def local_f(tau):
                return root_function(
                    tau, c, xq, yq, osc_state, args.rho1, args.rho2, args.omega
                )

            tau0 = 0.0
            R0 = local_f(tau0)
            assert R0 > 0
            tau1 = R0 / c
            while local_f(tau1) > 0:
                tau0 = tau1
                tau1 *= 2

            # print("starting bracket:", (tau0, tau1))
            taur, evals = bisect_bracket(local_f, tau0, tau1)
            # print("taur:", taur, "(evals:", evals, ")")

            froot = local_f(taur)
            if np.abs(froot) > max_froot:
                max_froot = np.abs(froot)

            total_evals += evals

        print("<evals> =", total_evals / args.test_bisections)
        print("maximum |f(taur)| =", max_froot)
