"""
Test/develop root-finding scheme/algorithm solving for the retarded time in the Lienard-Wiechert sense.

https://en.wikipedia.org/wiki/Li%C3%A9nard%E2%80%93Wiechert_potential

EXAMPLES:
  python3 retarded.py --tmax 10 --beta 0.99 --check-query
  python3 retarded.py --tmax 5 --beta 0.91 --check-field --xx 25.0 --yy 25.0
  python3 retarded.py --tmax 5 --beta 0.32 --check-field --xx 50.0 --yy 50.0

"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho1", type=float, default=1.0)
    parser.add_argument("--omega1", type=float, default=1.0)
    parser.add_argument("--phi1", type=float, default=0.0)

    parser.add_argument("--rho2", type=float, default=1.0)
    parser.add_argument("--omega2", type=float, default=1.0)
    parser.add_argument("--phi2", type=float, default=-1.0 * np.pi / 2)

    parser.add_argument("--check-velocity", action="store_true")
    parser.add_argument("--check-query", action="store_true")
    parser.add_argument("--nt", type=int, default=500)
    parser.add_argument("--tmax", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=0.15)

    parser.add_argument("--xq", type=float, default=-2.0)
    parser.add_argument("--yq", type=float, default=0.5)

    parser.add_argument("--check-field", action="store_true")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--xx", type=float, default=5.0)
    parser.add_argument("--yy", type=float, default=5.0)

    parser.add_argument("--rootfinder", type=str, default="bisect")

    args = parser.parse_args()

    print(args)

    def source_position(tau):
        xtau = args.rho1 * np.cos(args.omega1 * tau + args.phi1)
        ytau = args.rho2 * np.cos(args.omega2 * tau + args.phi2)
        return xtau, ytau

    def source_velocity(tau):
        vx = -1 * args.rho1 * args.omega1 * np.sin(args.omega1 * tau + args.phi1)
        vy = -1 * args.rho2 * args.omega2 * np.sin(args.omega2 * tau + args.phi2)
        return vx, vy

    def modified_R(xr, yr, vxr, vyr, xq, yq, c):
        Rx = xq - xr
        Ry = yq - yr
        vdotR = vxr * Rx + vyr * Ry
        return np.sqrt(Rx * Rx + Ry * Ry) - vdotR / c

    def ZFUNC(tr, t, c, xq, yq):
        assert c > 0
        tau = t - tr
        assert tau >= 0.0
        xs, ys = source_position(tr)
        dx, dy = xs - xq, ys - yq
        return np.sqrt(dx * dx + dy * dy) - c * tau

    # Find tr < t, so that c * (t - tr) = |rs(tr) - (X,Y)|
    def retarded_time(
        X, Y, c, t, ept=0.5e-6, epf=0.5e-6, verbose=False, max_evals=50, method="bisect"
    ):
        t0 = t
        f0 = ZFUNC(t0, t, c, X, Y)

        assert f0 > 0

        dt_stride = f0 / c
        t1 = t0 - dt_stride
        f1 = ZFUNC(t1, t, c, X, Y)

        if verbose:
            print(t0, f0)
            print(t1, f1)

        evals = 2

        # backtrack until zero has been crossed
        while f1 > 0 and evals < max_evals:
            t0, f0 = t1, f1
            t1 = t0 - dt_stride
            f1 = ZFUNC(t1, t, c, X, Y)
            evals += 1
            if verbose:
                print(t1, f1)

        assert f0 > 0 and f1 < 0
        assert t0 > t1

        if method == "secant":
            # Very fast when it works
            while evals < max_evals:
                if verbose:
                    print(t1, f1)
                DT = t1 - t0
                DF = f1 - f0
                if np.abs(DT) < ept or np.abs(f1) < epf:
                    break
                t2 = t1 - f1 * (DT / DF)
                f2 = ZFUNC(t2, t, c, X, Y)
                evals += 1
                t0, f0 = t1, f1
                t1, f1 = t2, f2

        elif method == "bisect":
            # Always works & simple
            while evals < max_evals:
                if verbose:
                    print((t1, t0), (f1, f0))
                if np.abs(t1 - t0) < ept:
                    break
                t2 = (t0 + t1) / 2
                f2 = ZFUNC(t2, t, c, X, Y)
                evals += 1
                if f2 > 0:
                    t0, f0 = t2, f2
                else:
                    t1, f1 = t2, f2

        elif method == "ridder":
            # Faster, a bit complex though
            while evals < max_evals:
                if verbose:
                    print((t1, t0), (f1, f0))
                if np.abs(t1 - t0) < ept:
                    break
                tmid = (t0 + t1) / 2
                fmid = ZFUNC(tmid, t, c, X, Y)
                trid = tmid + (tmid - t1) * np.sign(f1) * fmid / np.sqrt(
                    fmid * fmid - f1 * f0
                )
                frid = ZFUNC(trid, t, c, X, Y)
                evals += 2
                if np.abs(frid) < epf:
                    return trid, evals
                if frid >= 0:
                    t0, f0 = trid, frid
                    if fmid < 0:
                        t1, f1 = tmid, fmid
                else:
                    t1, f1 = trid, frid
                    if fmid >= 0:
                        t0, f0 = tmid, fmid

        elif method == "regula-falsi":
            # Fast when it works but not so reliable
            while evals < max_evals:
                if verbose:
                    print((t1, t0), (f1, f0))
                if np.abs(t1 - t0) < ept:
                    break
                t2 = (t1 * f0 - t0 * f1) / (f0 - f1)
                f2 = ZFUNC(t2, t, c, X, Y)
                evals += 1
                if np.abs(f2) < epf:
                    return t2, evals
                if f2 > 0:
                    t0, f0 = t2, f2
                else:
                    t1, f1 = t2, f2

        else:
            raise ValueError("method argument not recognized")

        assert evals < max_evals, "failed to converge"

        tr = (t0 + t1) / 2
        return tr, evals

    tvec = np.linspace(0.0, args.tmax, args.nt)
    dt = tvec[1] - tvec[0]

    xt, yt = source_position(tvec)
    vxt, vyt = source_velocity(tvec)
    vabs = np.sqrt(vxt * vxt + vyt * vyt)

    vupper = np.sqrt(
        args.rho1 * args.rho1 * args.omega1 * args.omega1
        + args.rho2 * args.rho2 * args.omega2 * args.omega2
    )

    if args.omega1 == args.omega2:
        # Tight bound for special case
        A = -args.rho1 * args.rho1 * args.omega1 * args.omega1 / 2
        B = -args.rho2 * args.rho2 * args.omega2 * args.omega2 / 2
        alfa = A * np.cos(2 * args.phi1) + B * np.cos(2 * args.phi2)
        beta = A * np.sin(2 * args.phi1) + B * np.sin(2 * args.phi2)
        C1max = np.sqrt(alfa * alfa + beta * beta)
        vupper = np.sqrt(vupper * vupper / 2 + C1max)

    print("max. velocity (safe upper bound):", vupper)
    print("max. velocity (snippet):", np.max(vabs))
    assert np.max(vabs) < vupper + 1.0e-6

    c = vupper / args.beta

    print("setting c:", c, "(beta:", args.beta, ")")

    if args.check_velocity:
        fd_vx = np.diff(xt) / dt
        fd_vy = np.diff(yt) / dt
        plt.plot(tvec[:-1], fd_vx, label="FD-vx")
        plt.plot(tvec[:-1], fd_vy, label="FD-vy")
        plt.plot(tvec, vxt, label="vx", linestyle="--")
        plt.plot(tvec, vyt, label="vy", linestyle="--")
        plt.plot(tvec, vabs, label="|v|", linestyle="-.")
        plt.axhline(
            y=vupper, linestyle=":", linewidth=2, alpha=0.50, label="safe upper bound"
        )
        plt.xlabel("time")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Show trajectory & also show an example query point

    plt.plot(xt, yt, linewidth=2.0, alpha=0.75, label="path")
    plt.plot(xt[0], yt[0], linestyle="none", marker="s", label="start")
    plt.plot(xt[-1], yt[-1], linestyle="none", marker="o", label="stop")
    plt.plot(args.xq, args.yq, linestyle="none", marker="*", label="query")
    plt.gca().set_aspect("equal")
    plt.grid(True)
    plt.xlabel("$x(t)$")
    plt.ylabel("$y(t)$")
    plt.title("Source Trajectory")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if args.check_query:
        # Given x(t), y(t); analytical expressions available; find the retarded time for different positions in space (X,Y)
        Xq = args.xq
        Yq = args.yq

        tr, _ = retarded_time(Xq, Yq, c, tvec[-1], verbose=True, method=args.rootfinder)

        R1 = c * (tvec[-1] - tvec)
        dx = xt - Xq
        dy = yt - Yq
        R2 = np.sqrt(dx * dx + dy * dy)

        plt.plot(tvec, R1, label="$R1$ (light-cone)")
        plt.plot(tvec, R2, label="$R2$ (source-distance)")
        plt.plot(tr, c * (tvec[-1] - tr), label="$R(tr)$", marker="o")
        plt.xlabel("retarded time")
        plt.ylabel("retarded distance")
        plt.grid(True)
        plt.legend()
        plt.title("Illustration: retarded time solver")
        plt.tight_layout()
        plt.show()

    if args.check_field:
        print("Evaluating tr on grid:", args.nx, "-by-", args.ny)
        print("Using x grid limits:", -args.xx, args.xx)
        print("Using y grid limits:", -args.yy, args.yy)

        # Evaluate the retarded distance to plot a full image of R(tr) for (x,y) @ t=tvec[-1]
        xvec = np.linspace(-args.xx, args.xx, args.nx)
        yvec = np.linspace(-args.yy, args.yy, args.ny)
        Tmat = np.tile(np.nan, (args.nx, args.ny))
        Rmat = np.copy(Tmat)
        Rmod = np.copy(Tmat)
        Amat = np.tile(np.nan, (args.nx, args.ny, 2))
        Imat = np.tile(np.nan, (args.nx, args.ny))

        for i, xi in enumerate(xvec):
            for j, yj in enumerate(yvec):
                tij, evals = retarded_time(xi, yj, c, tvec[-1], method=args.rootfinder)
                rij = c * (tvec[-1] - tij)
                Tmat[i, j] = tij
                Rmat[i, j] = rij
                Imat[i, j] = evals
                xr, yr = source_position(tij)
                vxr, vyr = source_velocity(tij)
                rij_ = modified_R(xr, yr, vxr, vyr, xi, yj, c)
                Rmod[i, j] = rij_
                Amat[i, j, 0] = (vxr / c) / (1.0 + Rmod[i, j])
                Amat[i, j, 1] = (vyr / c) / (1.0 + Rmod[i, j])

        # TODO: encode the triad potentials in the color-channels (phi, Ax, Ay)

        FIELDS = [
            {"name": "Retarded distance", "field": Rmat.T},
            {"name": "Modified retarded distance", "field": Rmod.T},
            {"name": "Function evaluations", "field": Imat.T},
            {"name": "$A_x$", "field": Amat[:, :, 0].T},
            {"name": "$A_y$", "field": Amat[:, :, 1].T},
            {"name": "$\\psi$", "field": 1 / (1 + Rmod.T)},
        ]

        for k, f in enumerate(FIELDS):
            print(k, f["name"])
            plt.imshow(
                f["field"],
                extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]],
                origin="lower",
                aspect="equal",
            )
            plt.plot(
                xt[-1], yt[-1], linestyle="none", marker="o", color="white", alpha=0.75
            )
            vx_, vy_ = vxt[-1], vyt[-1]
            v_norm = np.sqrt(vx_ * vx_ + vy_ * vy_)
            vhat = np.array([vx_, vy_]) / v_norm
            aa = 0.50
            plt.plot(
                [xt[-1], xt[-1] + aa * vhat[0]],
                [yt[-1], yt[-1] + aa * vhat[1]],
                linewidth=1.0,
                alpha=0.50,
                color="white",
            )
            plt.xlabel("$x$")
            plt.ylabel("$y$")
            plt.colorbar()
            plt.title("%s @ $\\beta$=%.3f" % (f["name"], args.beta))
            plt.tight_layout()
            plt.show()

    print("done.")
