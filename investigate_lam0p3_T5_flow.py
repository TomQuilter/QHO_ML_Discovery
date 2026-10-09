"""Intersection test at lambda=0.3, T=5. Does not use the lattice.

The Lefschetz thimble for e^{iS} is steepest ascent of Im S.
The dual cycle is steepest descent of Im S. A saddle meets the original
real contour when a dual trajectory, started on the dual tangent, reaches
a nearly real path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import investigate_lambda_up_0p3_to_1 as inv

ROOT = Path(__file__).resolve().parent
inv.NB_PATH = ROOT / "Complex_Time_QHO_Anharmonic_MU_lambda0p3_T4_morethimbles.ipynb"
SADDLES = np.load(ROOT / "runs" / "lambda0p3_T5_realcycle" / "saddles.npz")


def configure(g):
    g["LAMBDA"] = 0.3
    g["T"] = 5.0
    g["A_C"] = (5.0 - 1j * g["ETA"]) / g["N"]


def main():
    g = inv.load_notebook_namespace()
    configure(g)
    names = ["trivial"] + [f"b{i}" for i in range(1, 5)]
    saddles = {"trivial": np.zeros(g["D"], dtype=np.complex128)}
    for name in SADDLES.files:
        saddles[name] = SADDLES[name]

    print("Dual-flow intersection test", flush=True)
    for name in names:
        sad = saddles[name]
        s = g["action_np"](sad)
        try:
            tangent = g["tangent_map"](g["hessian_np"](sad))
        except RuntimeError as exc:
            print(f"  {name}: tangent failed ({exc})", flush=True)
            continue
        hits, closest, ran = flow_saddle(g, sad, tangent)
        print(
            f"  {name}: hits={hits}/{ran}  closest max|Im|={closest:.3e}  "
            f"S={s.real:+.3f}{s.imag:+.3f}i  weight=e^{-s.imag:+.2f}",
            flush=True,
        )


def flow_saddle(g, saddle, tangent):
    dim = g["D"]
    hits = 0
    closest = float("inf")
    ran = 0
    for axis in range(dim):
        for sign in (1.0, -1.0):
            z = np.zeros(dim)
            z[axis] = sign * 0.03
            start = saddle + 1j * tangent @ z
            hit, near = descend(g, saddle, start)
            ran += 1
            closest = min(closest, near)
            hits += int(hit)
    return hits, closest, ran


def descend(g, saddle, start):
    dim = g["D"]

    def rhs(_t, y):
        x = y[:dim] + 1j * y[dim:]
        dx = -1j * np.conjugate(g["gradient_np"](x))
        return np.r_[np.real(dx), np.imag(dx)]

    def hit_real(_t, y):
        x = y[:dim] + 1j * y[dim:]
        away = np.max(np.abs(x - saddle))
        return np.max(np.abs(np.imag(x))) - 0.08 if away > 0.3 else 1.0

    hit_real.terminal = True
    hit_real.direction = -1

    sol = solve_ivp(
        rhs,
        (0.0, 8.0),
        np.r_[np.real(start), np.imag(start)],
        events=hit_real,
        rtol=1e-6,
        atol=1e-8,
        max_step=0.05,
        dense_output=False,
    )
    imag = np.max(np.abs(sol.y[dim:, :]), axis=0)
    return bool(sol.t_events[0].size), float(np.min(imag))


if __name__ == "__main__":
    main()
