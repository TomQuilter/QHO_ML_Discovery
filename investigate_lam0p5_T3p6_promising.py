"""Train only the small-action two-mode saddles at lambda=0.5, T=3.6.

Added on top of the original five-thimble sum (trivial + two bounce pairs).
Saddles with a huge action are left out: the previous run showed they
integrate to zero, or to a noisy term that pulls K off the lattice.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import investigate_lam0p5_T3p6_morethimbles as base

OUT = Path(__file__).resolve().parent / "runs" / "lambda0p5_T3p6_morethimbles"
OLD = json.loads(
    (Path(__file__).resolve().parent / "runs" / "lambda_up_0p3_to_1" / "thimble_checkpoints.json").read_text(
        encoding="utf-8"
    )
)
LAM = 0.5
T_VALUE = 3.6


def main():
    g = base.load_notebook_namespace()
    g["LAMBDA"] = LAM
    g["T"] = T_VALUE
    g["A_C"] = (T_VALUE - 1j * g["ETA"]) / g["N"]
    k_lat = base.lattice_K_fft(LAM, (T_VALUE - 1j * 0.05) / 16)
    k0 = complex(OLD["0.50|3.60|trivial"]["K_re"], OLD["0.50|3.60|trivial"]["K_im"])
    kb1 = complex(OLD["0.50|3.60|b1"]["K_re"], OLD["0.50|3.60|b1"]["K_im"])
    kb2 = complex(OLD["0.50|3.60|b2"]["K_re"], OLD["0.50|3.60|b2"]["K_im"])
    base_k = k0 + 2.0 * kb1 + 2.0 * kb2
    base_err = abs(base_k - k_lat) / abs(k_lat)
    print(f"lattice {k_lat:+.6f}", flush=True)
    print(f"original 5-thimble {base_k:+.6f}  err={100*base_err:.3f}%", flush=True)

    print("Searching two-mode guesses", flush=True)
    raw = base.search_shapes(g, base.two_mode_shapes(g), np.linspace(0.8, 10.0, 16))
    # Also keep single-harmonic families so we don't retrain those.
    known = base.search_shapes(g, base.single_harmonic_shapes(g, 8), np.linspace(0.3, 14.0, 36))
    picked = []
    for x in raw:
        if any(base.same(x, old, 1e-3) for old in known):
            continue
        if any(base.same(x, old, 1e-3) for old in picked):
            continue
        s = g["action_np"](x)
        if abs(s) > 40.0 or abs(s.imag) > 12.0:
            continue
        picked.append(x)
    picked.sort(key=lambda x: abs(g["action_np"](x).imag))
    print(f"Promising new families: {len(picked)}", flush=True)
    base.describe(g, picked)
    np.savez(OUT / "promising_saddles.npz", **{f"p{i+1}": x for i, x in enumerate(picked)})

    ckpt_path = OUT / "promising_checkpoints.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.exists() else {}
    running = base_k
    best = (base_err, 5, base_k, "original 5")
    for i, sad in enumerate(picked):
        label = f"p{i+1}"
        try:
            k, ess, stderr = base.train_saddle(g, label, sad, g["SEED"] + 100 + i, ckpt)
        except Exception as exc:
            print(f"  skip {label}: {exc}", flush=True)
            continue
        # train_saddle writes the other checkpoint file; mirror into ours
        ckpt_path.write_text(json.dumps(ckpt), encoding="utf-8")
        alone = base_k + 2.0 * k
        alone_err = abs(alone - k_lat) / abs(k_lat)
        running = running + 2.0 * k
        run_err = abs(running - k_lat) / abs(k_lat)
        print(
            f"  {label}: K={k:+.6f} ESS/N={ess:.4g} stderr={stderr:.2e}  "
            f"added to original 5 -> {100*alone_err:.3f}%   "
            f"all promising so far -> {100*run_err:.3f}%",
            flush=True,
        )
        if alone_err < best[0]:
            best = (alone_err, 7, alone, label)
        if run_err < best[0]:
            best = (run_err, 5 + 2 * (i + 1), running, "cumulative")
        if min(alone_err, run_err) <= 0.01:
            print("WITHIN 1%.", flush=True)
            (OUT / "SUCCESS.txt").write_text(
                f"err={min(alone_err, run_err)}\nK_alone={alone}\nK_running={running}\nlabel={label}\n",
                encoding="utf-8",
            )
            return
    print(
        f"Best: err={100*best[0]:.3f}%  thimbles~{best[1]}  via {best[3]}  K={best[2]:+.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
