"""Lambda=0.3, T=5, with the real-cycle rule already used at T=4.

Same notebook, same rank-32 network, same training settings as the T=4 run.
A bounce is kept when its path is still mostly real:
max|Im x| < 0.5 and max|Im x| < max|Re x|.
The trivial saddle is always kept. Each kept bounce is counted twice.
The lattice is computed only after that sum is fixed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import investigate_lambda_up_0p3_to_1 as inv

ROOT = Path(__file__).resolve().parent
inv.NB_PATH = ROOT / "Complex_Time_QHO_Anharmonic_MU_lambda0p3_T4_morethimbles.ipynb"
OUT = ROOT / "runs" / "lambda0p3_T5_realcycle"
OUT.mkdir(parents=True, exist_ok=True)

LAM = 0.3
T_VALUE = 5.0
# Frozen from the T=4 geometry. Real-cycle bounces had Im/Re about 0.01-0.04.
# Paths that had left the real axis had max|Im| of 1.3 or about 2.6.
IM_CUT = 0.5


def on_real_cycle(x):
    re = float(np.max(np.abs(np.real(x))))
    im = float(np.max(np.abs(np.imag(x))))
    return im < IM_CUT and im < re, re, im


def configure(g):
    g["LAMBDA"] = LAM
    g["T"] = T_VALUE
    g["A_C"] = (T_VALUE - 1j * g["ETA"]) / g["N"]
    g["RANK"] = 32
    g["EPOCHS"] = 2000
    g["BATCH"] = 1024
    g["LEARNING_RATE"] = 1.0e-3
    g["N_MC"] = 200_000
    g["LOG_EVERY"] = 500


def train(g, label, saddle, seed, store):
    if label in store:
        row = store[label]
        k = complex(row["K_re"], row["K_im"])
        print(f"    reuse {label}: K={k:+.6f}  ESS/N={row['ESS']:.5f}", flush=True)
        return k, float(row["ESS"])
    print("\n" + "=" * 78, flush=True)
    print(f"*** STARTING ML TRAINING: {label} thimble at lambda={LAM}, T={T_VALUE} ***", flush=True)
    print("=" * 78, flush=True)
    model, _hist, seconds = g["train_thimble"](label, saddle, seed)
    estimates, esses = [], []
    for i in range(2):
        est, stderr, ess, _sc, _bl = g["estimate_thimble"](model, seed + 31 * i)
        estimates.append(est)
        esses.append(ess)
        print(f"  draw {i+1}: K={est:+.6f}  ESS/N={ess:.5f}  stderr={stderr:.2e}", flush=True)
    k = sum(estimates) / len(estimates)
    ess = float(np.median(esses))
    print(f"K_{label}={k:+.6f}  ESS/N={ess:.5f}  {seconds:.0f}s", flush=True)
    store[label] = {"K_re": k.real, "K_im": k.imag, "ESS": ess, "seconds": seconds}
    (OUT / "results.json").write_text(json.dumps(store, indent=1), encoding="utf-8")
    return k, ess


def main():
    g = inv.load_notebook_namespace()
    configure(g)

    print("Searching harmonic bounce families", flush=True)
    families = inv.collect_families(g, max_pairs=4)
    np.savez(OUT / "saddles.npz", **{f"b{i+1}": x for i, x in enumerate(families)})

    print("\nReal-cycle decision (lattice not consulted)", flush=True)
    chosen = []
    rows = []
    for i, x in enumerate(families):
        keep, re, im = on_real_cycle(x)
        s = g["action_np"](x)
        ratio = im / re if re > 0 else float("inf")
        name = f"b{i+1}"
        rows.append({
            "name": name,
            "keep": keep,
            "max_re": re,
            "max_im": im,
            "im_over_re": ratio,
            "S_re": float(s.real),
            "S_im": float(s.imag),
        })
        flag = "KEEP" if keep else "DROP"
        print(
            f"  {name}: max|Re|={re:.4f}  max|Im|={im:.4e}  Im/Re={ratio:.4f}  "
            f"S={s.real:+.4f}{s.imag:+.4f}i  {flag}",
            flush=True,
        )
        if keep:
            chosen.append((name, x))
    (OUT / "selection.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"Kept: trivial + {[name for name, _x in chosen]}", flush=True)

    store_path = OUT / "results.json"
    store = json.loads(store_path.read_text(encoding="utf-8")) if store_path.exists() else {}

    total = 0j
    k0, e0 = train(g, "trivial", np.zeros(g["D"], dtype=np.complex128), g["SEED"], store)
    total += k0
    print(f"  trivial  K={k0:+.6f}  ESS/N={e0:.5f}", flush=True)
    for i, (name, sad) in enumerate(chosen):
        k, ess = train(g, name, sad, g["SEED"] + 1 + i, store)
        total += 2 * k
        print(f"  {name} x2  K={k:+.6f}  ESS/N={ess:.5f}", flush=True)

    print(f"\nFixed sum K={total:+.6f}", flush=True)
    k_lat = inv.lattice_K_fft(LAM, (T_VALUE - 1j * 0.05) / 16)
    err = abs(total - k_lat) / abs(k_lat)
    print(f"lattice {k_lat:+.6f}", flush=True)
    print(f"error {100 * err:.3f}%", flush=True)
    (OUT / "summary.json").write_text(json.dumps({
        "K_re": total.real,
        "K_im": total.imag,
        "lattice_re": k_lat.real,
        "lattice_im": k_lat.imag,
        "rel_err": err,
        "kept": ["trivial"] + [name for name, _x in chosen],
    }, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
