"""Lambda=0.3, T=4: more thimbles plus the rank-32 network.

The seven-thimble sum was 271% off because the one-hump bounce
(ESS/N ~ 4e-4) swamped a sum that is otherwise within a few percent.
Same SaddleFlow, with rank 32, 2000 epochs, batch 1024, 200k samples.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import investigate_lambda_up_0p3_to_1 as inv
import investigate_lam0p5_T3p6_morethimbles as base

ROOT = Path(__file__).resolve().parent
inv.NB_PATH = ROOT / "Complex_Time_QHO_Anharmonic_MU_lambda0p3_T4_morethimbles.ipynb"
OUT = ROOT / "runs" / "lambda0p3_T4_morethimbles"
OUT.mkdir(parents=True, exist_ok=True)
SAVED = np.load(ROOT / "runs" / "lambda_up_0p3_to_1" / "saddles_lam0p3_T4p0.npz")


def configure(g):
    g["LAMBDA"] = 0.3
    g["T"] = 4.0
    g["A_C"] = (4.0 - 1j * g["ETA"]) / g["N"]
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
    print(f"*** STARTING ML TRAINING: {label} thimble at lambda=0.3, T=4 ***", flush=True)
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
    k_lat = inv.lattice_K_fft(0.3, (4.0 - 1j * 0.05) / 16)
    print(f"lattice {k_lat:+.6f}", flush=True)

    known = [SAVED[name] for name in ("b1", "b2", "b3", "b4")]
    print("Saved families", flush=True)
    base.describe(g, known)

    print("Searching two-mode guesses for extra small-action saddles", flush=True)
    raw = base.search_shapes(g, base.two_mode_shapes(g), np.linspace(0.8, 10.0, 12))
    extra = []
    for x in raw:
        if any(base.same(x, old, 1e-3) for old in known + extra):
            continue
        s = g["action_np"](x)
        if abs(s) > 30.0 or abs(s.imag) > 8.0:
            continue
        extra.append(x)
    extra.sort(key=lambda x: abs(g["action_np"](x)))
    extra = extra[:3]
    print(f"Extra families to train: {len(extra)}", flush=True)
    base.describe(g, extra)
    np.savez(OUT / "extra_saddles.npz", **{f"e{i+1}": x for i, x in enumerate(extra)})

    store_path = OUT / "results.json"
    store = json.loads(store_path.read_text(encoding="utf-8")) if store_path.exists() else {}

    pieces = []
    k0, e0 = train(g, "trivial", np.zeros(g["D"], dtype=np.complex128), g["SEED"], store)
    pieces.append(("trivial", k0, e0, 1))
    for i, name in enumerate(("b1", "b2", "b3", "b4")):
        k, ess = train(g, name, SAVED[name], g["SEED"] + 1 + i, store)
        pieces.append((name, k, ess, 2))
    for i, sad in enumerate(extra):
        k, ess = train(g, f"e{i+1}", sad, g["SEED"] + 10 + i, store)
        pieces.append((f"e{i+1}", k, ess, 2))

    print("\nCumulative sums", flush=True)
    print(f"{'thimbles':>8}  {'K':>26}  {'err%':>8}  {'ESS':>8}  label")
    total = 0j
    healthy = 0j
    n_th = 0
    n_healthy = 0
    for label, k, ess, mult in pieces:
        total = total + mult * k
        n_th += mult if label == "trivial" else mult
        if label == "trivial":
            n_th = 1
        err = abs(total - k_lat) / abs(k_lat)
        flag = ""
        if ess >= 0.01 or label == "trivial":
            healthy = healthy + (1 if label == "trivial" else 2) * k
            n_healthy = 1 + 2 * sum(
                1 for lab, _k, e, _m in pieces
                if lab != "trivial" and e >= 0.01 and pieces.index((lab, _k, e, _m)) <= pieces.index((label, k, ess, mult))
            )
            flag = "  [kept]"
        print(f"{n_th:8d}  {total:+.6f}  {100*err:7.3f}%  {ess:8.5f}  {label}{flag}", flush=True)
    # Healthy sum: trivial plus every bounce with ESS/N >= 0.01.
    healthy = k0 + 2.0 * sum((k for lab, k, ess, _m in pieces if lab != "trivial" and ess >= 0.01), 0j)
    n_h = 1 + 2 * sum(1 for lab, _k, ess, _m in pieces if lab != "trivial" and ess >= 0.01)
    eh = abs(healthy - k_lat) / abs(k_lat)
    print(f"\nFull sum:     {total:+.6f}  err={100*abs(total-k_lat)/abs(k_lat):.3f}%", flush=True)
    print(f"ESS>=0.01:    {healthy:+.6f}  err={100*eh:.3f}%  ({n_h} thimbles)", flush=True)
    (OUT / "summary.json").write_text(json.dumps({
        "lattice": [k_lat.real, k_lat.imag],
        "full": [total.real, total.imag],
        "full_err": abs(total - k_lat) / abs(k_lat),
        "healthy": [healthy.real, healthy.imag],
        "healthy_err": eh,
        "healthy_thimbles": n_h,
    }, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
