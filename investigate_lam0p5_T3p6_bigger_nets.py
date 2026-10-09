"""Lambda=0.5, T=3.6: bigger networks / longer training on the same SaddleFlow.

Only the constants change (RANK, EPOCHS, BATCH, LEARNING_RATE, N_MC).
The one-hump bounce is the piece whose sampler collapses, so the ladder is
run on it first. The best configuration is then used for every thimble, and
the 5-thimble and 7-thimble sums are compared with the lattice.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import investigate_lam0p5_T3p6_morethimbles as base

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "lambda0p5_T3p6_bigger_nets"
OUT.mkdir(parents=True, exist_ok=True)
SAD = np.load(ROOT / "runs" / "lambda_up_0p3_to_1" / "saddles_lam0p5_T3p6.npz")
PROM = np.load(ROOT / "runs" / "lambda0p5_T3p6_morethimbles" / "promising_saddles.npz")

LADDER = [
    # name, rank, epochs, batch, lr
    ("r16_e1000_b512 (baseline)", 16, 1000, 512, 1.0e-3),
    ("r32_e2000_b1024", 32, 2000, 1024, 1.0e-3),
    ("r32_e3000_b1024_lr5e-4", 32, 3000, 1024, 5.0e-4),
]
N_MC = 200_000


def configure(g, rank, epochs, batch, lr):
    g["RANK"] = rank
    g["EPOCHS"] = epochs
    g["BATCH"] = batch
    g["LEARNING_RATE"] = lr
    g["N_MC"] = N_MC
    g["LOG_EVERY"] = max(epochs // 4, 1)


def run(g, label, saddle, seed, store):
    if label in store:
        row = store[label]
        k = complex(row["K_re"], row["K_im"])
        print(f"    reuse {label}: K={k:+.6f}  ESS/N={row['ESS']:.5f}", flush=True)
        return k, float(row["ESS"]), float(row["scatter"])
    print("\n" + "=" * 78, flush=True)
    print(
        f"*** STARTING ML TRAINING: {label}  rank={g['RANK']} epochs={g['EPOCHS']} "
        f"batch={g['BATCH']} lr={g['LEARNING_RATE']}  lambda=0.5 T=3.6 ***",
        flush=True,
    )
    print("=" * 78, flush=True)
    model, _hist, seconds = g["train_thimble"](label, saddle, seed)
    estimates, esses = [], []
    for i in range(3):
        est, stderr, ess, _sc, _bl = g["estimate_thimble"](model, seed + 31 * i)
        estimates.append(est)
        esses.append(ess)
    k = sum(estimates) / 3
    scatter = float(np.std(np.array(estimates, dtype=np.complex128)))
    ess = float(np.median(esses))
    print(
        f"K_{label}={k:+.6f}  ESS/N(median of 3)={ess:.5f}  scatter={scatter:.2e}  {seconds:.0f}s",
        flush=True,
    )
    store[label] = {"K_re": k.real, "K_im": k.imag, "ESS": ess, "scatter": scatter, "seconds": seconds}
    (OUT / "results.json").write_text(json.dumps(store, indent=1), encoding="utf-8")
    return k, ess, scatter


def main():
    g = base.load_notebook_namespace()
    g["LAMBDA"] = 0.5
    g["T"] = 3.6
    g["A_C"] = (3.6 - 1j * g["ETA"]) / g["N"]
    k_lat = base.lattice_K_fft(0.5, (3.6 - 1j * 0.05) / 16)
    print(f"lattice {k_lat:+.6f}", flush=True)

    store_path = OUT / "results.json"
    store = json.loads(store_path.read_text(encoding="utf-8")) if store_path.exists() else {}
    b1 = SAD["b1"]
    print("\nLadder on the one-hump bounce", flush=True)
    ladder_results = []
    for name, rank, epochs, batch, lr in LADDER:
        configure(g, rank, epochs, batch, lr)
        k, ess, scatter = run(g, f"b1|{name}", b1, g["SEED"] + 1, store)
        ladder_results.append((name, rank, epochs, batch, lr, k, ess, scatter))

    print("\nOne-hump ladder summary", flush=True)
    for name, _r, _e, _b, _lr, k, ess, scatter in ladder_results:
        print(f"  {name:28s} K={k:+.6f}  ESS/N={ess:.5f}  scatter={scatter:.2e}", flush=True)
    best = max(ladder_results, key=lambda r: r[6])
    name, rank, epochs, batch, lr = best[:5]
    print(f"\nBest one-hump sampler: {name} (ESS/N={best[6]:.5f})", flush=True)

    configure(g, rank, epochs, batch, lr)
    k0, _, _ = run(g, f"trivial|{name}", np.zeros(g["D"], dtype=np.complex128), g["SEED"], store)
    kb2, _, _ = run(g, f"b2|{name}", SAD["b2"], g["SEED"] + 2, store)
    kb1 = best[5]
    k5 = k0 + 2 * kb1 + 2 * kb2
    e5 = abs(k5 - k_lat) / abs(k_lat)
    print(f"\n5 thimbles ({name}): K={k5:+.6f}  err={100*e5:.3f}%", flush=True)

    # Seventh thimble: the small-action two-hump saddle found earlier (p6 ~ p7 pair).
    p6 = PROM["p6"]
    kp, essp, _ = run(g, f"p6|{name}", p6, g["SEED"] + 6, store)
    k7 = k5 + 2 * kp
    e7 = abs(k7 - k_lat) / abs(k_lat)
    print(f"7 thimbles ({name}): K={k7:+.6f}  err={100*e7:.3f}%  (p6 ESS/N={essp:.4f})", flush=True)

    # Ninth: the next one-hump family b3 (second one found by the harmonic search).
    kb3, ess3, _ = run(g, f"b3|{name}", SAD["b3"], g["SEED"] + 3, store)
    k9 = k7 + 2 * kb3
    e9 = abs(k9 - k_lat) / abs(k_lat)
    print(f"9 thimbles ({name}): K={k9:+.6f}  err={100*e9:.3f}%  (b3 ESS/N={ess3:.4f})", flush=True)

    summary = {
        "config": name,
        "lattice": [k_lat.real, k_lat.imag],
        "K5": [k5.real, k5.imag],
        "err5": e5,
        "K7": [k7.real, k7.imag],
        "err7": e7,
        "K9": [k9.real, k9.imag],
        "err9": e9,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\nFINAL", json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
