"""More thimbles at lambda=0.5, T=3.6.

Uses the copied notebook's SaddleFlow, training, and saddle search unchanged.
The earlier scan found four bounce families here and only trained two.
This run trains every distinct family, reusing the two already trained.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.optimize import root

import investigate_lambda_up_0p3_to_1 as inv

ROOT = Path(__file__).resolve().parent
inv.NB_PATH = ROOT / "Complex_Time_QHO_Anharmonic_MU_lambda0p5_T3p6_morethimbles.ipynb"
load_notebook_namespace = inv.load_notebook_namespace
lattice_K_fft = inv.lattice_K_fft
OUT = ROOT / "runs" / "lambda0p5_T3p6_morethimbles"
OUT.mkdir(parents=True, exist_ok=True)
OLD_CKPT = ROOT / "runs" / "lambda_up_0p3_to_1" / "thimble_checkpoints.json"
OLD_SADDLES = ROOT / "runs" / "lambda_up_0p3_to_1" / "saddles_lam0p5_T3p6.npz"

LAM = 0.5
T_VALUE = 3.6
GATE = 0.01


def same(a, b, tol=1e-4):
    return np.max(np.abs(a - b)) < tol or np.max(np.abs(a + b)) < tol


def add_unique(pool, x, tol=1e-4):
    if np.max(np.abs(x)) < 1e-3:
        return
    if any(same(x, old, tol) for old in pool):
        return
    pool.append(x)


def search_shapes(g, shapes, amplitudes):
    found = []
    for shape in shapes:
        for amplitude in amplitudes:
            guess = np.r_[amplitude * shape, np.zeros(g["D"])]
            solved = root(g["packed_gradient"], guess, method="lm", options={"maxiter": 4000})
            x = solved.x[: g["D"]] + 1j * solved.x[g["D"] :]
            residual = np.max(np.abs(g["gradient_np"](x)))
            if residual < 1e-8:
                add_unique(found, x)
    return found


def single_harmonic_shapes(g, max_h):
    sites = np.arange(1, g["N"])
    return [np.sin(h * np.pi * sites / g["N"]) for h in range(1, max_h + 1)]


def two_mode_shapes(g):
    sites = np.arange(1, g["N"])
    shapes = []
    for h1, h2 in ((1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (1, 4)):
        s1 = np.sin(h1 * np.pi * sites / g["N"])
        s2 = np.sin(h2 * np.pi * sites / g["N"])
        for mix in (0.35, 0.7, 1.0, 1.4):
            shapes.append(s1 + mix * s2)
            shapes.append(s1 - mix * s2)
    return shapes


def describe(g, saddles):
    rows = []
    for i, x in enumerate(saddles):
        s = g["action_np"](x)
        rows.append((i, s, np.max(np.abs(x))))
        print(
            f"  family {i+1}: max|x|={np.max(np.abs(x)):.3f}  "
            f"S={s.real:+.4f}{s.imag:+.4f}i  |S|={abs(s):.4f}",
            flush=True,
        )
    return rows


def order_families(g, pool):
    """Harmonic-1 lowest-|S| first, then the rest by increasing Im S."""
    h1 = []
    sites = np.arange(1, g["N"])
    shape = np.sin(np.pi * sites / g["N"])
    # Re-solve is unnecessary: pick the pool member closest to a sine-1 profile
    # by overlap, among those already found from harmonic 1 via the caller.
    chosen = []
    rest = list(pool)
    # Prefer the saved b1 if present so labels match the earlier run.
    return rest


def train_saddle(g, label, saddle, seed, ckpt):
    key = label
    if key in ckpt:
        row = ckpt[key]
        print(
            f"    reuse {label}: K={complex(row['K_re'], row['K_im']):+.6f}  ESS/N={row['ESS']:.4g}",
            flush=True,
        )
        return complex(row["K_re"], row["K_im"]), float(row["ESS"]), float(row["stderr"])
    print("\n" + "=" * 78, flush=True)
    print(f"*** STARTING ML TRAINING: {label} thimble at lambda={LAM}, T={T_VALUE} ***", flush=True)
    print("=" * 78, flush=True)
    model, _history, seconds = g["train_thimble"](label, saddle, seed)
    estimate, stderr, ess, _scatter, _blocks = g["estimate_thimble"](model, seed)
    print(
        f"K_{label}={estimate:+.8f}  ESS/N={ess:.6f}  stderr={stderr:.3e}  {seconds:.0f}s",
        flush=True,
    )
    ckpt[key] = {
        "K_re": estimate.real,
        "K_im": estimate.imag,
        "ESS": ess,
        "stderr": stderr,
        "seconds": seconds,
    }
    (OUT / "checkpoints.json").write_text(json.dumps(ckpt), encoding="utf-8")
    return estimate, ess, stderr


def report(pieces, esses, labels, k_lat):
    total = 0j
    print("\nCumulative sum (each bounce counted twice)", flush=True)
    print(f"{'thimbles':>10}  {'K':>28}  {'err%':>10}  {'last ESS':>10}")
    best = None
    for i, (k, ess, label) in enumerate(zip(pieces, esses, labels)):
        total = total + (k if label == "trivial" else 2.0 * k)
        err = abs(total - k_lat) / abs(k_lat)
        n_th = 1 + 2 * i if label != "trivial" else 1
        if label == "trivial":
            n_th = 1
        else:
            n_th = 1 + 2 * sum(1 for name in labels[: i + 1] if name != "trivial")
        print(f"{n_th:10d}  {total:+.6f}  {100*err:9.3f}%  {ess:10.4g}  ({label})", flush=True)
        if best is None or err < best[0]:
            best = (err, n_th, total)
    return best


def main():
    print("Loading copied notebook", flush=True)
    g = load_notebook_namespace()
    g["LAMBDA"] = LAM
    g["T"] = T_VALUE
    g["A_C"] = (T_VALUE - 1j * g["ETA"]) / g["N"]
    ac = (T_VALUE - 1j * 0.05) / 16
    k_lat = lattice_K_fft(LAM, ac)
    print(f"lattice K = {k_lat:+.6f}", flush=True)

    print("\nSingle-harmonic search, harmonics 1..8", flush=True)
    amps = np.linspace(0.3, 14.0, 36)
    shapes = single_harmonic_shapes(g, 8)
    pool = []
    from_h1 = []
    for harmonic, shape in enumerate(shapes, start=1):
        found = search_shapes(g, [shape], amps)
        new = [x for x in found if not any(same(x, old) for old in pool)]
        print(f"  harmonic {harmonic}: {len(found)} solved, {len(new)} new", flush=True)
        for x in new:
            pool.append(x)
            if harmonic == 1:
                from_h1.append(x)
    first = min(from_h1, key=lambda x: abs(g["action_np"](x))) if from_h1 else None
    rest = [x for x in pool if first is None or not same(x, first)]
    rest.sort(key=lambda x: g["action_np"](x).imag)
    families = ([first] if first is not None else []) + rest
    print(f"Distinct bounce families: {len(families)}", flush=True)
    describe(g, families)

    old = json.loads(OLD_CKPT.read_text(encoding="utf-8")) if OLD_CKPT.exists() else {}
    old_sad = np.load(OLD_SADDLES) if OLD_SADDLES.exists() else None
    ckpt_path = OUT / "checkpoints.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.exists() else {}

    if "trivial" not in ckpt and "0.50|3.60|trivial" in old:
        ckpt["trivial"] = old["0.50|3.60|trivial"]

    labels = ["trivial"]
    saddles = [np.zeros(g["D"], dtype=np.complex128)]
    for i, sad in enumerate(families):
        labels.append(f"b{i+1}")
        saddles.append(sad)
        # If this saddle is an old saved one, copy that checkpoint onto this label.
        if old_sad is not None:
            for name in old_sad.files:
                src = f"0.50|3.60|{name}"
                if src in old and same(sad, old_sad[name]) and f"b{i+1}" not in ckpt:
                    ckpt[f"b{i+1}"] = old[src]

    pieces, esses = [], []
    kept_labels = []
    for i, (label, sad) in enumerate(zip(labels, saddles)):
        try:
            k, ess, _stderr = train_saddle(g, label, sad, g["SEED"] + i, ckpt)
        except Exception as exc:
            print(f"  skip {label}: {exc}", flush=True)
            traceback.print_exc()
            continue
        pieces.append(k)
        esses.append(ess)
        kept_labels.append(label)

    best = report(pieces, esses, kept_labels, k_lat)
    if best is not None:
        print(
            f"\nBest prefix: {best[1]} thimbles, err={100*best[0]:.3f}%, K={best[2]:+.6f}",
            flush=True,
        )
    full = pieces[0] + 2.0 * sum(pieces[1:], 0j) if pieces else 0j
    full_err = abs(full - k_lat) / abs(k_lat)
    print(f"Full sum err={100*full_err:.3f}%  ({1+2*(len(pieces)-1)} thimbles)", flush=True)

    if full_err <= GATE:
        print("WITHIN 1%. Single-harmonic families are enough.", flush=True)
        (OUT / "SUCCESS.txt").write_text(
            f"full_err={full_err}\nK={full}\nlattice={k_lat}\nthimbles={1+2*(len(pieces)-1)}\n",
            encoding="utf-8",
        )
        return

    print("\nSingle-harmonic sum is still outside 1%. Searching two-mode guesses.", flush=True)
    extra = search_shapes(g, two_mode_shapes(g), np.linspace(0.8, 10.0, 16))
    new = []
    for x in extra:
        if not any(same(x, old) for old in families):
            add_unique(new, x)
    new.sort(key=lambda x: g["action_np"](x).imag)
    print(f"New families from two-mode guesses: {len(new)}", flush=True)
    describe(g, new)
    for j, sad in enumerate(new):
        label = f"m{j+1}"
        try:
            k, ess, _stderr = train_saddle(g, label, sad, g["SEED"] + 20 + j, ckpt)
        except Exception as exc:
            print(f"  skip {label}: {exc}", flush=True)
            continue
        pieces.append(k)
        esses.append(ess)
        kept_labels.append(label)
        running = pieces[0] + 2.0 * sum(pieces[1:], 0j)
        err = abs(running - k_lat) / abs(k_lat)
        print(f"  after {label}: err={100*err:.3f}%  K={running:+.6f}", flush=True)
        if err <= GATE:
            print("WITHIN 1% after adding two-mode thimbles.", flush=True)
            (OUT / "SUCCESS.txt").write_text(
                f"full_err={err}\nK={running}\nlattice={k_lat}\n",
                encoding="utf-8",
            )
            return
    final = pieces[0] + 2.0 * sum(pieces[1:], 0j)
    final_err = abs(final - k_lat) / abs(k_lat)
    print(f"\nStill outside 1%. Final err={100*final_err:.3f}%  K={final:+.6f}", flush=True)
    report(pieces, esses, kept_labels, k_lat)


if __name__ == "__main__":
    main()
