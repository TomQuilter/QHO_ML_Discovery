"""Lambda-up investigation using the copied 5-thimble notebook.

Architecture (SaddleFlow, train_thimble, estimate_thimble, saddle search)
is loaded from Complex_Time_QHO_Anharmonic_MU_lambda_up_5to9thimble.ipynb
and is not modified.

For each lambda in 0.3, 0.5, 0.8, 1.0:
  probe T=2.6 with 5 thimbles (trivial + two bounce pairs);
  if the error vs the same-N lattice FFT is above 1%, add pairs
  up to 7 and then 9 thimbles;
  then scan T=2.0..4.0 step 0.2 with the thimble count that worked,
  or with the best count tried if none reached 1%.

Lattice targets are the N=16 FFT, same convention as exact_check_Time_Loop.
"""
from __future__ import annotations

import csv
import json
import math
import os
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
NB_PATH = ROOT / "Complex_Time_QHO_Anharmonic_MU_lambda_up_5to9thimble.ipynb"
OUT = ROOT / "runs" / "lambda_up_0p3_to_1"
OUT.mkdir(parents=True, exist_ok=True)

LAMBDAS = [0.3, 0.5, 0.8, 1.0]
DT = 0.2
T_LIST = [round(2.0 + i * DT, 2) for i in range(int(round((4.0 - 2.0) / DT)) + 1)]
PROBE_T = 2.6
# 2, 3, 4 bounce pairs -> 5, 7, 9 thimbles (each bounce is a ± pair).
PAIR_LADDER = [2, 3, 4]
MAX_HARMONIC = 6
GATE = 0.01
N_LATTICE = 16
ETA = 0.05
MASS = 1.0

FIELDNAMES = [
    "lambda",
    "T",
    "n_pairs_requested",
    "n_pairs_found",
    "n_thimbles",
    "K_trivial_re",
    "K_trivial_im",
    "K_ML_re",
    "K_ML_im",
    "K_lattice_re",
    "K_lattice_im",
    "complex_rel_err_vs_lattice",
    "trivial_ESS_over_N",
    "family_ESS",
    "family_K_re",
    "family_K_im",
    "status",
    "used_for_scan",
]


def load_notebook_namespace():
    import nbformat

    nb = nbformat.read(NB_PATH, as_version=4)
    # Definition cells only: setup, action, saddles, flow, training.
    wanted = {1, 3, 5, 7, 11}
    ns = {"__name__": "__notebook__"}
    for i, cell in enumerate(nb.cells):
        if i not in wanted or cell.cell_type != "code":
            continue
        source = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        lines = []
        for line in source.splitlines():
            if line.startswith("%"):
                continue
            if line.startswith("from __future__"):
                continue
            lines.append(line)
        exec(compile("\n".join(lines), f"{NB_PATH.name}:cell{i}", "exec"), ns)
    return ns


def fftconvolve(a, b):
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    full = np.fft.ifft(np.fft.fft(a, nfft) * np.fft.fft(b, nfft))[:n]
    return full[len(b) - 1 : len(a)]


def lattice_K_fft(lam, ac, L=10.0, M=2**15):
    x = np.linspace(-L, L, M)
    h = x[1] - x[0]
    pref = np.sqrt(MASS / (2 * np.pi * 1j * ac))
    u = np.linspace(-2 * L, 2 * L, 2 * M - 1)
    link = pref * np.exp(1j * MASS * u**2 / (2 * ac))
    f = pref * np.exp(1j * MASS * x**2 / (2 * ac))
    w = np.exp(-1j * ac * (0.5 * x**2 + 0.25 * lam * x**4))
    for _ in range(N_LATTICE - 1):
        f = h * fftconvolve(link, w * f)
    return complex(np.interp(0.0, x, f.real) + 1j * np.interp(0.0, x, f.imag))


def lattice_table(ns_unused=None):
    path = OUT / "lattice_T2to4_dt0.2.csv"
    table = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                table[(float(row["lambda"]), round(float(row["T"]), 2))] = complex(
                    float(row["K_re"]), float(row["K_im"])
                )
    missing = [(lam, T) for lam in LAMBDAS for T in T_LIST if (lam, T) not in table]
    if missing:
        print(f"Computing {len(missing)} lattice FFT points", flush=True)
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["lambda", "T", "K_re", "K_im", "abs_K"])
            if path.stat().st_size == 0:
                writer.writeheader()
            for lam, T in missing:
                ac = (T - 1j * ETA) / N_LATTICE
                K = lattice_K_fft(lam, ac)
                table[(lam, T)] = K
                writer.writerow({
                    "lambda": lam,
                    "T": T,
                    "K_re": K.real,
                    "K_im": K.imag,
                    "abs_K": abs(K),
                })
                handle.flush()
                print(f"  lattice lambda={lam} T={T:.1f}  K={K:+.6f}", flush=True)
    return table


def _same_saddle(a, b, tol):
    return np.max(np.abs(a - b)) < tol or np.max(np.abs(a + b)) < tol


def collect_families(g, max_pairs):
    """Distinct bounce saddles, ordered as in the 5-thimble notebook.

    Family 1 is the lowest-|S| harmonic-1 bounce. Later families are the
    remaining saddles of lowest Im S, searched through higher harmonics.
    """
    found = {}
    for harmonic in range(1, MAX_HARMONIC + 1):
        found[harmonic] = g["find_bounce_harmonic"](harmonic)
        print(f"    harmonic {harmonic}: {len(found[harmonic])} saddles", flush=True)

    chosen = []
    if found.get(1):
        chosen.append(min(found[1], key=lambda x: abs(g["action_np"](x))))

    pool = []
    for cands in found.values():
        for x in cands:
            if np.max(np.abs(x)) < 1e-3:
                continue
            if any(_same_saddle(x, old, 1e-5) for old in pool):
                continue
            pool.append(x)

    rest = [x for x in pool if not any(_same_saddle(x, c, 1e-3) for c in chosen)]
    while len(chosen) < max_pairs and rest:
        nxt = min(rest, key=lambda x: g["action_np"](x).imag)
        chosen.append(nxt)
        rest = [x for x in rest if not _same_saddle(x, nxt, 1e-5)]
    return chosen[:max_pairs]


def load_ckpt():
    path = OUT / "thimble_checkpoints.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_ckpt(ckpt):
    path = OUT / "thimble_checkpoints.json"
    path.write_text(json.dumps(ckpt), encoding="utf-8")


def train_one(g, label, saddle, seed, lam, T, ckpt):
    key = f"{lam:.2f}|{T:.2f}|{label}"
    if key in ckpt:
        row = ckpt[key]
        print(f"    reuse {label}: K={complex(row['K_re'], row['K_im']):+.6f}  ESS/N={row['ESS']:.4f}", flush=True)
        return complex(row["K_re"], row["K_im"]), float(row["ESS"]), float(row["stderr"])

    print("\n" + "=" * 78, flush=True)
    print(f"*** STARTING ML TRAINING: {label} thimble at lambda={lam}, T={T} ***", flush=True)
    print("=" * 78, flush=True)
    model, _history, seconds = g["train_thimble"](label, saddle, seed)
    estimate, stderr, ess, _scatter, _blocks = g["estimate_thimble"](model, seed)
    print(f"K_{label}={estimate:+.8f}  ESS/N={ess:.6f}  stderr={stderr:.3e}  {seconds:.0f}s", flush=True)
    ckpt[key] = {
        "K_re": estimate.real,
        "K_im": estimate.imag,
        "ESS": ess,
        "stderr": stderr,
        "seconds": seconds,
    }
    save_ckpt(ckpt)
    return estimate, ess, stderr


def load_results():
    path = OUT / "results.csv"
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["lambda"] = float(row["lambda"])
                row["T"] = float(row["T"])
                row["n_pairs_requested"] = int(row["n_pairs_requested"])
                row["n_pairs_found"] = int(row["n_pairs_found"])
                row["n_thimbles"] = int(row["n_thimbles"])
                row["complex_rel_err_vs_lattice"] = float(row["complex_rel_err_vs_lattice"])
                row["used_for_scan"] = row["used_for_scan"] == "True"
                rows.append(row)
    return rows


def write_results(rows):
    path = OUT / "results.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(g, lam, T, n_pairs, lattice, ckpt, family_cache):
    g["LAMBDA"] = float(lam)
    g["T"] = float(T)
    g["A_C"] = (T - 1j * g["ETA"]) / g["N"]
    cache_key = (float(lam), float(T))
    if cache_key not in family_cache:
        print(f"\nSearching saddles  lambda={lam}  T={T}", flush=True)
        family_cache[cache_key] = collect_families(g, max(PAIR_LADDER))
        tag = f"{lam:.1f}".replace(".", "p") + f"_T{T:.1f}".replace(".", "p")
        np.savez(
            OUT / f"saddles_lam{tag}.npz",
            **{f"b{i+1}": sad for i, sad in enumerate(family_cache[cache_key])},
        )
    families = family_cache[cache_key][:n_pairs]
    print(
        f"\nlambda={lam} T={T}: using {len(families)} bounce pair(s) "
        f"of {len(family_cache[cache_key])} found "
        f"({1 + 2 * len(families)} thimbles)",
        flush=True,
    )

    K0, ess0, _ = train_one(g, "trivial", np.zeros(g["D"], dtype=np.complex128), g["SEED"], lam, T, ckpt)
    pieces = []
    esses = []
    for i, saddle in enumerate(families):
        label = f"b{i+1}"
        Kb, ess, _ = train_one(g, label, saddle, g["SEED"] + 1 + i, lam, T, ckpt)
        pieces.append(Kb)
        esses.append(ess)
    K = K0 + 2.0 * sum(pieces, 0j)
    Klat = lattice[(float(lam), round(float(T), 2))]
    err = abs(K - Klat) / abs(Klat)
    status = "PASS" if err <= GATE else "FAIL"
    print(
        f"RESULT lambda={lam} T={T} thimbles={1 + 2 * len(families)}  "
        f"K={K:+.6f}  lattice={Klat:+.6f}  err={100 * err:.3f}%  {status}",
        flush=True,
    )
    return {
        "lambda": lam,
        "T": T,
        "n_pairs_requested": n_pairs,
        "n_pairs_found": len(families),
        "n_thimbles": 1 + 2 * len(families),
        "K_trivial_re": K0.real,
        "K_trivial_im": K0.imag,
        "K_ML_re": K.real,
        "K_ML_im": K.imag,
        "K_lattice_re": Klat.real,
        "K_lattice_im": Klat.imag,
        "complex_rel_err_vs_lattice": err,
        "trivial_ESS_over_N": ess0,
        "family_ESS": ";".join(f"{e:.6g}" for e in esses),
        "family_K_re": ";".join(f"{p.real:.8g}" for p in pieces),
        "family_K_im": ";".join(f"{p.imag:.8g}" for p in pieces),
        "status": status,
        "used_for_scan": False,
    }


def already(rows, lam, T, n_pairs):
    for row in rows:
        if (
            abs(row["lambda"] - lam) < 1e-9
            and abs(row["T"] - T) < 1e-9
            and row["n_pairs_requested"] == n_pairs
        ):
            return row
    return None


def choose_pairs(rows, lam):
    probes = [
        r for r in rows
        if abs(r["lambda"] - lam) < 1e-9 and abs(r["T"] - PROBE_T) < 1e-9
    ]
    if not probes:
        return None
    good = [r for r in probes if r["complex_rel_err_vs_lattice"] <= GATE]
    best = min(good or probes, key=lambda r: (r["complex_rel_err_vs_lattice"], r["n_pairs_requested"]))
    return best["n_pairs_requested"]


def investigate():
    print("Loading 5-thimble notebook functions", flush=True)
    g = load_notebook_namespace()
    lattice = lattice_table()
    ckpt = load_ckpt()
    rows = load_results()
    family_cache = {}

    for lam in LAMBDAS:
        print("\n" + "#" * 78, flush=True)
        print(f"LAMBDA {lam}: probe T={PROBE_T}", flush=True)
        print("#" * 78, flush=True)
        for n_pairs in PAIR_LADDER:
            row = already(rows, lam, PROBE_T, n_pairs)
            if row is None:
                try:
                    row = evaluate(g, lam, PROBE_T, n_pairs, lattice, ckpt, family_cache)
                except Exception:
                    traceback.print_exc()
                    print(f"Probe failed lambda={lam} pairs={n_pairs}", flush=True)
                    break
                rows.append(row)
                write_results(rows)
            else:
                print(
                    f"  already have lambda={lam} T={PROBE_T} pairs={n_pairs} "
                    f"err={100 * row['complex_rel_err_vs_lattice']:.3f}%",
                    flush=True,
                )
            if row["complex_rel_err_vs_lattice"] <= GATE:
                print(f"Within 1% at {row['n_thimbles']} thimbles. Stopping the ladder.", flush=True)
                break
            if row["n_pairs_found"] < n_pairs:
                print("No further distinct bounce. Stopping the ladder.", flush=True)
                break

        n_scan = choose_pairs(rows, lam)
        if n_scan is None:
            print(f"No successful probe at lambda={lam}; scan skipped", flush=True)
            continue
        print(f"Full T scan at lambda={lam} uses {n_scan} bounce pair(s)", flush=True)
        for T in T_LIST:
            row = already(rows, lam, T, n_scan)
            if row is None:
                try:
                    row = evaluate(g, lam, T, n_scan, lattice, ckpt, family_cache)
                except Exception:
                    traceback.print_exc()
                    print(f"Scan point failed lambda={lam} T={T}", flush=True)
                    continue
                rows.append(row)
                write_results(rows)
            row["used_for_scan"] = True
        # Mark only this lambda's scan rows. Earlier probe rows at other pair counts stay false
        # except the chosen T=2.6 row, which is also a scan point.
        for row in rows:
            if abs(row["lambda"] - lam) < 1e-9:
                row["used_for_scan"] = (
                    abs(row["T"] - PROBE_T) > 1e-9 and row["n_pairs_requested"] == n_scan
                ) or (
                    abs(row["T"] - PROBE_T) < 1e-9 and row["n_pairs_requested"] == n_scan
                )
        write_results(rows)
        plot_progress(rows, lattice)

    plot_progress(rows, lattice)
    print(f"\nDone. Results in {OUT}", flush=True)


def plot_progress(rows, lattice):
    scan = [r for r in rows if r["used_for_scan"]]
    if not scan:
        return
    lambdas = []
    for r in scan:
        if r["lambda"] not in lambdas:
            lambdas.append(r["lambda"])

    n = len(lambdas)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 7), sharex=True, squeeze=False)
    for col, lam in enumerate(lambdas):
        pts = sorted([r for r in scan if abs(r["lambda"] - lam) < 1e-9], key=lambda r: r["T"])
        T = np.array([r["T"] for r in pts])
        ml = np.array([complex(float(r["K_ML_re"]), float(r["K_ML_im"])) for r in pts])
        lat = np.array([complex(float(r["K_lattice_re"]), float(r["K_lattice_im"])) for r in pts])
        sane = np.array([abs(m) < 50.0 * max(abs(l), 1e-8) for m, l in zip(ml, lat)])
        n_th = pts[0]["n_thimbles"] if pts else "?"
        for row_i, part in enumerate(("real", "imag")):
            ax = axes[row_i, col]
            y_lat = getattr(lat, part)
            y_ml = getattr(ml, part)
            ax.plot(T, y_lat, "k*-", ms=7, lw=1.4, label="lattice FFT")
            ax.plot(T[sane], y_ml[sane], "C1s-", ms=6, lw=1.4, label=f"ML {n_th} thimbles")
            ax.axvline(np.pi, color="grey", ls=":", alpha=0.7)
            ax.grid(alpha=0.3)
            if row_i == 0:
                ax.set_title(rf"$\lambda={lam:g}$, {n_th} thimbles")
                ax.set_ylabel("Re K")
            else:
                ax.set_ylabel("Im K")
                ax.set_xlabel("T")
            if col == 0:
                ax.legend(fontsize=8)
    fig.suptitle(r"ML vs lattice, $N=16$, $\eta=0.05$, $T=2\to 4$")
    fig.tight_layout()
    fig.savefig(OUT / "summary_ReIm_vs_lattice.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lam in lambdas:
        pts = sorted([r for r in scan if abs(r["lambda"] - lam) < 1e-9], key=lambda r: r["T"])
        T = [r["T"] for r in pts]
        err = [100.0 * min(r["complex_rel_err_vs_lattice"], 10.0) for r in pts]
        ax.plot(T, err, "o-", label=rf"$\lambda={lam:g}$")
    ax.axhline(1.0, color="grey", ls="--", label="1%")
    ax.axvline(np.pi, color="grey", ls=":", alpha=0.7)
    ax.set_xlabel("T")
    ax.set_ylabel("error vs lattice (%)  [capped at 1000 for the axis]")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 2000)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    ax.set_title("Relative error of the multi-thimble ML sum")
    fig.tight_layout()
    fig.savefig(OUT / "summary_error_vs_T.png", dpi=140)
    plt.close(fig)

    probes = [r for r in rows if abs(r["T"] - PROBE_T) < 1e-9]
    if probes:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for lam in LAMBDAS:
            pts = sorted(
                [r for r in probes if abs(r["lambda"] - lam) < 1e-9],
                key=lambda r: r["n_thimbles"],
            )
            if not pts:
                continue
            ax.plot(
                [r["n_thimbles"] for r in pts],
                [100.0 * min(r["complex_rel_err_vs_lattice"], 10.0) for r in pts],
                "o-",
                label=rf"$\lambda={lam:g}$",
            )
        ax.axhline(1.0, color="grey", ls="--")
        ax.set_xlabel("thimbles in the sum")
        ax.set_ylabel("T=2.6 error vs lattice (%)")
        ax.set_yscale("log")
        ax.set_ylim(0.01, 2000)
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        ax.set_title(r"Probe at $T=2.6$: error versus number of thimbles")
        fig.tight_layout()
        fig.savefig(OUT / "probe_T2p6_error_vs_thimbles.png", dpi=140)
        plt.close(fig)
    print(f"Updated plots in {OUT}", flush=True)


if __name__ == "__main__":
    investigate()
