"""Overnight three-saddle scan, plus a five-saddle check near lambda = 0.5.

Training is the method in
Complex_Time_QHO_Anharmonic_MU_lambda0_0p01_T2to4.ipynb
(rank 16, 1000 epochs, batch 512, N_MC 80000, seed 52).
That notebook produced the lambda = 0.01 and 0.1 curves.

Three saddles means the saddle at zero plus one opposite pair:
K = K_trivial + 2 K_bounce.
If the bounce search finds nothing, only the zero saddle is used and
n_saddles is 1.

The second bounce is the search in
Complex_Time_QHO_Anharmonic_MU_Time_Loop_5_thimble _Inv.ipynb.
It is trained only near where five saddles previously helped:
lambda 0.40, 0.45, 0.50, 0.55, 0.60 at T = 1.80 to 2.40,
and lambda 0.30, 0.40, 0.45, 0.50, 0.55, 0.60 at T = 3.40 to 3.80,
step 0.05. There K_5 = K_3 + 2 K_bounce2.

Press overnight_dt0p05.bat, or from this folder:

    C:\\Users\\tomqu\\Documents\\MLandPython\\envs\\helios_env\\python.exe -u overnight_dt0p05.py

Run the same command again to continue a stopped night. Finished (lambda, T)
rows are skipped.
"""
from __future__ import annotations

import csv
import json
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import nbformat
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
NB_THREE = ROOT / "Complex_Time_QHO_Anharmonic_MU_lambda0_0p01_T2to4.ipynb"
NB_FIVE = ROOT / "Complex_Time_QHO_Anharmonic_MU_Time_Loop_5_thimble _Inv.ipynb"
OUT = ROOT / "runs" / "overnight_dt0p05_T0to4_N16"
OUT.mkdir(parents=True, exist_ok=True)

DT = 0.05
T_MAIN = [round(i * DT, 2) for i in range(int(round(4.0 / DT)) + 1)]  # 0, 0.05, ..., 4
LAMBDA_MAIN = [0.01, 0.1, 0.3, 0.5]
# Where five saddles improved the three-saddle answer: near T=2 at lambda=0.5,
# and near T=3.6 at lambda=0.3 and 0.5. Neighbour lambdas are included.
FIVE_EARLY = {
    "lambdas": [0.4, 0.45, 0.5, 0.55, 0.6],
    "times": [round(1.8 + i * DT, 2) for i in range(int(round(0.6 / DT)) + 1)],
}
FIVE_LATE = {
    "lambdas": [0.3, 0.4, 0.45, 0.5, 0.55, 0.6],
    "times": [round(3.4 + i * DT, 2) for i in range(int(round(0.4 / DT)) + 1)],
}

THREE_CSV = OUT / "three_saddles.csv"
FIVE_CSV = OUT / "five_saddles.csv"
LATTICE_CSV = OUT / "lattice_cache.csv"

THREE_FIELDS = [
    "lambda", "T", "N", "eta", "n_saddles", "has_bounce", "bounce_source",
    "K_trivial_re", "K_trivial_im", "K_bounce_re", "K_bounce_im",
    "K_ML_re", "K_ML_im", "K_lattice_re", "K_lattice_im",
    "complex_rel_err_vs_lattice", "one_thimble_err_vs_lattice",
    "trivial_ESS_over_N", "bounce_ESS_over_N",
    "weight_trivial", "weight_bounce", "status",
]
FIVE_FIELDS = [
    "lambda", "T", "n_saddles_three", "n_saddles_five", "has_bounce2",
    "err3", "err5", "improved",
    "K3_re", "K3_im", "K5_re", "K5_im",
    "K_bounce2_re", "K_bounce2_im", "K_lattice_re", "K_lattice_im",
    "bounce2_ESS_over_N", "weight_bounce2", "status",
]


def tkey(value):
    return round(float(value) + 1e-12, 2)


def load_method():
    """Execute the notebook definition cells. Training code is not rewritten."""
    nb = nbformat.read(NB_THREE, as_version=4)
    ns = {"__name__": "overnight_method"}
    for index in (1, 3, 5, 7, 11):
        source = nb.cells[index].source
        if not isinstance(source, str):
            source = "".join(source)
        lines = []
        for line in source.splitlines():
            if line.startswith("%"):
                continue
            if line.startswith("stamp ="):
                break
            lines.append(line)
        exec(compile("\n".join(lines), f"{NB_THREE.name}:cell{index}", "exec"), ns)

    five = nbformat.read(NB_FIVE, as_version=4)
    copied = []
    for cell in five.cells:
        source = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        lines = source.splitlines()
        for name in ("find_bounce_harmonic", "second_bounce"):
            starts = [i for i, line in enumerate(lines) if line.startswith(f"def {name}")]
            if not starts:
                continue
            start = starts[0]
            end = start + 1
            while end < len(lines) and not lines[end].startswith("def "):
                end += 1
            copied.append("\n".join(lines[start:end]))
    if [block.split("(", 1)[0] for block in copied] != ["def find_bounce_harmonic", "def second_bounce"]:
        raise RuntimeError("Could not copy find_bounce_harmonic and second_bounce")
    exec(compile("\n\n".join(copied), f"{NB_FIVE.name}:bounce_search", "exec"), ns)
    return ns


def read_done(path, fields):
    done = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "OK":
                done[(float(row["lambda"]), tkey(row["T"]))] = row
    return done


def append_row(path, fields, row):
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def load_existing_lattice(g):
    table = {}
    for lam, csv_path in g["EXACT_CSVS"].items():
        with Path(csv_path).open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                table[(float(lam), tkey(row["T"]))] = complex(
                    float(row["lattice_K_re"]), float(row["lattice_K_im"])
                )
    # lambda = 0.5 file only covers T = 0 to 2. Keep those points.
    lam05 = ROOT / "Re__Complex_Time_Code_" / "py_scripts_extracted" / "exact_check_Time_Loop_lambda0.5_dt0.01.csv"
    with lam05.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            table[(0.5, tkey(row["T"]))] = complex(float(row["lattice_K_re"]), float(row["lattice_K_im"]))
    if LATTICE_CSV.exists():
        with LATTICE_CSV.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                table[(float(row["lambda"]), tkey(row["T"]))] = complex(float(row["K_re"]), float(row["K_im"]))
    return table


def lattice_K_fft(lam, ac, n_sites, mass, L=10.0, M=2**15):
    """Same N=16 lattice FFT as exact_check_Time_Loop.py. Used only for missing (lambda, T)."""
    x = np.linspace(-L, L, M)
    h = x[1] - x[0]
    pref = np.sqrt(mass / (2 * np.pi * 1j * ac))
    u = np.linspace(-2 * L, 2 * L, 2 * M - 1)
    link = pref * np.exp(1j * mass * u**2 / (2 * ac))
    n = len(link) + len(x) - 1
    nfft = 1 << (n - 1).bit_length()
    link_fft = np.fft.fft(link, nfft)

    def convolve(values):
        full = np.fft.ifft(link_fft * np.fft.fft(values, nfft))[:n]
        return full[len(x) - 1 : len(link)]

    f = pref * np.exp(1j * mass * x**2 / (2 * ac))
    w = np.exp(-1j * ac * (0.5 * x**2 + 0.25 * lam * x**4))
    for _ in range(n_sites - 1):
        f = h * convolve(w * f)
    return complex(np.interp(0.0, x, f.real) + 1j * np.interp(0.0, x, f.imag))


def ensure_lattice(g, jobs):
    table = load_existing_lattice(g)
    missing = [(lam, t) for lam, t in jobs if (lam, t) not in table]
    if not missing:
        print(f"Lattice targets already stored for all {len(jobs)} points.", flush=True)
        return table
    print(f"Computing {len(missing)} missing lattice FFT points (not machine learning).", flush=True)
    new_file = not LATTICE_CSV.exists()
    with LATTICE_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["lambda", "T", "K_re", "K_im"])
        if new_file:
            writer.writeheader()
        for lam, t in missing:
            ac = (t - 1j * g["ETA"]) / g["N"]
            value = lattice_K_fft(lam, ac, g["N"], g["MASS"])
            table[(lam, t)] = value
            writer.writerow({"lambda": lam, "T": t, "K_re": value.real, "K_im": value.imag})
            handle.flush()
            print(f"  lattice lambda={lam} T={t:.2f}", flush=True)
    return table


def run_dir_for(lam, t):
    name = f"lam{lam:g}_T{t:.2f}".replace(".", "p")
    path = OUT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(g, run_dir, label, trained, lam, t, n_saddles):
    path = run_dir / f"{label}_model.pt"
    torch.save(
        {
            "state_dict": trained["model"].state_dict(),
            "saddle": trained["saddle"],
            "lambda": lam,
            "T": t,
            "N": g["N"],
            "rank": g["RANK"],
            "seed": trained["seed"],
            "label": label,
            "n_saddles": n_saddles,
        },
        path,
    )
    return str(path.relative_to(ROOT))


def train_saddle(g, label, saddle, seed):
    model, history, seconds = g["train_thimble"](label, saddle, seed)
    estimate, stderr, ess, block_scatter, blocks = g["estimate_thimble"](model, seed)
    print(f"K_{label}={estimate:+.6f}  ESS/N={ess:.4f}  ({seconds:.0f}s)", flush=True)
    return {
        "model": model,
        "history": history,
        "seconds": seconds,
        "K": estimate,
        "stderr": stderr,
        "ESS": ess,
        "block_scatter": block_scatter,
        "blocks": blocks,
        "saddle": saddle,
        "seed": seed,
    }


def set_physics(g, lam, t):
    g["LAMBDA"] = float(lam)
    g["T"] = float(t)
    g["A_C"] = (t - 1j * g["ETA"]) / g["N"]


def find_primary_bounce(g):
    try:
        return g["first_bounce"](), "first_bounce"
    except RuntimeError as exc:
        print(f"first_bounce missed: {exc}", flush=True)
    pool = []
    for harmonic in (1, 2, 3):
        pool.extend(g["find_bounce_harmonic"](harmonic))
    distinct = []
    for saddle in pool:
        if any(np.max(np.abs(saddle - old)) < 1e-5 or np.max(np.abs(saddle + old)) < 1e-5 for old in distinct):
            continue
        distinct.append(saddle)
    if not distinct:
        return None, "none"
    chosen = min(distinct, key=lambda saddle: g["action_np"](saddle).imag)
    return chosen, "harmonic_fallback"


def three_saddle_point(g, lam, t, lattice):
    set_physics(g, lam, t)
    run_dir = run_dir_for(lam, t)
    trivial = np.zeros(g["D"], dtype=np.complex128)
    bounce, source = find_primary_bounce(g)
    if bounce is not None:
        g["validate_saddles_and_jacobian"](trivial, bounce)
    trained_trivial = train_saddle(g, "trivial", trivial, g["SEED"])
    n_saddles = 1
    bounce_k = 0j
    bounce_ess = float("nan")
    weight_bounce = ""
    if bounce is not None:
        trained_bounce = train_saddle(g, "bounce", bounce, g["SEED"] + 1)
        n_saddles = 3
        bounce_k = trained_bounce["K"]
        bounce_ess = trained_bounce["ESS"]
        weight_bounce = save_model(g, run_dir, "bounce", trained_bounce, lam, t, n_saddles)
        np.savez(run_dir / "bounce_saddle.npz", x=bounce, action=g["action_np"](bounce))
    k_ml = trained_trivial["K"] + (2.0 * bounce_k if bounce is not None else 0j)
    err = abs(k_ml - lattice) / abs(lattice)
    one = abs(trained_trivial["K"] - lattice) / abs(lattice)
    weight_trivial = save_model(g, run_dir, "trivial", trained_trivial, lam, t, n_saddles)
    row = {
        "lambda": lam,
        "T": t,
        "N": g["N"],
        "eta": g["ETA"],
        "n_saddles": n_saddles,
        "has_bounce": bounce is not None,
        "bounce_source": source,
        "K_trivial_re": trained_trivial["K"].real,
        "K_trivial_im": trained_trivial["K"].imag,
        "K_bounce_re": bounce_k.real,
        "K_bounce_im": bounce_k.imag,
        "K_ML_re": k_ml.real,
        "K_ML_im": k_ml.imag,
        "K_lattice_re": lattice.real,
        "K_lattice_im": lattice.imag,
        "complex_rel_err_vs_lattice": err,
        "one_thimble_err_vs_lattice": one,
        "trivial_ESS_over_N": trained_trivial["ESS"],
        "bounce_ESS_over_N": bounce_ess,
        "weight_trivial": weight_trivial,
        "weight_bounce": weight_bounce,
        "status": "OK",
    }
    (run_dir / "results.json").write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    print(
        f"THREE lambda={lam} T={t:.2f}  n_saddles={n_saddles}  "
        f"err={100 * err:.2f}%  source={source}",
        flush=True,
    )
    return row, bounce


def five_saddle_point(g, lam, t, lattice, three_row, primary_bounce):
    set_physics(g, lam, t)
    run_dir = run_dir_for(lam, t)
    k3 = complex(float(three_row["K_ML_re"]), float(three_row["K_ML_im"]))
    n3 = int(three_row["n_saddles"])
    bounce2 = None
    if primary_bounce is not None:
        bounce2 = g["second_bounce"](primary_bounce)
    if bounce2 is None:
        row = {
            "lambda": lam, "T": t, "n_saddles_three": n3, "n_saddles_five": n3,
            "has_bounce2": False, "err3": three_row["complex_rel_err_vs_lattice"],
            "err5": three_row["complex_rel_err_vs_lattice"], "improved": False,
            "K3_re": k3.real, "K3_im": k3.imag, "K5_re": k3.real, "K5_im": k3.imag,
            "K_bounce2_re": 0.0, "K_bounce2_im": 0.0,
            "K_lattice_re": lattice.real, "K_lattice_im": lattice.imag,
            "bounce2_ESS_over_N": float("nan"), "weight_bounce2": "", "status": "OK",
        }
        print(f"FIVE  lambda={lam} T={t:.2f}  no second bounce", flush=True)
        return row
    g["validate_saddles_and_jacobian"](np.zeros(g["D"], dtype=np.complex128), bounce2)
    trained = train_saddle(g, "bounce2", bounce2, g["SEED"] + 2)
    k5 = k3 + 2.0 * trained["K"]
    err3 = abs(k3 - lattice) / abs(lattice)
    err5 = abs(k5 - lattice) / abs(lattice)
    weight = save_model(g, run_dir, "bounce2", trained, lam, t, n3 + 2)
    np.savez(run_dir / "bounce2_saddle.npz", x=bounce2, action=g["action_np"](bounce2))
    row = {
        "lambda": lam, "T": t, "n_saddles_three": n3, "n_saddles_five": n3 + 2,
        "has_bounce2": True, "err3": err3, "err5": err5, "improved": err5 < err3,
        "K3_re": k3.real, "K3_im": k3.imag, "K5_re": k5.real, "K5_im": k5.imag,
        "K_bounce2_re": trained["K"].real, "K_bounce2_im": trained["K"].imag,
        "K_lattice_re": lattice.real, "K_lattice_im": lattice.imag,
        "bounce2_ESS_over_N": trained["ESS"], "weight_bounce2": weight, "status": "OK",
    }
    print(
        f"FIVE  lambda={lam} T={t:.2f}  err3={100 * err3:.2f}%  err5={100 * err5:.2f}%  "
        f"improved={row['improved']}",
        flush=True,
    )
    return row


def five_jobs():
    jobs = []
    for block in (FIVE_EARLY, FIVE_LATE):
        for lam in block["lambdas"]:
            for t in block["times"]:
                jobs.append((lam, t))
    # Preserve order, drop duplicates.
    seen = set()
    unique = []
    for job in jobs:
        if job not in seen:
            seen.add(job)
            unique.append(job)
    return unique


def main():
    started = time.time()
    print("Loading the existing thimble trainer.", flush=True)
    g = load_method()
    print(f"device={g['DEVICE']}  rank={g['RANK']}  epochs={g['EPOCHS']}  batch={g['BATCH']}", flush=True)

    three_jobs = [(lam, t) for lam in LAMBDA_MAIN for t in T_MAIN]
    extra_five = five_jobs()
    all_jobs = list(dict.fromkeys(three_jobs + extra_five))
    if "--check" in sys.argv:
        table = load_existing_lattice(g)
        for sample in ((0.01, 0.0), (0.01, 0.05), (0.1, 4.0), (0.5, 2.0)):
            print(f"lattice {sample} present={sample in table}", flush=True)
        print(f"three-saddle points={len(three_jobs)}  five-saddle points={len(extra_five)}", flush=True)
        print("check ok; machine learning was not started", flush=True)
        return

    lattice = ensure_lattice(g, all_jobs)
    done3 = read_done(THREE_CSV, THREE_FIELDS)
    done5 = read_done(FIVE_CSV, FIVE_FIELDS)
    bounces = {}

    print(f"\nThree saddles: {len(three_jobs)} points, {len(done3)} already finished.", flush=True)
    for index, (lam, t) in enumerate(three_jobs, start=1):
        if (lam, t) in done3:
            continue
        print(f"\n[{index}/{len(three_jobs)}] three saddles  lambda={lam}  T={t:.2f}", flush=True)
        try:
            row, bounce = three_saddle_point(g, lam, t, lattice[(lam, t)])
            append_row(THREE_CSV, THREE_FIELDS, row)
            done3[(lam, t)] = row
            bounces[(lam, t)] = bounce
        except Exception:
            traceback.print_exc()
            print(f"FAILED three saddles lambda={lam} T={t:.2f}; continuing.", flush=True)

    print(f"\nFive saddles: {len(extra_five)} points, {len(done5)} already finished.", flush=True)
    for index, (lam, t) in enumerate(extra_five, start=1):
        if (lam, t) in done5:
            continue
        print(f"\n[{index}/{len(extra_five)}] five saddles  lambda={lam}  T={t:.2f}", flush=True)
        try:
            if (lam, t) not in done3:
                row3, bounce = three_saddle_point(g, lam, t, lattice[(lam, t)])
                append_row(THREE_CSV, THREE_FIELDS, row3)
                done3[(lam, t)] = row3
                bounces[(lam, t)] = bounce
            bounce = bounces.get((lam, t))
            if bounce is None and done3[(lam, t)]["has_bounce"] in (True, "True"):
                saddle_file = run_dir_for(lam, t) / "bounce_saddle.npz"
                if saddle_file.exists():
                    bounce = np.load(saddle_file)["x"]
            row5 = five_saddle_point(g, lam, t, lattice[(lam, t)], done3[(lam, t)], bounce)
            append_row(FIVE_CSV, FIVE_FIELDS, row5)
            done5[(lam, t)] = row5
        except Exception:
            traceback.print_exc()
            print(f"FAILED five saddles lambda={lam} T={t:.2f}; continuing.", flush=True)

    hours = (time.time() - started) / 3600
    print(f"\nDone in {hours:.2f} h", flush=True)
    print(f"Three-saddle table: {THREE_CSV}", flush=True)
    print(f"Five-saddle table:  {FIVE_CSV}", flush=True)
    print(f"Weights:            {OUT}", flush=True)


if __name__ == "__main__":
    main()
