"""Resample the noisy one-hump at lambda=0.5, T=3.6.

The extra small-action thimble (p6/p7) moved the five-thimble error from
14.4% to 7.4%. The leftover is about the size of the noise on the
original one-hump, whose ESS/N is 0.0035. Same network, more samples.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import investigate_lam0p5_T3p6_morethimbles as base

ROOT = Path(__file__).resolve().parent
OLD = json.loads((ROOT / "runs" / "lambda_up_0p3_to_1" / "thimble_checkpoints.json").read_text(encoding="utf-8"))
SADDLE = np.load(ROOT / "runs" / "lambda_up_0p3_to_1" / "saddles_lam0p5_T3p6.npz")["b1"]


def main():
    g = base.load_notebook_namespace()
    g["LAMBDA"] = 0.5
    g["T"] = 3.6
    g["A_C"] = (3.6 - 1j * g["ETA"]) / g["N"]
    g["N_MC"] = 400_000
    k_lat = base.lattice_K_fft(0.5, (3.6 - 1j * 0.05) / 16)
    k0 = complex(OLD["0.50|3.60|trivial"]["K_re"], OLD["0.50|3.60|trivial"]["K_im"])
    kb2 = complex(OLD["0.50|3.60|b2"]["K_re"], OLD["0.50|3.60|b2"]["K_im"])
    # Two independent estimates of the same new saddle.
    k_new = 0.5 * (
        complex(0.01074889, -0.00275587) + complex(0.01052796, -0.00262160)
    )
    print("\n" + "=" * 78, flush=True)
    print("*** STARTING ML TRAINING: b1 thimble at lambda=0.5, T=3.6 (resample) ***", flush=True)
    print("=" * 78, flush=True)
    model, _hist, seconds = g["train_thimble"]("b1", SADDLE, g["SEED"] + 1)
    print(f"trained in {seconds:.0f}s; drawing 6 estimates of 400k", flush=True)
    estimates = []
    for i in range(6):
        est, stderr, ess, _sc, _bl = g["estimate_thimble"](model, g["SEED"] + 1 + 17 * i)
        estimates.append(est)
        print(f"  sample {i+1}: K={est:+.6f}  ESS/N={ess:.5f}  stderr={stderr:.3e}", flush=True)
    kb1 = sum(estimates) / len(estimates)
    scatter = float(np.std(np.array(estimates, dtype=np.complex128)))
    k5 = k0 + 2.0 * kb1 + 2.0 * kb2
    k7 = k5 + 2.0 * k_new
    e5 = abs(k5 - k_lat) / abs(k_lat)
    e7 = abs(k7 - k_lat) / abs(k_lat)
    print(f"mean b1={kb1:+.6f}  scatter={scatter:.4f}", flush=True)
    print(f"lattice          {k_lat:+.6f}", flush=True)
    print(f"5 thimbles       {k5:+.6f}  err={100*e5:.3f}%", flush=True)
    print(f"5 + new thimble  {k7:+.6f}  err={100*e7:.3f}%", flush=True)
    out = ROOT / "runs" / "lambda0p5_T3p6_morethimbles" / "resample_b1.json"
    out.write_text(
        json.dumps({
            "kb1_re": kb1.real,
            "kb1_im": kb1.imag,
            "k5_err": e5,
            "k7_err": e7,
            "k7_re": k7.real,
            "k7_im": k7.imag,
            "scatter": scatter,
        }),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
