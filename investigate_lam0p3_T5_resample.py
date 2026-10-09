"""Two attempts at the noisy real bounce, lambda=0.3, T=5.

The dual-flow test already kept b3 and b4 and rejected b1 and b2.
Neither attempt below changes that sum. Both only replace the b3 integral.

1. A third tanh layer inside the same low-rank Jacobian.
2. The original two-layer flow at rank 48.

Trivial and b4 stay at the values already sampled. The lattice is read
only after both replacement integrals are stored.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import investigate_lambda_up_0p3_to_1 as inv

ROOT = Path(__file__).resolve().parent
inv.NB_PATH = ROOT / "Complex_Time_QHO_Anharmonic_MU_lambda0p3_T4_morethimbles.ipynb"
OUT = ROOT / "runs" / "lambda0p3_T5_resample"
OUT.mkdir(parents=True, exist_ok=True)
PREV = json.loads((ROOT / "runs" / "lambda0p3_T5_realcycle" / "results.json").read_text(encoding="utf-8"))
SADDLES = np.load(ROOT / "runs" / "lambda0p3_T5_realcycle" / "saddles.npz")


def configure(g, rank):
    g["LAMBDA"] = 0.3
    g["T"] = 5.0
    g["A_C"] = (5.0 - 1j * g["ETA"]) / g["N"]
    g["RANK"] = rank
    g["EPOCHS"] = 2000
    g["BATCH"] = 1024
    g["LEARNING_RATE"] = 1.0e-3
    g["N_MC"] = 400_000
    g["LOG_EVERY"] = 500


class DeepFlow:
    """Placeholder. The real class is built once the notebook namespace exists."""


def make_deep_flow(g, base):
    rank_of = g

    class DeepFlow(base):
        def __init__(self, saddle, tangent, seed):
            super().__init__(saddle, tangent, seed)
            width = rank_of["RANK"]
            qmat, _ = torch.linalg.qr(torch.randn(width, width, dtype=torch.float64, device=g["DEVICE"]))
            self.W3 = torch.nn.Parameter(qmat)

        def forward(self, z, return_cache=False):
            z = z.to(dtype=torch.float64, device=g["DEVICE"])
            a1 = torch.tanh(z @ self.W1.T)
            a2 = torch.tanh(a1 @ self.W2.T)
            a3 = torch.tanh(a2 @ self.W3.T)
            self._a3 = a3
            x = z.to(torch.complex128) @ self.A.T + self.saddle + a3.to(torch.complex128) @ self.C().T
            return (x, (a1, a2)) if return_cache else x

        def full_jacobian(self, a1, a2):
            a3 = self._a3
            d1 = 1.0 - a1**2
            d2 = 1.0 - a2**2
            d3 = 1.0 - a3**2
            h1 = d1.unsqueeze(-1) * self.W1.unsqueeze(0)
            h2 = d2.unsqueeze(-1) * torch.matmul(self.W2.unsqueeze(0), h1)
            h3 = d3.unsqueeze(-1) * torch.matmul(self.W3.unsqueeze(0), h2)
            correction = torch.matmul(self.C().unsqueeze(0), h3.to(torch.complex128))
            return self.A.unsqueeze(0) + correction

        def determinant_parts(self, a1, a2):
            a3 = self._a3
            width = rank_of["RANK"]
            sign_a, logabs_a = torch.linalg.slogdet(self.A)
            k_mat = self.W1.to(torch.complex128) @ torch.linalg.solve(self.A, self.C())
            d1 = (1.0 - a1**2).to(torch.complex128)
            d2 = (1.0 - a2**2).to(torch.complex128)
            d3 = (1.0 - a3**2).to(torch.complex128)
            m1 = d1.unsqueeze(-1) * k_mat.unsqueeze(0)
            m2 = torch.matmul(self.W2.to(torch.complex128).unsqueeze(0), m1)
            m3 = d2.unsqueeze(-1) * m2
            m4 = torch.matmul(self.W3.to(torch.complex128).unsqueeze(0), m3)
            correction = (
                torch.eye(width, dtype=torch.complex128, device=g["DEVICE"]).unsqueeze(0)
                + d3.unsqueeze(-1) * m4
            )
            sign_c, logabs_c = torch.linalg.slogdet(correction)
            return logabs_a + logabs_c, sign_a * sign_c

    return DeepFlow


def jacobian_ok(g, saddle, seed):
    tangent = g["tangent_map"](g["hessian_np"](saddle))
    model = g["SaddleFlow"](saddle, tangent, seed)
    z = torch.randn(8, g["D"], dtype=torch.float64, device=g["DEVICE"])
    _x, (a1, a2) = model(z, return_cache=True)
    logabs, sign = model.determinant_parts(a1, a2)
    exact = torch.linalg.det(model.full_jacobian(a1, a2))
    pred = torch.exp(logabs) * sign
    err = torch.max(torch.abs(pred - exact) / torch.clamp(torch.abs(exact), min=1e-30))
    print(f"  Jacobian check max relative error {float(err):.3e}", flush=True)
    return float(err) < 1e-6


def sample(g, label, saddle, seed, store):
    if label in store:
        row = store[label]
        print(f"    reuse {label}: ESS/N={row['ESS']:.5f}", flush=True)
        return
    print("\n" + "=" * 78, flush=True)
    print(f"*** STARTING ML TRAINING: {label} at lambda=0.3, T=5 ***", flush=True)
    print("=" * 78, flush=True)
    model, _hist, seconds = g["train_thimble"](label, saddle, seed)
    draws = []
    for i in range(4):
        est, stderr, ess, _sc, _bl = g["estimate_thimble"](model, seed + 31 * i)
        draws.append({"K_re": est.real, "K_im": est.imag, "ESS": ess, "stderr": stderr})
        print(f"  draw {i+1}: K={est:+.6f}  ESS/N={ess:.5f}  stderr={stderr:.2e}", flush=True)
    k = sum(complex(d["K_re"], d["K_im"]) for d in draws) / len(draws)
    ess = float(np.median([d["ESS"] for d in draws]))
    print(f"K_{label}={k:+.6f}  median ESS/N={ess:.5f}  {seconds:.0f}s", flush=True)
    store[label] = {
        "K_re": k.real,
        "K_im": k.imag,
        "ESS": ess,
        "seconds": seconds,
        "draws": draws,
    }
    (OUT / "results.json").write_text(json.dumps(store, indent=1), encoding="utf-8")


def main():
    g = inv.load_notebook_namespace()
    original = g["SaddleFlow"]
    saddle = SADDLES["b3"]
    store_path = OUT / "results.json"
    store = json.loads(store_path.read_text(encoding="utf-8")) if store_path.exists() else {}

    configure(g, 32)
    g["SaddleFlow"] = make_deep_flow(g, original)
    print("Technique 1: three-layer flow, rank 32", flush=True)
    if jacobian_ok(g, saddle, g["SEED"] + 101):
        sample(g, "b3_deep", saddle, g["SEED"] + 101, store)
    else:
        print("  Jacobian check failed. Three-layer flow not trained.", flush=True)

    configure(g, 48)
    g["SaddleFlow"] = original
    print("\nTechnique 2: two-layer flow, rank 48", flush=True)
    sample(g, "b3_rank48", saddle, g["SEED"] + 201, store)

    k0 = complex(PREV["trivial"]["K_re"], PREV["trivial"]["K_im"])
    k4 = complex(PREV["b4"]["K_re"], PREV["b4"]["K_im"])
    print("\nFixed sums, then the lattice", flush=True)
    rows = {}
    for label in ("b3_deep", "b3_rank48"):
        if label not in store:
            continue
        kb = complex(store[label]["K_re"], store[label]["K_im"])
        total = k0 + 2 * kb + 2 * k4
        rows[label] = [total.real, total.imag, store[label]["ESS"]]
        print(f"  {label}: K={total:+.6f}  b3 ESS/N={store[label]['ESS']:.5f}", flush=True)
    k_lat = inv.lattice_K_fft(0.3, (5.0 - 1j * 0.05) / 16)
    print(f"lattice {k_lat:+.6f}", flush=True)
    for label, (re, im, _ess) in rows.items():
        err = abs(complex(re, im) - k_lat) / abs(k_lat)
        print(f"  {label} error {100 * err:.3f}%", flush=True)
        rows[label].append(err)
    (OUT / "summary.json").write_text(json.dumps({
        "lattice": [k_lat.real, k_lat.imag],
        "sums": rows,
    }, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
