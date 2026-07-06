"""
Convergence study: does the HO-basis spectral sum converge at lambda > 0
when we push N_even to 800, 1200, 1600, 2000?

Tests lambda=0 (must match Mehler) and lambda=0.1 (the real question).
If |K| settles at lambda=0.1, bigger machines will solve this problem.
If it keeps oscillating, the HO basis won't work at epsilon=0.01.
"""
import numpy as np
import scipy.linalg as la
import time
import sys

omega = 1.0; mass = 1.0; hbar = 1.0; epsilon = 0.01; T_time = 1.0

def build_operators(Nev):
    n = 2.0 * np.arange(Nev)
    x2 = np.zeros((Nev, Nev))
    for j in range(Nev):
        nj = 2 * j
        x2[j, j] = nj + 0.5
        if j + 1 < Nev:
            val = np.sqrt((nj + 1) * (nj + 2)) / 2.0
            x2[j, j + 1] = val
            x2[j + 1, j] = val
    x4 = x2 @ x2
    return x2, x4

def build_psi_even(Nev):
    v = np.zeros(Nev)
    v[0] = np.pi ** (-0.25)
    for k in range(1, Nev):
        v[k] = -np.sqrt((2 * k - 1.0) / (2 * k)) * v[k - 1]
    return v

def make_H(lam, Nev, x2, x4):
    n = 2.0 * np.arange(Nev)
    H = np.diag((n + 0.5).astype(complex))
    H -= 0.5j * epsilon * x2[:Nev, :Nev]
    H += (lam / 4.0) * x4[:Nev, :Nev]
    return H

def K_spectral(lam, Nev, x2, x4, ve, floor=1e-10):
    H = make_H(lam, Nev, x2, x4)
    vals, vecs = la.eig(H)
    order = np.argsort(vals.real)
    vals = vals[order]
    vecs = vecs[:, order]
    amp = np.abs(np.exp(-1j * vals * T_time))
    keep = (vals.imag < 0) & (amp > floor)
    vals_k = vals[keep]
    vecs_k = vecs[:, keep]
    cnorm = np.einsum("in,in->n", vecs_k, vecs_k)
    phi0 = vecs_k.T @ ve[:Nev]
    phases = np.exp(-1j * vals_k * T_time)
    K = np.sum(phi0 ** 2 / cnorm * phases)
    return K, int(keep.sum())

def mehler():
    w = np.sqrt(omega ** 2 - 1j * epsilon)
    wT = w * T_time
    return np.sqrt(mass * w / (2j * np.pi * hbar * np.sin(wT)))

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    K_m = mehler()
    print("=" * 70, flush=True)
    print("CONVERGENCE STUDY: HO-basis spectral sum", flush=True)
    print(f"  omega={omega}, mass={mass}, hbar={hbar}, eps={epsilon}, T={T_time}", flush=True)
    print(f"  Mehler (exact HO): Re={K_m.real:.8f}  Im={K_m.imag:.8f}  |K|={abs(K_m):.8f}", flush=True)
    print("=" * 70, flush=True)

    NEV_LIST = [200, 400, 600, 800, 1000, 1200, 1600, 2000]
    TEST_LAMBDAS = [0.0, 0.1, 0.5, 1.0]

    MAX_NEV = max(NEV_LIST)
    print(f"\nBuilding operators up to Nev={MAX_NEV}...", end=" ", flush=True)
    t0 = time.time()
    x2, x4 = build_operators(MAX_NEV)
    ve = build_psi_even(MAX_NEV)
    print(f"done ({time.time() - t0:.1f}s)", flush=True)

    for lam in TEST_LAMBDAS:
        print(f"\n{'='*70}", flush=True)
        print(f"  lambda = {lam}", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"{'Nev':>6s}  {'M_kept':>6s}  {'Re(K)':>14s}  {'Im(K)':>14s}  "
              f"{'|K|':>12s}  {'|K| err%':>10s}  {'time':>7s}", flush=True)
        print("-" * 80, flush=True)

        prev_absK = None
        for Nev in NEV_LIST:
            t1 = time.time()
            K, M = K_spectral(lam, Nev, x2, x4, ve)
            dt = time.time() - t1

            absK = abs(K)
            if lam == 0.0:
                err_str = f"{abs(K - K_m) / abs(K_m) * 100:.4f}%"
            else:
                err_str = "---"

            delta_str = ""
            if prev_absK is not None:
                delta = absK - prev_absK
                delta_str = f"  (d|K|={delta:+.6f})"
            prev_absK = absK

            print(f"{Nev:6d}  {M:6d}  {K.real:+14.8f}  {K.imag:+14.8f}  "
                  f"{absK:12.8f}  {err_str:>10s}  {dt:6.1f}s{delta_str}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("DONE. If |K| is settling for lambda>0, the method converges.", flush=True)
    print("If |K| keeps jumping around, HO basis won't work at eps=0.01.", flush=True)
    print(f"{'='*70}", flush=True)
