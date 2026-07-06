"""
EXACT epsilon extrapolation:
For each eps value, build the FULL complex Hamiltonian (with ieps)
in the even-parity HO basis, diagonalise with eig, compute K.
At large eps, only ~20 states contribute -> small matrix -> eig is stable.
Then extrapolate K(eps) to eps=0.01.
"""
import numpy as np
import scipy.linalg as la
import time
import sys

omega = 1.0; mass = 1.0; hbar = 1.0; T_time = 1.0

def build_operators_even(Nev):
    """x^2 and x^4 in even-parity HO sector."""
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
    """psi_{2k}(0) for k=0,...,Nev-1."""
    v = np.zeros(Nev)
    v[0] = np.pi ** (-0.25)
    for k in range(1, Nev):
        v[k] = -np.sqrt((2 * k - 1.0) / (2 * k)) * v[k - 1]
    return v

def make_H_even(lam, eps, Nev, x2, x4):
    n = 2.0 * np.arange(Nev)
    H = np.diag((n + 0.5).astype(complex))
    H -= 0.5j * eps * x2[:Nev, :Nev]
    H += (lam / 4.0) * x4[:Nev, :Nev]
    return H

def K_spectral_exact(lam, eps, Nev, x2, x4, ve, floor=1e-12):
    """Exact non-Hermitian spectral sum."""
    H = make_H_even(lam, eps, Nev, x2, x4)
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

def mehler(eps):
    w = np.sqrt(omega ** 2 - 1j * eps)
    wT = w * T_time
    return np.sqrt(mass * w / (2j * np.pi * hbar * np.sin(wT)))

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 70, flush=True)
    print("EXACT EPSILON EXTRAPOLATION", flush=True)
    print("Use full non-Hermitian eig at each eps, then extrapolate.", flush=True)
    print("=" * 70, flush=True)

    # For large eps, use small basis (enough to capture all contributing states)
    # For smaller eps, use larger basis
    eps_configs = [
        (2.0,   50),
        (1.5,   60),
        (1.0,   80),
        (0.8,  100),
        (0.6,  120),
        (0.5,  150),
        (0.4,  180),
        (0.3,  250),
        (0.25, 300),
        (0.2,  400),
    ]

    # Pre-build the largest operators needed
    MAX_NEV = max(n for _, n in eps_configs)
    print(f"Building operators (Nev_max={MAX_NEV})...", end=" ", flush=True)
    x2, x4 = build_operators_even(MAX_NEV)
    ve = build_psi_even(MAX_NEV)
    print("done", flush=True)

    # --- Validate at lambda=0 ---
    print(f"\n--- Lambda=0 validation ---", flush=True)
    print(f"{'eps':>6s}  {'Nev':>5s}  {'M':>4s}  {'Re(K)':>14s}  {'Im(K)':>14s}  "
          f"{'|K|':>12s}  {'Mehler err%':>12s}", flush=True)

    eps_vals_0 = []
    K_vals_0 = []
    for eps, Nev in eps_configs:
        K, M = K_spectral_exact(0.0, eps, Nev, x2, x4, ve)
        Km = mehler(eps)
        err = abs(K - Km) / abs(Km) * 100
        print(f"{eps:6.2f}  {Nev:5d}  {M:4d}  {K.real:+14.8f}  {K.imag:+14.8f}  "
              f"{abs(K):12.8f}  {err:11.4f}%", flush=True)
        eps_vals_0.append(eps)
        K_vals_0.append(K)

    eps_arr = np.array(eps_vals_0)
    K_arr_0 = np.array(K_vals_0)

    Km_target = mehler(0.01)
    print(f"\nExtrapolation to eps=0.01:", flush=True)
    print(f"  Mehler target: Re={Km_target.real:+.8f} Im={Km_target.imag:+.8f} "
          f"|K|={abs(Km_target):.8f}", flush=True)
    for deg in [1, 2, 3, 4]:
        cr = np.polyfit(eps_arr, K_arr_0.real, deg)
        ci = np.polyfit(eps_arr, K_arr_0.imag, deg)
        K_ext = np.polyval(cr, 0.01) + 1j * np.polyval(ci, 0.01)
        err = abs(K_ext - Km_target) / abs(Km_target) * 100
        print(f"  deg {deg}: Re={K_ext.real:+.8f} Im={K_ext.imag:+.8f} "
              f"|K|={abs(K_ext):.8f}  err={err:.4f}%", flush=True)

    # --- Lambda=0.1 ---
    for lam_test in [0.1, 1.0, 8.0]:
        print(f"\n--- Lambda={lam_test} ---", flush=True)
        print(f"{'eps':>6s}  {'Nev':>5s}  {'M':>4s}  {'Re(K)':>14s}  {'Im(K)':>14s}  "
              f"{'|K|':>12s}", flush=True)

        K_vals = []
        for eps, Nev in eps_configs:
            K, M = K_spectral_exact(lam_test, eps, Nev, x2, x4, ve)
            print(f"{eps:6.2f}  {Nev:5d}  {M:4d}  {K.real:+14.8f}  {K.imag:+14.8f}  "
                  f"{abs(K):12.8f}", flush=True)
            K_vals.append(K)

        K_arr = np.array(K_vals)
        print(f"\n  Extrapolation to eps=0.01:", flush=True)
        for deg in [1, 2, 3, 4]:
            cr = np.polyfit(eps_arr, K_arr.real, deg)
            ci = np.polyfit(eps_arr, K_arr.imag, deg)
            K_ext = np.polyval(cr, 0.01) + 1j * np.polyval(ci, 0.01)
            print(f"  deg {deg}: Re={K_ext.real:+.8f} Im={K_ext.imag:+.8f} "
                  f"|K|={abs(K_ext):.8f}", flush=True)

    print(f"\nML references (eps=0.01):", flush=True)
    print(f"  lam=0.1: Re=+0.30804542 Im=-0.30680188 |K|=0.43476", flush=True)
    print(f"  lam=1.0: Re=+0.31527619 Im=-0.29817093 |K|=0.43394", flush=True)
    print(f"  lam=8.0: Re=+0.33545447 Im=-0.23218318 |K|=0.40797", flush=True)
    print("Done.", flush=True)
