"""
Exact epsilon extrapolation v2: use much larger basis sizes.
At each eps, use a basis large enough that the result is converged.
Only include converged eps values in the extrapolation.
"""
import numpy as np
import scipy.linalg as la
import time
import sys

omega = 1.0; mass = 1.0; hbar = 1.0; T_time = 1.0

def build_operators_even(Nev):
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

def make_H(lam, eps, Nev, x2, x4):
    n = 2.0 * np.arange(Nev)
    H = np.diag((n + 0.5).astype(complex))
    H -= 0.5j * eps * x2[:Nev, :Nev]
    H += (lam / 4.0) * x4[:Nev, :Nev]
    return H

def K_spectral(lam, eps, Nev, x2, x4, ve, floor=1e-12):
    H = make_H(lam, eps, Nev, x2, x4)
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

def converged_K(lam, eps, x2, x4, ve, Nev_list):
    """Run at multiple basis sizes and return the best converged value."""
    results = []
    for Nev in Nev_list:
        if Nev > x2.shape[0]:
            break
        t0 = time.time()
        K, M = K_spectral(lam, eps, Nev, x2, x4, ve)
        dt = time.time() - t0
        results.append((Nev, K, M, dt))

    if len(results) >= 2:
        K_last = results[-1][1]
        K_prev = results[-2][1]
        rel_change = abs(K_last - K_prev) / max(abs(K_last), 1e-15) * 100
        converged = rel_change < 1.0
    else:
        converged = False
        rel_change = float('inf')

    return results, converged, rel_change

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 70, flush=True)
    print("EXACT EPSILON EXTRAPOLATION v2 (larger basis)", flush=True)
    print("=" * 70, flush=True)

    MAX_NEV = 1000
    print(f"Building operators (Nev_max={MAX_NEV})...", end=" ", flush=True)
    t0 = time.time()
    x2, x4 = build_operators_even(MAX_NEV)
    ve = build_psi_even(MAX_NEV)
    print(f"done ({time.time()-t0:.1f}s)", flush=True)

    eps_list = [2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15]
    Nev_sizes = [100, 200, 300, 400, 600, 800, 1000]

    for lam_test in [0.0, 0.1, 0.5, 1.0, 4.0, 8.0]:
        print(f"\n{'='*70}", flush=True)
        print(f"  Lambda = {lam_test}", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"{'eps':>6s}  {'Nev':>5s}  {'M':>4s}  {'Re(K)':>14s}  {'Im(K)':>14s}  "
              f"{'|K|':>12s}  {'conv?':>6s}  {'d%':>8s}  {'time':>5s}", flush=True)

        good_eps = []
        good_K = []

        for eps in eps_list:
            results, conv, d_pct = converged_K(lam_test, eps, x2, x4, ve, Nev_sizes)
            Nev_best, K_best, M_best, dt_best = results[-1]
            conv_str = "YES" if conv else "no"
            print(f"{eps:6.2f}  {Nev_best:5d}  {M_best:4d}  {K_best.real:+14.8f}  "
                  f"{K_best.imag:+14.8f}  {abs(K_best):12.8f}  {conv_str:>6s}  "
                  f"{d_pct:7.3f}%  {dt_best:4.1f}s", flush=True)

            if conv:
                good_eps.append(eps)
                good_K.append(K_best)

        if len(good_eps) >= 3:
            ea = np.array(good_eps)
            Ka = np.array(good_K)
            print(f"\n  Converged at {len(good_eps)} eps values: {good_eps}", flush=True)
            for deg in [1, 2, 3]:
                if deg >= len(good_eps):
                    break
                cr = np.polyfit(ea, Ka.real, deg)
                ci = np.polyfit(ea, Ka.imag, deg)
                K_ext = np.polyval(cr, 0.01) + 1j * np.polyval(ci, 0.01)
                print(f"  deg {deg}: Re={K_ext.real:+.8f} Im={K_ext.imag:+.8f} "
                      f"|K|={abs(K_ext):.8f}", flush=True)
        else:
            print(f"\n  Only {len(good_eps)} converged eps values -- extrapolation unreliable", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("ML references (eps=0.01):", flush=True)
    refs = [(0.0, 0.307230, -0.307623, 0.434901),
            (0.1, 0.308045, -0.306802, 0.434764),
            (0.5, 0.311708, -0.302996, 0.434705),
            (1.0, 0.315276, -0.298171, 0.433941),
            (4.0, 0.329807, -0.267632, 0.424734),
            (8.0, 0.335454, -0.232183, 0.407969)]
    for lam, re, im, ak in refs:
        print(f"  lam={lam:4.1f}: Re={re:+.6f} Im={im:+.6f} |K|={ak:.6f}", flush=True)
    Km = mehler(0.01)
    print(f"  Mehler(eps=0.01): Re={Km.real:+.8f} Im={Km.imag:+.8f} |K|={abs(Km):.8f}", flush=True)
    print("Done.", flush=True)
