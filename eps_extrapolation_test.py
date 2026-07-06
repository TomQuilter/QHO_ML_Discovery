"""
EPSILON EXTRAPOLATION:
1. Diagonalise the REAL Hermitian H with eigh (stable, N=800)
2. For large epsilon values (0.5, 0.3, 0.2, ...) compute K(eps)
   using the perturbative damping: exp(-eps/2 * <n|x^2|n> * T)
   At large eps, only ~20-50 states contribute -> well-converged
3. Extrapolate K(eps) to eps=0.01

This avoids the non-Hermitian instability entirely.
"""
import numpy as np
from scipy.linalg import eigh
import time
import sys

omega = 1.0; mass = 1.0; hbar = 1.0; T_time = 1.0

def mehler(eps):
    w = np.sqrt(omega**2 - 1j * eps)
    wT = w * T_time
    return np.sqrt(mass * w / (2j * np.pi * hbar * np.sin(wT)))

def build_H(N, lam):
    n = np.arange(N)
    x_mat = np.zeros((N, N))
    s = np.sqrt(n[1:] / (2.0 * omega))
    x_mat[np.arange(N - 1), np.arange(1, N)] = s
    x_mat[np.arange(1, N), np.arange(N - 1)] = s
    x2 = x_mat @ x_mat
    x4 = x2 @ x2
    H = np.diag(omega * (n + 0.5)) + 0.25 * lam * x4
    return H, x2

def ho_wf_at_zero(N):
    phi = np.zeros(N)
    phi[0] = (omega / np.pi) ** 0.25
    if N > 1:
        phi[1] = 0.0  # sqrt(2*omega) * 0 * phi[0]
    for n in range(2, N):
        phi[n] = (np.sqrt(2.0 * omega / n) * 0.0 * phi[n - 1]
                  - np.sqrt((n - 1.0) / n) * phi[n - 2])
    return phi

def K_at_eps(evals, psi0_sq, x2_expect, eps):
    """Spectral sum with perturbative ieps damping."""
    damping = np.exp(-0.5 * eps * x2_expect * T_time)
    phases = np.exp(-1j * evals * T_time)
    return np.sum(psi0_sq * phases * damping)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 70, flush=True)
    print("EPSILON EXTRAPOLATION METHOD", flush=True)
    print("=" * 70, flush=True)

    N = 800
    eps_list = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05]
    eps_target = 0.01

    # --- Validate at lambda=0: compare K(eps) with exact Mehler(eps) ---
    print(f"\n--- Lambda=0 validation: K(eps) vs Mehler(eps), N={N} ---", flush=True)
    H, x2_mat = build_H(N, 0.0)
    evals, evecs = eigh(H)
    phi0 = ho_wf_at_zero(N)
    psi0 = evecs.T @ phi0
    psi0_sq = psi0 ** 2
    x2_expect = np.diag(evecs.T @ x2_mat @ evecs)

    print(f"{'eps':>8s}  {'Re(K)':>14s}  {'Im(K)':>14s}  {'|K|':>12s}  "
          f"{'Re(Meh)':>14s}  {'Im(Meh)':>14s}  {'err%':>10s}  {'N_eff':>6s}", flush=True)
    for eps in eps_list:
        K = K_at_eps(evals, psi0_sq, x2_expect, eps)
        Km = mehler(eps)
        err = abs(K - Km) / abs(Km) * 100
        n_eff = np.sum(np.exp(-0.5 * eps * x2_expect * T_time) > 0.01)
        print(f"{eps:8.3f}  {K.real:+14.8f}  {K.imag:+14.8f}  {abs(K):12.8f}  "
              f"{Km.real:+14.8f}  {Km.imag:+14.8f}  {err:9.4f}%  {n_eff:6d}", flush=True)

    # Extrapolation test at lambda=0
    K_vals_0 = np.array([K_at_eps(evals, psi0_sq, x2_expect, e) for e in eps_list])

    for deg in [1, 2, 3]:
        cr = np.polyfit(eps_list, K_vals_0.real, deg)
        ci = np.polyfit(eps_list, K_vals_0.imag, deg)
        K_ext = np.polyval(cr, eps_target) + 1j * np.polyval(ci, eps_target)
        Km_target = mehler(eps_target)
        err = abs(K_ext - Km_target) / abs(Km_target) * 100
        print(f"  Extrapolated to eps={eps_target} (deg {deg}): "
              f"Re={K_ext.real:+.8f} Im={K_ext.imag:+.8f} |K|={abs(K_ext):.8f} "
              f"err={err:.4f}%", flush=True)

    # --- Now test at lambda=0.1 ---
    print(f"\n--- Lambda=0.1, N={N} ---", flush=True)
    H, x2_mat = build_H(N, 0.1)
    evals, evecs = eigh(H)
    phi0 = ho_wf_at_zero(N)
    psi0 = evecs.T @ phi0
    psi0_sq = psi0 ** 2
    x2_expect = np.diag(evecs.T @ x2_mat @ evecs)

    print(f"{'eps':>8s}  {'Re(K)':>14s}  {'Im(K)':>14s}  {'|K|':>12s}  {'N_eff':>6s}", flush=True)
    K_vals_01 = []
    for eps in eps_list:
        K = K_at_eps(evals, psi0_sq, x2_expect, eps)
        n_eff = np.sum(np.exp(-0.5 * eps * x2_expect * T_time) > 0.01)
        print(f"{eps:8.3f}  {K.real:+14.8f}  {K.imag:+14.8f}  {abs(K):12.8f}  {n_eff:6d}", flush=True)
        K_vals_01.append(K)

    K_vals_01 = np.array(K_vals_01)
    for deg in [1, 2, 3]:
        cr = np.polyfit(eps_list, K_vals_01.real, deg)
        ci = np.polyfit(eps_list, K_vals_01.imag, deg)
        K_ext = np.polyval(cr, eps_target) + 1j * np.polyval(ci, eps_target)
        print(f"  Extrapolated to eps={eps_target} (deg {deg}): "
              f"Re={K_ext.real:+.8f} Im={K_ext.imag:+.8f} |K|={abs(K_ext):.8f}", flush=True)
    print(f"  ML reference (eps=0.01): Re=+0.30804542 Im=-0.30680188 |K|=0.43476", flush=True)

    # --- Lambda=1.0 ---
    print(f"\n--- Lambda=1.0, N={N} ---", flush=True)
    H, x2_mat = build_H(N, 1.0)
    evals, evecs = eigh(H)
    psi0 = evecs.T @ phi0
    psi0_sq = psi0 ** 2
    x2_expect = np.diag(evecs.T @ x2_mat @ evecs)

    print(f"{'eps':>8s}  {'Re(K)':>14s}  {'Im(K)':>14s}  {'|K|':>12s}  {'N_eff':>6s}", flush=True)
    K_vals_10 = []
    for eps in eps_list:
        K = K_at_eps(evals, psi0_sq, x2_expect, eps)
        n_eff = np.sum(np.exp(-0.5 * eps * x2_expect * T_time) > 0.01)
        print(f"{eps:8.3f}  {K.real:+14.8f}  {K.imag:+14.8f}  {abs(K):12.8f}  {n_eff:6d}", flush=True)
        K_vals_10.append(K)

    K_vals_10 = np.array(K_vals_10)
    for deg in [1, 2, 3]:
        cr = np.polyfit(eps_list, K_vals_10.real, deg)
        ci = np.polyfit(eps_list, K_vals_10.imag, deg)
        K_ext = np.polyval(cr, eps_target) + 1j * np.polyval(ci, eps_target)
        print(f"  Extrapolated to eps={eps_target} (deg {deg}): "
              f"Re={K_ext.real:+.8f} Im={K_ext.imag:+.8f} |K|={abs(K_ext):.8f}", flush=True)
    print(f"  ML reference (eps=0.01): Re=+0.31527619 Im=-0.29817093 |K|=0.43394", flush=True)

    # --- Lambda=8.0 ---
    print(f"\n--- Lambda=8.0, N={N} ---", flush=True)
    H, x2_mat = build_H(N, 8.0)
    evals, evecs = eigh(H)
    psi0 = evecs.T @ phi0
    psi0_sq = psi0 ** 2
    x2_expect = np.diag(evecs.T @ x2_mat @ evecs)

    print(f"{'eps':>8s}  {'Re(K)':>14s}  {'Im(K)':>14s}  {'|K|':>12s}  {'N_eff':>6s}", flush=True)
    K_vals_80 = []
    for eps in eps_list:
        K = K_at_eps(evals, psi0_sq, x2_expect, eps)
        n_eff = np.sum(np.exp(-0.5 * eps * x2_expect * T_time) > 0.01)
        print(f"{eps:8.3f}  {K.real:+14.8f}  {K.imag:+14.8f}  {abs(K):12.8f}  {n_eff:6d}", flush=True)
        K_vals_80.append(K)

    K_vals_80 = np.array(K_vals_80)
    for deg in [1, 2, 3]:
        cr = np.polyfit(eps_list, K_vals_80.real, deg)
        ci = np.polyfit(eps_list, K_vals_80.imag, deg)
        K_ext = np.polyval(cr, eps_target) + 1j * np.polyval(ci, eps_target)
        print(f"  Extrapolated to eps={eps_target} (deg {deg}): "
              f"Re={K_ext.real:+.8f} Im={K_ext.imag:+.8f} |K|={abs(K_ext):.8f}", flush=True)
    print(f"  ML reference (eps=0.01): Re=+0.33545447 Im=-0.23218318 |K|=0.40797", flush=True)

    print("\nDone.", flush=True)
