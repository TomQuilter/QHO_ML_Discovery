"""
KEY INSIGHT: Drop the i*epsilon from the Hamiltonian entirely.
Use eigh (stable Hermitian diag) and compute the spectral sum
WITHOUT any damping. The sum converges conditionally via oscillation.

This is what spectral_propagator_general_omega.py already does successfully.
Here we test convergence and compare with ML (which uses i*eps=0.01).
"""
import numpy as np
from scipy.linalg import eigh
import time
import sys

omega = 1.0; mass = 1.0; hbar = 1.0; T_time = 1.0; epsilon = 0.01

LAMBDAS = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
           0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]

ML = {0: (0.307230355, -0.307623399), 0.01: (0.307257564, -0.307601922),
      0.05: (0.307613868, -0.307246798), 0.1: (0.308045423, -0.306801876),
      0.15: (0.30853144, -0.306316107), 0.2: (0.309051999, -0.305777055),
      0.3: (0.311482275, -0.304049745), 0.4: (0.311483769, -0.30367083),
      0.5: (0.311707652, -0.302996468), 0.6: (0.312276772, -0.302112874),
      0.8: (0.313795351, -0.300164734), 1.0: (0.315276192, -0.298170929),
      1.5: (0.318673821, -0.293095975), 2.0: (0.321627281, -0.287947061),
      3.0: (0.326363101, -0.277650885), 4.0: (0.329806674, -0.267631712),
      6.0: (0.333883313, -0.248951901), 8.0: (0.335454471, -0.232183184)}

def build_H_ho_basis(N, lam):
    """Hermitian Hamiltonian in HO basis (full, not even-parity restricted)."""
    n = np.arange(N)
    x_mat = np.zeros((N, N))
    s = np.sqrt(n[1:] / (2.0 * omega))
    x_mat[np.arange(N - 1), np.arange(1, N)] = s
    x_mat[np.arange(1, N), np.arange(N - 1)] = s
    x2 = x_mat @ x_mat
    x4 = x2 @ x2
    H = np.diag(omega * (n + 0.5)) + 0.25 * lam * x4
    return H

def ho_wf_at_zero(N):
    """psi_n(0) for n=0,...,N-1 using HO eigenstates."""
    Phi = np.zeros((N,))
    x = np.array([0.0])
    alpha = omega
    phi = np.zeros((N, 1))
    phi[0] = (alpha / np.pi) ** 0.25 * np.exp(-0.5 * alpha * x**2)
    if N > 1:
        phi[1] = np.sqrt(2.0 * alpha) * x * phi[0]
    for n in range(2, N):
        phi[n] = (np.sqrt(2.0 * alpha / n) * x * phi[n - 1]
                  - np.sqrt((n - 1.0) / n) * phi[n - 2])
    return phi[:, 0]

def propagator_hermitian(lam, N):
    """K(0,0;T) using Hermitian HO-basis diag (NO i*epsilon)."""
    H = build_H_ho_basis(N, lam)
    evals, evecs = eigh(H)

    phi0 = ho_wf_at_zero(N)  # HO basis functions at x=0
    psi_at_0 = evecs.T @ phi0  # eigenstates at x=0

    K = np.sum(psi_at_0**2 * np.exp(-1j * evals * T_time))
    return K

def mehler_no_eps():
    """Exact HO propagator WITHOUT i*epsilon."""
    w = omega
    return np.sqrt(w / (2j * np.pi * np.sin(w * T_time)))

def mehler_with_eps():
    """Exact HO propagator WITH i*epsilon."""
    w = np.sqrt(omega**2 - 1j * epsilon)
    return np.sqrt(mass * w / (2j * np.pi * hbar * np.sin(w * T_time)))

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    K_m0 = mehler_no_eps()
    K_me = mehler_with_eps()

    print("=" * 70, flush=True)
    print("HERMITIAN SPECTRAL SUM (no i*epsilon)", flush=True)
    print(f"Mehler (no eps):   Re={K_m0.real:+.8f}  Im={K_m0.imag:+.8f}  |K|={abs(K_m0):.8f}", flush=True)
    print(f"Mehler (eps=0.01): Re={K_me.real:+.8f}  Im={K_me.imag:+.8f}  |K|={abs(K_me):.8f}", flush=True)
    print(f"Difference: {abs(K_m0 - K_me) / abs(K_me) * 100:.4f}%", flush=True)
    print("=" * 70, flush=True)

    # Convergence at lambda=0
    print("\n--- Convergence at lam=0 (vs Mehler no-eps) ---", flush=True)
    print(f"{'N':>6s}  {'Re(K)':>14s}  {'Im(K)':>14s}  {'|K|':>12s}  "
          f"{'err%':>10s}  {'time':>6s}", flush=True)
    for N in [100, 200, 400, 800, 1200, 1600]:
        t0 = time.time()
        K = propagator_hermitian(0.0, N)
        dt = time.time() - t0
        err = abs(K - K_m0) / abs(K_m0) * 100
        print(f"{N:6d}  {K.real:+14.8f}  {K.imag:+14.8f}  {abs(K):12.8f}  "
              f"{err:9.4f}%  {dt:5.1f}s", flush=True)

    # Convergence at lambda=0.1
    print("\n--- Convergence at lam=0.1 ---", flush=True)
    print(f"{'N':>6s}  {'Re(K)':>14s}  {'Im(K)':>14s}  {'|K|':>12s}  "
          f"{'d|K|':>12s}  {'time':>6s}", flush=True)
    prev = None
    for N in [100, 200, 400, 800, 1200, 1600]:
        t0 = time.time()
        K = propagator_hermitian(0.1, N)
        dt = time.time() - t0
        absK = abs(K)
        d = f"{absK - prev:+.8f}" if prev is not None else "---"
        prev = absK
        print(f"{N:6d}  {K.real:+14.8f}  {K.imag:+14.8f}  {absK:12.8f}  "
              f"{d:>12s}  {dt:5.1f}s", flush=True)

    # Convergence at lambda=1.0
    print("\n--- Convergence at lam=1.0 ---", flush=True)
    print(f"{'N':>6s}  {'Re(K)':>14s}  {'Im(K)':>14s}  {'|K|':>12s}  "
          f"{'d|K|':>12s}  {'time':>6s}", flush=True)
    prev = None
    for N in [100, 200, 400, 800, 1200, 1600]:
        t0 = time.time()
        K = propagator_hermitian(1.0, N)
        dt = time.time() - t0
        absK = abs(K)
        d = f"{absK - prev:+.8f}" if prev is not None else "---"
        prev = absK
        print(f"{N:6d}  {K.real:+14.8f}  {K.imag:+14.8f}  {absK:12.8f}  "
              f"{d:>12s}  {dt:5.1f}s", flush=True)

    # Full lambda scan at N=800
    N_USE = 800
    print(f"\n{'='*70}", flush=True)
    print(f"Lambda scan (Hermitian, N={N_USE}) vs ML (eps=0.01)", flush=True)
    print(f"{'lam':>6s}  {'Re_spec':>12s}  {'Re_ML':>12s}  {'err%':>8s}  "
          f"{'Im_spec':>12s}  {'Im_ML':>12s}  {'err%':>8s}  "
          f"{'|K|_spec':>10s}  {'|K|_ML':>10s}  {'err%':>8s}", flush=True)
    print("-" * 110, flush=True)

    for lam in LAMBDAS:
        K = propagator_hermitian(lam, N_USE)
        re_m, im_m = ML[lam]
        ak_m = np.sqrt(re_m**2 + im_m**2)
        er_r = 100 * (K.real - re_m) / abs(re_m)
        er_i = 100 * (K.imag - im_m) / abs(im_m)
        er_a = 100 * (abs(K) - ak_m) / ak_m
        print(f"{lam:6.2f}  {K.real:+12.6f}  {re_m:+12.6f}  {er_r:+7.3f}%  "
              f"{K.imag:+12.6f}  {im_m:+12.6f}  {er_i:+7.3f}%  "
              f"{abs(K):10.6f}  {ak_m:10.6f}  {er_a:+7.3f}%", flush=True)

    print("\nDone.", flush=True)
