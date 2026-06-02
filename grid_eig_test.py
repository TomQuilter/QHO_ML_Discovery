"""
Quick test: position-space grid eigendecomposition with finite-difference KE.
Tests lambda=0 (must match Mehler) and lambda=0.1 (the real question).
"""
import numpy as np
import scipy.linalg as la
import time
import sys

omega = 1.0; mass = 1.0; hbar = 1.0; epsilon = 0.01; T_time = 1.0

def mehler():
    w = np.sqrt(omega**2 - 1j * epsilon)
    wT = w * T_time
    return np.sqrt(mass * w / (2j * np.pi * hbar * np.sin(wT)))

def propagator_grid_eig(lam, N, L):
    """
    Build H on a position grid with 2nd-order finite-difference KE.
    Diagonalise, compute spectral sum K(0,0;T).
    """
    dx = 2 * L / (N + 1)
    x = -L + dx * np.arange(1, N + 1)  # interior points, Dirichlet BC

    j0 = N // 2  # grid point closest to x=0

    # Potential (diagonal)
    V = 0.5 * mass * (omega**2 - 1j * epsilon) * x**2 + (lam / 4.0) * x**4

    # Kinetic energy: -hbar^2/(2m) * d^2/dx^2, 2nd-order FD
    # T_{jj} = hbar^2/(m*dx^2), T_{j,j+1} = T_{j+1,j} = -hbar^2/(2m*dx^2)
    t_diag = hbar**2 / (mass * dx**2)
    t_off = -hbar**2 / (2 * mass * dx**2)

    H = np.diag(V + t_diag) + np.diag(np.full(N - 1, t_off), 1) + np.diag(np.full(N - 1, t_off), -1)

    t0 = time.time()
    vals, vecs = la.eig(H)
    t_eig = time.time() - t0

    # Spectral sum: K(0,0;T) = sum_n psi_n(0)^2 exp(-i E_n T) / dx
    # psi_n(0) = vecs[j0, n] (unnormalised grid eigenvector component)
    # Grid eigenvectors satisfy: sum_j |psi_n(x_j)|^2 * dx = 1 if properly normalised
    # la.eig returns column-normalised vectors (L2 norm = 1 in vector sense)
    # To get position-space normalisation: psi_phys = vecs / sqrt(dx)
    # But for bilinear (non-Hermitian) c-norm: c_n = sum_j vecs[j,n]^2 (no conjugate)

    cnorm = np.einsum("in,in->n", vecs, vecs)
    psi0_sq = vecs[j0, :] ** 2  # psi_n(0)^2

    phases = np.exp(-1j * vals * T_time / hbar)

    K = np.sum(psi0_sq / cnorm * phases) / dx

    return K, t_eig, x[j0]

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    K_m = mehler()
    print("=" * 70, flush=True)
    print("GRID EIGENDECOMPOSITION TEST (finite-difference KE)", flush=True)
    print(f"Mehler: Re={K_m.real:.8f}  Im={K_m.imag:.8f}  |K|={abs(K_m):.8f}", flush=True)
    print("=" * 70, flush=True)

    L = 15.0
    for lam in [0.0, 0.1]:
        print(f"\n--- lambda = {lam}, L = {L} ---", flush=True)
        print(f"{'N':>6s}  {'Re(K)':>14s}  {'Im(K)':>14s}  {'|K|':>12s}  "
              f"{'err%':>10s}  {'eig_time':>8s}  {'x0':>6s}", flush=True)
        for N in [512, 1024, 2048, 4096]:
            t_total = time.time()
            K, t_eig, x0 = propagator_grid_eig(lam, N, L)
            dt = time.time() - t_total

            if lam == 0.0:
                err = abs(K - K_m) / abs(K_m) * 100
                err_str = f"{err:.4f}%"
            else:
                err_str = "---"

            print(f"{N:6d}  {K.real:+14.8f}  {K.imag:+14.8f}  {abs(K):12.8f}  "
                  f"{err_str:>10s}  {t_eig:7.1f}s  {x0:+.3f}", flush=True)

    print("\nDone.", flush=True)
