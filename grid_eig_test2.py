"""
Grid eigendecomposition v2: diagonalise the HERMITIAN Hamiltonian (no i*eps)
with eigh (stable), then apply i*eps damping perturbatively to eigenvalues.

For the HO: E_n -> E_n - i*(eps/2)*<n|x^2|n>
First-order perturbation theory in epsilon gives the imaginary shift.
This avoids the non-Hermitian eig instability entirely.
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

def propagator_grid_hermitian(lam, N, L):
    """
    1. Build real Hermitian H (no ieps) on position grid with FD kinetic energy
    2. Diagonalise with eigh (guaranteed stable)
    3. Compute <n|x^2|n> for each eigenstate
    4. Apply complex energy shift: E_n -> E_n - i*(eps/2)*<n|x^2|n>
    5. Spectral sum with shifted energies
    """
    dx = 2 * L / (N + 1)
    x = -L + dx * np.arange(1, N + 1)

    # Ensure a grid point sits exactly at x=0
    j0 = np.argmin(np.abs(x))

    # Real Hermitian Hamiltonian
    V_real = 0.5 * mass * omega**2 * x**2 + (lam / 4.0) * x**4
    t_diag = hbar**2 / (mass * dx**2)
    t_off = -hbar**2 / (2 * mass * dx**2)

    H = np.diag(V_real + t_diag) + np.diag(np.full(N - 1, t_off), 1) + np.diag(np.full(N - 1, t_off), -1)

    t0 = time.time()
    vals, vecs = la.eigh(H)
    t_eig = time.time() - t0

    # vecs are real orthonormal: vecs[:,n] is the n-th eigenvector
    # Physical wavefunction: psi_n(x_j) = vecs[j,n] / sqrt(dx)
    # <n|x^2|n> = sum_j x_j^2 * |psi_n(x_j)|^2 * dx = sum_j x_j^2 * vecs[j,n]^2
    x2_expect = np.einsum("j,jn->n", x**2, vecs**2)

    # Complex energy: E_n -> E_n - i*(eps/2)*<n|x^2|n>
    E_complex = vals - 0.5j * epsilon * x2_expect

    # psi_n(0) = vecs[j0, n] / sqrt(dx)
    psi0 = vecs[j0, :] / np.sqrt(dx)

    # K(0,0;T) = sum_n psi_n(0)^2 * exp(-i E_n T / hbar)
    phases = np.exp(-1j * E_complex * T_time / hbar)
    K = np.sum(psi0**2 * phases)

    return K, t_eig, x[j0]

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    K_m = mehler()
    print("=" * 70, flush=True)
    print("GRID EIGENDECOMP v2: Hermitian + perturbative i*eps", flush=True)
    print(f"Mehler: Re={K_m.real:.8f}  Im={K_m.imag:.8f}  |K|={abs(K_m):.8f}", flush=True)
    print("=" * 70, flush=True)

    L = 15.0

    for lam in [0.0, 0.1]:
        print(f"\n--- lambda = {lam}, L = {L} ---", flush=True)
        print(f"{'N':>6s}  {'Re(K)':>14s}  {'Im(K)':>14s}  {'|K|':>12s}  "
              f"{'err%':>10s}  {'eig_time':>8s}  {'x0':>8s}", flush=True)
        prev_absK = None
        for N in [512, 1024, 2048, 4096]:
            t_total = time.time()
            K, t_eig, x0 = propagator_grid_hermitian(lam, N, L)
            dt = time.time() - t_total

            absK = abs(K)
            if lam == 0.0:
                err = abs(K - K_m) / abs(K_m) * 100
                err_str = f"{err:.4f}%"
            else:
                err_str = "---"

            delta_str = ""
            if prev_absK is not None:
                delta_str = f"  d|K|={absK - prev_absK:+.6f}"
            prev_absK = absK

            print(f"{N:6d}  {K.real:+14.8f}  {K.imag:+14.8f}  {absK:12.8f}  "
                  f"{err_str:>10s}  {t_eig:7.1f}s  {x0:+8.5f}{delta_str}", flush=True)

    print("\nML reference for lam=0.1: Re=0.308045, Im=-0.306802, |K|=0.434569", flush=True)
    print("Done.", flush=True)
