"""
Grid eigendecomposition v3:
- ODD grid so x=0 is exactly a grid point
- Hermitian H + perturbative i*eps
- Even-parity sector only (halves matrix size, doubles speed)
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

def propagator_grid_full(lam, N_half, L):
    """
    Full grid with x=0 exactly at center.
    N_total = 2*N_half + 1 interior points: -L, ..., -dx, 0, dx, ..., L
    with Dirichlet BC at +/-(L+dx).
    """
    N = 2 * N_half + 1
    dx = 2 * L / (N + 1)
    x = dx * np.arange(-N_half, N_half + 1)  # symmetric, x[N_half] = 0 exactly

    j0 = N_half  # x=0

    V_real = 0.5 * mass * omega**2 * x**2 + (lam / 4.0) * x**4
    t_diag = hbar**2 / (mass * dx**2)
    t_off = -hbar**2 / (2 * mass * dx**2)

    H = np.diag(V_real + t_diag)
    H += np.diag(np.full(N - 1, t_off), 1)
    H += np.diag(np.full(N - 1, t_off), -1)

    t0 = time.time()
    vals, vecs = la.eigh(H)
    t_eig = time.time() - t0

    x2_expect = np.einsum("j,jn->n", x**2, vecs**2)
    E_complex = vals - 0.5j * epsilon * x2_expect

    psi0 = vecs[j0, :] / np.sqrt(dx)
    phases = np.exp(-1j * E_complex * T_time / hbar)
    K = np.sum(psi0**2 * phases)

    return K, t_eig, x[j0], N

def propagator_grid_even(lam, M, L):
    """
    Even-parity sector on half-grid x = dx/2, 3dx/2, ..., (M-1/2)*dx
    with Neumann BC at x=0 and Dirichlet at x=L.
    This exploits parity symmetry: only even eigenstates contribute to K(0,0;T).

    Actually, for K(0,0;T) we need psi_n(0), and even-parity functions have
    psi'(0)=0. We use a grid that includes x=0 but restricted to x>=0,
    with the mirror BC built in.

    Grid: x_j = j*dx for j=0,1,...,M  (M+1 points, x_0=0, x_M=L)
    Dirichlet at x=L+dx (psi=0), Neumann at x=0 (psi(-dx)=psi(dx)).
    """
    dx = L / (M + 1)
    x = dx * np.arange(0, M + 1)  # x[0]=0, x[M]=M*dx

    V_real = 0.5 * mass * omega**2 * x**2 + (lam / 4.0) * x**4
    t_coeff = hbar**2 / (2 * mass * dx**2)

    N = M + 1
    H = np.zeros((N, N))

    for j in range(N):
        H[j, j] = V_real[j] + 2 * t_coeff
        if j + 1 < N:
            H[j, j + 1] = -t_coeff
            H[j + 1, j] = -t_coeff

    # Neumann BC at x=0: psi(-dx) = psi(dx), so the FD stencil at j=0 is
    # [-2*psi(0) + 2*psi(dx)] / dx^2 instead of [-2*psi(0) + psi(-dx) + psi(dx)] / dx^2
    # This means H[0,0] gets 2*t_coeff (already set), and H[0,1] = -2*t_coeff (mirror)
    H[0, 1] = -2 * t_coeff  # double the off-diagonal for Neumann

    t0 = time.time()
    vals, vecs = la.eigh(H)
    t_eig = time.time() - t0

    x2_expect = np.einsum("j,jn->n", x**2, vecs**2)
    E_complex = vals - 0.5j * epsilon * x2_expect

    # psi_n(0) from the even-parity half-grid eigenvector
    # The full-space normalisation: integral |psi|^2 dx over (-inf,inf) = 1
    # On the half-grid: integral_0^inf |psi|^2 dx = 1/2, so half-grid norm = 1/sqrt(2) of full
    # psi_n(0) = vecs[0, n] * sqrt(2) / sqrt(dx)
    psi0 = vecs[0, :] * np.sqrt(2.0 / dx)

    phases = np.exp(-1j * E_complex * T_time / hbar)
    K = np.sum(psi0**2 * phases)

    return K, t_eig, N

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    K_m = mehler()
    print("=" * 70, flush=True)
    print("GRID EIGENDECOMP v3: centered grid + even-parity sector", flush=True)
    print(f"Mehler: Re={K_m.real:.8f}  Im={K_m.imag:.8f}  |K|={abs(K_m):.8f}", flush=True)
    print("=" * 70, flush=True)

    L = 15.0

    # --- Full grid test (x=0 exactly centered) ---
    print("\n=== FULL GRID (x=0 exactly at center) ===", flush=True)
    for lam in [0.0, 0.1]:
        print(f"\n--- lambda = {lam}, L = {L} ---", flush=True)
        print(f"{'N_half':>7s}  {'N_total':>7s}  {'Re(K)':>14s}  {'Im(K)':>14s}  "
              f"{'|K|':>12s}  {'err%':>10s}  {'eig_t':>6s}", flush=True)
        prev = None
        for N_half in [256, 512, 1024, 2048]:
            K, t_eig, x0, N_tot = propagator_grid_full(lam, N_half, L)
            absK = abs(K)
            if lam == 0.0:
                err_str = f"{abs(K - K_m) / abs(K_m) * 100:.4f}%"
            else:
                err_str = "---"
            d = f"  d|K|={absK - prev:+.6f}" if prev is not None else ""
            prev = absK
            print(f"{N_half:7d}  {N_tot:7d}  {K.real:+14.8f}  {K.imag:+14.8f}  "
                  f"{absK:12.8f}  {err_str:>10s}  {t_eig:5.1f}s{d}", flush=True)

    # --- Even-parity sector test ---
    print("\n\n=== EVEN-PARITY HALF-GRID ===", flush=True)
    for lam in [0.0, 0.1]:
        print(f"\n--- lambda = {lam}, L = {L} ---", flush=True)
        print(f"{'M':>7s}  {'N_mat':>7s}  {'Re(K)':>14s}  {'Im(K)':>14s}  "
              f"{'|K|':>12s}  {'err%':>10s}  {'eig_t':>6s}", flush=True)
        prev = None
        for M in [512, 1024, 2048, 4096]:
            K, t_eig, N_mat = propagator_grid_even(lam, M, L)
            absK = abs(K)
            if lam == 0.0:
                err_str = f"{abs(K - K_m) / abs(K_m) * 100:.4f}%"
            else:
                err_str = "---"
            d = f"  d|K|={absK - prev:+.6f}" if prev is not None else ""
            prev = absK
            print(f"{M:7d}  {N_mat:7d}  {K.real:+14.8f}  {K.imag:+14.8f}  "
                  f"{absK:12.8f}  {err_str:>10s}  {t_eig:5.1f}s{d}", flush=True)

    print(f"\nML ref lam=0.1: Re=+0.308045 Im=-0.306802 |K|=0.434569", flush=True)
    print("Done.", flush=True)
