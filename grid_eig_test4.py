"""
Direct split-operator propagation of a delta function.
No eigendecomposition, no Gaussian extrapolation.

K(0,0;T) = <x=0| exp(-iHT) |x=0>

On the grid: start with psi[j] = delta_{j,j0}, propagate with
split-operator, read off K = psi_final[j0] / dx.
"""
import numpy as np
import time
import sys

omega = 1.0; mass = 1.0; hbar = 1.0; epsilon = 0.01; T_time = 1.0

def mehler():
    w = np.sqrt(omega**2 - 1j * epsilon)
    wT = w * T_time
    return np.sqrt(mass * w / (2j * np.pi * hbar * np.sin(wT)))

def propagator_delta(lam, N, L, N_steps):
    dx = 2 * L / N
    x = -L + (np.arange(N) + 0.5) * dx  # cell-centered so x=0 is between two points

    # Use odd N so we can center exactly on 0
    # Actually, let's just pick the grid point closest to 0
    j0 = np.argmin(np.abs(x))

    k = 2 * np.pi * np.fft.fftfreq(N, dx)
    dt = T_time / N_steps

    V = 0.5 * mass * (omega**2 - 1j * epsilon) * x**2 + (lam / 4.0) * x**4
    exp_V_half = np.exp(-0.5j * V * dt / hbar)
    exp_T = np.exp(-0.5j * hbar * k**2 / mass * dt)

    # Delta function initial state: psi[j] = delta_{j,j0}
    psi = np.zeros(N, dtype=complex)
    psi[j0] = 1.0

    for step in range(N_steps):
        psi *= exp_V_half
        psi = np.fft.ifft(exp_T * np.fft.fft(psi))
        psi *= exp_V_half

    # K(0,0;T) = psi_final[j0] / dx
    K = psi[j0] / dx

    return K, x[j0]

def propagator_delta_centered(lam, N_half, L, N_steps):
    """Grid with x=0 exactly at center. N_total = 2*N_half+1."""
    N = 2 * N_half + 1
    dx = 2 * L / N
    x = -L + dx * (np.arange(N) + 0.5)
    # Shift so center point is exactly 0
    x = dx * np.arange(-N_half, N_half + 1)

    j0 = N_half  # x[j0] = 0 exactly

    k = 2 * np.pi * np.fft.fftfreq(N, dx)
    dt = T_time / N_steps

    V = 0.5 * mass * (omega**2 - 1j * epsilon) * x**2 + (lam / 4.0) * x**4
    exp_V_half = np.exp(-0.5j * V * dt / hbar)
    exp_T = np.exp(-0.5j * hbar * k**2 / mass * dt)

    psi = np.zeros(N, dtype=complex)
    psi[j0] = 1.0

    for step in range(N_steps):
        psi *= exp_V_half
        psi = np.fft.ifft(exp_T * np.fft.fft(psi))
        psi *= exp_V_half

    K = psi[j0] / dx
    return K, x[j0]

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    K_m = mehler()
    print("=" * 70, flush=True)
    print("SPLIT-OPERATOR DELTA PROPAGATION (no extrapolation)", flush=True)
    print(f"Mehler: Re={K_m.real:.8f}  Im={K_m.imag:.8f}  |K|={abs(K_m):.8f}", flush=True)
    print("=" * 70, flush=True)

    L = 20.0

    # Convergence in N_steps at fixed grid
    print("\n=== Convergence in time steps (N_half=2048, L=20) ===", flush=True)
    for lam in [0.0]:
        print(f"\nlambda = {lam}", flush=True)
        print(f"{'N_steps':>8s}  {'Re(K)':>14s}  {'Im(K)':>14s}  {'|K|':>12s}  "
              f"{'err%':>10s}  {'time':>6s}", flush=True)
        for N_steps in [1000, 2000, 5000, 10000, 20000]:
            t0 = time.time()
            K, x0 = propagator_delta_centered(lam, 2048, L, N_steps)
            dt = time.time() - t0
            err = abs(K - K_m) / abs(K_m) * 100
            print(f"{N_steps:8d}  {K.real:+14.8f}  {K.imag:+14.8f}  {abs(K):12.8f}  "
                  f"{err:9.4f}%  {dt:5.1f}s", flush=True)

    # Convergence in grid size at fixed N_steps
    print("\n=== Convergence in grid size (N_steps=10000, L=20) ===", flush=True)
    for lam in [0.0, 0.1]:
        print(f"\nlambda = {lam}", flush=True)
        print(f"{'N_half':>7s}  {'N_total':>7s}  {'Re(K)':>14s}  {'Im(K)':>14s}  "
              f"{'|K|':>12s}  {'err%':>10s}  {'time':>6s}", flush=True)
        for N_half in [512, 1024, 2048, 4096]:
            t0 = time.time()
            K, x0 = propagator_delta_centered(lam, N_half, L, 10000)
            dt = time.time() - t0
            absK = abs(K)
            if lam == 0.0:
                err_str = f"{abs(K - K_m) / abs(K_m) * 100:.4f}%"
            else:
                err_str = "---"
            print(f"{N_half:7d}  {2*N_half+1:7d}  {K.real:+14.8f}  {K.imag:+14.8f}  "
                  f"{absK:12.8f}  {err_str:>10s}  {dt:5.1f}s", flush=True)

    print(f"\nML ref lam=0.1: Re=+0.308045 Im=-0.306802 |K|=0.434569", flush=True)
    print("Done.", flush=True)
