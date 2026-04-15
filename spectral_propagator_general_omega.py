"""
Spectral propagator for the quartic anharmonic oscillator with general omega.

Hamiltonian (m = hbar = 1):
    H = p^2/2 + (1/2) omega^2 x^2 + (lambda/4) x^4

Technique:  diagonalise H in a truncated HO basis of size N_BASIS,
            then sum   K(xb, t | xa, 0) = sum_n psi_n(xa) psi_n(xb) exp(-i E_n t).

Based on anharmonic_propagator_Ross_14_April.ipynb (which uses omega=1).
Generalised here so omega is an adjustable parameter.
"""

import csv
import numpy as np
from scipy.linalg import eigh

# ---------------------------------------------------------------------------
# Global physics parameters  (m = hbar = 1 throughout)
# ---------------------------------------------------------------------------
OMEGA   = 1.0        # HO frequency  — change to 2.0 etc. as needed
T_REF   = 1.0        # reference propagation time
N_BASIS = 400        # basis-truncation size


# ---------------------------------------------------------------------------
# 1. Hamiltonian in HO basis  (general omega, m = hbar = 1)
# ---------------------------------------------------------------------------
def build_hamiltonian(N, lam, omega):
    """
    H_{mn} = hbar*omega*(n+1/2) delta_{mn} + (lam/4)*(x^4)_{mn}
    x matrix element scale: sqrt(hbar / (2*m*omega)) = 1/sqrt(2*omega)
    """
    n = np.arange(N)
    x_mat = np.zeros((N, N))
    s = np.sqrt(n[1:] / (2.0 * omega))          # <n-1|x|n> = sqrt(n/(2*omega))
    x_mat[np.arange(N-1), np.arange(1, N)] = s
    x_mat[np.arange(1, N), np.arange(N-1)] = s
    x2 = x_mat @ x_mat
    x4 = x2 @ x2
    H = np.diag(omega * (n + 0.5)) + 0.25 * lam * x4
    return H, x_mat


def diagonalise(lam, omega, N=N_BASIS):
    H, x_mat = build_hamiltonian(N, lam, omega)
    evals, evecs = eigh(H)
    return evals, evecs, x_mat


# ---------------------------------------------------------------------------
# 2. HO basis wave-functions on a grid  (general omega, m = hbar = 1)
# ---------------------------------------------------------------------------
def ho_basis(N, x, omega):
    """
    phi_0 = (omega/pi)^{1/4} exp(-omega x^2 / 2)
    phi_1 = sqrt(2 omega) x phi_0
    phi_n = sqrt(2 omega / n) x phi_{n-1} - sqrt((n-1)/n) phi_{n-2}
    """
    alpha = omega                               # m*omega/hbar with m=hbar=1
    Phi = np.zeros((N, len(x)))
    Phi[0] = (alpha / np.pi)**0.25 * np.exp(-0.5 * alpha * x**2)
    if N > 1:
        Phi[1] = np.sqrt(2.0 * alpha) * x * Phi[0]
    for n in range(2, N):
        Phi[n] = (np.sqrt(2.0 * alpha / n) * x * Phi[n-1]
                  - np.sqrt((n - 1.0) / n) * Phi[n-2])
    return Phi


def eigenfunctions(C, x, omega):
    return C.T @ ho_basis(C.shape[0], x, omega)


# ---------------------------------------------------------------------------
# 3. Propagator  K(xb, t | xa, 0)  (general omega, m = hbar = 1)
# ---------------------------------------------------------------------------
def propagator(lam, xa, xb, t, omega, N=N_BASIS):
    xa = np.atleast_1d(np.asarray(xa, float))
    xb = np.atleast_1d(np.asarray(xb, float))
    ev, C, _ = diagonalise(lam, omega, N)
    Pa = eigenfunctions(C, xa, omega)
    Pb = eigenfunctions(C, xb, omega)
    K = np.einsum("ni,nj,n->ij", Pa, Pb, np.exp(-1j * ev * t))
    return K.item() if K.size == 1 else K


# ---------------------------------------------------------------------------
# 4. Mehler kernel  (exact HO propagator, m = hbar = 1, general omega)
# ---------------------------------------------------------------------------
def mehler(xa, xb, t, omega):
    w = omega
    sinwt = np.sin(w * t)
    coswt = np.cos(w * t)
    pref = np.sqrt(w / (2j * np.pi * sinwt))
    expo = (1j * w / (2.0 * sinwt)) * ((xa**2 + xb**2) * coswt - 2.0 * xa * xb)
    return pref * np.exp(expo)


# ---------------------------------------------------------------------------
# 5. Table helper
# ---------------------------------------------------------------------------
def print_table(xa, xb, t, omega, lam_list, label=""):
    print()
    print("=" * 82)
    print(f"  {label}    t={t}    (m=hbar=1, omega={omega}, N={N_BASIS})")
    print("=" * 82)
    print(f"{'lambda':>10}  {'Re K':>18}  {'Im K':>18}  {'|K|':>18}")
    print("-" * 70)
    for lam in lam_list:
        K = propagator(lam, xa, xb, t, omega)
        print(f"{lam:>10.4f}  {K.real:>+18.12f}  {K.imag:>+18.12f}  {abs(K):>18.12f}")


# ===================================================================
#  MAIN — run the requested checks
# ===================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------------
    # CHECK 1:  omega = 1, endpoints B (xa=0, xb=0), lambda = 0 and 0.01
    #   Expected (from Ross_14_April):
    #     0.0000  +0.300907724946  -0.319155419539   0.438640673848
    #     0.0100  +0.307095290114  -0.303034566927   0.431436514406
    # ------------------------------------------------------------------
    omega_check = 1.0
    lam_list_check = [0.0, 0.01]

    print("\n" + "=" * 82)
    print("CHECK 1:  omega=1, B: xa=0, xb=0  (should match Ross_14_April)")
    print("=" * 82)
    print(f"{'lambda':>10}  {'Re K':>18}  {'Im K':>18}  {'|K|':>18}")
    print("-" * 70)
    for lam in lam_list_check:
        K = propagator(lam, 0.0, 0.0, T_REF, omega_check)
        print(f"{lam:>10.4f}  {K.real:>+18.12f}  {K.imag:>+18.12f}  {abs(K):>18.12f}")

    print("\n  Expected:")
    print("    0.0000     +0.300907724946     -0.319155419539      0.438640673848")
    print("    0.0100     +0.307095290114     -0.303034566927      0.431436514406")

    # ------------------------------------------------------------------
    # CHECK 2:  omega = 1, endpoints A (xa=0.5, xb=0.7), lambda = 0 and 0.01
    #   Expected:
    #     0.0000  +0.237198508144  -0.357742088560   0.429234824068
    #     0.0100  +0.254021548964  -0.361768693084   0.442044720174
    # ------------------------------------------------------------------
    print("\n" + "=" * 82)
    print("CHECK 2:  omega=1, A: xa=0.5, xb=0.7  (should match Ross_14_April)")
    print("=" * 82)
    print(f"{'lambda':>10}  {'Re K':>18}  {'Im K':>18}  {'|K|':>18}")
    print("-" * 70)
    for lam in lam_list_check:
        K = propagator(lam, 0.5, 0.7, T_REF, omega_check)
        print(f"{lam:>10.4f}  {K.real:>+18.12f}  {K.imag:>+18.12f}  {abs(K):>18.12f}")

    print("\n  Expected:")
    print("    0.0000     +0.237198508144     -0.357742088560      0.429234824068")
    print("    0.0100     +0.254021548964     -0.361768693084      0.442044720174")

    # ------------------------------------------------------------------
    # CHECK 3:  omega = 2, lambda = 0, xa = 0, xb = 0, t = 1
    #   Expected Mehler ≈ 0.4183666762 - 0.4183666762 j
    # ------------------------------------------------------------------
    omega2 = 2.0
    K_spectral = propagator(0.0, 0.0, 0.0, T_REF, omega2)
    K_mehler   = mehler(0.0, 0.0, T_REF, omega2)

    print("\n" + "=" * 82)
    print("CHECK 3:  omega=2, lambda=0, xa=0, xb=0, t=1")
    print("=" * 82)
    print(f"  Mehler (exact):    Re={K_mehler.real:+.10f}, Im={K_mehler.imag:+.10f}")
    print(f"  Spectral (N=400):  Re={K_spectral.real:+.10f}, Im={K_spectral.imag:+.10f}")
    print(f"  |diff| = {abs(K_spectral - K_mehler):.3e}")
    print(f"\n  Expected Mehler:   0.4183666762-0.4183666762j")

    # ------------------------------------------------------------------
    # CHECK 4:  For ALL lambda=0, verify spectral == Mehler
    #           across several (omega, xa, xb) combos
    # ------------------------------------------------------------------
    test_cases = [
        (1.0, 0.0,  0.0,  "omega=1, xa=0,   xb=0"),
        (1.0, 0.5,  0.7,  "omega=1, xa=0.5, xb=0.7"),
        (1.0, 0.5,  0.0,  "omega=1, xa=0.5, xb=0"),
        (2.0, 0.0,  0.0,  "omega=2, xa=0,   xb=0"),
        (2.0, 0.5,  0.0,  "omega=2, xa=0.5, xb=0"),
        (2.0, 0.5,  0.7,  "omega=2, xa=0.5, xb=0.7"),
    ]

    print("\n" + "=" * 82)
    print("CHECK 4:  lambda=0 — spectral vs Mehler for various (omega, xa, xb)")
    print("=" * 82)
    print(f"  {'label':>30}  {'Re(spec)':>14}  {'Im(spec)':>14}  "
          f"{'Re(Meh)':>14}  {'Im(Meh)':>14}  {'|diff|':>10}")
    print("-" * 108)

    for omega_t, xa_t, xb_t, label_t in test_cases:
        Ks = propagator(0.0, xa_t, xb_t, T_REF, omega_t)
        Km = mehler(xa_t, xb_t, T_REF, omega_t)
        diff = abs(Ks - Km)
        print(f"  {label_t:>30}  {Ks.real:>+14.10f}  {Ks.imag:>+14.10f}  "
              f"{Km.real:>+14.10f}  {Km.imag:>+14.10f}  {diff:>10.3e}")

    # ==================================================================
    #  OMEGA = 2  propagator scan
    # ==================================================================
    omega_scan  = 2.0
    t_scan      = T_REF
    lam_scan    = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
                   0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 2.0, 3.0, 4.0, 8.0]
    endpoints   = [
        (0.0, 0.0,  "xa=0,   xb=0"),
        (0.5, 0.7,  "xa=0.5, xb=0.7"),
    ]

    csv_rows = []

    for xa_s, xb_s, ep_label in endpoints:
        print("\n" + "=" * 82)
        print(f"  OMEGA=2 SCAN — {ep_label}    t={t_scan}    "
              f"(m=hbar=1, omega={omega_scan}, N={N_BASIS})")
        print("=" * 82)
        print(f"{'lambda':>10}  {'Re K':>18}  {'Im K':>18}  {'|K|':>18}")
        print("-" * 70)

        for lam in lam_scan:
            K = propagator(lam, xa_s, xb_s, t_scan, omega_scan)
            print(f"{lam:>10.4f}  {K.real:>+18.12f}  {K.imag:>+18.12f}  "
                  f"{abs(K):>18.12f}")
            csv_rows.append([omega_scan, xa_s, xb_s, t_scan, lam,
                             K.real, K.imag, abs(K)])

    # ---- write CSV ----
    csv_path = "spectral_propagator_omega2_scan.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["omega", "xa", "xb", "t", "lambda",
                     "Re_K", "Im_K", "abs_K"])
        w.writerows(csv_rows)
    print(f"\nResults written to {csv_path}")
