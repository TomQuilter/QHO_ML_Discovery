"""
Eigendecomposition propagator for the quantum anharmonic oscillator.

Two methods compared:
  A) Matrix exponential  (v^T expm(-iHT) v)  -- includes truncation artifacts
  B) Selective spectral sum -- keeps only converged low-lying eigenstates

Uses HO basis, even-parity sector for K(0,0;T).
Convention:  V(x) = (1/2) m (omega^2 - i*epsilon) x^2 + (lambda/4) x^4
"""
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import pandas as pd
import time
import sys

omega = 1.0; mass = 1.0; hbar = 1.0; epsilon = 0.01; T_time = 1.0

LAMBDAS = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
           0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]

# ==================================================================
#  Build operators in even-parity HO basis
# ==================================================================

def build_operators(Nev):
    """Build x^2 and x^4 as dense matrices in even HO sector."""
    n = 2.0 * np.arange(Nev)
    x2 = np.zeros((Nev, Nev))
    for j in range(Nev):
        nj = 2*j
        x2[j, j] = nj + 0.5
        if j+1 < Nev:
            val = np.sqrt((nj+1)*(nj+2)) / 2.0
            x2[j, j+1] = val
            x2[j+1, j] = val
    x4 = x2 @ x2
    return x2, x4

def build_psi_even(Nev):
    v = np.zeros(Nev)
    v[0] = np.pi**(-0.25)
    for k in range(1, Nev):
        v[k] = -np.sqrt((2*k - 1.0) / (2*k)) * v[k-1]
    return v

def make_H(lam, Nev, x2, x4):
    n = 2.0 * np.arange(Nev)
    H = np.diag((n + 0.5).astype(complex))
    H -= 0.5j * epsilon * x2[:Nev, :Nev]
    H += (lam / 4.0) * x4[:Nev, :Nev]
    return H

# ==================================================================
#  Method A: matrix exponential (for reference)
# ==================================================================

def K_expm(lam, Nev, x2, x4, ve):
    H = make_H(lam, Nev, x2, x4)
    U = la.expm(-1j * H * T_time)
    v = ve[:Nev]
    return v @ U @ v

# ==================================================================
#  Method B: selective spectral sum
# ==================================================================

def K_spectral(lam, Nev, x2, x4, ve, floor=1e-10):
    H = make_H(lam, Nev, x2, x4)
    vals, vecs = la.eig(H)

    order = np.argsort(vals.real)
    vals = vals[order]; vecs = vecs[:, order]

    amp = np.abs(np.exp(-1j * vals * T_time))
    keep = (vals.imag < 0) & (amp > floor)

    vals_k = vals[keep]; vecs_k = vecs[:, keep]
    cnorm = np.einsum("in,in->n", vecs_k, vecs_k)
    phi0  = vecs_k.T @ ve[:Nev]
    phases = np.exp(-1j * vals_k * T_time)

    K = np.sum(phi0**2 / cnorm * phases)
    return K, int(keep.sum())

def mehler():
    w = np.sqrt(omega**2 - 1j*epsilon)
    wT = w * T_time
    return np.sqrt(mass * w / (2j * np.pi * hbar * np.sin(wT)))

# ==================================================================
#  Run
# ==================================================================
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 68)
    print("Eigendecomposition propagator  (HO basis, even-parity)")
    print("=" * 68)

    NEV = 600
    print(f"\nBuilding operators (Nev={NEV})...", end=" ", flush=True)
    t0 = time.time()
    x2, x4 = build_operators(NEV)
    ve = build_psi_even(NEV)
    print(f"done ({time.time()-t0:.2f}s)")

    K_m = mehler()
    print(f"\nMehler (exact HO, eps={epsilon}):")
    print(f"  Re={K_m.real:.8f}  Im={K_m.imag:.8f}  |K|={abs(K_m):.8f}")

    # -- Convergence: compare expm vs spectral at lam=0 --
    print(f"\n--- Convergence at lam=0: expm vs selective spectral ---")
    print(f"{'Nev':>5s}  {'Method':>8s}  {'Re(K)':>12s}  {'Im(K)':>12s}  "
          f"{'|K|':>10s}  {'Meh err%':>10s}  {'time':>5s}")
    for Nev in [100, 200, 300, 400, 500, 600]:
        t1 = time.time()
        Ka = K_expm(0.0, Nev, x2, x4, ve)
        dta = time.time() - t1
        ea = abs(Ka - K_m)/abs(K_m)*100

        t1 = time.time()
        Kb, M = K_spectral(0.0, Nev, x2, x4, ve)
        dtb = time.time() - t1
        eb = abs(Kb - K_m)/abs(K_m)*100

        print(f"{Nev:5d}  {'expm':>8s}  {Ka.real:+12.8f}  {Ka.imag:+12.8f}  "
              f"{abs(Ka):10.8f}  {ea:10.4e}  {dta:4.1f}s")
        print(f"{'':>5s}  {'spec':>8s}  {Kb.real:+12.8f}  {Kb.imag:+12.8f}  "
              f"{abs(Kb):10.8f}  {eb:10.4e}  {dtb:4.1f}s  (M={M})")

    # -- Convergence at lam=1: expm vs spectral --
    print(f"\n--- Convergence at lam=1.0: expm vs selective spectral ---")
    print(f"{'Nev':>5s}  {'Method':>8s}  {'Re(K)':>12s}  {'Im(K)':>12s}  "
          f"{'|K|':>10s}  {'time':>5s}")
    for Nev in [100, 200, 300, 400, 500, 600]:
        Ka = K_expm(1.0, Nev, x2, x4, ve)
        Kb, M = K_spectral(1.0, Nev, x2, x4, ve)
        print(f"{Nev:5d}  {'expm':>8s}  {Ka.real:+12.8f}  {Ka.imag:+12.8f}  "
              f"{abs(Ka):10.8f}")
        print(f"{'':>5s}  {'spec':>8s}  {Kb.real:+12.8f}  {Kb.imag:+12.8f}  "
              f"{abs(Kb):10.8f}  (M={M})")

    # -- Convergence at lam=8: expm vs spectral --
    print(f"\n--- Convergence at lam=8.0: expm vs selective spectral ---")
    print(f"{'Nev':>5s}  {'Method':>8s}  {'Re(K)':>12s}  {'Im(K)':>12s}  "
          f"{'|K|':>10s}  {'time':>5s}")
    for Nev in [100, 200, 300, 400, 500, 600]:
        Ka = K_expm(8.0, Nev, x2, x4, ve)
        Kb, M = K_spectral(8.0, Nev, x2, x4, ve)
        print(f"{Nev:5d}  {'expm':>8s}  {Ka.real:+12.8f}  {Ka.imag:+12.8f}  "
              f"{abs(Ka):10.8f}")
        print(f"{'':>5s}  {'spec':>8s}  {Kb.real:+12.8f}  {Kb.imag:+12.8f}  "
              f"{abs(Kb):10.8f}  (M={M})")

    # -- Full lambda scan with selective spectral sum --
    N_USE = 600
    print(f"\n{'='*68}")
    print(f"Full lambda scan  (selective spectral, Nev={N_USE})")
    print(f"{'lam':>6s}  {'M':>4s}  {'Re(K)':>12s}  {'Im(K)':>12s}  "
          f"{'|K|':>10s}  {'time':>5s}")
    rows = []
    for lam in LAMBDAS:
        t1 = time.time()
        K, M = K_spectral(lam, N_USE, x2, x4, ve)
        dt = time.time() - t1
        print(f"{lam:6.2f}  {M:4d}  {K.real:+12.6f}  {K.imag:+12.6f}  "
              f"{abs(K):10.6f}  {dt:4.1f}s")
        rows.append({"lambda": lam, "K_real": K.real,
                      "K_imag": K.imag, "abs_K": abs(K)})

    df = pd.DataFrame(rows)
    csv_path = "eigendecomp_propagator_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    # -- Comparison with ML data --
    ml = {0:    (0.307230355, -0.307623399),
          0.01: (0.307257564, -0.307601922),
          0.05: (0.307613868, -0.307246798),
          0.1:  (0.308045423, -0.306801876),
          0.15: (0.30853144,  -0.306316107),
          0.2:  (0.309051999, -0.305777055),
          0.3:  (0.311482275, -0.304049745),
          0.4:  (0.311483769, -0.30367083),
          0.5:  (0.311707652, -0.302996468),
          0.6:  (0.312276772, -0.302112874),
          0.8:  (0.313795351, -0.300164734),
          1.0:  (0.315276192, -0.298170929),
          1.5:  (0.318673821, -0.293095975),
          2.0:  (0.321627281, -0.287947061),
          3.0:  (0.326363101, -0.277650885),
          4.0:  (0.329806674, -0.267631712),
          6.0:  (0.333883313, -0.248951901),
          8.0:  (0.335454471, -0.232183184)}

    print(f"\n{'='*68}")
    print("Comparison with ML  (% err = (Eig-ML)/|ML| * 100)")
    print(f"{'lam':>6s}  {'Re_Eig':>10s}  {'Re_ML':>10s}  {'err%':>9s}  "
          f"{'Im_Eig':>10s}  {'Im_ML':>10s}  {'err%':>9s}  "
          f"{'|K|_Eig':>10s}  {'|K|_ML':>10s}  {'err%':>9s}")
    for r in rows:
        lam = r["lambda"]
        re_e, im_e, ak_e = r["K_real"], r["K_imag"], r["abs_K"]
        re_m, im_m = ml[lam]
        ak_m = np.sqrt(re_m**2 + im_m**2)
        er_r = 100*(re_e-re_m)/abs(re_m)
        er_i = 100*(im_e-im_m)/abs(im_m)
        er_a = 100*(ak_e-ak_m)/ak_m
        print(f"{lam:6.2f}  {re_e:10.6f}  {re_m:10.6f}  {er_r:+9.4f}  "
              f"{im_e:10.6f}  {im_m:10.6f}  {er_i:+9.4f}  "
              f"{ak_e:10.6f}  {ak_m:10.6f}  {er_a:+9.4f}")
