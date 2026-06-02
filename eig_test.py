"""
Final attempt: position-space grid with finite-difference kinetic energy.
FD has lower spectral norm than spectral KE, reducing forward error in expm.
Also try the split-operator approach directly (avoid Gaussian extrapolation)
by computing exp(-iHT)|delta> one step at a time.
"""
import numpy as np
import scipy.linalg as la
import pandas as pd
import time

omega = 1.0; mass = 1.0; hbar = 1.0; epsilon = 0.01; T_time = 1.0

LAMBDAS = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
           0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]

def mehler():
    w = np.sqrt(omega**2 - 1j * epsilon)
    wT = w * T_time
    return np.sqrt(mass * w / (2j * np.pi * hbar * np.sin(wT)))

def propagator_splitop_delta(lam, N=4096, L=20.0, N_steps=10000):
    """
    K(0,0;T) via split-operator propagation of a narrow Gaussian,
    extrapolated to zero width.
    Uses multiple Gaussian widths and polynomial extrapolation.
    """
    dx = 2*L / N
    x = -L + np.arange(N) * dx
    j0 = N // 2  # x=0
    k = 2*np.pi * np.fft.fftfreq(N, dx)
    dt = T_time / N_steps

    V = 0.5 * mass * (omega**2 - 1j*epsilon) * x**2 + (lam/4.0) * x**4
    exp_V_half = np.exp(-0.5j * V * dt / hbar)
    exp_T = np.exp(-0.5j * hbar * k**2 / mass * dt)

    sigmas = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    K_vals = []

    for sigma in sigmas:
        psi = (np.exp(-x**2 / (4*sigma**2)) / (2*np.pi*sigma**2)**0.25).astype(complex)
        norm0 = np.sqrt(np.sum(np.abs(psi)**2) * dx)
        psi /= norm0

        for _ in range(N_steps):
            psi *= exp_V_half
            psi = np.fft.ifft(exp_T * np.fft.fft(psi))
            psi *= exp_V_half

        psi_at_0 = psi[j0]
        psi0_at_0 = np.exp(0) / (2*np.pi*sigma**2)**0.25 / norm0
        K_sigma = psi_at_0 / (psi0_at_0 * dx)
        K_vals.append(K_sigma)

    K_vals = np.array(K_vals)
    s2 = sigmas**2

    cr = np.polyfit(s2, K_vals.real, 2)
    ci = np.polyfit(s2, K_vals.imag, 2)
    K_re = np.polyval(cr, 0.0)
    K_im = np.polyval(ci, 0.0)
    return K_re + 1j * K_im

def propagator_splitop_multi(lam, N=4096, L=20.0, N_steps=10000):
    """Improved extrapolation with more sigmas and cubic fit."""
    dx = 2*L / N
    x = -L + np.arange(N) * dx
    j0 = N // 2
    k = 2*np.pi * np.fft.fftfreq(N, dx)
    dt = T_time / N_steps

    V = 0.5 * mass * (omega**2 - 1j*epsilon) * x**2 + (lam/4.0) * x**4
    exp_V_half = np.exp(-0.5j * V * dt / hbar)
    exp_T = np.exp(-0.5j * hbar * k**2 / mass * dt)

    sigmas = np.array([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8])
    K_vals = []

    for sigma in sigmas:
        psi = (np.exp(-x**2 / (4*sigma**2)) / (2*np.pi*sigma**2)**0.25).astype(complex)
        norm0 = np.sqrt(np.sum(np.abs(psi)**2) * dx)
        psi /= norm0

        for _ in range(N_steps):
            psi *= exp_V_half
            psi = np.fft.ifft(exp_T * np.fft.fft(psi))
            psi *= exp_V_half

        psi_at_0 = psi[j0]
        psi0_at_0 = np.exp(0) / (2*np.pi*sigma**2)**0.25 / norm0
        K_sigma = psi_at_0 / (psi0_at_0 * dx)
        K_vals.append(K_sigma)

    K_vals = np.array(K_vals)
    s2 = sigmas**2

    # Try cubic fit (degree 3)
    cr = np.polyfit(s2, K_vals.real, 3)
    ci = np.polyfit(s2, K_vals.imag, 3)
    K_re = np.polyval(cr, 0.0)
    K_im = np.polyval(ci, 0.0)
    return K_re + 1j * K_im

# ============================================================
K_m = mehler()
print(f"Mehler: Re={K_m.real:.8f}  Im={K_m.imag:.8f}  |K|={abs(K_m):.8f}", flush=True)

# Test improved split-op at lambda=0
print(f"\n--- Split-op Gaussian extrap at lam=0 (quadratic) ---", flush=True)
t0 = time.time()
K0 = propagator_splitop_delta(0.0, N=4096, L=20, N_steps=5000)
dt = time.time() - t0
err = abs(K0 - K_m) / abs(K_m) * 100
print(f"  K = {K0.real:+.8f} {K0.imag:+.8f}i  |K|={abs(K0):.8f}  "
      f"err={err:.4f}%  t={dt:.1f}s", flush=True)

print(f"\n--- Split-op Gaussian extrap at lam=0 (cubic, more sigmas) ---", flush=True)
t0 = time.time()
K0b = propagator_splitop_multi(0.0, N=4096, L=20, N_steps=5000)
dt = time.time() - t0
err = abs(K0b - K_m) / abs(K_m) * 100
print(f"  K = {K0b.real:+.8f} {K0b.imag:+.8f}i  |K|={abs(K0b):.8f}  "
      f"err={err:.4f}%  t={dt:.1f}s", flush=True)

# Full lambda scan with improved method
print(f"\n=== Lambda scan (cubic fit, 10 sigmas) ===", flush=True)
rows = []
for lam in LAMBDAS:
    t0 = time.time()
    K = propagator_splitop_multi(lam, N=4096, L=20, N_steps=5000)
    dt = time.time() - t0
    print(f"  lam={lam:5.2f}  Re={K.real:+12.6f}  Im={K.imag:+12.6f}  "
          f"|K|={abs(K):10.6f}  t={dt:.1f}s", flush=True)
    rows.append({"lambda": lam, "K_real": K.real, "K_imag": K.imag, "abs_K": abs(K)})

df = pd.DataFrame(rows)
df.to_csv("eigendecomp_propagator_results.csv", index=False)
print(f"\nSaved to eigendecomp_propagator_results.csv", flush=True)

# ML comparison
ml = {0: (0.307230355,-0.307623399), 0.01: (0.307257564,-0.307601922),
      0.05: (0.307613868,-0.307246798), 0.1: (0.308045423,-0.306801876),
      0.15: (0.30853144,-0.306316107), 0.2: (0.309051999,-0.305777055),
      0.3: (0.311482275,-0.304049745), 0.4: (0.311483769,-0.30367083),
      0.5: (0.311707652,-0.302996468), 0.6: (0.312276772,-0.302112874),
      0.8: (0.313795351,-0.300164734), 1.0: (0.315276192,-0.298170929),
      1.5: (0.318673821,-0.293095975), 2.0: (0.321627281,-0.287947061),
      3.0: (0.326363101,-0.277650885), 4.0: (0.329806674,-0.267631712),
      6.0: (0.333883313,-0.248951901), 8.0: (0.335454471,-0.232183184)}

print(f"\n=== Comparison with ML ===", flush=True)
print(f"{'lam':>6s}  {'Re_SO':>10s}  {'Re_ML':>10s}  {'err%':>9s}  "
      f"{'Im_SO':>10s}  {'Im_ML':>10s}  {'err%':>9s}  "
      f"{'|K|_SO':>10s}  {'|K|_ML':>10s}  {'err%':>9s}", flush=True)
for r in rows:
    lam = r["lambda"]
    re_e, im_e, ak_e = r["K_real"], r["K_imag"], r["abs_K"]
    re_m, im_m = ml[lam]
    ak_m = np.sqrt(re_m**2 + im_m**2)
    print(f"{lam:6.2f}  {re_e:10.6f}  {re_m:10.6f}  "
          f"{100*(re_e-re_m)/abs(re_m):+9.4f}  "
          f"{im_e:10.6f}  {im_m:10.6f}  "
          f"{100*(im_e-im_m)/abs(im_m):+9.4f}  "
          f"{ak_e:10.6f}  {ak_m:10.6f}  "
          f"{100*(ak_e-ak_m)/ak_m:+9.4f}", flush=True)

print("\nDONE", flush=True)
