"""Exact (non-ML) extended time scan at fixed lambda.

Same lattice FFT and continuum spectral evaluations as exact_check.py,
with lambda held at 0.5.  T starts at 2.0 and steps by 0.5 up to 8.0,
skipping points within 0.5 of the harmonic caustics n*pi.
Everything else matches the original check: N=16, eta=0.05, m=omega=1,
x_i=x_f=0, V = x^2/2 + (lambda/4) x^4.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fftconvolve(a, b, mode="valid"):
    """numpy-only linear convolution via FFT, 'valid' mode, len(a) >= len(b)."""
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    full = np.fft.ifft(np.fft.fft(a, nfft) * np.fft.fft(b, nfft))[:n]
    return full[len(b) - 1: len(a)]


m, omega, eta, N = 1.0, 1.0, 0.05, 16
lam = 0.5
D = N - 1


def V(x, lam_val):
    return 0.5 * m * omega**2 * x**2 + 0.25 * lam_val * x**4


def lattice_K_fft(lam_val, ac, L=10.0, M=2**15):
    x = np.linspace(-L, L, M)
    h = x[1] - x[0]
    pref = np.sqrt(m / (2 * np.pi * 1j * ac))
    u = np.linspace(-2 * L, 2 * L, 2 * M - 1)
    link = pref * np.exp(1j * m * u**2 / (2 * ac))
    xi, xf = 0.0, 0.0
    f = pref * np.exp(1j * m * (x - xi)**2 / (2 * ac))
    w = np.exp(-1j * ac * V(x, lam_val))
    for _ in range(N - 1):
        f = h * fftconvolve(link, w * f, mode="valid")
    return np.interp(xf, x, f.real) + 1j * np.interp(xf, x, f.imag)


def continuum_eigenpairs(lam_val, L=14.0, Mg=1500, nmax=600):
    """T-independent spectral data for the continuum propagator."""
    x = np.linspace(-L, L, Mg, endpoint=False)
    h = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(Mg, d=h)
    F = np.fft.fft(np.eye(Mg), axis=0)
    Kin = np.real(np.fft.ifft((k**2)[:, None] * F, axis=0)) / (2 * m)
    H = Kin + np.diag(V(x, lam_val))
    E, psi = np.linalg.eigh(H)
    psi = psi / np.sqrt(h)
    psi0 = np.array([np.interp(0.0, x, psi[:, n]) for n in range(nmax)])
    return E[:nmax], psi0


def continuum_K_from_spectrum(E, psi0, Tc):
    return np.sum(psi0**2 * np.exp(-1j * E * Tc))


CAUSTIC_KEEP_AWAY = 0.5
candidates = [round(2.0 + 0.5 * i, 2) for i in range(13)]  # 2.0, 2.5, ..., 8.0
time_list = []
for T_cand in candidates:
    dist = min(abs(T_cand - n * np.pi) for n in (1, 2, 3))
    if dist >= CAUSTIC_KEEP_AWAY:
        time_list.append(T_cand)

out_dir = Path(__file__).resolve().parent
out_path = out_dir / "exact_check_Time_Loop_Extended_lambda0.5.csv"
plot_path = out_dir / "exact_check_Time_Loop_Extended_lambda0.5_ReIm.png"

print("STARTING exact_check_Time_Loop_Extended", flush=True)
print(f"lambda = {lam} (fixed),  N = {N},  eta = {eta}")
print(f"T candidates 2..8 step 0.5; skipped if |T-n*pi| < {CAUSTIC_KEEP_AWAY}")
print(f"Scanning {len(time_list)} times: {time_list}")
print(f"CSV: {out_path}", flush=True)

E, psi0 = continuum_eigenpairs(lam)
rows = []
fieldnames = [
    "lambda", "N", "T", "eta", "T_c_re", "T_c_im",
    "lattice_K_re", "lattice_K_im", "lattice_abs_K",
    "continuum_K_re", "continuum_K_im", "continuum_abs_K",
    "lattice_vs_continuum_rel_diff",
]
for i, T in enumerate(time_list):
    Tc = T - 1j * eta
    ac = Tc / N
    Kf = lattice_K_fft(lam, ac)
    Kc = continuum_K_from_spectrum(E, psi0, Tc)
    rel = abs(Kf - Kc) / abs(Kc) if abs(Kc) else float("nan")
    print(
        f"T={T:5.2f}  lattice={Kf:+.8f}  continuum={Kc:+.8f}  rel={rel:.3e}",
        flush=True,
    )
    rows.append({
        "lambda": lam,
        "N": N,
        "T": T,
        "eta": eta,
        "T_c_re": Tc.real,
        "T_c_im": Tc.imag,
        "lattice_K_re": Kf.real,
        "lattice_K_im": Kf.imag,
        "lattice_abs_K": abs(Kf),
        "continuum_K_re": Kc.real,
        "continuum_K_im": Kc.imag,
        "continuum_abs_K": abs(Kc),
        "lattice_vs_continuum_rel_diff": rel,
    })
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

Ts = np.array([row["T"] for row in rows])
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axes[0].plot(Ts, [row["lattice_K_re"] for row in rows], lw=1.6, label="lattice FFT Re K")
axes[0].plot(Ts, [row["continuum_K_re"] for row in rows], "--", lw=1.2, label="continuum Re K")
axes[0].set_ylabel("Re K")
axes[0].set_title(rf"$\lambda={lam}$, $N={N}$, $\eta={eta}$: Re $K(0,0;T-i\eta)$")
axes[0].grid(alpha=0.3)
axes[0].legend()
axes[1].plot(Ts, [row["lattice_K_im"] for row in rows], lw=1.6, label="lattice FFT Im K")
axes[1].plot(Ts, [row["continuum_K_im"] for row in rows], "--", lw=1.2, label="continuum Im K")
axes[1].set_xlabel("T")
axes[1].set_ylabel("Im K")
axes[1].set_title(rf"$\lambda={lam}$, $N={N}$, $\eta={eta}$: Im $K(0,0;T-i\eta)$")
axes[1].grid(alpha=0.3)
axes[1].legend()
fig.tight_layout()
fig.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Saved {len(rows)} rows to {out_path}")
print(f"Saved plot to {plot_path}")
