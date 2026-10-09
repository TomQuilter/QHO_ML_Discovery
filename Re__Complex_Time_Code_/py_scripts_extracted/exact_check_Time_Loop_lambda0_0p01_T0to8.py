"""Exact (non-ML) time scan at lambda = 0 and 0.01, T = 0..8 step 0.01.

Copied from exact_check_Time_Loop.py. Originals are not modified.
Same lattice: N=16, eta=0.05, m=omega=1, x_i=x_f=0,
V = x^2/2 + (lambda/4) x^4.
Writes one CSV per lambda and a combined Re/Im plot of both.
"""
from __future__ import annotations

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
    return full[len(b) - 1 : len(a)]


m, omega, eta, N = 1.0, 1.0, 0.05, 16
LAMBDAS = [0.0, 0.01, 0.1]
dt = 0.01
T_MAX = 8.0
time_list = [round(i * dt, 2) for i in range(int(round(T_MAX / dt)) + 1)]

out_dir = Path(__file__).resolve().parent
plot_path = out_dir / "exact_check_Time_Loop_lambda0_0p01_T0to8_ReIm.png"

fieldnames = [
    "lambda",
    "N",
    "T",
    "eta",
    "T_c_re",
    "T_c_im",
    "lattice_K_re",
    "lattice_K_im",
    "lattice_abs_K",
    "continuum_K_re",
    "continuum_K_im",
    "continuum_abs_K",
    "lattice_vs_continuum_rel_diff",
]


def V(x, lam_val):
    return 0.5 * m * omega**2 * x**2 + 0.25 * lam_val * x**4


def lattice_K_fft(lam_val, ac, L=10.0, M=2**15):
    x = np.linspace(-L, L, M)
    h = x[1] - x[0]
    pref = np.sqrt(m / (2 * np.pi * 1j * ac))
    u = np.linspace(-2 * L, 2 * L, 2 * M - 1)
    link = pref * np.exp(1j * m * u**2 / (2 * ac))
    xi, xf = 0.0, 0.0
    f = pref * np.exp(1j * m * (x - xi) ** 2 / (2 * ac))
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


def lam_tag(lam):
    if lam == 0.0:
        return "0"
    return f"{lam:g}".replace(".", "p")


def scan_lambda(lam):
    out_path = out_dir / f"exact_check_Time_Loop_lambda{lam_tag(lam)}_T0to8_dt0.01.csv"
    print("\n" + "=" * 72, flush=True)
    print(f"STARTING exact scan  lambda={lam}, N={N}, eta={eta}", flush=True)
    print(f"Scanning {len(time_list)} times: T = 0 .. {T_MAX} step {dt}", flush=True)
    print(f"CSV: {out_path}", flush=True)

    rows = []
    done_T = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({
                    "lambda": float(row["lambda"]),
                    "N": int(row["N"]),
                    "T": float(row["T"]),
                    "eta": float(row["eta"]),
                    "T_c_re": float(row["T_c_re"]),
                    "T_c_im": float(row["T_c_im"]),
                    "lattice_K_re": float(row["lattice_K_re"]),
                    "lattice_K_im": float(row["lattice_K_im"]),
                    "lattice_abs_K": float(row["lattice_abs_K"]),
                    "continuum_K_re": float(row["continuum_K_re"]),
                    "continuum_K_im": float(row["continuum_K_im"]),
                    "continuum_abs_K": float(row["continuum_abs_K"]),
                    "lattice_vs_continuum_rel_diff": float(row["lattice_vs_continuum_rel_diff"]),
                })
                done_T.add(round(float(row["T"]), 2))
        print(f"Resuming: already have {len(rows)} rows up to T={max(done_T) if done_T else 'n/a'}", flush=True)
        if len(done_T) >= len(time_list):
            print("Already complete; skipping lattice scan.", flush=True)
            return rows, out_path

    print("Building continuum eigenpairs (once)...", flush=True)
    E, psi0 = continuum_eigenpairs(lam)
    for i, T in enumerate(time_list):
        if round(T, 2) in done_T:
            continue
        Tc = T - 1j * eta
        ac = Tc / N
        Kf = lattice_K_fft(lam, ac)
        Kc = continuum_K_from_spectrum(E, psi0, Tc)
        rel = abs(Kf - Kc) / abs(Kc) if abs(Kc) else float("nan")
        if i % 50 == 0 or i + 1 == len(time_list):
            print(
                f"lam={lam}  T={T:5.2f}  lattice={Kf:+.8f}  "
                f"continuum={Kc:+.8f}  rel={rel:.3e}",
                flush=True,
            )
        rows.append(
            {
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
            }
        )
        # checkpoint every 50 points so a crash does not lose the scan
        if i % 50 == 0 or i + 1 == len(time_list):
            with out_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {out_path}", flush=True)
    return rows, out_path


def make_combined_plot(results):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    styles = {
        0.0: {"color": "C0", "lw": 1.8, "label": r"$\lambda=0$ lattice"},
        0.01: {"color": "C1", "lw": 1.6, "label": r"$\lambda=0.01$ lattice"},
        0.1: {"color": "C3", "lw": 1.5, "label": r"$\lambda=0.1$ lattice"},
    }
    for lam, rows in results.items():
        Ts = np.array([row["T"] for row in rows])
        style = styles[lam]
        axes[0].plot(Ts, [row["lattice_K_re"] for row in rows], **style)
        axes[1].plot(Ts, [row["lattice_K_im"] for row in rows], **style)

    axes[0].set_ylabel(r"Re $K(0,0;T-i\eta)$")
    axes[0].set_title(
        rf"Lattice FFT: $\lambda=0$, $0.01$, $0.1$, $N={N}$, $\eta={eta}$, "
        rf"$T=0\to{T_MAX:g}$"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[0].axvline(np.pi, color="grey", ls=":", alpha=0.7, label=r"$T=\pi$")
    axes[0].axvline(2 * np.pi, color="grey", ls="--", alpha=0.7, label=r"$T=2\pi$")
    # redraw legend with caustic markers once
    handles, labels = axes[0].get_legend_handles_labels()
    # de-duplicate caustic entries that would otherwise appear twice
    seen = set()
    uniq = []
    for h, lab in zip(handles, labels):
        if lab in seen:
            continue
        seen.add(lab)
        uniq.append((h, lab))
    axes[0].legend([h for h, _ in uniq], [lab for _, lab in uniq])

    axes[1].axvline(np.pi, color="grey", ls=":", alpha=0.7)
    axes[1].axvline(2 * np.pi, color="grey", ls="--", alpha=0.7)
    axes[1].set_xlabel("T")
    axes[1].set_ylabel(r"Im $K(0,0;T-i\eta)$")
    axes[1].set_title(r"Im $K$")
    axes[1].grid(alpha=0.3)
    axes[1].legend([styles[lam]["label"] for lam in LAMBDAS])
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined plot to {plot_path}", flush=True)


if __name__ == "__main__":
    print(
        "STARTING exact_check_Time_Loop_lambda0_0p01_T0to8 "
        "(lattice FFT + continuum, not ML)",
        flush=True,
    )
    results = {}
    for lam in LAMBDAS:
        rows, _ = scan_lambda(lam)
        results[lam] = rows
    make_combined_plot(results)
    print("DONE", flush=True)
