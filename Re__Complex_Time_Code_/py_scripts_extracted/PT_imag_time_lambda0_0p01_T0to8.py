"""T-scan from the IT/PT notebook methods: lambda=0 and 0.01, T=0..8 step 0.01.

Copied logic from PT_propagator_imag_time_Latest_01_Oct_2026.ipynb
(propagator_ho_Tscan / Mehler). Original notebook is not modified.
This is NOT the lattice-FFT exact_check_Time_Loop script.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh


OMEGA = 1.0
ETA = 0.05
N_BASIS = 500
LAMBDAS = [0.0, 0.01]
dt = 0.01
T_MAX = 8.0
time_list = np.array([round(i * dt, 2) for i in range(int(round(T_MAX / dt)) + 1)])

out_dir = Path(__file__).resolve().parent
plot_path = out_dir / "PT_imag_time_lambda0_0p01_T0to8_ReIm.png"


def K0_exact(xf, xi, T, omega=OMEGA, eta=ETA):
    T_c = T - 1j * eta
    wT = omega * T_c
    sin_wT = np.sin(wT)
    cos_wT = np.cos(wT)
    prefactor = np.sqrt(omega / (2 * np.pi * 1j * sin_wT))
    exponent = 1j * omega / (2 * sin_wT) * ((xf**2 + xi**2) * cos_wT - 2 * xf * xi)
    return prefactor * np.exp(exponent)


def ho_wfn_array(x, omega=OMEGA, N=N_BASIS):
    xi = np.sqrt(omega) * x
    phi = np.zeros(N)
    phi[0] = (omega / np.pi) ** 0.25 * np.exp(-omega * x**2 / 2.0)
    if N > 1:
        phi[1] = np.sqrt(2.0) * xi * phi[0]
    for n in range(1, N - 1):
        phi[n + 1] = np.sqrt(2.0 / (n + 1)) * xi * phi[n] - np.sqrt(n / (n + 1)) * phi[n - 1]
    return phi


def x4_matrix_elements(N, omega=OMEGA):
    mat = np.zeros((N, N))
    inv4w2 = 1.0 / (4.0 * omega**2)
    for n in range(N):
        mat[n, n] = (6 * n * n + 6 * n + 3) * inv4w2
        if n + 2 < N:
            val = 2 * (2 * n + 3) * np.sqrt((n + 1) * (n + 2)) * inv4w2
            mat[n, n + 2] = val
            mat[n + 2, n] = val
        if n + 4 < N:
            val = np.sqrt((n + 1) * (n + 2) * (n + 3) * (n + 4)) * inv4w2
            mat[n, n + 4] = val
            mat[n + 4, n] = val
    return mat


def propagator_ho_Tscan(T_array, lam, omega=OMEGA, eta=ETA, N=None):
    """Same HO-basis T-scan as the IT notebook."""
    if N is None:
        N = max(N_BASIS, int(np.ceil(23.0 / (eta * omega))) + 50)
        N = min(N, 3000)
    T_arr = np.asarray(T_array, dtype=float)
    E0_diag = omega * (np.arange(N) + 0.5)
    H_mat = np.diag(E0_diag) + (lam / 4.0) * x4_matrix_elements(N, omega)
    eigenvalues, U = eigh(H_mat)
    phi_0 = ho_wfn_array(0.0, omega=omega, N=N)
    psi_0 = U.T @ phi_0
    weights = psi_0**2
    T_c = T_arr - 1j * eta
    return np.exp(-1j * eigenvalues[np.newaxis, :] * T_c[:, np.newaxis]) @ weights


def lam_tag(lam: float) -> str:
    if lam == 0.0:
        return "0"
    return f"{lam:g}".replace(".", "p")


def scan_lambda(lam: float):
    out_path = out_dir / f"PT_imag_time_lambda{lam_tag(lam)}_T0to8_dt0.01.csv"
    print("=" * 72, flush=True)
    print(f"STARTING IT/HO T-scan  lambda={lam}, eta={ETA}", flush=True)
    print(f"T = 0 .. {T_MAX} step {dt}  ({len(time_list)} points)", flush=True)
    N = max(N_BASIS, int(np.ceil(23.0 / (ETA * OMEGA))) + 50)
    N = min(N, 3000)
    print(f"HO basis N={N}", flush=True)
    K = propagator_ho_Tscan(time_list, lam, eta=ETA, N=N)
    mehler = np.array([K0_exact(0.0, 0.0, T) for T in time_list])
    rows = []
    fieldnames = [
        "lambda",
        "T",
        "eta",
        "K_re",
        "K_im",
        "abs_K",
        "mehler_re",
        "mehler_im",
        "abs_mehler",
        "vs_mehler_rel",
    ]
    for T, Kv, Km in zip(time_list, K, mehler):
        rel = abs(Kv - Km) / abs(Km) if abs(Km) else float("nan")
        rows.append(
            {
                "lambda": lam,
                "T": float(T),
                "eta": ETA,
                "K_re": Kv.real,
                "K_im": Kv.imag,
                "abs_K": abs(Kv),
                "mehler_re": Km.real,
                "mehler_im": Km.imag,
                "abs_mehler": abs(Km),
                "vs_mehler_rel": rel,
            }
        )
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {out_path}", flush=True)
    # spot checks
    for Tshow in (0.0, 1.0, np.pi, 2 * np.pi, 8.0):
        idx = int(round(Tshow / dt))
        idx = min(idx, len(time_list) - 1)
        print(
            f"  T={time_list[idx]:5.2f}  K={K[idx]:+.8f}  Mehler={mehler[idx]:+.8f}  "
            f"rel={abs(K[idx]-mehler[idx])/abs(mehler[idx]):.3e}",
            flush=True,
        )
    return rows, K


def make_plot(results):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    styles = {
        0.0: {"color": "C0", "lw": 1.8, "label": r"$\lambda=0$ HO exact"},
        0.01: {"color": "C1", "lw": 1.6, "label": r"$\lambda=0.01$ HO exact"},
    }
    for lam, rows in results.items():
        Ts = np.array([r["T"] for r in rows])
        style = styles[lam]
        axes[0].plot(Ts, [r["K_re"] for r in rows], **style)
        axes[1].plot(Ts, [r["K_im"] for r in rows], **style)
    # Mehler guide (exact for lambda=0 continuum)
    mehler = np.array([K0_exact(0.0, 0.0, T) for T in time_list])
    axes[0].plot(time_list, mehler.real, "k--", lw=1.0, alpha=0.55, label="Mehler")
    axes[1].plot(time_list, mehler.imag, "k--", lw=1.0, alpha=0.55, label="Mehler")

    for ax in axes:
        ax.axvline(np.pi, color="grey", ls=":", alpha=0.7)
        ax.axvline(2 * np.pi, color="grey", ls="--", alpha=0.7)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"Re $K(0,0;T-i\eta)$")
    axes[0].set_title(
        rf"IT notebook HO spectral: $\lambda=0$ and $0.01$, $\eta={ETA}$, $T=0\to{T_MAX:g}$"
    )
    axes[0].legend()
    axes[1].set_xlabel("T")
    axes[1].set_ylabel(r"Im $K(0,0;T-i\eta)$")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined plot to {plot_path}", flush=True)


if __name__ == "__main__":
    print(
        "STARTING PT/IT T-scan copy (HO spectral from imag-time notebook, not lattice FFT)",
        flush=True,
    )
    results = {}
    for lam in LAMBDAS:
        rows, _ = scan_lambda(lam)
        results[lam] = rows
    make_plot(results)
    print("DONE", flush=True)
