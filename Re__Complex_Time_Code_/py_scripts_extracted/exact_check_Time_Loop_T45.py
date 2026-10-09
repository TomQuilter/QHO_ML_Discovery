"""Exact (non-ML) lattice FFT plus saddle diagnostic at T=4.5.

Copied from exact_check_Time_Loop_Extended.py and restricted to T=4.5
(past the first harmonic caustic at pi, before 2*pi).  Does not modify
the original Extended checker.

Same lattice as the ML notebook: N=16, eta=0.05, m=omega=1, lambda=0.5,
x_i=x_f=0, V = x^2/2 + (lambda/4) x^4.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.optimize import root


m, omega, eta, N = 1.0, 1.0, 0.05, 16
lam = 0.5
D = N - 1
T = 4.5
A_C = (T - 1j * eta) / N


def V(x, lam_val):
    return 0.5 * m * omega**2 * x**2 + 0.25 * lam_val * x**4


def fftconvolve(a, b, mode="valid"):
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    full = np.fft.ifft(np.fft.fft(a, nfft) * np.fft.fft(b, nfft))[:n]
    return full[len(b) - 1: len(a)]


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


def action_np(x: np.ndarray) -> complex:
    padded = np.r_[0.0 + 0.0j, x, 0.0 + 0.0j]
    dx = np.diff(padded)
    return (
        m / (2.0 * A_C) * np.sum(dx**2)
        - A_C * m * omega**2 / 2.0 * np.sum(x**2)
        - A_C * lam / 4.0 * np.sum(x**4)
    )


def gradient_np(x: np.ndarray) -> np.ndarray:
    padded = np.r_[0.0 + 0.0j, x, 0.0 + 0.0j]
    return (
        (m / A_C) * (2.0 * x - padded[:-2] - padded[2:])
        - A_C * m * omega**2 * x
        - A_C * lam * x**3
    )


def hessian_np(x: np.ndarray) -> np.ndarray:
    hessian = np.zeros((D, D), dtype=np.complex128)
    np.fill_diagonal(
        hessian,
        2.0 * m / A_C - A_C * m * omega**2 - 3.0 * A_C * lam * x**2,
    )
    np.fill_diagonal(hessian[1:], -m / A_C)
    np.fill_diagonal(hessian[:, 1:], -m / A_C)
    return hessian


def packed_gradient(y: np.ndarray) -> np.ndarray:
    x = y[:D] + 1j * y[D:]
    grad = gradient_np(x)
    return np.r_[grad.real, grad.imag]


def gaussian_saddle_estimate(saddle: np.ndarray) -> complex:
    hessian = hessian_np(saddle)
    norm = (m / (2.0 * np.pi * 1j * A_C)) ** (N / 2)
    return complex(
        norm * np.exp(1j * action_np(saddle))
        * (2.0 * np.pi * 1j) ** (D / 2) / np.sqrt(np.linalg.det(hessian))
    )


def find_bounce_harmonic(harmonic: int) -> list:
    sites = np.arange(1, N)
    shape = np.sin(harmonic * np.pi * sites / N)
    candidates = []
    for amplitude in np.linspace(0.3, 16.0, 40):
        guess = np.r_[amplitude * shape, np.zeros(D)]
        solved = root(packed_gradient, guess, method="lm", options={"maxiter": 5_000})
        x = solved.x[:D] + 1j * solved.x[D:]
        residual = np.max(np.abs(gradient_np(x)))
        if residual < 1e-8 and np.max(np.abs(x)) > 1e-3:
            if not any(np.max(np.abs(x - old)) < 1e-6 for old in candidates):
                if not any(np.max(np.abs(x + old)) < 1e-6 for old in candidates):
                    candidates.append(x)
    return candidates


def report_saddle(label: str, x: np.ndarray) -> dict:
    S = action_np(x)
    H = hessian_np(x)
    eigs = np.linalg.eigvals(H)
    abs_eigs = np.sort(np.abs(eigs))
    K_g = gaussian_saddle_estimate(x)
    row = {
        "label": label,
        "max_abs_x": float(np.max(np.abs(x))),
        "ImS": float(S.imag),
        "exp_minus_ImS": float(np.exp(-S.imag)),
        "min_abs_eig_H": float(abs_eigs[0]),
        "cond_abs_H": float(abs_eigs[-1] / max(abs_eigs[0], 1e-300)),
        "det_H": complex(np.linalg.det(H)),
        "K_gauss": K_g,
        "grad_res": float(np.max(np.abs(gradient_np(x)))),
    }
    print(
        f"{label:10s}  max|x|={row['max_abs_x']:.4f}  ImS={row['ImS']:+.6f}  "
        f"e^(-ImS)={row['exp_minus_ImS']:.3e}  min|eig H|={row['min_abs_eig_H']:.3e}  "
        f"cond={row['cond_abs_H']:.3e}  K_gauss={K_g:+.8f}",
        flush=True,
    )
    return row


def main():
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "exact_check_Time_Loop_T45_lambda0.5.csv"

    print("STARTING exact_check_Time_Loop_T45 (lattice FFT, not ML)", flush=True)
    print(f"lambda={lam}, N={N}, eta={eta}, T={T}")
    print(f"distance to pi={abs(T - np.pi):.3f}, to 2pi={abs(T - 2 * np.pi):.3f}")
    print(f"CSV: {out_path}", flush=True)

    Tc = T - 1j * eta
    ac = Tc / N
    Kf = lattice_K_fft(lam, ac)
    E, psi0 = continuum_eigenpairs(lam)
    Kc = continuum_K_from_spectrum(E, psi0, Tc)
    rel = abs(Kf - Kc) / abs(Kc) if abs(Kc) else float("nan")
    print(f"T={T:.2f}  lattice={Kf:+.10f}  continuum={Kc:+.10f}  rel={rel:.3e}", flush=True)

    fieldnames = [
        "lambda", "N", "T", "eta", "T_c_re", "T_c_im",
        "lattice_K_re", "lattice_K_im", "lattice_abs_K",
        "continuum_K_re", "continuum_K_im", "continuum_abs_K",
        "lattice_vs_continuum_rel_diff",
    ]
    row = {
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
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    print(f"saved {out_path}", flush=True)

    print("\n*** SADDLE DIAGNOSTIC (no NN) ***", flush=True)
    trivial = np.zeros(D, dtype=np.complex128)
    s0 = report_saddle("trivial", trivial)

    pool = []
    for harmonic in (1, 2, 3, 4):
        found = find_bounce_harmonic(harmonic)
        print(f"harmonic {harmonic}: {len(found)} distinct nonzero saddles", flush=True)
        pool.extend(found)

    distinct = []
    for x in pool:
        if any(np.max(np.abs(x - old)) < 1e-5 or np.max(np.abs(x + old)) < 1e-5 for old in distinct):
            continue
        distinct.append(x)
    distinct.sort(key=lambda x: action_np(x).imag)
    print(f"unique nonzero saddles (mod parity): {len(distinct)}", flush=True)

    saddles = [("trivial", trivial)]
    for i, x in enumerate(distinct, start=1):
        saddles.append((f"bounce{i}", x))
        report_saddle(f"bounce{i}", x)

    K0 = s0["K_gauss"]
    print(f"\nGaussian K0 = {K0:+.10f}", flush=True)
    if len(distinct) >= 1:
        Kb1 = gaussian_saddle_estimate(distinct[0])
        K3 = K0 + 2.0 * Kb1
        err3 = abs(K3 - Kf) / abs(Kf)
        print(f"Gaussian K0+2 b1 = {K3:+.10f}  vs lattice err={100 * err3:.3f}%", flush=True)
    if len(distinct) >= 2:
        Kb2 = gaussian_saddle_estimate(distinct[1])
        K5 = K3 + 2.0 * Kb2
        err5 = abs(K5 - Kf) / abs(Kf)
        print(f"Gaussian K0+2 b1+2 b2 = {K5:+.10f}  vs lattice err={100 * err5:.3f}%", flush=True)
    print(
        "Note: Gaussian is only a softness/suppression diagnostic; at T=2 it was a bad K predictor.",
        flush=True,
    )

    soft = [name for name, x in saddles if np.min(np.abs(np.linalg.eigvals(hessian_np(x)))) < 1e-3]
    if soft:
        print(f"SOFT HESSIAN (min|eig|<1e-3): {soft}  -- extra nets will not fix these.", flush=True)
    else:
        print("No Hessian with min|eig|<1e-3.  ML on this T is worth trying.", flush=True)


if __name__ == "__main__":
    main()
