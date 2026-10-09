"""Exact (non-ML) cross-checks for Tom's complex-time thimble notebook.

Three independent evaluations of K(0, 0; T_c = T - i*eta):

1. LATTICE EXACT (transfer matrix / FFT convolution):
   Evaluates *exactly the same* discretized path integral as the notebook
   (same N, same complex a_c, potential on interior slices only, same
   normalisation (m/(2*pi*i*a_c))^(N/2)) by deterministic quadrature.
   Any difference between this and Tom's ML number is ML/contour error.

2. LATTICE EXACT (Gaussian determinant), lambda = 0 only:
   Closed-form for the discretized harmonic integral. Validates method 1.

3. CONTINUUM SPECTRAL: diagonalize H = p^2/2m + m w^2 x^2/2 + lam x^4 / 4
   (Fourier spectral), K = sum_n |psi_n(0)|^2 exp(-i E_n T_c).
   Difference between 1 and 3 is pure discretization error (what limits
   agreement with a continuum benchmark even if the ML part is perfect).
"""
import numpy as np


def fftconvolve(a, b, mode="valid"):
    """numpy-only linear convolution via FFT, 'valid' mode, len(a) >= len(b)."""
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    full = np.fft.ifft(np.fft.fft(a, nfft) * np.fft.fft(b, nfft))[:n]
    return full[len(b) - 1: len(a)]

m, omega, T, eta, N = 1.0, 1.0, 1.0, 0.05, 16
Tc = T - 1j * eta 
ac = Tc / N
D = N - 1


def V(x, lam):
    return 0.5 * m * omega**2 * x**2 + 0.25 * lam * x**4


# ------------------------------------------------------------------
# 1. Transfer matrix on a grid (FFT convolution), matching Tom's action
# ------------------------------------------------------------------
def lattice_K_fft(lam, L=10.0, M=2**15):
    x = np.linspace(-L, L, M)
    h = x[1] - x[0]
    # link kernel with per-link normalisation sqrt(m/(2 pi i ac))
    pref = np.sqrt(m / (2 * np.pi * 1j * ac))
    u = np.linspace(-2 * L, 2 * L, 2 * M - 1)
    link = pref * np.exp(1j * m * u**2 / (2 * ac))
    xi, xf = 0.0, 0.0
    f = pref * np.exp(1j * m * (x - xi)**2 / (2 * ac))  # after first link
    w = np.exp(-1j * ac * V(x, lam))                    # interior potential
    for _ in range(N - 1):
        f = h * fftconvolve(link, w * f, mode="valid")  # length M again
    # f is now K(x_f = grid point); interpolate at xf = 0
    return np.interp(xf, x, f.real) + 1j * np.interp(xf, x, f.imag)


# ------------------------------------------------------------------
# 2. Exact lattice Gaussian (lambda = 0)
# ------------------------------------------------------------------
def lattice_K_gauss():
    Hm = np.zeros((D, D), dtype=complex)
    dv = 2 * m / ac - ac * m * omega**2
    ov = -m / ac
    for j in range(D):
        Hm[j, j] = dv
        if j > 0:
            Hm[j, j - 1] = ov
        if j < D - 1:
            Hm[j, j + 1] = ov
    d = np.linalg.eigvals(Hm)
    # each mode: int e^{i d z^2 / 2} dz = sqrt(2 pi i / d), principal branch
    gauss = np.prod(np.sqrt(2 * np.pi * 1j / d))
    return (m / (2 * np.pi * 1j * ac))**(N / 2) * gauss


# ------------------------------------------------------------------
# 3. Continuum spectral sum (Fourier spectral Hamiltonian)
# ------------------------------------------------------------------
def continuum_K(lam, L=14.0, Mg=1500, nmax=600):
    x = np.linspace(-L, L, Mg, endpoint=False)
    h = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(Mg, d=h)
    F = np.fft.fft(np.eye(Mg), axis=0)
    Kin = np.real(np.fft.ifft((k**2)[:, None] * F, axis=0)) / (2 * m)
    H = Kin + np.diag(V(x, lam))
    E, psi = np.linalg.eigh(H)
    psi = psi / np.sqrt(h)          # normalize: sum |psi|^2 h = 1
    i0 = np.argmin(np.abs(x))       # x = 0 (grid contains 0 since endpoint=False, symmetric? check)
    # interpolate psi at exactly 0 to be safe
    psi0 = np.array([np.interp(0.0, x, psi[:, n]) for n in range(nmax)])
    w = np.exp(-1j * E[:nmax] * Tc)
    return np.sum(psi0**2 * w), E[:5]


def mehler():
    wT = omega * Tc
    return np.sqrt(m * omega / (2 * np.pi * 1j * np.sin(wT))) * np.exp(0j)


print(f"T_c = {Tc},  a_c = {ac},  N = {N}, D = {D}\n")

print("=== lambda = 0 validation ===")
Kg = lattice_K_gauss()
Kf0 = lattice_K_fft(0.0)
Kc0, E0 = continuum_K(0.0)
Km = mehler()
print(f"lattice exact (det)   : {Kg:.10f}   |K| = {abs(Kg):.8f}")
print(f"lattice exact (FFT)   : {Kf0:.10f}   |K| = {abs(Kf0):.8f}")
print(f"continuum spectral    : {Kc0:.10f}   |K| = {abs(Kc0):.8f}")
print(f"continuum Mehler      : {Km:.10f}   |K| = {abs(Km):.8f}")
print(f"first HO eigenvalues  : {E0}")
print(f"lattice-vs-continuum rel. diff (discretization error at N=32): "
      f"{abs(Kg - Km) / abs(Km):.3e}\n")

for lam in [4]:
    Kf = lattice_K_fft(lam)
    Kc, _ = continuum_K(lam)
    print(f"=== lambda = {lam} ===")
    print(f"lattice exact (FFT)   : {Kf:.10f}   |K| = {abs(Kf):.8f}")
    print(f"continuum spectral    : {Kc:.10f}   |K| = {abs(Kc):.8f}")
    print(f"lattice-vs-continuum rel. diff: {abs(Kf - Kc)/abs(Kc):.3e}")
    if lam == 0.5:
        Ktom = 0.3170553726 - 0.2982940369j
        print(f"Tom ML (seed 42)      : {Ktom:.10f}   |K| = {abs(Ktom):.8f}")
        print(f"Tom-vs-lattice-exact rel. diff: {abs(Ktom - Kf)/abs(Kf):.3e}")
    print()
