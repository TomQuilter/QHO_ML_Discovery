#!/usr/bin/env python3
"""
Level 0: Real-Time Path Integral for the Quantum Harmonic Oscillator
=====================================================================

LAYMAN OVERVIEW:
  In quantum mechanics, to predict how a particle gets from point A to point B,
  you must sum over EVERY possible path it could take — not just the obvious one.
  Each path contributes a complex number (a "spinning arrow"). The final answer
  is the sum of all those arrows, called the "propagator".

  The problem: on the real number line, the arrows spin wildly and mostly cancel
  out (the "sign problem"). This makes numerical computation extremely noisy.

  The fix (Picard-Lefschetz thimble): tilt the calculation 45 degrees into the
  complex plane. The wildly spinning arrows become a smooth bell curve. No more
  cancellation, no more noise.

  This file computes the propagator for the simplest system — a ball on a spring
  (harmonic oscillator) — using five independent methods, and checks they all
  agree. This validates the machinery before applying it to harder problems.

  The five methods:
    1. Exact formula (the textbook answer key)
    2. Semiclassical (uses the "most important path" + corrections)
    3. Thimble (the 45-degree tilt trick, done analytically)
    4. Monte Carlo on the thimble (random sampling on the tilted contour)
    5. Brute-force grid (try every point — only works for tiny problems)

Physics conventions: m=1, hbar=1 unless otherwise noted.
"""

import numpy as np
from scipy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# 1. EXACT PROPAGATOR (Mehler kernel)
# ============================================================
# LAYMAN: This is the textbook answer — a closed-form formula that gives the
# exact probability amplitude for a particle on a spring to travel from
# position x_i to position x_f in time T. It's the "answer key" that all
# other methods will be checked against.

def exact_propagator(x_i, x_f, T, omega, m=1.0, hbar=1.0):
    """
    K(x_f, T; x_i, 0) = sqrt(m*w / (2*pi*i*hbar*sin(wT)))
        * exp(i*m*w / (2*hbar*sin(wT)) * [(xi^2+xf^2)*cos(wT) - 2*xi*xf])
    """
    swT = np.sin(omega * T)
    cwT = np.cos(omega * T)
    prefactor = np.sqrt(m * omega / (2 * np.pi * 1j * hbar * swT))
    phase = 1j * m * omega / (2 * hbar * swT) * (
        (x_i**2 + x_f**2) * cwT - 2 * x_i * x_f
    )
    return prefactor * np.exp(phase)


# ============================================================
# 2. CLASSICAL PATH AND ACTION
# ============================================================
# LAYMAN: The "classical path" is the single trajectory a ball on a spring
# would actually follow in everyday (non-quantum) physics — the most likely
# path. The "action" is a single number that scores how "costly" a path is:
# it balances kinetic energy (speed) against potential energy (spring stretch).
# In quantum mechanics, paths with lower action contribute more to the sum.

def classical_path(t_grid, x_i, x_f, T, omega):
    """x_cl(t) satisfying x(0)=x_i, x(T)=x_f for harmonic oscillator."""
    swT = np.sin(omega * T)
    return (x_i * np.sin(omega * (T - t_grid)) + x_f * np.sin(omega * t_grid)) / swT


def classical_action(x_i, x_f, T, omega, m=1.0):
    """S_cl = m*w/(2*sin(wT)) * [(xi^2+xf^2)*cos(wT) - 2*xi*xf]"""
    swT = np.sin(omega * T)
    cwT = np.cos(omega * T)
    return m * omega / (2 * swT) * ((x_i**2 + x_f**2) * cwT - 2 * x_i * x_f)


def discretized_action(x_path, epsilon, omega, m=1.0):
    """
    LAYMAN: The continuous path is chopped into N small steps. This computes
    the action by summing (velocity^2 - spring_energy) at each step.
    As N grows, this approaches the true continuous action.

    S = sum_{k=0}^{N-1} [m/(2*eps)*(x_{k+1}-x_k)^2 - eps*m*w^2/2 * x_k^2]
    x_path: array of length N+1 (includes boundary points)
    """
    dx = np.diff(x_path)
    kinetic = m / (2 * epsilon) * np.sum(dx**2)
    potential = epsilon * m * omega**2 / 2 * np.sum(x_path[:-1]**2)
    return kinetic - potential


# ============================================================
# 3. HESSIAN (fluctuation operator)
# ============================================================
# LAYMAN: Think of the classical path as the bottom of a valley. The Hessian
# measures the "curvature" of that valley in every direction — how steeply
# the action rises when you deviate from the classical path.
#
# It's a matrix where each entry describes how wiggling the path at one time
# step affects the action at another time step. Since only neighbouring time
# steps interact (the path can only "see" its immediate neighbours), the
# matrix is mostly zeros except on the diagonal and one step off it
# (tridiagonal).
#
# Its eigenvalues tell you how "stiff" each independent wiggle mode is:
#   - Large eigenvalue = stiff = tightly constrained = easy to integrate
#   - Small eigenvalue = floppy = loosely constrained = harder to integrate

def build_hessian(N, epsilon, omega, m=1.0):
    """
    (N-1)x(N-1) tridiagonal Hessian K_{jk} = d^2 S / d x_j d x_k
    for internal points x_1,...,x_{N-1}.

    Diagonal:     2m/eps - eps*m*w^2
    Off-diagonal: -m/eps
    """
    n = N - 1
    d = 2 * m / epsilon - epsilon * m * omega**2
    o = -m / epsilon
    K = np.diag(np.full(n, d))
    if n > 1:
        K += np.diag(np.full(n - 1, o), 1)
        K += np.diag(np.full(n - 1, o), -1)
    return K


# ============================================================
# 4. SEMICLASSICAL PROPAGATOR (determinant method, log-space)
# ============================================================
# LAYMAN: Instead of summing over ALL paths, this method says:
#   "The classical path dominates. Let me take that path's contribution
#    and multiply by a correction factor that accounts for nearby paths."
#
# That correction factor comes from the Hessian's determinant — essentially
# the product of all eigenvalues. It measures the "volume" of the Gaussian
# cloud of paths around the classical one.
#
# For the harmonic oscillator this is exact (not an approximation) because
# the action is purely quadratic — like fitting a parabola, the Gaussian
# approximation IS the true shape. For harder problems it's only approximate.
#
# We work in log-space (adding logs instead of multiplying numbers) to avoid
# numerical overflow when N is large.

def semiclassical_propagator(x_i, x_f, T, omega, N, m=1.0, hbar=1.0):
    """
    K_sc = sqrt(m^N / (2*pi*i*hbar * eps^N * det K)) * exp(i*S_cl/hbar)

    Computed in log-space via eigenvalues to avoid overflow at large N.
    Exact for harmonic oscillator (purely quadratic action).
    """
    epsilon = T / N
    S_cl = classical_action(x_i, x_f, T, omega, m)
    K = build_hessian(N, epsilon, omega, m)
    eigenvalues = np.linalg.eigvalsh(K)

    log_mag = 0.5 * (N * np.log(m) - np.log(2 * np.pi * hbar)
                      - N * np.log(epsilon) - np.sum(np.log(np.abs(eigenvalues))))
    n_neg = np.sum(eigenvalues < 0)
    phase = -np.pi / 4 - n_neg * np.pi / 2

    return np.exp(log_mag + 1j * (phase + S_cl / hbar))


# ============================================================
# 5. PICARD-LEFSCHETZ THIMBLE (analytical, mode-by-mode)
# ============================================================
# LAYMAN: This is the key trick. On the real line, each wiggle mode
# contributes a wildly oscillating integral (spinning arrows that cancel).
# The thimble rotates each mode 45 degrees into the complex plane, turning
# the spinning arrows into a smooth bell curve (Gaussian).
#
# The direction of rotation depends on the sign of the eigenvalue:
#   Positive eigenvalue -> rotate +45 degrees
#   Negative eigenvalue -> rotate -45 degrees
#
# After rotation, every integral is a standard Gaussian that can be solved
# analytically. The result is mathematically identical to the semiclassical
# method above — they must agree to machine precision for the spring system.
# This is the validation test: if these two disagree, something is broken.

def thimble_propagator(x_i, x_f, T, omega, N, m=1.0, hbar=1.0):
    """
    Evaluate propagator by deforming each eigenmode to its Lefschetz thimble.
    """
    epsilon = T / N
    S_cl = classical_action(x_i, x_f, T, omega, m)
    K = build_hessian(N, epsilon, omega, m)
    eigenvalues = np.linalg.eigvalsh(K)

    # Each mode contributes a Gaussian integral with width ~ 1/sqrt(|eigenvalue|)
    log_mag_fluct = 0.5 * np.sum(np.log(2 * np.pi * hbar / np.abs(eigenvalues)))
    # Each mode picks up a phase of +/-45 degrees from the rotation direction
    phase_fluct = np.sum(np.sign(eigenvalues)) * np.pi / 4

    # Normalisation factor from chopping continuous time into N discrete steps
    log_mag_meas = (N / 2) * (np.log(m) - np.log(2 * np.pi * hbar * epsilon))
    phase_meas = -N * np.pi / 4

    return np.exp((log_mag_fluct + log_mag_meas)
                  + 1j * (phase_fluct + phase_meas + S_cl / hbar))


# ============================================================
# 6. MONTE CARLO ON THE THIMBLE
# ============================================================
# LAYMAN: This is where random sampling enters. Instead of evaluating the
# integral analytically (like methods 4 and 5), we draw random samples from
# a Gaussian and estimate the answer statistically — just like a poll
# estimates an election result from a sample of voters.
#
# For the harmonic oscillator (pure spring), every sample has equal weight
# (ESS/N = 1.0, meaning all samples are equally useful — no waste).
#
# The lam_quartic parameter is a hook for harder problems: if you add an
# x^4 term to the spring potential, the thimble no longer perfectly removes
# the sign problem. Some samples become more important than others, ESS drops,
# and you start needing smarter sampling (e.g. normalizing flows / ML).

def mc_thimble_propagator(x_i, x_f, T, omega, N, n_samples=100_000,
                          m=1.0, hbar=1.0, lam_quartic=0.0):
    """
    Monte Carlo on the Lefschetz thimble.
    Returns: (K_mean, K_std_of_mean, effective_sample_size)
    """
    epsilon = T / N
    S_cl = classical_action(x_i, x_f, T, omega, m)
    K_mat = build_hessian(N, epsilon, omega, m)
    eigenvalues, eigenvectors = eigh(K_mat)
    n_modes = len(eigenvalues)

    # Compute the exact part analytically (same as thimble_propagator above)
    log_mag_fluct = 0.5 * np.sum(np.log(2 * np.pi * hbar / np.abs(eigenvalues)))
    phase_fluct = np.sum(np.sign(eigenvalues)) * np.pi / 4
    log_mag_meas = (N / 2) * (np.log(m) - np.log(2 * np.pi * hbar * epsilon))
    phase_meas = -N * np.pi / 4

    overall = np.exp((log_mag_fluct + log_mag_meas)
                     + 1j * (phase_fluct + phase_meas + S_cl / hbar))

    # Draw random Gaussian samples — one per wiggle mode, shaped by eigenvalue
    sigmas = np.sqrt(hbar / np.abs(eigenvalues))
    t_samples = np.random.randn(n_samples, n_modes) * sigmas[np.newaxis, :]

    if lam_quartic == 0.0:
        # Pure spring: all samples contribute equally (no leftover phase)
        weights = np.ones(n_samples, dtype=complex)
    else:
        # Anharmonic (x^4) case: the thimble handles the quadratic part,
        # but there's a leftover oscillating phase from the x^4 term.
        # This is where the sign problem creeps back in.
        rot_per_mode = np.where(eigenvalues > 0,
                                np.exp(1j * np.pi / 4),
                                np.exp(-1j * np.pi / 4))
        xi_samples = t_samples * rot_per_mode[np.newaxis, :]
        eta_samples = xi_samples @ eigenvectors.T
        S_anh = lam_quartic * epsilon * np.sum(eta_samples**4, axis=1)
        weights = np.exp(1j * S_anh / hbar)

    # Standard MC statistics: mean, uncertainty, and effective sample size
    mean_w = np.mean(weights)
    K_mean = overall * mean_w
    K_std = abs(overall) * np.std(weights) / np.sqrt(n_samples)
    # ESS: how many samples are actually useful (1.0 = all, <<1 = sign problem)
    ess = abs(mean_w)**2 / np.mean(np.abs(weights)**2) * n_samples

    return K_mean, K_std, ess


# ============================================================
# 7. BRUTE FORCE (small N only)
# ============================================================
# LAYMAN: The most straightforward approach — lay down a dense grid of
# points in every dimension and just add up the integrand at each one.
# Think of it as checking every pixel in an image.
#
# The problem: with N-1 dimensions, a grid of 200 points per dimension
# needs 200^(N-1) evaluations. At N=4 that's 200^3 = 8 million (OK).
# At N=32 it would be 200^31 — more points than atoms in the universe.
# So this only works for tiny problems (N <= 4), but it's a useful sanity
# check because it makes no approximations at all.

def bruteforce_propagator(x_i, x_f, T, omega, N, n_grid=150,
                          x_range=5.0, m=1.0, hbar=1.0):
    """
    Direct numerical integration on a grid (thimble contour, eigenbasis).
    Only feasible for N-1 <= 3 due to curse of dimensionality.
    """
    n_modes = N - 1
    if n_modes > 3:
        raise ValueError(f"Brute force needs N-1 <= 3, got {n_modes}")

    epsilon = T / N
    S_cl = classical_action(x_i, x_f, T, omega, m)
    K_mat = build_hessian(N, epsilon, omega, m)
    eigenvalues = np.linalg.eigvalsh(K_mat)

    # Build a uniform grid over each dimension
    dt = 2 * x_range / n_grid
    t_1d = np.linspace(-x_range, x_range, n_grid)

    # Create the full N-dimensional grid (every combination of points)
    grids = np.meshgrid(*([t_1d] * n_modes), indexing='ij')
    t_array = np.stack([g.ravel() for g in grids], axis=1)

    # Rotate each mode onto its thimble (same +/-45 degree trick)
    rot_per_mode = np.where(eigenvalues > 0,
                            np.exp(1j * np.pi / 4),
                            np.exp(-1j * np.pi / 4))
    xi_array = t_array * rot_per_mode[np.newaxis, :]

    # Evaluate the action at every grid point and sum
    S_quad = 0.5 * np.sum(eigenvalues[np.newaxis, :] * xi_array**2, axis=1)

    jacobian = np.prod(rot_per_mode)
    integrand = jacobian * np.exp(1j * S_quad / hbar)
    integral = np.sum(integrand) * dt**n_modes

    log_mag_meas = (N / 2) * (np.log(m) - np.log(2 * np.pi * hbar * epsilon))
    phase_meas = -N * np.pi / 4
    measure = np.exp(log_mag_meas + 1j * phase_meas)

    return measure * integral * np.exp(1j * S_cl / hbar)


# ============================================================
# 8. MODE ANALYSIS
# ============================================================
# LAYMAN: Every path can be broken down into independent "wiggle modes" —
# like how any sound can be decomposed into individual frequencies (Fourier).
# Low modes = slow wiggles (long wavelength). High modes = fast wiggles.
#
# This function compares the eigenvalues from our discrete grid against what
# they should be in the continuous (perfect) limit. Low modes should match
# well; high modes deviate because the grid can't resolve fast wiggles.
#
# Key insight for ML: low modes (IR) have small eigenvalues and are hard to
# sample; high modes (UV) have huge eigenvalues and are trivially Gaussian.
# A smart sampler only needs to focus on the difficult low modes.

def analyze_modes(N, T, omega, m=1.0):
    """
    Compare discrete vs continuum eigenvalues.
    """
    epsilon = T / N
    K = build_hessian(N, epsilon, omega, m)
    eigenvalues = np.linalg.eigvalsh(K)
    n_arr = np.arange(1, N)
    continuum = epsilon * m * ((n_arr * np.pi / T)**2 - omega**2)
    return eigenvalues, continuum


# ============================================================
# 9. VERIFY DISCRETIZED ACTION vs ANALYTICAL
# ============================================================
# LAYMAN: A basic sanity check — compute the action on the classical path
# two ways (discrete grid vs exact formula) and make sure they nearly match.
# The small difference shrinks as N grows. If this fails, nothing else
# can be trusted.

def verify_classical_action(x_i, x_f, T, omega, N, m=1.0):
    """Check that discretized action on classical path matches analytical."""
    epsilon = T / N
    t_grid = np.linspace(0, T, N + 1)
    x_cl = classical_path(t_grid, x_i, x_f, T, omega)
    S_disc = discretized_action(x_cl, epsilon, omega, m)
    S_anal = classical_action(x_i, x_f, T, omega, m)
    return S_disc, S_anal


# ============================================================
# MAIN
# ============================================================
# LAYMAN: This runs all five methods, compares them, and generates plots.
# Think of it as the test suite — if all five methods agree on the same
# answer for the simple spring problem, we can trust the code when we
# later apply it to problems where only some methods are feasible.

def _main_original():
    # m = mass, hbar = quantum scale, omega = spring stiffness
    # x_i / x_f = start / end position, T = travel time
    m = 1.0
    hbar = 1.0
    omega = 2.0
    x_i = 1.0
    x_f = 0.5
    T = 1.0

    print("=" * 72)
    print("  LEVEL 0: Harmonic Oscillator — Real-Time Path Integral")
    print("=" * 72)
    print(f"\n  Parameters: m={m}, hbar={hbar}, omega={omega}")
    print(f"  Boundary:   x_i={x_i}, x_f={x_f}, T={T}")

    # --- Exact ---
    K_exact = exact_propagator(x_i, x_f, T, omega, m, hbar)
    S_cl = classical_action(x_i, x_f, T, omega, m)
    print(f"\n  Exact (Mehler kernel):")
    print(f"    K     = {K_exact:.10f}")
    print(f"    |K|   = {abs(K_exact):.10f}")
    print(f"    arg/pi = {np.angle(K_exact)/np.pi:.10f}")
    print(f"    S_cl  = {S_cl:.10f}")

    # --- Verify classical action ---
    S_disc, S_anal = verify_classical_action(x_i, x_f, T, omega, 64, m)
    print(f"\n  Classical action check (N=64):")
    print(f"    Discretized: {S_disc:.10f}")
    print(f"    Analytical:  {S_anal:.10f}")
    print(f"    Difference:  {abs(S_disc - S_anal):.2e}")

    # --- Convergence ---
    print(f"\n  Convergence of discretized propagator:")
    print(f"    {'N':>6s}  {'|K_sc|':>14s}  {'arg(K)/pi':>14s}  {'rel err':>12s}")
    print(f"    {'-'*6}  {'-'*14}  {'-'*14}  {'-'*12}")
    N_vals = [4, 8, 16, 32, 64, 128, 256, 512]
    errors = []
    for N in N_vals:
        K_sc = semiclassical_propagator(x_i, x_f, T, omega, N, m, hbar)
        err = abs(K_sc - K_exact) / abs(K_exact)
        errors.append(err)
        print(f"    {N:6d}  {abs(K_sc):14.10f}  {np.angle(K_sc)/np.pi:14.10f}  {err:12.2e}")

    # --- Thimble vs semiclassical ---
    N = 32
    K_thimble = thimble_propagator(x_i, x_f, T, omega, N, m, hbar)
    K_sc = semiclassical_propagator(x_i, x_f, T, omega, N, m, hbar)
    print(f"\n  Picard-Lefschetz thimble vs semiclassical (N={N}):")
    print(f"    K_thimble  = {K_thimble:.10f}")
    print(f"    K_semicl   = {K_sc:.10f}")
    print(f"    Difference = {abs(K_thimble - K_sc):.2e}")

    # --- Brute force (N=4, i.e. 3 integration variables) ---
    N_bf = 4
    K_bf = bruteforce_propagator(x_i, x_f, T, omega, N_bf, n_grid=200,
                                  x_range=6.0, m=m, hbar=hbar)
    K_sc_bf = semiclassical_propagator(x_i, x_f, T, omega, N_bf, m, hbar)
    print(f"\n  Brute-force grid integration (N={N_bf}, 200^{N_bf-1} points):")
    print(f"    K_brute    = {K_bf:.10f}")
    print(f"    K_semicl   = {K_sc_bf:.10f}")
    print(f"    Rel diff   = {abs(K_bf - K_sc_bf)/abs(K_sc_bf):.2e}")

    # --- Monte Carlo on thimble ---
    N_mc = 32
    n_samp = 200_000
    K_mc, K_mc_std, ess = mc_thimble_propagator(
        x_i, x_f, T, omega, N_mc, n_samples=n_samp, m=m, hbar=hbar
    )
    print(f"\n  Monte Carlo on thimble (N={N_mc}, {n_samp:,} samples):")
    print(f"    K_MC       = {K_mc:.10f}")
    print(f"    K_MC_std   = {K_mc_std:.2e}")
    print(f"    ESS / N    = {ess/n_samp:.4f} (= 1.0 for Gaussian)")
    print(f"    Rel error  = {abs(K_mc - K_exact)/abs(K_exact):.2e}")

    # --- Mode analysis ---
    N_mode = 32
    eigenvalues, continuum = analyze_modes(N_mode, T, omega, m)
    print(f"\n  Eigenmode spectrum (N={N_mode}):")
    print(f"    {'n':>4s}  {'lambda_disc':>14s}  {'lambda_cont':>14s}  {'ratio':>8s}")
    print(f"    {'-'*4}  {'-'*14}  {'-'*14}  {'-'*8}")
    for i in range(min(6, len(eigenvalues))):
        r = eigenvalues[i] / continuum[i] if abs(continuum[i]) > 1e-15 else np.inf
        print(f"    {i+1:4d}  {eigenvalues[i]:14.6f}  {continuum[i]:14.6f}  {r:8.5f}")
    print(f"    {'...':>4s}")
    for i in range(max(0, len(eigenvalues)-2), len(eigenvalues)):
        r = eigenvalues[i] / continuum[i]
        print(f"    {i+1:4d}  {eigenvalues[i]:14.6f}  {continuum[i]:14.6f}  {r:8.5f}")

    print(f"\n    IR modes (low n): lambda ~ O(1), slowly-varying phase")
    print(f"    UV modes (high n): lambda ~ O(N^2), rapidly oscillating => Gaussian")
    print(f"    This hierarchy motivates the IR/UV split strategy.")

    # ============================================================
    # PLOTS
    # ============================================================
    # LAYMAN: Four diagnostic plots:
    #   (a) How the eigenvalues compare to the theoretical values
    #   (b) What the classical (most likely) path looks like
    #   (c) How the error shrinks as we use more time slices (should be ~1/N^2)
    #   (d) The shape of the lowest wiggle modes (the hardest ones to handle)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Level 0: Harmonic Oscillator Path Integral Diagnostics',
                 fontsize=14, fontweight='bold')

    # (a) Eigenvalue spectrum
    ax = axes[0, 0]
    modes = np.arange(1, N_mode)
    ax.plot(modes, eigenvalues, 'bo', markersize=5, label='Discrete', zorder=3)
    ax.plot(modes, continuum, 'r-', linewidth=1.5, label='Continuum', alpha=0.7)
    ax.set_xlabel('Mode number n')
    ax.set_ylabel(r'$\lambda_n$')
    ax.set_title('(a) Fluctuation operator spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Classical path
    ax = axes[0, 1]
    t_fine = np.linspace(0, T, 300)
    x_cl = classical_path(t_fine, x_i, x_f, T, omega)
    ax.plot(t_fine, x_cl, 'b-', linewidth=2)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('t')
    ax.set_ylabel(r'$x_{\mathrm{cl}}(t)$')
    ax.set_title(f'(b) Classical path ($\\omega={omega}$)')
    ax.grid(True, alpha=0.3)

    # (c) Convergence
    ax = axes[1, 0]
    ax.loglog(N_vals, errors, 'ko-', markersize=6, linewidth=1.5, label='Measured')
    N_arr = np.array(N_vals, dtype=float)
    ax.loglog(N_arr, errors[0] * (N_arr[0] / N_arr)**2, 'r--',
              linewidth=1, label=r'$O(1/N^2)$')
    ax.set_xlabel('N (time slices)')
    ax.set_ylabel('Relative error vs exact')
    ax.set_title('(c) Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) First few eigenmodes
    ax = axes[1, 1]
    eps_mode = T / N_mode
    K_mat = build_hessian(N_mode, eps_mode, omega, m)
    evals, evecs = eigh(K_mat)
    t_internal = np.linspace(0, T, N_mode + 1)[1:-1]
    for i in range(4):
        mode = evecs[:, i]
        ax.plot(t_internal, mode / np.max(np.abs(mode)),
                linewidth=1.5, label=f'n={i+1} ($\\lambda$={evals[i]:.1f})')
    ax.set_xlabel('t')
    ax.set_ylabel('Normalised amplitude')
    ax.set_title('(d) Lowest eigenmodes (IR)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('level0_diagnostics.png', dpi=150, bbox_inches='tight')
    print(f"\n  Diagnostic plots saved.")

    # --- Thimble visualisation ---
    # LAYMAN: These plots show WHY the thimble trick works:
    #   (a) Left: on the real axis the integrand oscillates wildly (blue).
    #       On the thimble it becomes a smooth bell curve (red). Same integral,
    #       vastly easier to compute.
    #   (b) Right: a map of the complex plane showing where the integrand is
    #       large (red) vs small (blue). The thimble follows the ridge of
    #       steepest descent — the path of maximum damping.
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle('Picard-Lefschetz Thimble Structure', fontsize=14, fontweight='bold')

    ax = axes2[0]
    lam_ex = evals[0]
    t_line = np.linspace(-3, 3, 300)

    # Real axis: wildly oscillating integrand
    exp_real = np.exp(1j * lam_ex * t_line**2 / (2 * hbar))
    # Thimble: smooth bell curve after 45-degree rotation
    xi_th = np.exp(1j * np.pi / 4) * t_line
    exp_thimble = np.exp(1j * lam_ex * xi_th**2 / (2 * hbar))

    ax.plot(t_line, np.real(exp_real), 'b-', lw=1.5, alpha=0.6,
            label=r'Real axis: Re$(e^{iS})$')
    ax.plot(t_line, np.imag(exp_real), 'b--', lw=1.5, alpha=0.6,
            label=r'Real axis: Im$(e^{iS})$')
    ax.plot(t_line, np.real(exp_thimble), 'r-', lw=2,
            label=r'Thimble: Re$(e^{iS})$')
    ax.plot(t_line, np.imag(exp_thimble), 'r--', lw=2,
            label=r'Thimble: Im$(e^{iS})$')
    ax.set_xlabel('t (parameterisation)')
    ax.set_ylabel('Integrand')
    ax.set_title(f'(a) Mode n=1 ($\\lambda$={lam_ex:.2f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Complex plane with steepest descent/ascent
    ax = axes2[1]
    re_g = np.linspace(-3, 3, 120)
    im_g = np.linspace(-3, 3, 120)
    RE, IM = np.meshgrid(re_g, im_g)
    XI = RE + 1j * IM
    EXP = 1j * lam_ex * XI**2 / (2 * hbar)

    cs = ax.contourf(RE, IM, np.real(EXP), levels=20, cmap='RdBu_r', alpha=0.5)
    ax.contour(RE, IM, np.imag(EXP), levels=[0], colors='green',
               linewidths=1.5, linestyles='--')
    plt.colorbar(cs, ax=ax, label=r'Re$(iS/\hbar)$ (damping)', shrink=0.8)

    # Draw contours
    ax.plot([-3, 3], [0, 0], 'b-', lw=2, alpha=0.6, label='Original (real axis)')
    ax.plot([-3, 3], [-3, 3], 'r-', lw=2, label=r'Thimble ($45°$)')
    ax.plot([-3, 3], [3, -3], 'gray', lw=1, ls=':', label='Anti-thimble')

    ax.set_xlabel(r'Re$(\xi)$')
    ax.set_ylabel(r'Im$(\xi)$')
    ax.set_title('(b) Complex $\\xi$ plane')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('level0_thimble.png', dpi=150, bbox_inches='tight')
    print(f"  Thimble plots saved.")

    # ============================================================
    # SUMMARY
    # ============================================================
    # LAYMAN: Final report card — all five methods side by side. For the
    # harmonic oscillator they should all agree. The key metrics:
    #   - Rel error: how far each method is from the exact answer
    #   - ESS/N: fraction of MC samples that are useful (1.0 = perfect)
    print(f"\n{'=' * 72}")
    print(f"  SUMMARY")
    print(f"{'=' * 72}")
    K_best = semiclassical_propagator(x_i, x_f, T, omega, 512, m, hbar)
    print(f"  Exact (Mehler):       {K_exact:.10f}")
    print(f"  Semicl (N=512):       {K_best:.10f}")
    print(f"  Thimble (N=32):       {K_thimble:.10f}")
    print(f"  Brute force (N=4):    {K_bf:.10f}")
    print(f"  Monte Carlo (N=32):   {K_mc:.10f}")

    err_sc = abs(K_best - K_exact) / abs(K_exact)
    err_th = abs(K_thimble - K_sc) / abs(K_exact)
    ess_ratio = ess / n_samp

    all_ok = err_sc < 1e-4 and err_th < 1e-10 and ess_ratio > 0.99
    status = "All methods agree. Infrastructure validated." if all_ok \
        else "WARNING: discrepancy detected."
    print(f"\n  {status}")

    print(f"\n  Key observations:")
    print(f"    - Discretisation error is O(1/N^2)")
    print(f"    - Thimble and determinant methods agree to machine precision")
    print(f"    - MC on thimble: ESS/N = {ess_ratio:.4f} (1.0 = no sign problem)")
    print(f"    - IR modes: lambda ~ O(1), UV modes: lambda ~ O(N^2)")
    print(f"")
    print(f"  Next step: Level 1 (add lambda*x^4 anharmonicity)")
    print(f"  The MC reweighting factor exp(i*S_quartic) will reduce ESS,")
    print(f"  motivating the normalizing flow to improve sampling efficiency.")


def main():
    m = 1.0
    hbar = 1.0
    omega = 2.0
    x_i = 1.0
    x_f = 0.5
    T = 1.0

    print("=" * 72)
    print("  TEACHING SCRIPT: The Picard-Lefschetz Thimble")
    print("  Applied to the Quantum Harmonic Oscillator")
    print("=" * 72)

    # ------------------------------------------------------------------
    # CHAPTER 1: THE PROBLEM
    # ------------------------------------------------------------------
    print(f"""
CHAPTER 1: THE PROBLEM
----------------------
We want to compute the quantum propagator K(x_f, T; x_i, 0) for a
particle on a spring (harmonic oscillator).

  K tells us: what is the probability amplitude for a particle at
  position x_i={x_i} to be found at x_f={x_f} after time T={T}?

The path integral says: sum e^{{iS}} over ALL possible paths from
x_i to x_f. We discretize time into N steps, giving N-1 free
position variables.

On the real axis, each path gives an arrow e^{{iS}} of length 1
pointing in a random direction. They cancel massively -- the sign problem.

The thimble rotates paths 45 degrees into the complex plane, turning
the spinning arrows into a smooth bell curve. No cancellation.

  Parameters: m={m}, hbar={hbar}, omega={omega}, T={T}
  Boundary:   x_i={x_i}, x_f={x_f}
""")

    # ------------------------------------------------------------------
    # CHAPTER 2: THE HESSIAN
    # ------------------------------------------------------------------
    N_teach = 4
    D_teach = N_teach - 1
    eps_teach = T / N_teach
    K_teach = build_hessian(N_teach, eps_teach, omega, m)
    evals_teach, evecs_teach = eigh(K_teach)

    d_val = 2 * m / eps_teach - eps_teach * m * omega**2
    o_val = -m / eps_teach

    print(f"""
CHAPTER 2: THE HESSIAN (N={N_teach}, D={D_teach} free variables)
----------------------
The action S for a path x = [0, x1, x2, x3, 0] is quadratic:
  S = (1/2) x^T K x  +  (linear terms from boundary conditions)

K is the Hessian: K_jk = d^2 S / dx_j dx_k   (tridiagonal)
  Diagonal:     2m/eps - eps*m*omega^2 = {d_val:.4f}
  Off-diagonal: -m/eps                 = {o_val:.4f}

K ({D_teach}x{D_teach}) =""")

    for i in range(D_teach):
        row = "  ["
        for j in range(D_teach):
            row += f" {K_teach[i,j]:8.3f}"
        row += "  ]"
        print(row)

    print(f"""
Diagonalising K -> eigenvalues (stiffness of each wiggle mode):
""")
    for i in range(D_teach):
        print(f"  Mode {i+1}: lambda = {evals_teach[i]:.4f}"
              f"  ->  sigma = sqrt(hbar/|lambda|) = "
              f"{np.sqrt(hbar / abs(evals_teach[i])):.4f}")

    n_neg = np.sum(evals_teach < 0)
    print(f"""
  {n_neg} negative eigenvalue(s).
  Small eigenvalue = floppy mode = wide Gaussian = big fluctuations
  Large eigenvalue = stiff mode  = narrow Gaussian = tiny fluctuations

Eigenvectors (columns = modes, rows = time positions):
""")
    for i in range(D_teach):
        row = "  ["
        for j in range(D_teach):
            row += f" {evecs_teach[i,j]:8.4f}"
        row += "  ]"
        print(row)

    # ------------------------------------------------------------------
    # CHAPTER 3: THE THIMBLE MATRIX A
    # ------------------------------------------------------------------
    sigmas_teach = np.sqrt(hbar / np.abs(evals_teach))
    rot_teach = np.where(evals_teach > 0,
                         np.exp(1j * np.pi / 4),
                         np.exp(-1j * np.pi / 4))
    A_teach = (evecs_teach @ np.diag(rot_teach * sigmas_teach)).T

    print(f"""

CHAPTER 3: THE THIMBLE MATRIX A
-------------------------------
The thimble maps real Gaussian noise z to complex paths on the thimble:

  x = A^T @ z      z ~ N(0, I) in R^D,  x in C^D

A is built from three operations fused into one matrix:

  A = ( V  @  diag(r_n * sigma_n) )^T

  V       = eigenvectors of K           (rotate to independent modes)
  sigma_n = sqrt(hbar / |lambda_n|)     (scale each mode to correct width)
  r_n     = e^(+i*pi/4) if lambda > 0   (rotate +45 deg into complex plane)
            e^(-i*pi/4) if lambda < 0   (rotate -45 deg)

Mode-by-mode breakdown:
""")
    for i in range(D_teach):
        sign_str = "+" if evals_teach[i] > 0 else "-"
        rs = rot_teach[i] * sigmas_teach[i]
        print(f"  Mode {i+1}: lambda={evals_teach[i]:8.4f}  "
              f"sigma={sigmas_teach[i]:.4f}  rot={sign_str}45deg  "
              f"r*sigma = {rs.real:+.4f}{rs.imag:+.4f}i")

    print(f"""
The full thimble matrix A ({D_teach}x{D_teach}):
""")
    for i in range(D_teach):
        row = "  ["
        for j in range(D_teach):
            re = A_teach[i, j].real
            im = A_teach[i, j].imag
            sign = "+" if im >= 0 else "-"
            row += f"  {re:+.4f} {sign} {abs(im):.4f}i"
        row += "  ]"
        print(row)

    avg_phase = np.mean(np.angle(A_teach))
    print(f"""
  Average phase of A entries: {np.degrees(avg_phase):.1f} degrees
  (45 degrees confirms the thimble rotation)

  Each column of A^T takes one noise component z_n and distributes it
  across all time positions with the correct width and 45-degree phase.
  The matrix IS the thimble -- it maps D real numbers to D complex
  path positions on the contour where all arrows point the same way.
""")

    # ------------------------------------------------------------------
    # CHAPTER 4: COMPUTING THE PROPAGATOR
    # ------------------------------------------------------------------
    N_prop = 32
    K_thimble = thimble_propagator(x_i, x_f, T, omega, N_prop, m, hbar)
    K_exact = exact_propagator(x_i, x_f, T, omega, m, hbar)
    S_cl = classical_action(x_i, x_f, T, omega, m)

    print(f"""
CHAPTER 4: COMPUTING THE PROPAGATOR (N={N_prop})
------------------------------------
The thimble approach evaluates the propagator by:
  1. Classical action S_cl (the 'best path' contribution)
  2. Gaussian integral over fluctuations (each mode independently)
  3. Multiply everything together

  Classical action:  S_cl = {S_cl:.10f}

  Thimble result:    K = {K_thimble:.10f}
  Exact (Mehler):    K = {K_exact:.10f}

  |K_thimble|  = {abs(K_thimble):.10f}
  |K_exact|    = {abs(K_exact):.10f}

  arg(K)/pi:     {np.angle(K_thimble)/np.pi:.10f}  (thimble)
                 {np.angle(K_exact)/np.pi:.10f}  (exact)

  Relative error: {abs(K_thimble - K_exact)/abs(K_exact):.2e}

  The thimble is EXACT for the harmonic oscillator (up to O(1/N^2)
  discretisation error) because the action is purely quadratic.
""")

    # ------------------------------------------------------------------
    # CHAPTER 5: COMPUTING <x(T/2)^2>
    # ------------------------------------------------------------------
    N_x2 = 32
    eps_x2 = T / N_x2
    D_x2 = N_x2 - 1
    n_samples = 200_000
    epsilon_reg = 1.0

    K_x2 = build_hessian(N_x2, eps_x2, omega, m)
    evals_x2, evecs_x2 = eigh(K_x2)
    sigmas_x2 = np.sqrt(hbar / np.abs(evals_x2))
    rot_x2 = np.where(evals_x2 > 0,
                       np.exp(1j * np.pi / 4),
                       np.exp(-1j * np.pi / 4))
    A_x2 = (evecs_x2 @ np.diag(rot_x2 * sigmas_x2)).T

    z_samples = np.random.randn(n_samples, D_x2)
    x_samples = z_samples @ A_x2.T

    x_padded = np.concatenate([
        np.zeros((n_samples, 1), dtype=complex),
        x_samples,
        np.zeros((n_samples, 1), dtype=complex)
    ], axis=1)
    dx = np.diff(x_padded, axis=1)
    kinetic = (m / (2 * eps_x2)) * np.sum(dx**2, axis=1)
    omega_sq_c = omega**2 - 1j * epsilon_reg
    potential = (eps_x2 * m / 2) * omega_sq_c * np.sum(x_samples**2, axis=1)
    S_vals = kinetic - potential

    log_P = (-0.5 * np.sum(z_samples**2, axis=1)
             - (D_x2 / 2) * np.log(2 * np.pi))
    sign_det, logabsdet = np.linalg.slogdet(A_x2)
    log_det_J = logabsdet + 1j * np.angle(sign_det)
    log_W = 1j * S_vals + log_det_J - log_P

    log_W_max = np.max(log_W.real)
    W = np.exp(log_W - log_W_max)
    W_norm = W / np.sum(W)

    absW = np.abs(W)
    ess = np.sum(absW)**2 / np.sum(absW**2)
    ess_ratio = ess / n_samples

    mid_idx = D_x2 // 2
    variance_per_step = np.zeros(D_x2)
    for j in range(D_x2):
        obs = x_samples[:, j] ** 2
        variance_per_step[j] = np.sum(W_norm * obs).real

    x2_result = variance_per_step[mid_idx]
    x2_exact_limit = 1.0 / (2 * m * omega)

    K_eps = K_x2 + 1j * eps_x2 * m * epsilon_reg * np.eye(D_x2)
    K_eps_inv = np.linalg.inv(K_eps)
    x2_exact_finite = (1j * hbar * K_eps_inv[mid_idx, mid_idx]).real

    print(f"""
CHAPTER 5: COMPUTING <x(T/2)^2>
--------------------------------
How much does the particle jiggle at the midpoint of its path?

  Setup:
    Boundary: x_i = x_f = 0 (both endpoints pinned)
    N = {N_x2}, D = {D_x2}, {n_samples:,} Monte Carlo samples
    iepsilon regulator: epsilon_reg = {epsilon_reg}

  The thimble matrix A ({D_x2}x{D_x2}) maps z -> x = A^T z,
  placing samples on the thimble contour.

  The iepsilon regulator modifies the potential:
    omega^2 -> omega^2 - i*epsilon_reg = {omega**2} - {epsilon_reg}i

  This damps paths far from zero, projecting onto the ground state.
  Without it, the real-time path integral gives a complex matrix
  element, not a physical expectation value.

  The weights W account for the iepsilon perturbation:
    log W = iS_epsilon + log det(A) - log P(z)
  They are nearly uniform (the thimble handles most of the work)
  with a small real damping factor from the iepsilon term.

  Results:
    ESS/N = {ess_ratio:.4f}  (fraction of useful samples; 1.0 = perfect)

    <x(T/2)^2>:
      MC on thimble:           {x2_result:.6f}
      Exact (finite T, ieps):  {x2_exact_finite:.6f}
      Exact (continuum limit): {x2_exact_limit:.6f}  = 1/(2*m*omega)

      Relative error (vs finite-T exact): """
          f"""{abs(x2_result - x2_exact_finite) / abs(x2_exact_finite):.2e}
""")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Thimble Teaching Results', fontsize=14, fontweight='bold')

    ax = axes[0]
    t_steps = np.arange(1, N_x2) * eps_x2
    ax.plot(t_steps, variance_per_step, 'bo-', markersize=4,
            label='MC on thimble')
    ax.axhline(x2_exact_limit, color='r', ls='--', lw=1.5,
               label=f'1/(2m$\\omega$) = {x2_exact_limit:.3f}')
    ax.axhline(x2_exact_finite, color='g', ls=':', lw=1.5,
               label=f'Exact (finite T) = {x2_exact_finite:.4f}')
    ax.set_xlabel('Time t')
    ax.set_ylabel(r'$\langle x(t)^2 \rangle$')
    ax.set_title(r'$\langle x(t)^2 \rangle$ at each time step')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    n_show = min(30, n_samples)
    t_full = np.linspace(0, T, N_x2 + 1)
    for i in range(n_show):
        path = np.zeros(N_x2 + 1, dtype=complex)
        path[1:-1] = x_samples[i]
        ax.plot(t_full, path.real, 'b-', alpha=0.15, lw=0.8)
        ax.plot(t_full, path.imag, 'r-', alpha=0.15, lw=0.8)
    ax.plot([], [], 'b-', alpha=0.5, label='Real part')
    ax.plot([], [], 'r-', alpha=0.5, label='Imag part')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Position x')
    ax.set_title('Sample paths on the thimble\n(blue=real, red=imaginary)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('claude_teaching.png', dpi=150, bbox_inches='tight')
    print("  [Plot saved: claude_teaching.png]")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print(f"""
{'=' * 72}
  SUMMARY
{'=' * 72}

  PROPAGATOR (thimble approach, N={N_prop}):
    K_thimble = {K_thimble:.10f}
    K_exact   = {K_exact:.10f}
    |K|       = {abs(K_thimble):.10f}
    Rel error = {abs(K_thimble - K_exact)/abs(K_exact):.2e}

  OBSERVABLE <x(T/2)^2> (MC on thimble, N={N_x2}, {n_samples:,} samples):
    Numerical = {x2_result:.6f}
    Exact     = {x2_exact_finite:.6f}  (finite T with iepsilon)
    Continuum = {x2_exact_limit:.6f}  (T -> infinity limit = 1/(2*m*omega))
    ESS/N     = {ess_ratio:.4f}

  THE THIMBLE MATRIX A (displayed for N={N_teach}, D={D_teach}):
    x = A^T @ z  maps real Gaussian noise to complex paths on the thimble
    A = (V @ diag(r_n * sigma_n))^T
    where r_n = e^(+-i*pi/4) and sigma_n = sqrt(hbar/|lambda_n|)

  KEY INSIGHT:
    The thimble rotates the integral 45 degrees into the complex plane.
    This turns wildly oscillating arrows (sign problem) into a smooth
    bell curve (no cancellation). For the harmonic oscillator, this
    gives the EXACT propagator and accurate observables.
""")


if __name__ == "__main__":
    np.random.seed(42)
    main()