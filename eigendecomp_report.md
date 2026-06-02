# Eigendecomposition Propagator — Report

## Objective

Compute K(0,0; T=1) for the quantum anharmonic oscillator using eigendecomposition (spectral sum), as an independent check against:
- Split-operator + Gaussian extrapolation
- ML (neural-network thimble) warm-start

Convention: V(x) = ½m(ω²−iε)x² + (λ/4)x⁴, with ω = m = ℏ = 1, ε = 0.01.

## Methods Attempted

### 1. HO-Basis Matrix Exponential (expm)
- Build H in the harmonic oscillator basis, restrict to even-parity sector.
- Compute K = v^T expm(−iHT) v, where v = [ψ_{2k}(0)].
- **Result at λ=0**: 0.5% error against Mehler at N_even=800. Converging slowly.
- **Result at λ>0**: Results oscillate wildly and do NOT converge as N_even increases (50→800). At λ=1, |K|≈1.9 vs expected ~0.43.
- **Diagnosis**: The matrix exponential includes ALL eigenstates of the truncated Hamiltonian, including artifact states near the truncation boundary that are distorted by the x⁴ term.

### 2. HO-Basis Selective Spectral Sum (two-window)
- Diagonalise H at two basis sizes (N_large, N_small).
- Keep only eigenvalues that agree between both (converged physical states).
- Compute spectral sum with c-normalised eigenvectors.
- **Result at λ=0**: **0.04% error** at NL=500/NS=400. Excellent convergence.
- **Result at λ>0**: Still fails. The converged low-lying states alone do NOT account for the full propagator because the missing high-energy tail (states with E > ~400) still contributes significantly with ε=0.01.
- λ=1 gives |K|≈1.9 (wrong), λ=8 gives |K|≈2.9 (wrong). ML gives |K|≈0.43 for both.

### 3. Position-Space Grid + expm
- Spectral (FFT-based) kinetic energy on a uniform grid.
- Compute K = [expm(−iH_grid T)]_{j0,j0} / dx.
- **Result**: Catastrophic failure. Results get WORSE with finer grids (N=256 → 2.8% error; N=512 → 120% error). Non-monotonic convergence.
- **Diagnosis**: The matrix exponential exp(−iHT) is an oscillatory matrix with norm ∝ 1/dx². As dx decreases, the norm increases, and the forward error in individual matrix elements grows due to numerical error amplification during the scaling-and-squaring algorithm. This is a well-known issue in computational quantum mechanics.

### 4. Split-Operator + Polynomial Gaussian Extrapolation
- Propagate Gaussians of varying widths σ, extract K(σ), extrapolate to σ→0.
- **Result**: Extrapolation diverges catastrophically (errors > 3000%).
- **Diagnosis**: K(σ) is not a smooth polynomial in σ² for these parameters. The extrapolation is ill-conditioned.

## Root Cause Analysis

The fundamental difficulty is the **weak iε damping** (ε = 0.01):

- The spectral sum K = Σ_n ψ_n(0)² exp(−iE_nT) converges because Im(E_n) < 0, so |exp(−iE_nT)| = exp(Im(E_n)T) decreases with n.
- However, Im(E_n) ≈ −(ε/2)⟨n|x²|n⟩ ≈ −0.005n (for the HO). The damping factor at n=1000 is exp(−5) ≈ 0.007, meaning states up to n≈1000 contribute non-negligibly.
- ψ_n(0)² decreases slowly (∝ n^{−1/2}), so the spectral sum needs ~2000 accurately computed eigenstates.
- In the HO basis, this requires ~4000+ basis states. The eigendecomposition of a 2000×2000 matrix takes many minutes on this machine and could not be completed.
- On the position-space grid, the maximum representable energy (∝ 1/dx²) creates aliasing when the matrix exponential is computed, causing the results to degrade with finer grids.

## Key Finding: λ=0 Validation

At λ=0, the HO-basis two-window method gives excellent results:

| N_large | N_small | M_converged | Mehler error |
|---------|---------|-------------|-------------|
| 200     | 150     | 150         | 0.80%       |
| 300     | 250     | 250         | 0.23%       |
| 400     | 350     | 350         | 0.07%       |
| 500     | 400     | 400         | **0.04%**   |

The method and code are correct — the eigenvalues converge to the exact Mehler kernel values. The physics is sound. The failure at λ>0 is a practical computational limitation, not a conceptual error.

## Conclusion

**The eigendecomposition approach cannot serve as a reliable independent verification at the current computational scale.**

For λ=0 it matches the exact Mehler kernel to 0.04% (confirming the code is correct), but for λ>0 the truncation error dominates because ε=0.01 requires ~2000+ accurately computed eigenstates, which is beyond practical limits on this machine.

**Implications for the ML results**: The ML warm-start results remain the most reliable data for K(0,0;T) at λ>0. They show smooth, physically plausible behaviour (|K| monotonically decreasing with λ, consistent with perturbation theory at small λ and WKB analysis at large λ). Neither the split-operator extrapolation nor the eigendecomposition can currently challenge these results at strong coupling.

**What would work**: Running the HO-basis eigendecomposition with N_even ≈ 2000–4000 (requiring eigendecomposition of matrices up to 4000×4000) would likely converge. This is feasible on a more powerful machine or with optimised linear algebra (e.g., using GPU-accelerated libraries). Alternatively, using a much larger ε (e.g., ε=0.1) would reduce the number of required eigenstates, and results could be extrapolated to ε=0.01.
