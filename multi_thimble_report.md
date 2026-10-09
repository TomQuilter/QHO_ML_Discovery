# Multi-Thimble Overnight Exploration Report

*Report will be overwritten when the overnight script completes.*

## Pre-run Notes

### Key Findings So Far

1. **lam=0 sanity check PASSED** (single thimble): K = 0.3121495 - 0.3021658j
   matches target 0.312 - 0.302j to within 0.05%.

2. **Non-trivial saddle points do NOT exist** for these parameters (T=1, omega=1,
   N=32, x_i=x_f=0). This is because T < pi/omega, so there are no bounce
   solutions of the classical EOM. The continuation method (homotopy from
   uncoupled to fully-coupled action) confirmed this -- all initial guesses
   converged back to the trivial saddle x=0.

3. **The sign ambiguity** is the main issue: the network can converge to +K or -K
   with equal probability and equal loss.

### Overnight Exploration Plan

The script tests three strategies:

**Strategy A** - Baseline: single thimble (trivial saddle), 5 seeds, WITH the
determinant bug (A^{-T}C). This characterises how often we get the right/wrong sign.

**Strategy B** - Multi-thimble via different A-matrix initialisations: instead of
using the standard square-root branch sqrt(i/eigenvalue), we flip the sign of the
square root for the lowest eigenvalue mode(s). This effectively initialises the
network on a "different sheet" of the thimble. 3 branches x 1 seed.

**Strategy C** - Bug fix test: same as A but with the corrected Jacobian determinant
(A^{-1}C instead of A^{-T}C, as recommended in note_for_tom.pdf). 5 seeds.
This tests whether fixing the bug resolves the sign issue and improves accuracy.

### Estimated Runtime

- Strategy A: 3 lambdas x 5 seeds x ~20min = ~300 min
- Strategy B: 3 lambdas x 3 branches x ~20min = ~180 min
- Strategy C: 3 lambdas x 5 seeds x ~20min = ~300 min
- **Total: ~13 hours**

### λ = 0.7 (Attempt 1)

**Target**: K_re = 0.268, K_im = -0.288, |K| = 0.394

**Per-thimble results**:

| Thimble | K_re | K_im | |K| | Loss | ESS/N | Time |
|---------|------|------|-----|------|-------|------|
| trivial | -0.319042 | 0.296511 | 0.435553 | 4.56e-04 | 0.9996 | 3041s |

**Combination weights**: [1]
**Combined K**: -0.3190418966+0.2965107323j
**|K_combined|**: 0.43555292
**Match**: NO -- err_re=19.05%, err_im=2.96%, err_|K|=10.55%

---

### λ = 0.7 (Attempt 1)

**Target**: K_re = 0.268, K_im = -0.288, |K| = 0.394

**Per-thimble results**:

| Thimble | K_re | K_im | |K| | Loss | ESS/N | Time |
|---------|------|------|-----|------|-------|------|
| trivial | -0.319042 | 0.296511 | 0.435553 | 4.56e-04 | 0.9996 | 3056s |

**Combination weights**: [1]
**Combined K**: -0.3190418966+0.2965107323j
**|K_combined|**: 0.43555292
**Match**: NO -- err_re=19.05%, err_im=2.96%, err_|K|=10.55%

---
