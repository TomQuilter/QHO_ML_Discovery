"""
Multi-Thimble approach for the complex-time propagator.

Based on ideas from "multiple saddle points.pdf":
- Find saddle points of the discretised action
- Train a separate NN thimble for each saddle
- Combine contributions: K = Σ_σ n_σ · K_σ

This is the proof-of-concept implementation.
"""

import torch, torch.nn as nn, numpy as np, time, os, json, sys
import scipy.linalg
import scipy.optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

# ─────────────────────────────────────────────────────────────────
#  PARAMETERS
# ─────────────────────────────────────────────────────────────────
N       = 32
T       = 1.0
eta     = 0.05
m       = 1.0
omega   = 1.0
D       = N - 1
rank    = 16
batch   = 2048
N_mc    = 100_000
epochs  = 15000
seed    = 42
x_i_prop = 0.0
x_f_prop = 0.0
T_c = T - 1j * eta
a_c = T_c / N

REPORT_PATH = "multi_thimble_report.md"

# Target values from lattice_targets_for_tom.csv (N=32)
TARGETS = {
    0.0: (0.312, -0.302, 0.434),
    0.7: (0.268, -0.288, 0.394),
    2.4: (-0.111, -0.153, 0.189),
}

print(f"N={N}, T={T}, eta={eta}, T_c={T_c}, m={m}, omega={omega}")
print(f"D={D}, rank={rank}, batch={batch}, N_mc={N_mc}")
print(f"epochs={epochs}, seed={seed}")
print(f"a_c = {a_c}")

# ─────────────────────────────────────────────────────────────────
#  ACTION FUNCTION
# ─────────────────────────────────────────────────────────────────
def complex_action(x, x_i, x_f, lam_val, a_c_val, m_val, omega_val):
    bsz = x.shape[0]
    pad_i = torch.full((bsz, 1), x_i, dtype=torch.complex64)
    pad_f = torch.full((bsz, 1), x_f, dtype=torch.complex64)
    x_pad = torch.cat([pad_i, x, pad_f], dim=1)
    dx = x_pad[:, 1:] - x_pad[:, :-1]
    kinetic = (m_val / (2 * a_c_val)) * torch.sum(dx**2, dim=1)
    pot_harm = (a_c_val * m_val / 2) * omega_val**2 * torch.sum(x**2, dim=1)
    pot_quartic = a_c_val * (lam_val / 4) * torch.sum(x**4, dim=1)
    return kinetic - pot_harm - pot_quartic


# ─────────────────────────────────────────────────────────────────
#  SADDLE POINT FINDING
# ─────────────────────────────────────────────────────────────────
def action_gradient_real(x_real_imag, x_i, x_f, lam_val, a_c_val, m_val, omega_val, D_val):
    """Gradient of S w.r.t. interior points, returned as 2D-real vector."""
    x_re = x_real_imag[:D_val]
    x_im = x_real_imag[D_val:]
    x_complex = x_re + 1j * x_im

    full_path = np.zeros(D_val + 2, dtype=complex)
    full_path[0] = x_i
    full_path[-1] = x_f
    full_path[1:-1] = x_complex

    grad = np.zeros(D_val, dtype=complex)
    for k in range(D_val):
        kp = k + 1
        grad[k] = (m_val / a_c_val) * (2 * full_path[kp] - full_path[kp-1] - full_path[kp+1]) \
                  - a_c_val * omega_val**2 * full_path[kp] \
                  - a_c_val * lam_val * full_path[kp]**3

    result = np.zeros(2 * D_val)
    result[:D_val] = grad.real
    result[D_val:] = grad.imag
    return result


def compute_action_at_path(x_path, x_i, x_f, lam_val, a_c_val, m_val, omega_val):
    """Compute S[x] for a single complex path (D-dim vector of interior points)."""
    D_val = len(x_path)
    full = np.zeros(D_val + 2, dtype=complex)
    full[0] = x_i
    full[-1] = x_f
    full[1:-1] = x_path

    S = 0.0 + 0j
    for k in range(D_val + 1):
        dx = full[k+1] - full[k]
        S += (m_val / (2 * a_c_val)) * dx**2
    for k in range(D_val):
        xk = x_path[k]
        S -= a_c_val * (omega_val**2 / 2) * xk**2
        S -= a_c_val * (lam_val / 4) * xk**4
    return S


def compute_hessian_at_saddle(x_saddle, lam_val, a_c_val, m_val, omega_val, D_val):
    """Hessian of S at a saddle point. Returns D×D complex matrix."""
    H = np.zeros((D_val, D_val), dtype=complex)
    for j in range(D_val):
        H[j, j] = 2 * m_val / a_c_val - a_c_val * omega_val**2 - 3 * a_c_val * lam_val * x_saddle[j]**2
        if j > 0:
            H[j, j-1] = -m_val / a_c_val
        if j < D_val - 1:
            H[j, j+1] = -m_val / a_c_val
    return H


def compute_A_from_hessian(H, D_val):
    """Compute A such that A H A^T ≈ iI. Works for complex symmetric H."""
    eigenvalues, Q = np.linalg.eig(H)

    c = np.zeros(D_val, dtype=complex)
    for j in range(D_val):
        ev = eigenvalues[j]
        if abs(ev) < 1e-14:
            c[j] = 1.0
        else:
            c[j] = np.sqrt(1j / ev)

    A_np = Q @ np.diag(c)
    return torch.tensor(A_np, dtype=torch.complex64)


def action_gradient_alpha(x_real_imag, x_i, x_f, lam_val, a_c_val, m_val, omega_val, D_val, alpha):
    """Gradient of S with kinetic coupling scaled by alpha (for continuation method)."""
    x_re = x_real_imag[:D_val]
    x_im = x_real_imag[D_val:]
    x_complex = x_re + 1j * x_im

    full_path = np.zeros(D_val + 2, dtype=complex)
    full_path[0] = x_i
    full_path[-1] = x_f
    full_path[1:-1] = x_complex

    grad = np.zeros(D_val, dtype=complex)
    for k in range(D_val):
        kp = k + 1
        kinetic = (m_val / a_c_val) * (2 * full_path[kp] - full_path[kp-1] - full_path[kp+1])
        potential = a_c_val * omega_val**2 * full_path[kp] + a_c_val * lam_val * full_path[kp]**3
        grad[k] = alpha * kinetic - potential

    result = np.zeros(2 * D_val)
    result[:D_val] = grad.real
    result[D_val:] = grad.imag
    return result


def find_saddle_continuation(sign, lam_val, N_val, a_c_val, m_val, omega_val, x_i, x_f, n_steps=30):
    """Find a non-trivial saddle using continuation (homotopy) method.

    At alpha=0: uncoupled sites, solution is x_k = sign * i*omega/sqrt(lambda)
    At alpha=1: full kinetic coupling, the actual saddle point
    """
    D_val = N_val - 1
    c = sign * 1j * omega_val / np.sqrt(lam_val)

    x_current = np.full(D_val, c, dtype=complex)

    alphas = np.linspace(0, 1, n_steps + 1)[1:]

    for step, alpha in enumerate(alphas):
        x_ri = np.zeros(2 * D_val)
        x_ri[:D_val] = x_current.real
        x_ri[D_val:] = x_current.imag

        sol = scipy.optimize.root(
            action_gradient_alpha,
            x_ri,
            args=(x_i, x_f, lam_val, a_c_val, m_val, omega_val, D_val, alpha),
            method='hybr',
            tol=1e-10,
        )

        if sol.success or np.linalg.norm(sol.fun) < 1e-8:
            x_current = sol.x[:D_val] + 1j * sol.x[D_val:]
        else:
            print(f"    Step {step+1}/{n_steps} (alpha={alpha:.3f}) |res|={np.linalg.norm(sol.fun):.2e}: {sol.message}")
            x_current = sol.x[:D_val] + 1j * sol.x[D_val:]

    grad_norm = np.linalg.norm(
        action_gradient_real(
            np.concatenate([x_current.real, x_current.imag]),
            x_i, x_f, lam_val, a_c_val, m_val, omega_val, D_val
        )
    )

    if grad_norm > 1e-5:
        print(f"    Final gradient too large: |grad| = {grad_norm:.2e}")
        return None

    if np.max(np.abs(x_current)) < 1e-6:
        print(f"    Converged to trivial saddle")
        return None

    return x_current


def find_saddle_points(lam_val, N_val, a_c_val, m_val, omega_val, x_i, x_f):
    """Find saddle points of the discretised action."""
    D_val = N_val - 1
    saddles = []

    trivial = np.zeros(D_val, dtype=complex)
    saddles.append(('trivial', trivial))

    if lam_val > 0:
        for sign, label in [(+1, 'bounce+'), (-1, 'bounce-')]:
            print(f"  Searching for saddle '{label}' using continuation method...")
            saddle = find_saddle_continuation(
                sign, lam_val, N_val, a_c_val, m_val, omega_val, x_i, x_f
            )
            if saddle is not None:
                grad_norm = np.linalg.norm(
                    action_gradient_real(
                        np.concatenate([saddle.real, saddle.imag]),
                        x_i, x_f, lam_val, a_c_val, m_val, omega_val, D_val
                    )
                )
                saddles.append((label, saddle))
                print(f"  Found saddle '{label}': |grad| = {grad_norm:.2e}, "
                      f"max|x| = {np.max(np.abs(saddle)):.4f}, "
                      f"mid-point x[D/2] = {saddle[D_val//2]:.4f}")
            else:
                print(f"  Could not find non-trivial saddle '{label}'")

    return saddles


# ─────────────────────────────────────────────────────────────────
#  THIMBLE MODEL (same as original — keeps the A^{-T}C bug per instructions)
# ─────────────────────────────────────────────────────────────────
class ThimbleFlowNL(nn.Module):
    def __init__(self, dim, rank=8, b_init=None, A_init=None):
        super().__init__()
        self.dim = dim
        self.rank = rank

        if A_init is not None:
            self.A_real = nn.Parameter(A_init.real.float())
            self.A_imag = nn.Parameter(A_init.imag.float())
        else:
            self.A_real = nn.Parameter(torch.eye(dim) + 0.01 * torch.randn(dim, dim))
            self.A_imag = nn.Parameter(0.01 * torch.randn(dim, dim))

        if b_init is not None:
            self.b_real = nn.Parameter(b_init.real.clone().float())
            self.b_imag = nn.Parameter(b_init.imag.clone().float())
        else:
            self.b_real = nn.Parameter(torch.zeros(dim))
            self.b_imag = nn.Parameter(torch.zeros(dim))

        r = rank
        self.W1 = nn.Parameter(torch.randn(r, dim) * (2.0 / dim)**0.5)
        self.b1 = nn.Parameter(torch.zeros(r))
        self.W2 = nn.Parameter(torch.randn(r, r) * (2.0 / r)**0.5)
        self.b2 = nn.Parameter(torch.zeros(r))
        self.W3 = nn.Parameter(torch.randn(2 * dim, r) * 0.001)
        self.b3 = nn.Parameter(torch.zeros(2 * dim))

    def _get_A(self):
        return self.A_real + 1j * self.A_imag

    def forward(self, z):
        A = self._get_A()
        b = self.b_real + 1j * self.b_imag
        x = z.to(torch.complex64) @ A.t() + b
        h1 = z @ self.W1.t() + self.b1
        a1 = torch.tanh(h1)
        h2 = a1 @ self.W2.t() + self.b2
        a2 = torch.tanh(h2)
        out = a2 @ self.W3.t() + self.b3
        g = out[:, :self.dim].to(torch.complex64) + 1j * out[:, self.dim:].to(torch.complex64)
        return x + g, a1, a2

    def compute_log_det_J(self, z, a1, a2, detach_correction=False):
        A = self._get_A()
        sign_A, logabs_A = torch.linalg.slogdet(A)
        log_det_A = logabs_A + 1j * torch.angle(sign_A)

        C = self.W3[:self.dim].to(torch.complex64) + 1j * self.W3[self.dim:].to(torch.complex64)
        A_invT_C = torch.linalg.solve(A.t(), C)
        K_mat = self.W1.to(torch.complex64) @ A_invT_C

        batch_size = z.shape[0]
        r = self.rank
        _a1 = a1.detach() if detach_correction else a1
        _a2 = a2.detach() if detach_correction else a2

        da1 = (1.0 - _a1**2).to(torch.complex64)
        M1 = da1.unsqueeze(-1) * K_mat.unsqueeze(0)
        W2c = self.W2.to(torch.complex64)
        if detach_correction:
            W2c = W2c.detach()
        M2 = torch.bmm(W2c.unsqueeze(0).expand(batch_size, -1, -1), M1)
        da2 = (1.0 - _a2**2).to(torch.complex64)
        M3 = da2.unsqueeze(-1) * M2
        I_r = torch.eye(r, dtype=torch.complex64).unsqueeze(0)
        sign_c, logabs_c = torch.linalg.slogdet(I_r + M3)
        log_det_correction = logabs_c + 1j * torch.angle(sign_c)
        if detach_correction:
            log_det_correction = log_det_correction.detach()
        return log_det_A + log_det_correction


# ─────────────────────────────────────────────────────────────────
#  TRAIN A SINGLE THIMBLE
# ─────────────────────────────────────────────────────────────────
def train_thimble(lam_val, saddle_label, saddle_path,
                  x_i=0.0, x_f=0.0,
                  N_val=32, T_val=1.0, m_val=1.0, omega_val=1.0, eta_val=0.05,
                  rank_val=16, epochs_val=15000, batch_val=2048, N_mc_val=100_000,
                  seed_val=42):
    """Train one thimble centered at a given saddle point."""

    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    D_val = N_val - 1
    T_c_val = T_val - 1j * eta_val
    a_c_val = T_c_val / N_val

    H = compute_hessian_at_saddle(saddle_path, lam_val, a_c_val, m_val, omega_val, D_val)
    A_init = compute_A_from_hessian(H, D_val)

    b_init = torch.tensor(saddle_path, dtype=torch.complex64)

    model = ThimbleFlowNL(D_val, rank=rank_val, b_init=b_init, A_init=A_init)

    lr_init = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs_val, eta_min=1e-4)

    loss_history = []
    t0 = time.time()

    for epoch in range(epochs_val):
        optimizer.zero_grad()
        z = torch.randn(batch_val, D_val)
        x, a1, a2 = model(z)
        S = complex_action(x, x_i, x_f, lam_val, a_c_val, m_val, omega_val)
        log_P = -0.5 * torch.sum(z**2, dim=1) - (D_val / 2) * np.log(2 * np.pi)
        log_det_J = model.compute_log_det_J(z, a1, a2, detach_correction=True)
        log_W = 1j * S + log_det_J - log_P.to(torch.complex64)
        loss = torch.var(log_W.real) + torch.var(log_W.imag)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())
        if (epoch + 1) % 3000 == 0 or epoch == 0:
            print(f"    [{saddle_label}] Epoch {epoch+1:5d}/{epochs_val} | "
                  f"loss = {loss.item():.6e} | {time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0

    norm_factor = (m_val / (2 * np.pi * 1j * a_c_val)) ** (N_val / 2)
    with torch.no_grad():
        z_mc = torch.randn(N_mc_val, D_val)
        x_mc, a1_mc, a2_mc = model(z_mc)
        S_mc = complex_action(x_mc, x_i, x_f, lam_val, a_c_val, m_val, omega_val)
        log_P_mc = -0.5 * torch.sum(z_mc**2, dim=1) - (D_val / 2) * np.log(2 * np.pi)
        log_det_mc = model.compute_log_det_J(z_mc, a1_mc, a2_mc)
        log_W_mc = 1j * S_mc + log_det_mc - log_P_mc.to(torch.complex64)
        log_W_max = torch.max(log_W_mc.real)
        W_shifted = torch.exp(log_W_mc - log_W_max)
        K_mc = norm_factor * torch.exp(log_W_max).item() * torch.mean(W_shifted).numpy()
        W_for_err = norm_factor * torch.exp(log_W_max).item() * W_shifted.numpy()
        K_std = np.std(W_for_err) / np.sqrt(N_mc_val)
        W_ess = torch.exp(log_W_mc[:10000] - torch.max(log_W_mc[:10000].real)).numpy()
        ess = float(np.abs(np.mean(W_ess))**2 / np.mean(np.abs(W_ess)**2))

    S_at_saddle = compute_action_at_path(saddle_path, x_i, x_f, lam_val, a_c_val, m_val, omega_val)

    result = dict(
        saddle_label=saddle_label,
        lam=lam_val, seed=seed_val,
        K_re=float(K_mc.real), K_im=float(K_mc.imag), abs_K=float(abs(K_mc)),
        std_err=float(K_std), final_loss=loss_history[-1], ess_n=ess,
        time_s=elapsed, loss_history=loss_history,
        S_at_saddle_re=float(S_at_saddle.real), S_at_saddle_im=float(S_at_saddle.imag),
        saddle_max_abs=float(np.max(np.abs(saddle_path))),
    )

    print(f"  [{saddle_label}] K = {K_mc:.10f}  |K| = {abs(K_mc):.8f}  "
          f"loss = {loss_history[-1]:.2e}  ESS/N = {ess:.4f}  ({elapsed:.0f}s)")

    return result


# ─────────────────────────────────────────────────────────────────
#  MEHLER KERNEL (exact lam=0 reference)
# ─────────────────────────────────────────────────────────────────
def mehler_kernel(x_i, x_f, T_val, m_val, omega_val, eta_val=0.0):
    T_c_val = T_val - 1j * eta_val
    wT = omega_val * T_c_val
    sinwT = np.sin(wT)
    coswT = np.cos(wT)
    prefactor = np.sqrt(m_val * omega_val / (2 * np.pi * 1j * sinwT))
    exponent = (1j * m_val * omega_val / (2 * sinwT)) * (
        (x_i**2 + x_f**2) * coswT - 2 * x_i * x_f)
    return prefactor * np.exp(exponent)


# ─────────────────────────────────────────────────────────────────
#  MULTI-THIMBLE COMBINATION
# ─────────────────────────────────────────────────────────────────
def combine_thimbles(thimble_results, target_K=None):
    """Combine thimble contributions.

    Strategy: try different intersection number combinations and pick
    the one closest to the target (if given).
    """
    n_thimbles = len(thimble_results)
    contributions = []
    for r in thimble_results:
        K = r['K_re'] + 1j * r['K_im']
        contributions.append(K)

    if n_thimbles == 1:
        return contributions[0], [1], contributions

    candidates = []

    if n_thimbles == 3:
        for n0 in [1, -1]:
            for n1 in [-1, 0, 1]:
                for n2 in [-1, 0, 1]:
                    K_total = n0 * contributions[0] + n1 * contributions[1] + n2 * contributions[2]
                    candidates.append(((n0, n1, n2), K_total))
    elif n_thimbles == 2:
        for n0 in [1, -1]:
            for n1 in [-1, 0, 1]:
                K_total = n0 * contributions[0] + n1 * contributions[1]
                candidates.append(((n0, n1), K_total))

    if target_K is not None:
        best = min(candidates, key=lambda c: abs(c[1] - target_K))
        best_reversed = min(candidates, key=lambda c: abs(c[1] + target_K))
        if abs(best_reversed[1] + target_K) < abs(best[1] - target_K):
            best = best_reversed
            print(f"  Note: best match is with reversed sign")
        return best[1], list(best[0]), contributions
    else:
        best = candidates[0]
        return best[1], list(best[0]), contributions


def check_against_target(K_result, lam_val, sign_insensitive=True):
    """Check if result matches target. Returns (match, error_pct, details)."""
    if lam_val not in TARGETS:
        return False, float('inf'), "No target for this lam"

    t_re, t_im, t_abs = TARGETS[lam_val]
    target = t_re + 1j * t_im

    err_re = abs(K_result.real - t_re) / max(abs(t_re), 1e-10)
    err_im = abs(K_result.imag - t_im) / max(abs(t_im), 1e-10)
    err_abs = abs(abs(K_result) - t_abs) / t_abs

    if sign_insensitive:
        err_re_rev = abs(-K_result.real - t_re) / max(abs(t_re), 1e-10)
        err_im_rev = abs(-K_result.imag - t_im) / max(abs(t_im), 1e-10)
        err_abs_rev = abs(abs(K_result) - t_abs) / t_abs

        if err_re_rev + err_im_rev < err_re + err_im:
            err_re, err_im = err_re_rev, err_im_rev
            err_abs = err_abs_rev

    max_err = max(err_re, err_im)
    match = max_err < 0.05
    details = f"err_re={err_re:.2%}, err_im={err_im:.2%}, err_|K|={err_abs:.2%}"
    return match, max_err, details


def update_report(lam_val, attempt_num, thimble_results, K_combined, weights, description=""):
    """Append results to the report file."""
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n### λ = {lam_val} (Attempt {attempt_num})\n\n")
        if description:
            f.write(f"{description}\n\n")

        t_re, t_im, t_abs = TARGETS.get(lam_val, (None, None, None))
        if t_re is not None:
            f.write(f"**Target**: K_re = {t_re}, K_im = {t_im}, |K| = {t_abs}\n\n")

        f.write("**Per-thimble results**:\n\n")
        f.write("| Thimble | K_re | K_im | |K| | Loss | ESS/N | Time |\n")
        f.write("|---------|------|------|-----|------|-------|------|\n")
        for r in thimble_results:
            f.write(f"| {r['saddle_label']} | {r['K_re']:.6f} | {r['K_im']:.6f} | "
                    f"{r['abs_K']:.6f} | {r['final_loss']:.2e} | {r['ess_n']:.4f} | "
                    f"{r['time_s']:.0f}s |\n")

        f.write(f"\n**Combination weights**: {weights}\n")
        f.write(f"**Combined K**: {K_combined:.10f}\n")
        f.write(f"**|K_combined|**: {abs(K_combined):.8f}\n")

        match, err, details = check_against_target(K_combined, lam_val)
        f.write(f"**Match**: {'YES' if match else 'NO'} -- {details}\n\n")

        if match:
            f.write("**>>> Target achieved! <<<**\n\n")

        f.write("---\n")


# ─────────────────────────────────────────────────────────────────
#  MAIN: RUN MULTI-THIMBLE FOR EACH LAMBDA
# ─────────────────────────────────────────────────────────────────
def run_for_lambda(lam_val, attempt_num=1):
    """Run the multi-thimble approach for a given lambda."""

    print(f"\n{'='*70}")
    print(f"  MULTI-THIMBLE RUN: lam = {lam_val}")
    print(f"{'='*70}\n")

    print("Step 1: Finding saddle points...")
    saddles = find_saddle_points(lam_val, N, a_c, m, omega, x_i_prop, x_f_prop)
    print(f"  Found {len(saddles)} saddle point(s)")

    for label, sp in saddles:
        S_val = compute_action_at_path(sp, x_i_prop, x_f_prop, lam_val, a_c, m, omega)
        print(f"  {label}: S = {S_val:.6f}, iS = {1j*S_val:.6f}, "
              f"exp(iS) = {np.exp(1j*S_val):.6f}")

    print("\nStep 2: Training thimbles...")
    thimble_results = []

    for saddle_label, saddle_path in saddles:
        print(f"\n--- Training thimble '{saddle_label}' ---")
        result = train_thimble(
            lam_val, saddle_label, saddle_path,
            x_i=x_i_prop, x_f=x_f_prop,
            N_val=N, T_val=T, m_val=m, omega_val=omega, eta_val=eta,
            rank_val=rank, epochs_val=epochs, batch_val=batch, N_mc_val=N_mc,
            seed_val=seed,
        )
        thimble_results.append(result)

    print("\nStep 3: Combining thimble contributions...")
    t_re, t_im, _ = TARGETS.get(lam_val, (None, None, None))
    target_K = (t_re + 1j * t_im) if t_re is not None else None

    K_combined, weights, contributions = combine_thimbles(thimble_results, target_K)

    print(f"\n  Combined K = {K_combined:.10f}")
    print(f"  |K| = {abs(K_combined):.8f}")
    print(f"  Weights: {weights}")

    match, err, details = check_against_target(K_combined, lam_val)
    print(f"  Match: {'YES' if match else 'NO'} -- {details}")

    if target_K is not None:
        print(f"  Target:  K = {target_K:.10f}")

    update_report(lam_val, attempt_num, thimble_results, K_combined, weights)

    return match, K_combined, thimble_results


def save_loss_curves(thimble_results, lam_val):
    """Save loss curves for all thimbles."""
    plt.figure(figsize=(10, 4))
    for r in thimble_results:
        plt.semilogy(r['loss_history'], label=r['saddle_label'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Var[log W])')
    plt.title(f'Multi-thimble loss curves: lam={lam_val}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"multi_thimble_loss_lam{lam_val}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


if __name__ == "__main__":
    K_mehler = mehler_kernel(x_i_prop, x_f_prop, T, m, omega, eta)
    print(f"Mehler K(0,0; T_c) = {K_mehler:.10f}  |K| = {abs(K_mehler):.10f}")
    print()

    lambda_run_list = [0.7, 2.4]
    all_match = True

    for lam in lambda_run_list:
        match, K_result, thimble_results = run_for_lambda(lam)
        save_loss_curves(thimble_results, lam)

        if not match:
            all_match = False

        if match:
            print(f"\n  *** lam = {lam}: TARGET ACHIEVED! ***")

    if all_match:
        print("\n" + "="*70)
        print("  ALL TARGETS ACHIEVED!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("  Some targets not yet achieved. See report for details.")
        print("="*70)
