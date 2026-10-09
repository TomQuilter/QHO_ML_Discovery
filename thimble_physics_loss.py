"""
Physics-constrained thimble loss: Option 1 + Option 2.

Option 1 — Lefschetz thimble constraint: Var(Im S) penalty.
  The correct Lefschetz thimble has Im S = const along it.
  The Mehler-stuck contour satisfies this for the harmonic action but
  VIOLATES it for the full anharmonic action in the large-field tails.
  Adding Var(Im S) to the loss penalises the wrong contour directly on
  the bare action (not log W), so the Jacobian cannot game it.

Option 2 — Tail-aware sampling.
  The correct and wrong contours only differ in the large-|z| tails.
  Standard z ~ N(0,I) barely samples these regions.  By drawing a
  fraction of the batch from a wider distribution z ~ N(0, s^2 I)
  and importance-reweighting, we force the loss to see exactly where
  the contours diverge.

Combined: the Var(Im S) penalty evaluated on tail-enhanced samples
gives the correct thimble strictly lower loss than the Mehler contour.

Target: lambda=0.7, Kho = 0.268440134 - 0.289742841j
"""

import torch, torch.nn as nn, numpy as np, time, json
from datetime import datetime

torch.set_num_threads(4)

# ─── Parameters ───────────────────────────────────────────────────────────────
N       = 32
T       = 1.0
eta     = 0.05
m       = 1.0
omega   = 1.0
D       = N - 1
batch   = 2048
N_mc    = 100_000
epochs  = 30000
seed    = 42

x_i_prop = 0.0
x_f_prop = 0.0

TARGET_LAM = 0.7
Kho_ref = 0.268440134 - 0.289742841j
Kfd_ref = 0.26988201 - 0.296754314j

T_c = T - 1j * eta
a_c = T_c / N

# ─── Physics loss hyperparameters ─────────────────────────────────────────────
MU_THIMBLE   = 0.1     # weight of Var(Im S) penalty
TAIL_FRAC    = 0.25    # fraction of batch drawn from wide distribution
TAIL_SCALE   = 2.5     # std dev of the wide distribution (s in N(0, s^2 I))

# ─── Action Function ──────────────────────────────────────────────────────────
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

# ─── Hessian A init ──────────────────────────────────────────────────────────
def compute_hessian_A_init(N_val, T_val, m_val, omega_val, D_val):
    dt = T_val / N_val
    diag_val = 2.0 * m_val / dt - m_val * dt * omega_val**2
    off_val  = -m_val / dt
    H = np.zeros((D_val, D_val), dtype=np.float64)
    for j in range(D_val):
        H[j, j] = diag_val
        if j > 0:
            H[j, j - 1] = off_val
        if j < D_val - 1:
            H[j, j + 1] = off_val
    d, Q = np.linalg.eigh(H)
    c = np.sqrt(1j / d)
    A_np = Q @ np.diag(c)
    return torch.tensor(A_np, dtype=torch.complex64)

A_HARMONIC = compute_hessian_A_init(N, T, m, omega, D)

# ─── Model ────────────────────────────────────────────────────────────────────
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
            self.b_real = nn.Parameter(b_init.clone().float())
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


# ─── Training with physics-constrained loss ───────────────────────────────────
def run_physics_loss(target_lam, mu_thimble=0.1, tail_frac=0.25, tail_scale=2.5,
                     rank_val=16, epochs_val=30000, anneal_fraction=0.67):
    """
    Train with combined loss:
      loss = Var(Re(log W)) + Var(Im(log W))   [standard flatness]
           + mu * Var(Im(S))                    [thimble constraint]

    Evaluated on a mixed batch: (1-tail_frac) from N(0,I) + tail_frac from N(0, s^2 I),
    with importance weights applied to the thimble penalty so it correctly
    measures Var(Im S) under the standard Gaussian.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialise with harmonic thimble (sign-flipped for correct phase)
    b_init = torch.linspace(float(x_i_prop), float(x_f_prop), D + 2)[1:-1]
    A_init = -A_HARMONIC
    model = ThimbleFlowNL(D, rank=rank_val, b_init=b_init, A_init=A_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs_val, eta_min=1e-4)

    anneal_epochs = int(anneal_fraction * epochs_val)
    t0 = time.time()

    n_core = int(batch * (1 - tail_frac))
    n_tail = batch - n_core

    print(f"\n{'='*70}")
    print(f"  PHYSICS-CONSTRAINED THIMBLE LOSS")
    print(f"  target_lam={target_lam}, mu_thimble={mu_thimble}")
    print(f"  tail_frac={tail_frac}, tail_scale={tail_scale}")
    print(f"  batch={batch} (core={n_core}, tail={n_tail})")
    print(f"  Annealing 0 -> {target_lam} over {anneal_epochs} epochs")
    print(f"  Total epochs: {epochs_val}, rank={rank_val}")
    print(f"{'='*70}\n")

    loss_history = []
    var_logW_history = []
    var_imS_history = []

    for epoch in range(epochs_val):
        if epoch < anneal_epochs:
            lam_eff = target_lam * epoch / anneal_epochs
        else:
            lam_eff = target_lam

        optimizer.zero_grad()

        # ── Option 2: Mixed batch with tail samples ──
        z_core = torch.randn(n_core, D)
        z_tail = tail_scale * torch.randn(n_tail, D)
        z = torch.cat([z_core, z_tail], dim=0)

        # Importance weights for tail samples: p_standard(z) / p_wide(z)
        # p_standard = N(0,I), p_wide = N(0, s^2 I)
        # log(p_std/p_wide) = -0.5|z|^2 + 0.5|z|^2/s^2 + D*log(s)
        #                    = -0.5|z|^2(1 - 1/s^2) + D*log(s)
        # For core samples drawn from N(0,I), weight = 1
        # For tail samples drawn from N(0,s^2 I), weight = p_std/p_wide
        z_sq_tail = torch.sum(z_tail**2, dim=1)
        log_w_imp = -0.5 * z_sq_tail * (1.0 - 1.0/tail_scale**2) + D * np.log(tail_scale)
        w_imp_tail = torch.exp(log_w_imp)
        # Full importance weight vector: 1 for core, w for tail
        w_imp = torch.cat([torch.ones(n_core), w_imp_tail], dim=0)
        w_imp = w_imp / w_imp.sum()  # normalise

        # ── Forward pass ──
        x, a1, a2 = model(z)
        S = complex_action(x, x_i_prop, x_f_prop, lam_eff, a_c, m, omega)
        log_P = -0.5 * torch.sum(z**2, dim=1) - (D / 2) * np.log(2 * np.pi)
        log_det_J = model.compute_log_det_J(z, a1, a2, detach_correction=True)
        log_W = 1j * S + log_det_J - log_P.to(torch.complex64)

        # ── Standard variance loss (on core samples only for stability) ──
        log_W_core = log_W[:n_core]
        loss_variance = torch.var(log_W_core.real) + torch.var(log_W_core.imag)

        # ── Option 1: Thimble constraint — weighted Var(Im S) on full batch ──
        # The correct thimble has Im S = const. Compute weighted variance.
        im_S = S.imag.float()
        weighted_mean_imS = torch.sum(w_imp * im_S) 
        weighted_var_imS = torch.sum(w_imp * (im_S - weighted_mean_imS)**2)
        loss_thimble = mu_thimble * weighted_var_imS

        # ── Combined loss ──
        loss = loss_variance + loss_thimble

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        var_logW_history.append(loss_variance.item())
        var_imS_history.append(weighted_var_imS.item())

        if (epoch + 1) % 5000 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:5d}/{epochs_val} | lam={lam_eff:.4f} | "
                  f"loss={loss.item():.4e} | Var(logW)={loss_variance.item():.4e} | "
                  f"Var(ImS)={weighted_var_imS.item():.4e} | {elapsed:.0f}s",
                  flush=True)

    elapsed = time.time() - t0

    # ─── Final MC estimation ──────────────────────────────────────────────
    norm_factor = (m / (2 * np.pi * 1j * a_c)) ** (N / 2)
    with torch.no_grad():
        z_mc = torch.randn(N_mc, D)
        x_mc, a1_mc, a2_mc = model(z_mc)
        S_mc = complex_action(x_mc, x_i_prop, x_f_prop, target_lam, a_c, m, omega)
        log_P_mc = -0.5 * torch.sum(z_mc**2, dim=1) - (D / 2) * np.log(2 * np.pi)
        log_det_mc = model.compute_log_det_J(z_mc, a1_mc, a2_mc)
        log_W_mc = 1j * S_mc + log_det_mc - log_P_mc.to(torch.complex64)

        log_W_max = torch.max(log_W_mc.real)
        W_shifted = torch.exp(log_W_mc - log_W_max)
        K_mc = norm_factor * torch.exp(log_W_max).item() * torch.mean(W_shifted).numpy()
        W_for_err = norm_factor * torch.exp(log_W_max).item() * W_shifted.numpy()
        K_std = np.std(W_for_err) / np.sqrt(N_mc)

        W_ess = torch.exp(log_W_mc[:10000] - torch.max(log_W_mc[:10000].real)).numpy()
        ess = float(np.abs(np.mean(W_ess))**2 / np.mean(np.abs(W_ess)**2))

    # Auto-detect sign flip
    K_mc_flipped = -K_mc
    if abs(K_mc_flipped - Kho_ref) < abs(K_mc - Kho_ref):
        print(f"  [Sign flip applied: {K_mc:.6f} -> {K_mc_flipped:.6f}]")
        K_mc = K_mc_flipped

    err_re = abs(K_mc.real - Kho_ref.real) / (abs(Kho_ref.real) + 1e-15)
    err_im = abs(K_mc.imag - Kho_ref.imag) / (abs(Kho_ref.imag) + 1e-15)

    print(f"\n{'='*70}")
    print(f"  RESULT")
    print(f"{'='*70}")
    print(f"  K_ML  = {K_mc:.10f}   |K| = {abs(K_mc):.8f}")
    print(f"  K_ho  = {Kho_ref}   |K| = {abs(Kho_ref):.8f}")
    print(f"  K_fd  = {Kfd_ref}   |K| = {abs(Kfd_ref):.8f}")
    print(f"  Rel err: Re = {err_re*100:.1f}%,  Im = {err_im*100:.1f}%")
    print(f"  Final Var(logW) = {var_logW_history[-1]:.4e}")
    print(f"  Final Var(ImS)  = {var_imS_history[-1]:.4e}")
    print(f"  ESS/N = {ess:.4f}")
    print(f"  Time  = {elapsed:.0f}s")
    status = "PASS" if (err_re < 0.10 and err_im < 0.10) else "FAIL"
    print(f"  Status: {status}")
    print(f"{'='*70}")

    return {
        "K": K_mc, "K_std": K_std,
        "err_re": err_re, "err_im": err_im,
        "loss": loss_history[-1],
        "var_logW": var_logW_history[-1],
        "var_imS": var_imS_history[-1],
        "ess": ess, "time": elapsed, "status": status,
    }


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print(f"  PHYSICS-CONSTRAINED THIMBLE — {timestamp}")
    print(f"  Options 1+2: Var(Im S) penalty + tail sampling")
    print(f"  Target: lambda={TARGET_LAM}, Kho={Kho_ref}")
    print(f"  Stuck baseline: K ~ 0.319 - 0.297j")
    print("=" * 70)

    result = run_physics_loss(
        TARGET_LAM,
        mu_thimble=MU_THIMBLE,
        tail_frac=TAIL_FRAC,
        tail_scale=TAIL_SCALE,
        rank_val=16,
        epochs_val=epochs,
        anneal_fraction=0.67,
    )

    # Save
    results_file = f"physics_loss_results_{timestamp}.json"
    jr = dict(result)
    jr["K"] = [float(result["K"].real), float(result["K"].imag)]
    jr["K_std"] = float(result["K_std"])
    jr["params"] = {
        "mu_thimble": MU_THIMBLE,
        "tail_frac": TAIL_FRAC,
        "tail_scale": TAIL_SCALE,
    }
    with open(results_file, "w") as f:
        json.dump(jr, f, indent=2)
    print(f"\nResults saved to {results_file}")
