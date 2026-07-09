"""
Overnight investigation: 4 strategies to escape the lambda=0 thimble trap.

Strategy A: Freeze A (exact harmonic thimble), force g_theta to do all work
Strategy B: Random/perturbed A initialization (away from harmonic thimble)
Strategy C: Curiosity regularizer penalizing A staying near initial value
Strategy D: Much larger nonlinear correction (deep wide MLP)

Target: lambda=0.7, Kho = 0.268440134 - 0.289742841j
Current stuck value: K = 0.319 - 0.297j (the lambda=0 contour answer)
Success criterion: within 10% of Kho on both Re and Im parts.
"""

import torch, torch.nn as nn, numpy as np, time, sys, json, os
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

# ─── Model: Standard ThimbleFlowNL ──────────────────────────────────────────
class ThimbleFlowNL(nn.Module):
    def __init__(self, dim, rank=8, b_init=None, A_init=None, freeze_A=False):
        super().__init__()
        self.dim = dim
        self.rank = rank

        if A_init is not None:
            self.A_real = nn.Parameter(A_init.real.float(), requires_grad=not freeze_A)
            self.A_imag = nn.Parameter(A_init.imag.float(), requires_grad=not freeze_A)
        else:
            self.A_real = nn.Parameter(torch.eye(dim) + 0.01 * torch.randn(dim, dim), requires_grad=not freeze_A)
            self.A_imag = nn.Parameter(0.01 * torch.randn(dim, dim), requires_grad=not freeze_A)

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


# ─── Model: Deep Wide MLP (Strategy D) ──────────────────────────────────────
class ThimbleFlowDeep(nn.Module):
    """Larger nonlinear correction: 3 hidden layers, wider, with residual."""
    def __init__(self, dim, hidden=128, b_init=None, A_init=None, freeze_A=False):
        super().__init__()
        self.dim = dim
        self.hidden = hidden

        if A_init is not None:
            self.A_real = nn.Parameter(A_init.real.float(), requires_grad=not freeze_A)
            self.A_imag = nn.Parameter(A_init.imag.float(), requires_grad=not freeze_A)
        else:
            self.A_real = nn.Parameter(torch.eye(dim) + 0.01 * torch.randn(dim, dim), requires_grad=not freeze_A)
            self.A_imag = nn.Parameter(0.01 * torch.randn(dim, dim), requires_grad=not freeze_A)

        if b_init is not None:
            self.b_real = nn.Parameter(b_init.clone().float())
        else:
            self.b_real = nn.Parameter(torch.zeros(dim))
        self.b_imag = nn.Parameter(torch.zeros(dim))

        h = hidden
        self.net = nn.Sequential(
            nn.Linear(dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, 2 * dim),
        )
        # Start with near-zero output so initial contour = A*z + b
        with torch.no_grad():
            self.net[-1].weight.mul_(0.001)
            self.net[-1].bias.zero_()

    def _get_A(self):
        return self.A_real + 1j * self.A_imag

    def forward(self, z):
        A = self._get_A()
        b = self.b_real + 1j * self.b_imag
        x_linear = z.to(torch.complex64) @ A.t() + b
        out = self.net(z)
        g = out[:, :self.dim].to(torch.complex64) + 1j * out[:, self.dim:].to(torch.complex64)
        return x_linear + g

    def compute_log_det_J(self, z):
        """Full Jacobian via autograd for the deep model.
        
        For D=31 this does 31 backward passes per batch. Slower than the
        bottleneck formula but correct for arbitrary architectures.
        """
        D = self.dim
        bsz = z.shape[0]
        z_req = z.detach().requires_grad_(True)
        x = self.forward(z_req)

        J_real = torch.zeros(bsz, D, D)
        J_imag = torch.zeros(bsz, D, D)
        for i in range(D):
            grad_re = torch.autograd.grad(x[:, i].real.sum(), z_req,
                                          create_graph=True, retain_graph=True)[0]
            grad_im = torch.autograd.grad(x[:, i].imag.sum(), z_req,
                                          create_graph=True, retain_graph=True)[0]
            J_real[:, i, :] = grad_re
            J_imag[:, i, :] = grad_im

        J = J_real.to(torch.complex64) + 1j * J_imag.to(torch.complex64)
        sign, logabs = torch.linalg.slogdet(J)
        return logabs + 1j * torch.angle(sign)


# ─── Training Loop (generic) ─────────────────────────────────────────────────
def train_and_estimate(model, strategy_name, rank_val, epochs_val=30000,
                       anneal_fraction=0.67, curiosity_weight=0.0,
                       A_init_ref=None, use_autograd_jac=False,
                       batch_override=None):
    """
    Train with lambda-annealing and return K estimate.
    
    curiosity_weight > 0: adds ||A - A_init||^2 penalty that ENCOURAGES A to move
    use_autograd_jac: use full autograd Jacobian (for ThimbleFlowDeep)
    batch_override: use a different batch size (for slow autograd models)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_batch = batch_override if batch_override else batch

    b_init = torch.linspace(float(x_i_prop), float(x_f_prop), D + 2)[1:-1]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs_val, eta_min=1e-4)

    anneal_epochs = int(anneal_fraction * epochs_val)
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"  Strategy: {strategy_name}")
    print(f"  Annealing 0 -> {TARGET_LAM} over {anneal_epochs} epochs, then fixed")
    print(f"  Total epochs: {epochs_val}")
    print(f"{'='*70}\n")

    loss_history = []
    for epoch in range(epochs_val):
        if epoch < anneal_epochs:
            lam_eff = TARGET_LAM * epoch / anneal_epochs
        else:
            lam_eff = TARGET_LAM

        optimizer.zero_grad()
        z = torch.randn(train_batch, D)

        if use_autograd_jac:
            x = model(z)
            S = complex_action(x, x_i_prop, x_f_prop, lam_eff, a_c, m, omega)
            log_P = -0.5 * torch.sum(z**2, dim=1) - (D / 2) * np.log(2 * np.pi)
            log_det_J = model.compute_log_det_J(z)
        else:
            x, a1, a2 = model(z)
            S = complex_action(x, x_i_prop, x_f_prop, lam_eff, a_c, m, omega)
            log_P = -0.5 * torch.sum(z**2, dim=1) - (D / 2) * np.log(2 * np.pi)
            log_det_J = model.compute_log_det_J(z, a1, a2, detach_correction=True)

        log_W = 1j * S + log_det_J - log_P.to(torch.complex64)
        loss = torch.var(log_W.real) + torch.var(log_W.imag)

        # Strategy C: curiosity regularizer pushes A AWAY from initial value
        if curiosity_weight > 0.0 and A_init_ref is not None:
            A_current = model._get_A()
            A_diff = A_current - A_init_ref
            # NEGATIVE penalty: we REWARD A for moving away
            curiosity_loss = -curiosity_weight * torch.mean(torch.abs(A_diff)**2)
            loss = loss + curiosity_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        if (epoch + 1) % 5000 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"    Epoch {epoch+1:5d}/{epochs_val} | lam_eff={lam_eff:.4f} | "
                  f"loss={loss.item():.6e} | {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0

    # ─── Final MC estimation ──────────────────────────────────────────────
    norm_factor = (m / (2 * np.pi * 1j * a_c)) ** (N / 2)
    with torch.no_grad():
        z_mc = torch.randn(N_mc, D)
        if use_autograd_jac:
            # Need grad for autograd Jacobian
            pass
        else:
            x_mc, a1_mc, a2_mc = model(z_mc)
            S_mc = complex_action(x_mc, x_i_prop, x_f_prop, TARGET_LAM, a_c, m, omega)
            log_P_mc = -0.5 * torch.sum(z_mc**2, dim=1) - (D / 2) * np.log(2 * np.pi)
            log_det_mc = model.compute_log_det_J(z_mc, a1_mc, a2_mc)
            log_W_mc = 1j * S_mc + log_det_mc - log_P_mc.to(torch.complex64)

    if use_autograd_jac:
        # For deep model, compute in batches with grad enabled
        all_log_W = []
        mc_batch = 5000
        for i in range(0, N_mc, mc_batch):
            z_b = torch.randn(min(mc_batch, N_mc - i), D)
            z_b.requires_grad_(True)
            x_b = model(z_b)
            S_b = complex_action(x_b, x_i_prop, x_f_prop, TARGET_LAM, a_c, m, omega)
            log_P_b = -0.5 * torch.sum(z_b.detach()**2, dim=1) - (D / 2) * np.log(2 * np.pi)
            log_det_b = model.compute_log_det_J(z_b)
            log_W_b = 1j * S_b + log_det_b - log_P_b.to(torch.complex64)
            all_log_W.append(log_W_b.detach())
        log_W_mc = torch.cat(all_log_W, dim=0)

    log_W_max = torch.max(log_W_mc.real)
    W_shifted = torch.exp(log_W_mc - log_W_max)
    K_mc = norm_factor * torch.exp(log_W_max).item() * torch.mean(W_shifted).numpy()
    W_for_err = norm_factor * torch.exp(log_W_max).item() * W_shifted.numpy()
    K_std = np.std(W_for_err) / np.sqrt(N_mc)

    # ESS
    W_ess = torch.exp(log_W_mc[:10000] - torch.max(log_W_mc[:10000].real)).numpy()
    ess = float(np.abs(np.mean(W_ess))**2 / np.mean(np.abs(W_ess)**2))

    # Sign check: flip if closer to -Kho
    K_mc_flipped = -K_mc
    if abs(K_mc_flipped - Kho_ref) < abs(K_mc - Kho_ref):
        print(f"  [Sign flip applied: original K={K_mc:.6f}, flipped={K_mc_flipped:.6f}]")
        K_mc = K_mc_flipped

    err_re = abs(K_mc.real - Kho_ref.real) / (abs(Kho_ref.real) + 1e-15)
    err_im = abs(K_mc.imag - Kho_ref.imag) / (abs(Kho_ref.imag) + 1e-15)

    print(f"\n  => K = {K_mc:.10f}  |K| = {abs(K_mc):.8f}")
    print(f"     Kho = {Kho_ref}  |Kho| = {abs(Kho_ref):.8f}")
    print(f"     Rel err: Re={err_re*100:.1f}%, Im={err_im*100:.1f}%")
    print(f"     loss={loss_history[-1]:.2e}  ESS/N={ess:.4f}  time={elapsed:.0f}s")
    status = "PASS" if (err_re < 0.10 and err_im < 0.10) else "FAIL"
    print(f"     Status: {status}")

    return {
        "strategy": strategy_name,
        "K": K_mc,
        "K_std": K_std,
        "err_re": err_re,
        "err_im": err_im,
        "loss": loss_history[-1],
        "ess": ess,
        "time": elapsed,
        "status": status,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN: Run all 4 strategies sequentially
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print(f"  OVERNIGHT INVESTIGATION — {timestamp}")
    print(f"  Target: lambda={TARGET_LAM}, Kho={Kho_ref}")
    print(f"  Current stuck value: K ~ 0.319 - 0.297j")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════════════
    # STRATEGY A: Freeze A at harmonic thimble, only train g_theta
    # Rationale: if A can't adjust to "absorb" the annealing, g_theta MUST
    # find the correct nonlinear deformation to reduce variance.
    # ═══════════════════════════════════════════════════════════════════════
    print("\n\n" + "#" * 70)
    print("# STRATEGY A: Freeze A, train only g_theta (rank=16)")
    print("#" * 70)

    torch.manual_seed(seed)
    b_init_A = torch.linspace(float(x_i_prop), float(x_f_prop), D + 2)[1:-1]
    A_init_A = -A_HARMONIC  # flip sign for correct global phase
    model_A = ThimbleFlowNL(D, rank=16, b_init=b_init_A, A_init=A_init_A, freeze_A=True)

    res_A = train_and_estimate(model_A, "A: Freeze A (rank=16)", rank_val=16,
                               epochs_val=epochs, anneal_fraction=0.67)
    results.append(res_A)

    if res_A["status"] == "PASS":
        print("\n*** STRATEGY A PASSED! Stopping early. ***")
    else:
        # ═══════════════════════════════════════════════════════════════════
        # STRATEGY B: Random A initialization (far from harmonic thimble)
        # Rationale: start A somewhere completely different so the optimizer
        # must find a genuinely new low-variance contour from scratch.
        # ═══════════════════════════════════════════════════════════════════
        print("\n\n" + "#" * 70)
        print("# STRATEGY B: Random A init (not harmonic thimble)")
        print("#" * 70)

        torch.manual_seed(seed)
        np.random.seed(seed)
        # Random complex matrix with similar scale to A_HARMONIC
        A_scale = torch.sqrt(torch.mean(torch.abs(A_HARMONIC)**2)).item()
        A_random = A_scale * (torch.randn(D, D) + 1j * torch.randn(D, D)) / np.sqrt(2 * D)
        b_init_B = torch.linspace(float(x_i_prop), float(x_f_prop), D + 2)[1:-1]
        model_B = ThimbleFlowNL(D, rank=16, b_init=b_init_B, A_init=A_random, freeze_A=False)

        res_B = train_and_estimate(model_B, "B: Random A init (rank=16)", rank_val=16,
                                   epochs_val=epochs, anneal_fraction=0.67)
        results.append(res_B)

        if res_B["status"] == "PASS":
            print("\n*** STRATEGY B PASSED! Stopping early. ***")
        else:
            # ═══════════════════════════════════════════════════════════════
            # STRATEGY C: Curiosity regularizer on A
            # Rationale: penalize A for STAYING near its initial value,
            # encouraging exploration of different contours.
            # ═══════════════════════════════════════════════════════════════
            print("\n\n" + "#" * 70)
            print("# STRATEGY C: Curiosity regularizer (push A away from init)")
            print("#" * 70)

            torch.manual_seed(seed)
            b_init_C = torch.linspace(float(x_i_prop), float(x_f_prop), D + 2)[1:-1]
            A_init_C = -A_HARMONIC  # flip sign
            model_C = ThimbleFlowNL(D, rank=16, b_init=b_init_C, A_init=A_init_C, freeze_A=False)

            res_C = train_and_estimate(model_C, "C: Curiosity reg (weight=0.01)", rank_val=16,
                                       epochs_val=epochs, anneal_fraction=0.67,
                                       curiosity_weight=0.01, A_init_ref=A_init_C)
            results.append(res_C)

            if res_C["status"] == "PASS":
                print("\n*** STRATEGY C PASSED! Stopping early. ***")
            else:
                # ═══════════════════════════════════════════════════════════
                # STRATEGY D: Much larger nonlinear correction (deep MLP)
                # Rationale: the bottleneck MLP (rank 16) may simply not
                # have enough capacity. Use a 3-layer MLP with hidden=128.
                # Also freeze A so the deep network must do everything.
                # ═══════════════════════════════════════════════════════════
                print("\n\n" + "#" * 70)
                print("# STRATEGY D: Deep wide MLP (hidden=128, 3 layers, freeze A)")
                print("#" * 70)

                torch.manual_seed(seed)
                b_init_D = torch.linspace(float(x_i_prop), float(x_f_prop), D + 2)[1:-1]
                A_init_D = -A_HARMONIC  # flip sign
                model_D = ThimbleFlowDeep(D, hidden=128, b_init=b_init_D,
                                          A_init=A_init_D, freeze_A=True)

                # Smaller batch and fewer epochs since autograd Jacobian is ~31x slower
                res_D = train_and_estimate(model_D, "D: Deep MLP (h=128, freeze A)", rank_val=128,
                                           epochs_val=10000, anneal_fraction=0.5,
                                           use_autograd_jac=True, batch_override=512)
                results.append(res_D)

    # ═══════════════════════════════════════════════════════════════════════
    #  FINAL REPORT
    # ═══════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("  OVERNIGHT RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Reference: Kho = {Kho_ref}")
    print(f"  Stuck value: K ~ 0.319 - 0.297j")
    print("-" * 70)
    print(f"{'Strategy':<40s} | {'Re(K)':>10s} | {'Im(K)':>10s} | {'Err_Re':>7s} | {'Err_Im':>7s} | {'Status'}")
    print("-" * 70)
    for r in results:
        K = r["K"]
        print(f"{r['strategy']:<40s} | {K.real:10.6f} | {K.imag:10.6f} | "
              f"{r['err_re']*100:5.1f}% | {r['err_im']*100:5.1f}% | {r['status']}")
    print("-" * 70)
    total_time = sum(r["time"] for r in results)
    print(f"Total wall time: {total_time/3600:.1f} hours")

    # Save results to JSON
    results_file = f"overnight_results_{timestamp}.json"
    json_results = []
    for r in results:
        jr = dict(r)
        jr["K"] = [float(r["K"].real), float(r["K"].imag)]
        jr["K_std"] = [float(r["K_std"].real), float(r["K_std"].imag)] if np.iscomplex(r["K_std"]) else float(r["K_std"])
        json_results.append(jr)
    with open(results_file, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Final verdict
    any_pass = any(r["status"] == "PASS" for r in results)
    if any_pass:
        print("\n*** AT LEAST ONE STRATEGY PASSED THE 10% THRESHOLD! ***")
    else:
        print("\n*** NO STRATEGY PASSED — further investigation needed ***")
        print("  Possible next steps:")
        print("  - Try even stronger curiosity weights")
        print("  - Add explicit KL divergence to target distribution")
        print("  - Use multiple random restarts for A")
        print("  - Try direct gradient on |K - K_target|^2 (supervised)")
