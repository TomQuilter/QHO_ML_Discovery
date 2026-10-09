"""
NUCLEAR OPTION: Supervised training with known Kho targets.

Tests whether the ThimbleFlowNL architecture (rank=16 bottleneck MLP)
can represent a contour that yields the correct propagator K.

If this works: the architecture is fine, the unsupervised loss is the problem.
If this fails: the architecture can't reach the correct homology class.

Two targets:
  lambda=0.7:  Kho = 0.268440134 - 0.289742841j
  lambda=2.5:  Kho = -0.111802649 - 0.157738326j

Loss = Var(Re logW) + Var(Im logW) + alpha * min(|K_est - Kho|^2, |K_est + Kho|^2)

alpha is held at 0 for WARMUP_PURE epochs (pure variance loss), then ramped
0 -> ALPHA_MAX over ALPHA_RAMP epochs, so sampling stabilises before the
anchor kicks in.
"""

import torch, torch.nn as nn, numpy as np, time, json
import matplotlib.pyplot as plt
from datetime import datetime

torch.set_num_threads(4)

# ─── Parameters ───────────────────────────────────────────────────────────────
N       = 32
T       = 1.0
eta     = 0.05
m       = 1.0
omega   = 1.0
D       = N - 1
rank    = 16
batch   = 2048
N_mc    = 100_000
epochs  = 30000
seed    = 42

x_i_prop = 0.0
x_f_prop = 0.0

T_c = T - 1j * eta
a_c = T_c / N
norm_factor = (m / (2 * np.pi * 1j * a_c)) ** (N / 2)

# Supervised targets
#TARGETS = {
#    0.7: 0.268440134 - 0.289742841j,
#    2.5: -0.111802649 - 0.157738326j,
#}

TARGETS = {
    #2.5: 0.1561802649 + 0.07738326j,  
     2.5: -0.111802649 - 0.157738326j,  # Ross actual answer 
}

# Supervised loss hyperparameters
ALPHA_MAX = 5.0         # max weight of supervised term (relative loss is O(1))
WARMUP_PURE = 3000      # epochs of PURE variance loss (no supervised graph at all)
ALPHA_RAMP = 5000       # epochs to ramp alpha 0 -> ALPHA_MAX after pure warmup

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


# ─── Supervised Training ──────────────────────────────────────────────────────
def run_supervised(lam_val, K_target):
    """
    Train with variance loss + supervised |K - Kho|^2 anchor.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    b_init = torch.linspace(float(x_i_prop), float(x_f_prop), D + 2)[1:-1]
    A_init = compute_hessian_A_init(N, T, m, omega, D)
    model = ThimbleFlowNL(D, rank=rank, b_init=b_init, A_init=A_init)
 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-4)

    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"  SUPERVISED TEST: lambda={lam_val}")
    print(f"  K_target = {K_target}")
    print(f"  alpha_max={ALPHA_MAX}, pure_warmup={WARMUP_PURE}, ramp={ALPHA_RAMP} epochs")
    print(f"  rank={rank}, epochs={epochs}, batch={batch}")
    print(f"{'='*70}\n")

    loss_history = []
    K_est_history = []

    K_target_t = torch.tensor(K_target, dtype=torch.complex128)
    Kho_abs2 = torch.abs(K_target_t)**2
    nf_64 = torch.tensor(complex(norm_factor), dtype=torch.complex128)
    n_nan_skips = 0

    for epoch in range(epochs):
        # Alpha schedule: pure variance warmup, then linear ramp, then constant
        if epoch < WARMUP_PURE:
            alpha = 0.0
        elif epoch < WARMUP_PURE + ALPHA_RAMP:
            alpha = ALPHA_MAX * (epoch - WARMUP_PURE) / ALPHA_RAMP
        else:
            alpha = ALPHA_MAX

        optimizer.zero_grad()
        z = torch.randn(batch, D)
        x, a1, a2 = model(z)
        S = complex_action(x, x_i_prop, x_f_prop, lam_val, a_c, m, omega)
        log_P = -0.5 * torch.sum(z**2, dim=1) - (D / 2) * np.log(2 * np.pi)
        log_det_J = model.compute_log_det_J(z, a1, a2, detach_correction=True)
        log_W = 1j * S + log_det_J - log_P.to(torch.complex64)

        # Standard variance loss
        loss_var = torch.var(log_W.real) + torch.var(log_W.imag)

        # Supervised term only built when alpha > 0 (avoids unstable graph in warmup)
        if alpha > 0.0:
            # Differentiable K estimate in float64 to avoid float32 overflow
            log_W_64 = log_W.to(torch.complex128)
            log_W_max = torch.max(log_W_64.real).detach()
            W_shifted = torch.exp(log_W_64 - log_W_max)
            K_est = nf_64 * torch.exp(log_W_max) * torch.mean(W_shifted)
            # Sign-robust, RELATIVE loss (O(1))
            err_plus = torch.abs(K_est - K_target_t)**2 / Kho_abs2
            err_minus = torch.abs(K_est + K_target_t)**2 / Kho_abs2
            loss_sup = torch.min(err_plus, err_minus).float()
            # loss = loss_var + alpha * loss_sup
            loss = alpha * loss_sup 
        else:
            # Cheap no-grad K estimate for logging only
            with torch.no_grad():
                log_W_64 = log_W.to(torch.complex128)
                log_W_max = torch.max(log_W_64.real)
                W_shifted = torch.exp(log_W_64 - log_W_max)
                K_est = nf_64 * torch.exp(log_W_max) * torch.mean(W_shifted)
            loss_sup = torch.zeros(1)
            loss = loss_var

        # Guard 1: skip if loss itself is non-finite
        if not torch.isfinite(loss):
            n_nan_skips += 1
            optimizer.zero_grad()
            loss_history.append(float('nan'))
            K_est_history.append(complex(K_est.detach()))
            if n_nan_skips <= 5 or n_nan_skips % 500 == 0:
                print(f"  [non-finite LOSS at epoch {epoch+1}, skip #{n_nan_skips}]", flush=True)
            continue

        loss.backward()

        # Guard 2: skip step if ANY gradient is non-finite (prevents clip from
        # propagating NaN/Inf into every weight and corrupting the whole model)
        grads_finite = all(
            (p.grad is None) or torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        if not grads_finite:
            n_nan_skips += 1
            optimizer.zero_grad()
            loss_history.append(float('nan'))
            K_est_history.append(complex(K_est.detach()))
            if n_nan_skips <= 5 or n_nan_skips % 500 == 0:
                print(f"  [non-finite GRAD at epoch {epoch+1}, skip #{n_nan_skips}]", flush=True)
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        K_est_history.append(complex(K_est.detach()))

        if (epoch + 1) % 1000 == 0 or epoch == 0:
            K_now = K_est_history[-1]
            elapsed = time.time() - t0
            err_re = abs(K_now.real - K_target.real) / (abs(K_target.real) + 1e-15)
            err_im = abs(K_now.imag - K_target.imag) / (abs(K_target.imag) + 1e-15)
            logW_max = log_W.real.max().item()
            sup_val = loss_sup.item() if loss_sup.numel() == 1 else float(loss_sup)
            print(f"  Epoch {epoch+1:5d}/{epochs} | loss={loss.item():.4e} | "
                  f"Var={loss_var.item():.4e} | sup={sup_val:.4e} | "
                  f"a={alpha:.2f} | logW_max={logW_max:.1f}", flush=True)
            print(f"         K_est={K_now:.6f} | err Re={err_re*100:.1f}% Im={err_im*100:.1f}% | "
                  f"{elapsed:.0f}s", flush=True)

    if n_nan_skips > 0:
        print(f"  [Total non-finite skips: {n_nan_skips}/{epochs}]", flush=True)

    elapsed = time.time() - t0

    # ─── Final MC estimation (100k, no supervised bias) ───────────────────
    with torch.no_grad():
        z_mc = torch.randn(N_mc, D)
        x_mc, a1_mc, a2_mc = model(z_mc)
        S_mc = complex_action(x_mc, x_i_prop, x_f_prop, lam_val, a_c, m, omega)
        log_P_mc = -0.5 * torch.sum(z_mc**2, dim=1) - (D / 2) * np.log(2 * np.pi)
        log_det_mc = model.compute_log_det_J(z_mc, a1_mc, a2_mc)
        log_W_mc = 1j * S_mc + log_det_mc - log_P_mc.to(torch.complex64)

        log_W_max_mc = torch.max(log_W_mc.real)
        W_shifted_mc = torch.exp(log_W_mc - log_W_max_mc)
        K_final = norm_factor * torch.exp(log_W_max_mc).item() * torch.mean(W_shifted_mc).numpy()
        W_for_err = norm_factor * torch.exp(log_W_max_mc).item() * W_shifted_mc.numpy()
        K_std = np.std(W_for_err) / np.sqrt(N_mc)

        W_ess = torch.exp(log_W_mc[:10000] - torch.max(log_W_mc[:10000].real)).numpy()
        ess = float(np.abs(np.mean(W_ess))**2 / np.mean(np.abs(W_ess)**2))

    # Check sign
    K_final_flip = -K_final
    if abs(K_final_flip - K_target) < abs(K_final - K_target):
        print(f"  [Sign flip: {K_final:.6f} -> {K_final_flip:.6f}]")
        K_final = K_final_flip

    err_re = abs(K_final.real - K_target.real) / (abs(K_target.real) + 1e-15)
    err_im = abs(K_final.imag - K_target.imag) / (abs(K_target.imag) + 1e-15)
    status = "PASS" if (err_re < 0.10 and err_im < 0.10) else "FAIL"

    print(f"\n{'='*70}")
    print(f"  FINAL RESULT (100k MC, lambda={lam_val})")
    print(f"{'='*70}")
    print(f"  K_ML     = {K_final:.10f}   |K| = {abs(K_final):.8f}")
    print(f"  K_target = {K_target}   |K| = {abs(K_target):.8f}")
    print(f"  Rel err: Re = {err_re*100:.2f}%,  Im = {err_im*100:.2f}%")
    print(f"  Var(logW) = {loss_var.item():.4e}")
    print(f"  ESS/N = {ess:.4f}")
    print(f"  Time = {elapsed:.0f}s")
    print(f"  Status: {status}")
    print(f"{'='*70}")

    # ─── Arrow output (Option A) for lambda=2.5 ─────────────────────────
    # Save normalized arrow data for before/after contour.
    if abs(lam_val - 2.5) < 1e-12:
        n_arrows = 200
        with torch.no_grad():
            z_arr = torch.randn(n_arrows, D)
            S_before = complex_action(
                z_arr.to(torch.complex64), x_i_prop, x_f_prop, lam_val, a_c, m, omega
            )
            W_before = torch.exp(1j * S_before).numpy()
            W_before_norm = W_before / np.max(np.abs(W_before))

            x_arr, a1_arr, a2_arr = model(z_arr)
            S_after = complex_action(x_arr, x_i_prop, x_f_prop, lam_val, a_c, m, omega)
            log_P_arr = -0.5 * torch.sum(z_arr**2, dim=1) - (D / 2) * np.log(2 * np.pi)
            log_det_arr = model.compute_log_det_J(z_arr, a1_arr, a2_arr)
            log_W_arr = 1j * S_after + log_det_arr - log_P_arr.to(torch.complex64)
            W_after = torch.exp(log_W_arr - torch.max(log_W_arr.real)).numpy()
            W_after_norm = W_after / np.max(np.abs(W_after))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        arrows_file = f"supervised_kho_results_lambda{lam_val:.1f}_arrows_{timestamp}.npz"
        np.savez(
            arrows_file,
            W_before_norm=W_before_norm,
            W_after_norm=W_after_norm,
            lam_val=lam_val,
            n_arrows=n_arrows,
        )
        print(f"\nArrow data saved to {arrows_file}")

        # Arrow diagram (matches notebook style)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, W, title in [
            (axes[0], W_before_norm, "BEFORE training (raw e^{iS})"),
            (axes[1], W_after_norm, "AFTER training (thimble)"),
        ]:
            ax.set_aspect("equal")
            for w in W:
                ax.annotate(
                    "",
                    xy=(w.real, w.imag),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="steelblue", alpha=0.3),
                )
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.axhline(0, color="grey", lw=0.5)
            ax.axvline(0, color="grey", lw=0.5)
            ax.set_xlabel("Re(W)")
            ax.set_ylabel("Im(W)")
            ax.set_title(title)
        fig.suptitle(f"Weight arrows: lam={lam_val:.1f}, T_c={T}-{eta}i", fontsize=13)
        plt.tight_layout()
        arrows_png = f"supervised_kho_results_lambda{lam_val:.1f}_arrows_{timestamp}.png"
        plt.savefig(arrows_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Arrow PNG saved to {arrows_png}")

        # Loss curve (matches notebook style)
        loss_arr = np.array(loss_history, dtype=np.float64)
        valid = np.isfinite(loss_arr)
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, len(loss_arr) + 1)[valid], loss_arr[valid], lw=1.5)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Var[log W])")
        plt.title(f"Loss curve: lam={lam_val:.1f}, T_c={T}-{eta}i")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_png = f"supervised_kho_results_lambda{lam_val:.1f}_loss_curve_{timestamp}.png"
        plt.savefig(loss_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Loss curve PNG saved to {loss_png}")

    # ─── Print final model weights ────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  FINAL MODEL WEIGHTS (lambda={lam_val})")
    print(f"{'─'*70}")

    A_final = model._get_A().detach().cpu().numpy()
    print(f"\n  A matrix ({D}x{D} complex):")
    print(f"    |A|_fro = {np.linalg.norm(A_final):.6f}")
    print(f"    det(A)  = {np.linalg.det(A_final):.6e}")
    print(f"    A[0:5, 0:5] (top-left corner):")
    for i in range(min(5, D)):
        row = "    " + "  ".join(f"{A_final[i,j]:.4f}" for j in range(min(5, D)))
        print(row)

    print(f"\n  b (bias):")
    b_final = (model.b_real + 1j * model.b_imag).detach().cpu().numpy()
    print(f"    b[0:5] = {b_final[:5]}")
    print(f"    |b| = {np.linalg.norm(b_final):.6f}")

    print(f"\n  MLP weights (bottleneck rank={rank}):")
    W1 = model.W1.detach().cpu().numpy()
    W2 = model.W2.detach().cpu().numpy()
    W3 = model.W3.detach().cpu().numpy()
    b1 = model.b1.detach().cpu().numpy()
    b2 = model.b2.detach().cpu().numpy()
    b3 = model.b3.detach().cpu().numpy()
    print(f"    W1: shape={W1.shape}, |W1|_fro={np.linalg.norm(W1):.6f}, max={np.max(np.abs(W1)):.6f}")
    print(f"    b1: shape={b1.shape}, |b1|={np.linalg.norm(b1):.6f}")
    print(f"    W2: shape={W2.shape}, |W2|_fro={np.linalg.norm(W2):.6f}, max={np.max(np.abs(W2)):.6f}")
    print(f"    b2: shape={b2.shape}, |b2|={np.linalg.norm(b2):.6f}")
    print(f"    W3: shape={W3.shape}, |W3|_fro={np.linalg.norm(W3):.6f}, max={np.max(np.abs(W3)):.6f}")
    print(f"    b3: shape={b3.shape}, |b3|={np.linalg.norm(b3):.6f}")

    # Measure how much g_theta contributes relative to A*z
    with torch.no_grad():
        z_test = torch.randn(1000, D)
        x_full, _, _ = model(z_test)
        A_t = model._get_A()
        b_t = model.b_real + 1j * model.b_imag
        x_linear = z_test.to(torch.complex64) @ A_t.t() + b_t
        g_contrib = x_full - x_linear
        ratio = torch.mean(torch.abs(g_contrib)) / torch.mean(torch.abs(x_linear))
    print(f"\n  g_theta contribution: |g|/|Az+b| = {ratio.item():.4f} (ratio of mean magnitudes)")

    print(f"{'─'*70}\n")

    return {
        "lam": lam_val,
        "K_final": K_final,
        "K_target": K_target,
        "K_std": K_std,
        "err_re": err_re,
        "err_im": err_im,
        "loss_var": loss_var.item(),
        "ess": ess,
        "time": elapsed,
        "status": status,
        "model_state": {k: v.detach().clone() for k, v in model.state_dict().items()},
    }


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print(f"  NUCLEAR OPTION: Supervised with Kho — {timestamp}")
    print(f"  Can the architecture REPRESENT the correct contour?")
    print("=" * 70)

    all_results = []

    for lam_val, K_target in TARGETS.items():
        res = run_supervised(lam_val, K_target)
        all_results.append(res)
        print()

    # ─── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"{'Lambda':<8} | {'K_ML':<28} | {'K_target':<28} | {'Err_Re':>7} | {'Err_Im':>7} | {'Status'}")
    print("-" * 100)
    for r in all_results:
        K = r["K_final"]
        Kt = r["K_target"]
        print(f"{r['lam']:<8.1f} | {K.real:+.8f}{K.imag:+.8f}j | "
              f"{Kt.real:+.8f}{Kt.imag:+.8f}j | "
              f"{r['err_re']*100:5.1f}% | {r['err_im']*100:5.1f}% | {r['status']}")
    print("-" * 100)

    # Verdict
    all_pass = all(r["status"] == "PASS" for r in all_results)
    if all_pass:
        print("\n*** ALL PASSED: Architecture CAN represent the correct contour! ***")
        print("    => The unsupervised variance loss is the bottleneck, not capacity.")
    else:
        passed = [r for r in all_results if r["status"] == "PASS"]
        failed = [r for r in all_results if r["status"] == "FAIL"]
        if passed:
            print(f"\n  PARTIAL: {len(passed)} passed, {len(failed)} failed")
            for r in failed:
                print(f"    FAIL: lambda={r['lam']} (err Re={r['err_re']*100:.1f}%, Im={r['err_im']*100:.1f}%)")
        else:
            print("\n*** ALL FAILED: Architecture may be too restrictive! ***")
            print("    => Need larger rank, different parameterisation, or different approach.")

    # Save
    results_file = f"supervised_kho_results_{timestamp}.json"
    json_out = []
    for r in all_results:
        jr = {
            "lam": r["lam"],
            "K_final": [float(r["K_final"].real), float(r["K_final"].imag)],
            "K_target": [float(r["K_target"].real), float(r["K_target"].imag)],
            "err_re": r["err_re"],
            "err_im": r["err_im"],
            "loss_var": r["loss_var"],
            "ess": r["ess"],
            "time": r["time"],
            "status": r["status"],
        }
        json_out.append(jr)
    with open(results_file, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nResults saved to {results_file}")
    print(f"Total time: {sum(r['time'] for r in all_results)/60:.1f} min")
