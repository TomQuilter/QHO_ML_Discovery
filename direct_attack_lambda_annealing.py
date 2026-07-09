"""
Direct attack on lambda=0.7 and lambda=0.9 using lambda-annealing.

Strategy: Instead of training at a fixed lambda, we anneal lambda from 0 to the
target value during training. This tracks the correct thimble branch continuously
within a single training run.
"""

import torch, torch.nn as nn, numpy as np, time, sys

torch.manual_seed(42)
np.random.seed(42)

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

TARGET_LAMBDAS = [0.7]

# Reference values for comparison
REFERENCES = {
    0.7: {"Kfd": 0.26988201 - 0.296754314j, "Kho": 0.268440134 - 0.289742841j},
}

T_c = T - 1j * eta

# ─── Action Function ──────────────────────────────────────────────────────────
def complex_action(x, x_i, x_f, lam_val, a_c, m_val, omega_val):
    bsz = x.shape[0]
    pad_i = torch.full((bsz, 1), x_i, dtype=torch.complex64)
    pad_f = torch.full((bsz, 1), x_f, dtype=torch.complex64)
    x_pad = torch.cat([pad_i, x, pad_f], dim=1)
    dx = x_pad[:, 1:] - x_pad[:, :-1]
    kinetic = (m_val / (2 * a_c)) * torch.sum(dx**2, dim=1)
    pot_harm = (a_c * m_val / 2) * omega_val**2 * torch.sum(x**2, dim=1)
    pot_quartic = a_c * (lam_val / 4) * torch.sum(x**4, dim=1)
    return kinetic - pot_harm - pot_quartic

# ─── Initialization Helpers ───────────────────────────────────────────────────
def compute_hessian_A_init(N_val, T_val, m_val, omega_val, D_val, flip_sign=False):
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
    if flip_sign:
        A_np = -A_np
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

# ─── Training with Lambda-Annealing ──────────────────────────────────────────
def run_annealed(target_lam, x_i=0.0, x_f=0.0,
                 N_val=32, T_val=1.0, m_val=1.0, omega_val=1.0, eta_val=0.05,
                 rank_val=16, epochs_val=30000, batch_val=2048, N_mc_val=100_000,
                 seed_val=42, anneal_fraction=0.67, flip_sign=False):
    """
    Train with lambda-annealing: ramp lambda from 0 to target_lam over the first
    anneal_fraction of epochs, then hold at target_lam for the remainder.
    """
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)

    D_val = N_val - 1
    T_c_val = T_val - 1j * eta_val
    a_c = T_c_val / N_val

    # Use improved initialization (exact harmonic thimble)
    b_init = torch.linspace(float(x_i), float(x_f), D_val + 2)[1:-1]
    A_init = compute_hessian_A_init(N_val, T_val, m_val, omega_val, D_val, flip_sign=flip_sign)

    model = ThimbleFlowNL(D_val, rank=rank_val, b_init=b_init, A_init=A_init)

    lr_init = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs_val, eta_min=1e-4)

    loss_history = []
    anneal_epochs = int(anneal_fraction * epochs_val)
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"  Lambda-Annealing: target_lam={target_lam}, flip_sign={flip_sign}")
    print(f"  Annealing 0 -> {target_lam} over epochs 0-{anneal_epochs}")
    print(f"  Fixed at {target_lam} for epochs {anneal_epochs}-{epochs_val}")
    print(f"  Using improved init (exact harmonic thimble)")
    print(f"{'='*70}\n")

    for epoch in range(epochs_val):
        # Lambda annealing schedule
        if epoch < anneal_epochs:
            lam_eff = target_lam * epoch / anneal_epochs
        else:
            lam_eff = target_lam

        optimizer.zero_grad()
        z = torch.randn(batch_val, D_val)
        x, a1, a2 = model(z)
        S = complex_action(x, x_i, x_f, lam_eff, a_c, m_val, omega_val)
        log_P = -0.5 * torch.sum(z**2, dim=1) - (D_val / 2) * np.log(2 * np.pi)
        log_det_J = model.compute_log_det_J(z, a1, a2, detach_correction=True)
        log_W = 1j * S + log_det_J - log_P.to(torch.complex64)

        loss = torch.var(log_W.real) + torch.var(log_W.imag)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        if (epoch + 1) % 5000 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:5d}/{epochs_val} | lam_eff={lam_eff:.4f} | "
                  f"loss = {loss.item():.6e} | {time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0

    # ─── Final MC estimation at TARGET lambda ─────────────────────────────
    norm_factor = (m_val / (2 * np.pi * 1j * a_c)) ** (N_val / 2)
    with torch.no_grad():
        z_mc = torch.randn(N_mc_val, D_val)
        x_mc, a1_mc, a2_mc = model(z_mc)
        S_mc = complex_action(x_mc, x_i, x_f, target_lam, a_c, m_val, omega_val)
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

    print(f"\n  => K = {K_mc:.10f}  |K| = {abs(K_mc):.8f}")
    print(f"     loss = {loss_history[-1]:.2e}  ESS/N = {ess:.4f}  ({elapsed:.0f}s)")

    return {
        "K": K_mc,
        "K_std": K_std,
        "loss": loss_history[-1],
        "ess": ess,
        "time": elapsed,
    }

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  DIRECT ATTACK: Lambda-Annealing + rank=64 (high expressivity)")
    print("=" * 70)
    print("  Runs 1-3: all converge to K=0.319-0.297i regardless of loss/rank")
    print("  Hypothesis: bottleneck MLP too constrained. Trying rank=64.")

    results = {}
    for target_lam in TARGET_LAMBDAS:
        res = run_annealed(
            target_lam,
            x_i=x_i_prop, x_f=x_f_prop,
            N_val=N, T_val=T, m_val=m, omega_val=omega, eta_val=eta,
            rank_val=64, epochs_val=epochs, batch_val=batch,
            N_mc_val=N_mc, seed_val=seed,
            anneal_fraction=0.67,
            flip_sign=True,
        )
        results[target_lam] = res

    # ─── Report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS COMPARISON")
    print("=" * 70)
    print(f"{'lam':>6s} | {'Method':>10s} | {'Re(K)':>12s} | {'Im(K)':>12s} | {'|K|':>10s}")
    print("-" * 70)

    all_within_10pct = True
    for lam in TARGET_LAMBDAS:
        ref = REFERENCES[lam]
        K_ml = results[lam]["K"]

        print(f"{lam:6.1f} | {'Kfd':>10s} | {ref['Kfd'].real:12.9f} | {ref['Kfd'].imag:12.9f} | {abs(ref['Kfd']):10.8f}")
        print(f"{lam:6.1f} | {'Kho':>10s} | {ref['Kho'].real:12.9f} | {ref['Kho'].imag:12.9f} | {abs(ref['Kho']):10.8f}")
        print(f"{lam:6.1f} | {'ML_anneal':>10s} | {K_ml.real:12.9f} | {K_ml.imag:12.9f} | {abs(K_ml):10.8f}")

        # Check 10% criterion against Kho
        Kho = ref["Kho"]
        err_re = abs(K_ml.real - Kho.real) / (abs(Kho.real) + 1e-15)
        err_im = abs(K_ml.imag - Kho.imag) / (abs(Kho.imag) + 1e-15)
        err_abs = abs(abs(K_ml) - abs(Kho)) / abs(Kho)
        within = err_re < 0.10 and err_im < 0.10

        print(f"       | {'Rel Err':>10s} | Re: {err_re*100:.1f}%     | Im: {err_im*100:.1f}%     | |K|: {err_abs*100:.1f}%")
        print(f"       | {'Status':>10s} | {'PASS (<10%)' if within else 'FAIL (>10%)'}")
        print("-" * 70)

        if not within:
            all_within_10pct = False

    if all_within_10pct:
        print("\n*** SUCCESS: All results within 10% of reference! ***")
    else:
        print("\n*** Some results exceed 10% error threshold ***")

    print(f"\nTotal time: {sum(r['time'] for r in results.values()):.0f}s")
