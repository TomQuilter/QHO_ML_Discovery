"""
Anharmonic Oscillator: Nonlinear Thimble via Neural Network
============================================================
Extends the linear thimble  x = A^T z + b  with a low-rank NN correction:

    x = A^T z + b + g_theta(z)

where g is a bottleneck MLP  z -[D,r]-> tanh -[r,r]-> tanh -[r,2D]-> (re,im).
Because g's Jacobian has rank <= r, the matrix-determinant lemma lets us
compute  det(A^T + dg/dz)  via an r x r determinant instead of D x D,
making training nearly as fast as the purely linear model.

Usage:
    python QHO_Anharmonic_ML_NN.py --lam 0.0          # sanity check
    python QHO_Anharmonic_ML_NN.py --lam 0.05
    python QHO_Anharmonic_ML_NN.py --lam 2.0
    python QHO_Anharmonic_ML_NN.py --lam 8.0
    python QHO_Anharmonic_ML_NN.py --lam 0.2 --linear_only
"""

import argparse, time
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

def pr(*a, **kw):
    print(*a, **kw, flush=True)

# ── CLI ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--lam",    type=float, default=0.2)
parser.add_argument("--epochs", type=int,   default=15000)
parser.add_argument("--batch",  type=int,   default=2048)
parser.add_argument("--rank",   type=int,   default=8,
                    help="Bottleneck rank for NN correction (default 8)")
parser.add_argument("--n_mc",   type=int,   default=100_000)
parser.add_argument("--linear_only", action="store_true",
                    help="Disable NN correction (pure linear baseline)")
args = parser.parse_args()

# ── physics ───────────────────────────────────────────────────────────
N       = 16    
T       = 1.0
a_dt    = T / N
m       = 1.0
omega   = 2.0
epsilon = 0.01
D       = N - 1
lam     = args.lam

pr(f"Parameters: N={N}, T={T}, m={m}, omega={omega}, eps={epsilon}, lam={lam}")
pr(f"D={D}, epochs={args.epochs}, batch={args.batch}, rank={args.rank}")
pr(f"Mode: {'LINEAR ONLY (baseline)' if args.linear_only else 'LINEAR + NN CORRECTION'}")
pr()

# ── action ────────────────────────────────────────────────────────────
def complex_action(x):
    bsz = x.shape[0]
    zeros = torch.zeros((bsz, 1), dtype=torch.complex64)
    x_pad = torch.cat([zeros, x, zeros], dim=1)
    dx = x_pad[:, 1:] - x_pad[:, :-1]
    kinetic = (m / (2 * a_dt)) * torch.sum(dx**2, dim=1)
    omega_sq = omega**2 - 1j * epsilon
    pot_harm = (a_dt * m / 2) * omega_sq * torch.sum(x**2, dim=1)
    pot_quartic = a_dt * (lam / 4) * torch.sum(x**4, dim=1)
    return kinetic - pot_harm - pot_quartic


class ThimbleFlowNL(nn.Module):
    """
    x = A^T z + b  +  g(z)

    g is a rank-r bottleneck MLP:
        z -[W1: r x D]-> tanh -[W2: r x r]-> tanh -[W3: 2D x r]-> split -> complex

    Jacobian:  J = A^T + dg/dz  where  dg/dz = C @ D2 @ W2 @ D1 @ W1
    with C = W3[:D]+iW3[D:] shape [D,r].

    Matrix-determinant lemma reduces  det J  to  det(A^T) * det(I_r + V^T A^{-T} C)
    where V^T = D2 @ W2 @ D1 @ W1 is [r,D] per-sample.
    So we only ever compute  slogdet  of [batch, r, r] matrices.
    """
    def __init__(self, dim, rank=8, use_nn=True):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.use_nn = use_nn

        self.A_real = nn.Parameter(torch.eye(dim) + 0.01 * torch.randn(dim, dim))
        self.A_imag = nn.Parameter(0.01 * torch.randn(dim, dim))
        self.b_real = nn.Parameter(torch.zeros(dim))
        self.b_imag = nn.Parameter(torch.zeros(dim))

        if use_nn:
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
        """Returns x [batch, D] complex, and intermediates (a1, a2) for Jacobian."""
        A = self._get_A()
        b = self.b_real + 1j * self.b_imag
        x = z.to(torch.complex64) @ A.t() + b

        if not self.use_nn:
            return x, None, None

        h1 = z @ self.W1.t() + self.b1       # [batch, r]
        a1 = torch.tanh(h1)
        h2 = a1 @ self.W2.t() + self.b2      # [batch, r]
        a2 = torch.tanh(h2)
        out = a2 @ self.W3.t() + self.b3      # [batch, 2D]
        g = out[:, :self.dim].to(torch.complex64) + 1j * out[:, self.dim:].to(torch.complex64)
        x = x + g
        return x, a1, a2

    def compute_log_det_J(self, z, a1, a2, detach_correction=False):
        """
        log det J  using the matrix-determinant lemma.

        J = A^T + C @ D2 @ W2 @ D1 @ W1
        det J = det(A^T) * det(I_r + D2 @ W2 @ D1 @ (W1 @ A^{-T} @ C))

        If detach_correction=True the per-sample correction is detached from
        the autograd graph (faster backprop; NN still learns through the action).
        Returns [batch] complex tensor.
        """
        A = self._get_A()
        sign_A, logabs_A = torch.linalg.slogdet(A)
        log_det_A = logabs_A + 1j * torch.angle(sign_A)

        if not self.use_nn:
            return log_det_A.expand(z.shape[0])

        C = self.W3[:self.dim].to(torch.complex64) + 1j * self.W3[self.dim:].to(torch.complex64)
        A_invT_C = torch.linalg.solve(A.t(), C)
        K = self.W1.to(torch.complex64) @ A_invT_C

        batch = z.shape[0]
        r = self.rank

        _a1 = a1.detach() if detach_correction else a1
        _a2 = a2.detach() if detach_correction else a2

        da1 = (1.0 - _a1**2).to(torch.complex64)
        M1 = da1.unsqueeze(-1) * K.unsqueeze(0)

        W2c = self.W2.to(torch.complex64)
        if detach_correction:
            W2c = W2c.detach()
        M2 = torch.bmm(W2c.unsqueeze(0).expand(batch, -1, -1), M1)

        da2 = (1.0 - _a2**2).to(torch.complex64)
        M3 = da2.unsqueeze(-1) * M2

        I_r = torch.eye(r, dtype=torch.complex64).unsqueeze(0)
        sign_c, logabs_c = torch.linalg.slogdet(I_r + M3)
        log_det_correction = logabs_c + 1j * torch.angle(sign_c)

        if detach_correction:
            log_det_correction = log_det_correction.detach()

        return log_det_A + log_det_correction


# ── build ─────────────────────────────────────────────────────────────
model = ThimbleFlowNL(D, rank=args.rank, use_nn=not args.linear_only)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=1e-4)

n_params = sum(p.numel() for p in model.parameters())
pr(f"Trainable parameters: {n_params}")
pr()

# ── train ─────────────────────────────────────────────────────────────
loss_history = []
t0 = time.time()

for epoch in range(args.epochs):
    optimizer.zero_grad()

    z = torch.randn(args.batch, D)
    x, a1, a2 = model(z)
    S = complex_action(x)

    log_P = -0.5 * torch.sum(z**2, dim=1) - (D / 2) * np.log(2 * np.pi)

    # Jacobian: keep gradient through det(A) but detach the per-sample
    # NN correction — the NN optimises primarily through the action S(x).
    # This makes backprop ~2x faster with negligible accuracy cost.
    log_det_J = model.compute_log_det_J(z, a1, a2, detach_correction=True)
    log_W = 1j * S + log_det_J - log_P.to(torch.complex64)

    loss = torch.var(log_W.real) + torch.var(log_W.imag)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    scheduler.step()

    loss_history.append(loss.item())

    if (epoch + 1) % 1000 == 0 or epoch == 0:
        elapsed = time.time() - t0
        pr(f"  Epoch {epoch+1:5d}/{args.epochs} | loss = {loss.item():.6e} | {elapsed:.0f}s")

elapsed = time.time() - t0
pr(f"\nTraining complete in {elapsed:.1f}s")
pr(f"  Final loss: {loss_history[-1]:.6e}")
pr()

# ── MC propagator ─────────────────────────────────────────────────────
N_mc = args.n_mc
norm_factor = (m / (2 * np.pi * 1j * a_dt)) ** (N / 2)

with torch.no_grad():
    z_mc = torch.randn(N_mc, D)
    x_mc, a1_mc, a2_mc = model(z_mc)
    S_mc = complex_action(x_mc)
    log_P_mc = -0.5 * torch.sum(z_mc**2, dim=1) - (D / 2) * np.log(2 * np.pi)
    log_det_mc = model.compute_log_det_J(z_mc, a1_mc, a2_mc)
    log_W_mc = 1j * S_mc + log_det_mc - log_P_mc.to(torch.complex64)
    log_W_max = torch.max(log_W_mc.real)
    W_shifted = torch.exp(log_W_mc - log_W_max)
    K_mc = norm_factor * torch.exp(log_W_max).item() * torch.mean(W_shifted).numpy()
    W_for_err = norm_factor * torch.exp(log_W_max).item() * W_shifted.numpy()
    K_std = np.std(W_for_err) / np.sqrt(N_mc)

pr(f"MC propagator K(0, T; 0, 0):")
pr(f"  K_mc       = {K_mc:.10f}")
pr(f"  |K_mc|     = {abs(K_mc):.10f}")
pr(f"  std error  = {K_std:.2e}")
pr(f"  ({N_mc:,} samples)")
pr()

# ── ESS ───────────────────────────────────────────────────────────────
with torch.no_grad():
    W_ess = torch.exp(log_W_mc[:10000] - torch.max(log_W_mc[:10000].real)).numpy()
    ess = np.abs(np.mean(W_ess))**2 / np.mean(np.abs(W_ess)**2)
    pr(f"ESS/N = {ess:.4f}  (on 10k samples)")
    pr()

# ── Mehler sanity (lam=0) ────────────────────────────────────────────
K_mehler = np.sqrt(m * omega / (2 * np.pi * 1j * np.sin(omega * T)))
if abs(lam) < 1e-12:
    rel = abs(K_mc - K_mehler) / abs(K_mehler)
    pr(f"SANITY CHECK (lam=0):")
    pr(f"  K_mc    = {K_mc:.10f}")
    pr(f"  Mehler  = {K_mehler:.10f}")
    pr(f"  rel err = {rel:.2e}")
    pr(f"  >> {'PASSED' if rel < 0.01 else 'FAILED'}")
    pr()

# ── summary ───────────────────────────────────────────────────────────
pr("=" * 60)
pr(f"SUMMARY  lam={lam}")
pr(f"  Mode:       {'linear only' if args.linear_only else f'linear + NN (rank={args.rank})'}")
pr(f"  Final loss: {loss_history[-1]:.6e}")
pr(f"  ESS/N:      {ess:.4f}")
pr(f"  K_mc:       {K_mc:.10f}")
pr(f"  std error:  {K_std:.2e}")
pr(f"  Time:       {elapsed:.1f}s")
pr("=" * 60)
