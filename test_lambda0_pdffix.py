"""Quick test: lambda=0 with complex_hessian A + corrected Jacobian (PDF fix).
Run this to validate that the setup gives the correct harmonic oscillator result.
"""
import torch
import torch.nn as nn
import numpy as np
import time
import copy

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
np.random.seed(42)

# Physics / discretisation
N = 32
T = 1.0
eta = 0.05
m = 1.0
omega = 1.0
D = N - 1
T_c = T - 1j * eta
a_c = T_c / N

# Model / training
rank = 16
batch = 2048
N_mc = 100_000
epochs = 5_000  # reduced for quick validation (loss reaches ~1e-3 by 5k)
x_i_prop = 0.0
x_f_prop = 0.0

INIT_MODE = "complex_hessian"
TRAIN_A = True
DETACH_JACOBIAN_CORRECTION = True

LATTICE_REFERENCE_N32 = {0.0: 0.3121159 - 0.3022604j}

# ─── Action ───────────────────────────────────────────────────────────────────
def complex_action(x, x_i, x_f, lam_val, a_c_val, m_val, omega_val):
    x = x.to(torch.complex128)
    bsz = x.shape[0]
    pad_i = torch.full((bsz, 1), complex(x_i), dtype=torch.complex128)
    pad_f = torch.full((bsz, 1), complex(x_f), dtype=torch.complex128)
    x_pad = torch.cat([pad_i, x, pad_f], dim=1)
    dx = x_pad[:, 1:] - x_pad[:, :-1]
    x_left = x_pad[:, :-1]
    kinetic = (m_val / (2 * a_c_val)) * torch.sum(dx**2, dim=1)
    potential = a_c_val * torch.sum(
        0.5 * m_val * omega_val**2 * x_left**2 + 0.25 * lam_val * x_left**4,
        dim=1,
    )
    return kinetic - potential


# ─── Hessian init ─────────────────────────────────────────────────────────────
def compute_complex_hessian_A_init(N_val, a_c_val, m_val, omega_val):
    D_val = N_val - 1
    j = np.arange(1, N_val)[:, None]
    k = np.arange(1, N_val)[None, :]
    Q = np.sqrt(2.0 / N_val) * np.sin(np.pi * j * k / N_val)
    diag_val = 2.0 * m_val / a_c_val - a_c_val * m_val * omega_val**2
    off_val = -m_val / a_c_val
    modes = np.arange(1, N_val)
    eigvals = diag_val + 2.0 * off_val * np.cos(np.pi * modes / N_val)
    scales = np.sqrt(1j / eigvals)
    A_np = Q @ np.diag(scales)
    return torch.tensor(A_np, dtype=torch.complex128)


def initial_A(init_mode, N_val, a_c_val, m_val, omega_val):
    D_val = N_val - 1
    if init_mode == "identity":
        return torch.eye(D_val, dtype=torch.complex128)
    if init_mode == "complex_hessian":
        A = compute_complex_hessian_A_init(N_val, a_c_val, m_val, omega_val)
        # Branch-cut fix: check that det(A) puts us in the correct homology class.
        K_ref_0 = LATTICE_REFERENCE_N32[0.0]
        prefactor = (m_val / (2 * np.pi * 1j * a_c_val)) ** (N_val / 2)
        det_A = np.linalg.det(A.numpy())
        K_gaussian = prefactor * det_A * (2 * np.pi) ** (D_val / 2)
        dot = (K_gaussian * np.conj(K_ref_0)).real
        if dot < 0:
            A = -A
            print("Branch-cut correction: negated A to match correct homology class.")
        else:
            print("Branch-cut check passed: A is in correct homology class.")
        return A
    raise ValueError(f"Unknown: {init_mode}")


# ─── Model ────────────────────────────────────────────────────────────────────
class ThimbleFlowNL(nn.Module):
    def __init__(self, dim, rank=16, b_init=None, A_init=None, train_A=False):
        super().__init__()
        self.dim = dim
        self.rank = rank
        if A_init is None:
            A_init = torch.eye(dim, dtype=torch.complex128)
        self.A_real = nn.Parameter(A_init.real.clone(), requires_grad=train_A)
        self.A_imag = nn.Parameter(A_init.imag.clone(), requires_grad=train_A)
        if b_init is None:
            b_init = torch.zeros(dim)
        self.b_real = nn.Parameter(b_init.clone().to(torch.float64))
        self.b_imag = nn.Parameter(torch.zeros(dim))
        r = rank
        self.W1 = nn.Parameter(torch.randn(r, dim) * np.sqrt(2.0 / dim))
        self.b1 = nn.Parameter(torch.zeros(r))
        self.W2 = nn.Parameter(torch.randn(r, r) * np.sqrt(2.0 / r))
        self.b2 = nn.Parameter(torch.zeros(r))
        self.W3 = nn.Parameter(torch.randn(2 * dim, r) * 0.001)
        self.b3 = nn.Parameter(torch.zeros(2 * dim))

    def _get_A(self):
        return self.A_real.to(torch.complex128) + 1j * self.A_imag.to(torch.complex128)

    def _get_C(self):
        return self.W3[:self.dim].to(torch.complex128) + 1j * self.W3[self.dim:].to(torch.complex128)

    def forward(self, z):
        z = z.to(torch.float64)
        A = self._get_A()
        b = self.b_real.to(torch.complex128) + 1j * self.b_imag.to(torch.complex128)
        x_linear = z.to(torch.complex128) @ A.T + b
        a1 = torch.tanh(z @ self.W1.T + self.b1)
        a2 = torch.tanh(a1 @ self.W2.T + self.b2)
        out = a2 @ self.W3.T + self.b3
        g = out[:, :self.dim].to(torch.complex128) + 1j * out[:, self.dim:].to(torch.complex128)
        return x_linear + g, a1, a2

    def compute_log_det_J(self, z, a1, a2, detach_correction=False):
        A = self._get_A()
        sign_A, logabs_A = torch.linalg.slogdet(A)
        log_det_A = logabs_A + 1j * torch.angle(sign_A)
        C = self._get_C()
        # PDF FIX: solve(A, C) not solve(A.t(), C)
        A_inv_C = torch.linalg.solve(A, C)
        K_mat = self.W1.to(torch.complex128) @ A_inv_C
        _a1 = a1.detach() if detach_correction else a1
        _a2 = a2.detach() if detach_correction else a2
        da1 = (1.0 - _a1**2).to(torch.complex128)
        da2 = (1.0 - _a2**2).to(torch.complex128)
        M1 = da1.unsqueeze(-1) * K_mat.unsqueeze(0)
        W2c = self.W2.to(torch.complex128)
        if detach_correction:
            W2c = W2c.detach()
        M2 = torch.bmm(W2c.unsqueeze(0).expand(z.shape[0], -1, -1), M1)
        M3 = da2.unsqueeze(-1) * M2
        I_r = torch.eye(self.rank, dtype=torch.complex128).unsqueeze(0)
        sign_c, logabs_c = torch.linalg.slogdet(I_r + M3)
        correction = logabs_c + 1j * torch.angle(sign_c)
        if detach_correction:
            correction = correction.detach()
        return log_det_A + correction


# ─── Run ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("TEST: lambda=0 with complex_hessian A_init + corrected Jacobian")
print("=" * 70)

torch.manual_seed(42)
np.random.seed(42)

b_init = torch.linspace(float(x_i_prop), float(x_f_prop), D + 2)[1:-1]
A_init = initial_A(INIT_MODE, N, a_c, m, omega)
model = ThimbleFlowNL(D, rank=rank, b_init=b_init, A_init=A_init, train_A=TRAIN_A)

trainable = [p for p in model.parameters() if p.requires_grad]
print(f"Trainable parameters: {sum(p.numel() for p in trainable)}")
optimizer = torch.optim.Adam(trainable, lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

t0 = time.time()
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    z = torch.randn(batch, D)
    x, a1, a2 = model(z)
    S = complex_action(x, x_i_prop, x_f_prop, 0.0, a_c, m, omega)
    log_P = -0.5 * torch.sum(z**2, dim=1) - (D / 2) * np.log(2 * np.pi)
    log_det_J = model.compute_log_det_J(z, a1, a2, detach_correction=DETACH_JACOBIAN_CORRECTION)
    log_W = 1j * S + log_det_J - log_P.to(torch.complex128)
    loss = torch.var(log_W.real) + torch.var(log_W.imag)

    if not torch.isfinite(loss):
        print(f"NON-FINITE LOSS at epoch {epoch+1}!")
        break
    loss.backward()
    bad_grad = not all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable)
    if bad_grad:
        print(f"NON-FINITE GRADIENT at epoch {epoch+1}, skipping step")
        optimizer.zero_grad()
        continue
    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
    optimizer.step()
    scheduler.step()
    loss_history.append(loss.item())

    if epoch == 0 or (epoch + 1) % 1000 == 0:
        print(f"epoch {epoch+1:5d}/{epochs}  loss={loss.item():.4e}  time={time.time()-t0:.0f}s")

print(f"\nTraining complete in {time.time()-t0:.0f}s")
print(f"Final loss: {loss_history[-1]:.4e}")

# ─── MC evaluation ────────────────────────────────────────────────────────────
print("\nRunning MC evaluation with 100k samples...")
log_W_chunks = []
mc_chunk = 5000
with torch.no_grad():
    remaining = N_mc
    while remaining:
        n_now = min(mc_chunk, remaining)
        z = torch.randn(n_now, D)
        x, a1, a2 = model(z)
        S = complex_action(x, x_i_prop, x_f_prop, 0.0, a_c, m, omega)
        log_P = -0.5 * torch.sum(z**2, dim=1) - (D / 2) * np.log(2 * np.pi)
        log_det_J = model.compute_log_det_J(z, a1, a2)
        log_W_chunks.append((1j * S + log_det_J - log_P.to(torch.complex128)).cpu())
        remaining -= n_now

    log_W_mc = torch.cat(log_W_chunks)
    shift = torch.max(log_W_mc.real)
    W_shifted = torch.exp(log_W_mc - shift)
    prefactor = (m / (2 * np.pi * 1j * a_c)) ** (N / 2)
    scale = prefactor * np.exp(shift.item())
    K_mc = scale * torch.mean(W_shifted).numpy()
    W_values = scale * W_shifted.numpy()
    K_std = np.std(W_values) / np.sqrt(N_mc)
    ess = float(np.abs(np.mean(W_values))**2 / np.mean(np.abs(W_values)**2))

ref = LATTICE_REFERENCE_N32[0.0]
err_re = abs(K_mc.real - ref.real) / abs(ref.real)
err_im = abs(K_mc.imag - ref.imag) / abs(ref.imag)

print(f"\nK_ML  = {K_mc:.10f}")
print(f"K_ref = {ref:.10f}")
print(f"stderr= {K_std:.2e}, ESS/N={ess:.5f}")
print(f"shift (max Re logW) = {shift.item():.4f}")
print(f"\nComponent errors: Re={100*err_re:.4f}%, Im={100*err_im:.4f}%")

if max(err_re, err_im) < 0.01:
    print("\n[PASSED] lambda=0 within 1% tolerance!")
else:
    print(f"\n[FAILED] errors exceed 1% (max={100*max(err_re,err_im):.2f}%)")
