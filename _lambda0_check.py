"""Quick λ=0 diagnostic: complex-Hessian init should already give the lattice answer."""
import time
import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

N = 32
T = 1.0
eta = 0.05
m = 1.0
omega = 1.0
D = N - 1
T_c = T - 1j * eta
a_c = T_c / N
rank = 16
LATTICE_REF = 0.3121159 - 0.3022604j


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


def compute_complex_hessian_A_init(N_val, a_c_val, m_val, omega_val):
    j = np.arange(1, N_val)[:, None]
    k = np.arange(1, N_val)[None, :]
    Q = np.sqrt(2.0 / N_val) * np.sin(np.pi * j * k / N_val)
    diag_val = 2.0 * m_val / a_c_val - a_c_val * m_val * omega_val**2
    off_val = -m_val / a_c_val
    modes = np.arange(1, N_val)
    eigvals = diag_val + 2.0 * off_val * np.cos(np.pi * modes / N_val)
    scales = np.sqrt(1j / eigvals)
    return torch.tensor(Q @ np.diag(scales), dtype=torch.complex128)


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
        # Near-zero nonlinear correction so λ=0 starts at the exact thimble.
        self.W3 = nn.Parameter(torch.randn(2 * dim, r) * 1e-8)
        self.b3 = nn.Parameter(torch.zeros(2 * dim))

    def _get_A(self):
        return self.A_real.to(torch.complex128) + 1j * self.A_imag.to(torch.complex128)

    def _get_C(self):
        return self.W3[: self.dim].to(torch.complex128) + 1j * self.W3[self.dim :].to(
            torch.complex128
        )

    def forward(self, z):
        z = z.to(torch.float64)
        A = self._get_A()
        b = self.b_real.to(torch.complex128) + 1j * self.b_imag.to(torch.complex128)
        x_linear = z.to(torch.complex128) @ A.T + b
        a1 = torch.tanh(z @ self.W1.T + self.b1)
        a2 = torch.tanh(a1 @ self.W2.T + self.b2)
        out = a2 @ self.W3.T + self.b3
        g = out[:, : self.dim].to(torch.complex128) + 1j * out[:, self.dim :].to(
            torch.complex128
        )
        return x_linear + g, a1, a2

    def compute_log_det_J(self, z, a1, a2, detach_correction=False):
        A = self._get_A()
        sign_A, logabs_A = torch.linalg.slogdet(A)
        log_det_A = logabs_A + 1j * torch.angle(sign_A)
        C = self._get_C()
        A_inv_C = torch.linalg.solve(A, C)  # PDF fix
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


def estimate_K(model, N_mc=50_000, lam=0.0):
    with torch.no_grad():
        z = torch.randn(N_mc, D)
        x, a1, a2 = model(z)
        S = complex_action(x, 0.0, 0.0, lam, a_c, m, omega)
        log_P = -0.5 * torch.sum(z**2, dim=1) - (D / 2) * np.log(2 * np.pi)
        log_det = model.compute_log_det_J(z, a1, a2)
        log_W = 1j * S + log_det - log_P.to(torch.complex128)
        shift = torch.max(log_W.real)
        W = torch.exp(log_W - shift)
        pref = (m / (2 * np.pi * 1j * a_c)) ** (N / 2)
        K = pref * np.exp(shift.item()) * torch.mean(W).numpy()
        loss = (torch.var(log_W.real) + torch.var(log_W.imag)).item()
        ess = float(np.abs(np.mean(W.numpy())) ** 2 / np.mean(np.abs(W.numpy()) ** 2))
    return K, loss, ess


def report(tag, K, loss, ess):
    err_re = abs(K.real - LATTICE_REF.real) / abs(LATTICE_REF.real)
    err_im = abs(K.imag - LATTICE_REF.imag) / abs(LATTICE_REF.imag)
    print(f"\n=== {tag} ===")
    print(f"K_ML     = {K:.10f}")
    print(f"K_lattice= {LATTICE_REF:.10f}")
    print(f"loss={loss:.4e}, ESS/N={ess:.5f}")
    print(f"err Re={100*err_re:.4f}%, Im={100*err_im:.4f}%")
    return err_re, err_im


# --- Test A: identity init (should FAIL — reproduces notebook bug) ---
torch.manual_seed(42)
np.random.seed(42)
model_id = ThimbleFlowNL(
    D, rank=rank, A_init=torch.eye(D, dtype=torch.complex128), train_A=False
)
K, loss, ess = estimate_K(model_id, N_mc=20_000)
report("identity A, no training", K, loss, ess)

# --- Test B: complex Hessian init, g~0, no training (should PASS) ---
torch.manual_seed(42)
np.random.seed(42)
A_hess = -compute_complex_hessian_A_init(N, a_c, m, omega)  # flip for +lattice branch
model_h = ThimbleFlowNL(D, rank=rank, A_init=A_hess, train_A=False)
K, loss, ess = estimate_K(model_h, N_mc=50_000)
err_re, err_im = report("complex_hessian -A, no training", K, loss, ess)

# --- Test C: short training with frozen A ---
torch.manual_seed(42)
np.random.seed(42)
model_t = ThimbleFlowNL(D, rank=rank, A_init=A_hess.clone(), train_A=False)
trainable = [p for p in model_t.parameters() if p.requires_grad]
opt = torch.optim.Adam(trainable, lr=1e-3)
t0 = time.time()
for epoch in range(500):
    opt.zero_grad()
    z = torch.randn(2048, D)
    x, a1, a2 = model_t(z)
    S = complex_action(x, 0.0, 0.0, 0.0, a_c, m, omega)
    log_P = -0.5 * torch.sum(z**2, dim=1) - (D / 2) * np.log(2 * np.pi)
    log_det = model_t.compute_log_det_J(z, a1, a2, detach_correction=True)
    log_W = 1j * S + log_det - log_P.to(torch.complex128)
    loss = torch.var(log_W.real) + torch.var(log_W.imag)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
    opt.step()
    if (epoch + 1) % 100 == 0:
        print(f"  train epoch {epoch+1}: loss={loss.item():.4e}", flush=True)
print(f"training time: {time.time()-t0:.0f}s")
K, loss, ess = estimate_K(model_t, N_mc=50_000)
err_re2, err_im2 = report("complex_hessian -A, 500 epochs, A frozen", K, loss, ess)

ok = max(err_re, err_im) < 0.01 and max(err_re2, err_im2) < 0.01
print("\nPASS" if ok else "\nFAIL")
raise SystemExit(0 if ok else 1)
