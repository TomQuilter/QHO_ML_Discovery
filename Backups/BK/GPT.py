#!/usr/bin/env python3
"""
LAYMAN OVERVIEW:
  This is the most ambitious of the three files. While Claude.py and Gemini.py
  compute a simple "particle goes from A to B" amplitude, this tackles real-time
  quantum mechanics at finite temperature using the Schwinger-Keldysh contour.

  What is the Schwinger-Keldysh contour?
  Imagine you want to know how a quantum system at some temperature behaves
  over time. You need THREE legs:
    1. FORWARD: evolve the system forward in real time (0 -> T)
    2. BACKWARD: evolve it back in real time (T -> 0)
    3. EUCLIDEAN: a trip through "imaginary time" to prepare the thermal state
       (this is how you set the temperature — longer trip = colder system)

  The forward-backward pair is needed because quantum expectation values
  require evolving both the "bra" and "ket" (both sides of the density matrix).

  The ML component learns two rotation angles (theta_f, theta_b) that tilt
  the forward and backward branches into the complex plane, plus a Cholesky-
  based sampler for the base distribution. This is the same "learn a good
  contour" idea as Gemini.py, but applied to a harder problem structure.
"""

import math
import numpy as np
import torch

# ----------------------------
# User parameters
# ----------------------------
m = 1.0       # Particle mass
omega = 1.0   # Spring frequency
hbar = 1.0    # Quantum scale (Planck's constant / 2*pi)

beta = 6.0    # Inverse temperature: larger = colder system, closer to ground state
T = 4.0       # How far forward in real time we evolve

N_t = 64      # Time steps on each real-time branch (forward and backward)
N_tau = 64    # Time steps on the imaginary-time (thermal) branch

dt = T / N_t       # Real-time step size
d_tau = beta / N_tau  # Imaginary-time step size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype_r = torch.float64
dtype_c = torch.complex128

torch.manual_seed(0)

# ----------------------------
# Indexing of the flattened variable vector v
# ----------------------------
# LAYMAN: All the path variables are packed into one flat vector v of length D.
# Think of it as concatenating three separate paths into one array:
#
#   v = [x0, xT, forward_interior..., backward_interior..., euclidean_interior...]
#
# where x0 and xT are the shared junction points between branches.
# The forward branch goes 0->T, backward goes T->0, euclidean goes 0->beta.
# Endpoints are pinned (shared or periodic), so only interior points are free.
#
# Total free variables D = 2 (shared endpoints) + (N_t-1) + (N_t-1) + (N_tau-1)
#                        = 2 + 63 + 63 + 63 = 191 for these settings.
D = 2 + (N_t - 1) + (N_t - 1) + (N_tau - 1)
idx_x0 = 0
idx_xT = 1
idx_f0 = 2
idx_f1 = idx_f0 + (N_t - 1)
idx_b0 = idx_f1
idx_b1 = idx_b0 + (N_t - 1)
idx_e0 = idx_b1
idx_e1 = idx_e0 + (N_tau - 1)
assert idx_e1 == D

idx_forward_internal = torch.arange(idx_f0, idx_f1, device=device)
idx_backward_internal = torch.arange(idx_b0, idx_b1, device=device)
idx_euclid_internal = torch.arange(idx_e0, idx_e1, device=device)

# LAYMAN: Unpacks the flat vector back into three separate paths (forward,
# backward, euclidean), stitching in the shared endpoints x0 and xT.
def unpack_paths(vc: torch.Tensor):
    """
    vc: (B, D) complex tensor -> returns three branches with endpoints attached.
    """
    B = vc.shape[0]
    x0 = vc[:, idx_x0]
    xT = vc[:, idx_xT]

    f_int = vc[:, idx_f0:idx_f1]  # (B, N_t-1)
    b_int = vc[:, idx_b0:idx_b1]  # (B, N_t-1)
    e_int = vc[:, idx_e0:idx_e1]  # (B, N_tau-1)

    # Forward: [x0, f_int..., xT]
    x_f = torch.zeros((B, N_t + 1), dtype=dtype_c, device=device)
    x_f[:, 0] = x0
    x_f[:, 1:N_t] = f_int
    x_f[:, N_t] = xT

    # Backward: [xT, b_int..., x0]
    x_b = torch.zeros((B, N_t + 1), dtype=dtype_c, device=device)
    x_b[:, 0] = xT
    x_b[:, 1:N_t] = b_int
    x_b[:, N_t] = x0

    # Euclidean: [x0, e_int..., x0] (periodic endpoints)
    x_e = torch.zeros((B, N_tau + 1), dtype=dtype_c, device=device)
    x_e[:, 0] = x0
    x_e[:, 1:N_tau] = e_int
    x_e[:, N_tau] = x0

    return x_f, x_b, x_e


def action_components(vc: torch.Tensor):
    """
    Compute:
      S_E  : Euclidean action on imaginary leg (real, >=0 in continuum)
      S_rt : real-time contour action difference S_forward - S_backward (complex in general)
    using a straightforward nearest-neighbor discretization.
    """
    x_f, x_b, x_e = unpack_paths(vc)

    # Real-time forward action S_+
    dx_f = x_f[:, 1:] - x_f[:, :-1]  # (B, N_t)
    S_plus = (m * (dx_f ** 2) / (2.0 * dt)).sum(dim=1) \
             - (0.5 * m * omega ** 2 * (x_f[:, :-1] ** 2) * dt).sum(dim=1)

    # Real-time backward action S_-  (same discretization, but contour contributes with minus sign)
    dx_b = x_b[:, 1:] - x_b[:, :-1]
    S_minus = (m * (dx_b ** 2) / (2.0 * dt)).sum(dim=1) \
              - (0.5 * m * omega ** 2 * (x_b[:, :-1] ** 2) * dt).sum(dim=1)

    S_rt = S_plus - S_minus

    # Euclidean action on imaginary leg:
    # S_E = ∫ dτ [ m/2 (dx/dτ)^2 + m ω^2/2 x^2 ]
    dx_e = x_e[:, 1:] - x_e[:, :-1]  # (B, N_tau)
    S_E = (m * (dx_e ** 2) / (2.0 * d_tau)).sum(dim=1) \
          + (0.5 * m * omega ** 2 * (x_e[:, :-1] ** 2) * d_tau).sum(dim=1)

    return S_E, S_rt


# ----------------------------
# ML model: linear flow v = mu + L z (z ~ N(0, I))
# with trainable contour phases theta_f, theta_b on real-time internal points.
# ----------------------------
class LinearFlowSampler(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mu = torch.nn.Parameter(torch.zeros(dim, dtype=dtype_r, device=device))

        # Strictly lower-triangular part
        self.L_lower = torch.nn.Parameter(torch.zeros((dim, dim), dtype=dtype_r, device=device))

        # Diagonal (positive via softplus)
        self.L_diag_raw = torch.nn.Parameter(torch.zeros(dim, dtype=dtype_r, device=device))

        # Contour deformation phases (scalars)
        self.theta_f = torch.nn.Parameter(torch.tensor(0.0, dtype=dtype_r, device=device))
        self.theta_b = torch.nn.Parameter(torch.tensor(0.0, dtype=dtype_r, device=device))

    def build_L(self):
        # Lower triangle excluding diagonal
        L = torch.tril(self.L_lower, diagonal=-1)
        diag = torch.nn.functional.softplus(self.L_diag_raw) + 1e-3
        L = L + torch.diag(diag)
        logdet = torch.log(diag).sum()
        return L, logdet

    def sample(self, batch_size: int):
        z = torch.randn((batch_size, self.dim), dtype=dtype_r, device=device)
        L, logdet = self.build_L()
        v = self.mu + z @ L.T  # (B, D)
        return z, v, logdet

    def deform(self, v: torch.Tensor):
        """
        v is real (B, D). Return complex deformed vc and complex logdet_deform.
        Endpoints and Euclidean points remain real.
        Internal forward/back points get rotated by exp(i theta_f/b).
        """
        B = v.shape[0]
        vc = v.to(dtype_c)

        # Apply phase rotation only on internal real-time points
        pf = torch.exp(1j * self.theta_f.to(dtype_c))
        pb = torch.exp(1j * self.theta_b.to(dtype_c))
        vc[:, idx_forward_internal] *= pf
        vc[:, idx_backward_internal] *= pb

        # Jacobian determinant of this diagonal complex rotation on the rotated coordinates
        logdet_def = 1j * ((N_t - 1) * self.theta_f + (N_t - 1) * self.theta_b)
        logdet_def = logdet_def.to(dtype_c)  # scalar complex
        return vc, logdet_def

    def logw(self, z: torch.Tensor, v: torch.Tensor, logdet_flow: torch.Tensor):
        """
        Compute complex log-weight:
          log w = -S_E/hbar + i*S_rt/hbar + log det(dx/dz) - log p0(z)  (constants dropped)
        """
        vc, logdet_def = self.deform(v)
        S_E, S_rt = action_components(vc)

        # -log p0(z) up to additive constant: + 0.5 ||z||^2
        minus_log_p0 = 0.5 * (z ** 2).sum(dim=1)  # (B,)

        logdet_total = logdet_flow.to(dtype_c) + logdet_def  # scalar complex
        logdet_total = logdet_total.expand_as(S_E.to(dtype_c))

        logw = (-S_E.to(dtype_c) / hbar) + 1j * (S_rt / hbar) + logdet_total + minus_log_p0.to(dtype_c)
        return logw, vc


def stable_avg_phase(logw: torch.Tensor):
    """
    Average phase = |E[w]| / E[|w|], computed stably per batch.
    """
    a = logw.real
    a0 = torch.max(a)
    w_tilde = torch.exp(logw - a0)  # scale out big real part
    mean_w = torch.mean(w_tilde)
    mean_abs = torch.mean(torch.abs(w_tilde))
    avg_phase = torch.abs(mean_w) / (mean_abs + 1e-300)
    return avg_phase, mean_w, mean_abs


def estimate_correlator(samples_vc: torch.Tensor, weights: torch.Tensor):
    """
    Compute G(t_k) = < x_+(t_k) x_+(0) > on the forward branch for k=0..N_t,
    using self-normalized complex reweighting.
    """
    x_f, _, _ = unpack_paths(samples_vc)
    # Observable matrix: O_i,k = x_i(t_k)*x_i(0)
    O = x_f * x_f[:, 0:1]  # (B, N_t+1)
    w = weights.reshape(-1, 1)
    num = torch.sum(w * O, dim=0)
    den = torch.sum(w, dim=0)  # same scalar repeated
    G = num / den
    return G


def analytic_G(t: np.ndarray):
    """
    Thermal time-ordered correlator for HO for 0<=t<=T:
      G(t) = (hbar/2mω) * cosh( ω(β/2 - i t) ) / sinh( ωβ/2 )
    """
    pref = hbar / (2.0 * m * omega)
    denom = np.sinh(omega * beta / 2.0)
    return pref * np.cosh(omega * (beta / 2.0 - 1j * t)) / denom


# ----------------------------
# Train the sampler to reduce phase fluctuations
# ----------------------------
model = LinearFlowSampler(D).to(device)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)

print(f"Device: {device}")
print(f"Dim D = {D}  (N_t={N_t}, N_tau={N_tau})")

n_steps = 2000
batch = 512

for step in range(1, n_steps + 1):
    z, v, logdet_flow = model.sample(batch)
    logw, _ = model.logw(z, v, logdet_flow)

    avg_phase, mean_w, mean_abs = stable_avg_phase(logw)

    # Loss: maximize avg_phase -> minimize -log(avg_phase)
    loss = -torch.log(avg_phase + 1e-15)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 200 == 0 or step == 1:
        with torch.no_grad():
            # A quick ESS proxy using |w|
            a = logw.real
            a0 = torch.max(a)
            wtil = torch.exp(logw - a0)
            absw = torch.abs(wtil)
            ess = (absw.sum() ** 2) / (torch.sum(absw ** 2) + 1e-300)
            ess = ess.item()
            print(f"step {step:4d} | loss={loss.item():.4f} | avg_phase={avg_phase.item():.4e} | ESS≈{ess:8.1f} | "
                  f"theta_f={model.theta_f.item():+.3f}, theta_b={model.theta_b.item():+.3f}")

# ----------------------------
# Generate an ensemble and compute correlator
# ----------------------------
with torch.no_grad():
    n_samples = 20000
    z, v, logdet_flow = model.sample(n_samples)
    logw, vc = model.logw(z, v, logdet_flow)
    # stable weights (common scale cancels in ratio)
    a = logw.real
    a0 = torch.max(a)
    w = torch.exp(logw - a0)  # complex weights up to a global factor

    G_est = estimate_correlator(vc, w).cpu().numpy()
    tgrid = np.arange(N_t + 1) * dt
    G_ex = analytic_G(tgrid)

    # Print a few values as a sanity check
    print("\nCompare correlator G(t)=<x(t)x(0)> at a few times (est vs exact):")
    for k in [0, 1, 2, 4, 8, 16, 32, 48, 64]:
        ge = G_est[k]
        gx = G_ex[k]
        print(f"t={tgrid[k]:.3f} | est={ge.real:+.6e}{ge.imag:+.6e}i | exact={gx.real:+.6e}{gx.imag:+.6e}i")

    # Return the forward-branch trajectories as "ensemble"
    # Note: these are complex (due to contour deformation) and must be used with weights.
    x_f, _, _ = unpack_paths(vc)
    # x_f is (n_samples, N_t+1) complex torch tensor
    # You can save x_f and w for later correlation measurements.
    print(f"\nEnsemble generated: x_forward shape = {tuple(x_f.shape)}, weights shape = {tuple(w.shape)}")