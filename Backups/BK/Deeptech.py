"""
LAYMAN OVERVIEW:
  This script uses a neural network (normalizing flow) to learn a good
  integration contour for the quantum path integral of a harmonic oscillator
  (ball on a spring).

  The idea: instead of hand-deriving the optimal 45-degree tilt (like Claude.py
  does), let a neural network figure it out by trial and error. The network
  starts with random samples on the real line and learns a complex-valued
  rotation matrix that pushes them into the complex plane where the integral
  becomes well-behaved.

  For the spring this is overkill (we already know the answer), but it
  validates the ML approach before applying it to harder problems.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# Physics Parameters
# -----------------------------------------------------------------------
N = 16    # Number of time steps (how finely we chop up time)
T = 2.0   # Total time for the particle to travel
a = T / N # Time step size (smaller = more accurate, more expensive)
m = 1.0   # Particle mass
omega = 1.0   # Spring stiffness (frequency of oscillation)
epsilon = 0.1 # Small damping factor that nudges the integral toward convergence
D = N - 1 # Number of free variables (endpoints are fixed, so N-1 interior points)


# -----------------------------------------------------------------------
# Action Definition
# -----------------------------------------------------------------------
# LAYMAN: The "action" scores each path — it's the physics cost function.
# It balances kinetic energy (how fast the particle moves between steps)
# against potential energy (how much the spring is stretched).
#
# The path is pinned at both ends: x(0) = 0 and x(T) = 0 (Dirichlet BCs).
# The i*epsilon term adds a tiny imaginary part that helps the integral
# converge — think of it as a gentle nudge that prevents numerical blowup.
#
# This function is batched (processes many paths at once), which is standard
# for GPU-friendly ML training.

def complex_action(x):
    """
    Computes the discretized real-time action S[x].
    x: Tensor of shape (batch_size, D) of complex64 types.
    """
    batch_size = x.shape[0]

    # Pin endpoints to zero: [0, x_1, x_2, ..., x_{N-1}, 0]
    zeros = torch.zeros((batch_size, 1), dtype=torch.complex64)
    x_padded = torch.cat([zeros, x, zeros], dim=1)

    # Velocity ~ difference between consecutive positions
    dx = x_padded[:, 1:] - x_padded[:, :-1]

    # Kinetic energy: how fast the particle moves (sum of velocity^2)
    K = (m / (2 * a)) * torch.sum(dx ** 2, dim=1)

    # Potential energy: how stretched the spring is (sum of position^2)
    omega_complex_sq = omega ** 2 - 1j * epsilon
    V = (a * m / 2) * omega_complex_sq * torch.sum(x ** 2, dim=1)

    # Action = kinetic - potential (Lagrangian formulation)
    return K - V


# -----------------------------------------------------------------------
# Normalizing Flow (Affine Complex Map)
# -----------------------------------------------------------------------
# LAYMAN: This is the "learnable lens" — a neural network that takes simple
# real Gaussian noise z ~ N(0, I) and maps it to complex-valued paths x.
#
# Architecture: just a linear transformation x = z * A + b, where A is a
# complex matrix (real part + imaginary part) and b is a complex bias.
#
# Initialisation: A starts near the identity matrix (i.e. "do nothing"),
# so initially the output is close to real Gaussian samples. During training,
# the network learns to rotate/stretch the samples into the complex plane
# to find a good integration contour.
#
# For the harmonic oscillator, the optimal A is a 45-degree rotation matrix
# (the thimble). The network should learn something close to this.
#
# This is a LINEAR flow — sufficient for the spring (where the thimble is a
# linear rotation), but would need nonlinear layers (RealNVP, coupling layers)
# for harder problems with curved thimbles.

class ThimbleFlow(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.A_real = torch.nn.Parameter(torch.eye(dim) + 0.01 * torch.randn(dim, dim))
        self.A_imag = torch.nn.Parameter(0.01 * torch.randn(dim, dim))
        self.b_real = torch.nn.Parameter(torch.zeros(dim))
        self.b_imag = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        """
        Maps real base distribution z ~ N(0, I) to complex contour x.
        """
        A = self.A_real + 1j * self.A_imag
        b = self.b_real + 1j * self.b_imag

        x = torch.matmul(z.to(torch.complex64), A.t()) + b
        return x, A


# -----------------------------------------------------------------------
# Loss Function & Training
# -----------------------------------------------------------------------
# LAYMAN: The training loop. At each step:
#   1. Draw random Gaussian noise z (the "darts")
#   2. Push z through the learnable matrix A to get complex paths x
#   3. Compute the physics action S for each path
#   4. Compute the "importance weight" W for each sample — this measures
#      how much that sample contributes to the integral
#   5. Loss = variance of log(W). If all samples contribute equally,
#      log(W) is constant and variance is zero. High variance means some
#      samples dominate while others are wasted (the sign problem).
#
# The optimizer (Adam) adjusts A to minimize this variance, i.e. to find
# a contour where all samples are equally useful.

model = ThimbleFlow(D)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 1000
batch_size = 512

print("Training Contour Deformation...")
for epoch in range(epochs):
    optimizer.zero_grad()

    # 1. Draw random Gaussian samples (the "base distribution")
    z = torch.randn(batch_size, D)

    # 2. Push through the learned complex rotation into the complex plane
    x, A = model(z)

    # 3. Score each path using the physics action
    S = complex_action(x)

    # 4. Compute importance weights: how much does each sample contribute?
    # log P(z) is the probability of drawing z from the Gaussian
    log_P = -0.5 * torch.sum(z ** 2, dim=1) - (D / 2) * np.log(2 * np.pi)

    # log det(A) accounts for the volume change from the transformation
    sign, logabsdet = torch.linalg.slogdet(A)
    log_det_J = logabsdet + 1j * torch.angle(sign)

    # Full weight = physics factor * Jacobian / base probability
    log_W = 1j * S + log_det_J - log_P.to(torch.complex64)

    # 5. Loss: we want all log_W values to be the same (zero variance)
    loss = torch.var(log_W.real) + torch.var(log_W.imag)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4e}")

# -----------------------------------------------------------------------
# Sampling and Observable Evaluation
# -----------------------------------------------------------------------
# LAYMAN: Training is done. Now we use the trained model to actually compute
# a physical quantity: the two-point correlator, which measures how much the
# particle's position at one time is correlated with its position at another.
#
# Specifically, we estimate <x(T/2)^2> — how much the particle jiggles at
# the midpoint. The exact answer for a quantum spring is 1/(2*m*omega) = 0.5.
#
# We generate 10,000 samples, compute their importance weights, and take a
# weighted average. If the learned contour is good, the weights are roughly
# uniform and the estimate is accurate.

print("\nGenerating Ensembles and Computing Correlators...")

with torch.no_grad():
    N_samples = 10000
    z = torch.randn(N_samples, D)
    x, A = model(z)
    S = complex_action(x)

    log_P = -0.5 * torch.sum(z ** 2, dim=1) - (D / 2) * np.log(2 * np.pi)
    sign, logabsdet = torch.linalg.slogdet(A)
    log_det_J = logabsdet + 1j * torch.angle(sign)

    log_W = 1j * S + log_det_J - log_P.to(torch.complex64)

    # Numerical stability trick: subtract the max before exponentiating
    # (same idea as the log-sum-exp trick common in ML)
    log_W_max = torch.max(log_W.real)
    W = torch.exp(log_W - log_W_max)

    # Normalise to get proper probability weights that sum to 1
    W_norm = W / torch.sum(W)

    # Weighted correlation matrix: C_{ij} = sum(W_k * x_{k,i} * x_{k,j})
    x_weighted = x * torch.sqrt(W_norm).unsqueeze(1)
    correlation_matrix = torch.matmul(x_weighted.t(), x_weighted)

# The diagonal of the correlation matrix gives <x(t)^2> at each time step
print("\nCentral lattice point variance <x(T/2)^2>:")
mid_idx = D // 2
analytic_approx = 1.0 / (2 * m * omega)  # Known exact answer = 0.5
numeric_var = correlation_matrix[mid_idx, mid_idx].real.item()

print(f"Numerical (Path Integral): {numeric_var:.5f}")
print(f"Analytic (Continuum):      {analytic_approx:.5f}")