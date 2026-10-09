"""Numerically verify Tom's compute_log_det_J against the exact Jacobian.

Replicates ThimbleFlowNL.forward and compute_log_det_J in numpy for random
parameters, and compares against the determinant of the analytically exact
Jacobian of the forward map  x(z) = z @ A^T + b + g(z).
"""
import numpy as np

rng = np.random.default_rng(0)
dim, r = 7, 3

A = (np.eye(dim) + 0.05 * rng.standard_normal((dim, dim))
     + 1j * 0.05 * rng.standard_normal((dim, dim)))
W1 = rng.standard_normal((r, dim)) * 0.5
b1 = rng.standard_normal(r) * 0.1
W2 = rng.standard_normal((r, r)) * 0.5
b2 = rng.standard_normal(r) * 0.1
W3 = rng.standard_normal((2 * dim, r)) * 0.3   # deliberately not tiny
b3 = rng.standard_normal(2 * dim) * 0.1
C = W3[:dim] + 1j * W3[dim:]                    # (dim, r)

z = rng.standard_normal(dim)

# forward, exactly as in the notebook: x = z @ A.t() + g(z)  (i.e. x = A z)
a1 = np.tanh(W1 @ z + b1)
a2 = np.tanh(W2 @ a1 + b2)

# exact Jacobian of the forward map
D1 = np.diag(1 - a1**2)
D2 = np.diag(1 - a2**2)
J_exact = A + C @ D2 @ W2 @ D1 @ W1            # since x = A z + C a2(z)
logdet_exact = np.log(np.linalg.det(J_exact))

# Tom's formula: log det A + log det(I_r + D2 W2 D1 W1 A^{-T} C)
K_tom = W1 @ np.linalg.solve(A.T, C)           # W1 A^{-T} C
M_tom = D2 @ W2 @ D1 @ K_tom
logdet_tom = np.log(np.linalg.det(A)) + np.log(np.linalg.det(np.eye(r) + M_tom))

# corrected formula: A^{-1} C
K_fix = W1 @ np.linalg.solve(A, C)
M_fix = D2 @ W2 @ D1 @ K_fix
logdet_fix = np.log(np.linalg.det(A)) + np.log(np.linalg.det(np.eye(r) + M_fix))

print(f"exact  log det J : {logdet_exact:.10f}")
print(f"Tom's  formula   : {logdet_tom:.10f}   diff = {abs(logdet_tom - logdet_exact):.2e}")
print(f"fixed  formula   : {logdet_fix:.10f}   diff = {abs(logdet_fix - logdet_exact):.2e}")

# finite-difference cross-check of J_exact
eps = 1e-6
J_fd = np.zeros((dim, dim), dtype=complex)
for k in range(dim):
    zp = z.copy(); zp[k] += eps
    zm = z.copy(); zm[k] -= eps
    def fwd(zz):
        aa1 = np.tanh(W1 @ zz + b1)
        aa2 = np.tanh(W2 @ aa1 + b2)
        return A @ zz + C @ aa2
    J_fd[:, k] = (fwd(zp) - fwd(zm)) / (2 * eps)
print(f"finite-diff check: |J_fd - J_exact|_max = {np.abs(J_fd - J_exact).max():.2e}")
