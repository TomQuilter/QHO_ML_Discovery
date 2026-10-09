"""
Overnight Multi-Thimble Exploration
====================================
Trains multiple thimble networks per lambda value, each initialised on a
different "branch" of the linear thimble map (by flipping the sign of
sqrt(i/eigenvalue) for selected Hessian eigenvalue modes).  The full
propagator is then reconstructed as a weighted sum:

    K = sum_k  n_k * K_k        (n_k integer intersection numbers)

After training all branches, every integer-weight combination is tested
and the best match to the exact lattice value is reported.

The determinant bug (A^{-T}C -> A^{-1}C) is FIXED throughout.

Results save incrementally to overnight_multi_thimble_results.json.
Final report in multi_thimble_report.md.
"""

import torch, torch.nn as nn, numpy as np, time, json, os
import scipy.linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product as iproduct

torch.set_default_dtype(torch.float32)

# =====================================================================
#  PARAMETERS
# =====================================================================
N       = 32
T       = 1.0
eta     = 0.05
m       = 1.0
omega   = 1.0
D       = N - 1
rank    = 16
batch   = 2048
N_mc    = 100_000
x_i_prop = 0.0
x_f_prop = 0.0
T_c = T - 1j * eta
a_c = T_c / N

REPORT_PATH = "multi_thimble_report.md"
RESULTS_PATH = "overnight_multi_thimble_results.json"

EXACT_LATTICE = {
    0.0:  0.3121159 - 0.3022604j,
    0.7:  0.2684413 - 0.2878695j,
    2.4: -0.1109040 - 0.1529262j,
}

# =====================================================================
#  ACTION
# =====================================================================
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


# =====================================================================
#  A-MATRIX INIT WITH BRANCH SELECTION
# =====================================================================
def compute_hessian_A_branch(N_val, T_val, m_val, omega_val, D_val,
                             flip_modes=()):
    """Build A from the harmonic Hessian, optionally flipping the sign of
    sqrt(i/eigenvalue) for selected mode indices.

    flip_modes: tuple of mode indices (0-based, sorted by eigenvalue) whose
                square-root branch is negated.  () = standard branch.
    """
    dt = T_val / N_val
    diag_val = 2.0 * m_val / dt - m_val * dt * omega_val**2
    off_val  = -m_val / dt
    H = np.zeros((D_val, D_val), dtype=np.float64)
    for j in range(D_val):
        H[j, j] = diag_val
        if j > 0:       H[j, j-1] = off_val
        if j < D_val-1: H[j, j+1] = off_val
    d, Q = np.linalg.eigh(H)
    c = np.sqrt(1j / d)
    for idx in flip_modes:
        if 0 <= idx < D_val:
            c[idx] = -c[idx]
    A_np = Q @ np.diag(c)
    return torch.tensor(A_np, dtype=torch.complex64)


# =====================================================================
#  THIMBLE MODEL  (det bug FIXED: uses A^{-1}C)
# =====================================================================
class ThimbleFlowNL(nn.Module):
    def __init__(self, dim, rank=8, b_init=None, A_init=None):
        super().__init__()
        self.dim = dim
        self.rank = rank
        if A_init is not None:
            self.A_real = nn.Parameter(A_init.real.float())
            self.A_imag = nn.Parameter(A_init.imag.float())
        else:
            self.A_real = nn.Parameter(torch.eye(dim) + 0.01*torch.randn(dim,dim))
            self.A_imag = nn.Parameter(0.01*torch.randn(dim,dim))
        if b_init is not None:
            self.b_real = nn.Parameter(b_init.clone().float())
        else:
            self.b_real = nn.Parameter(torch.zeros(dim))
        self.b_imag = nn.Parameter(torch.zeros(dim))
        r = rank
        self.W1 = nn.Parameter(torch.randn(r, dim) * (2.0/dim)**0.5)
        self.b1 = nn.Parameter(torch.zeros(r))
        self.W2 = nn.Parameter(torch.randn(r, r) * (2.0/r)**0.5)
        self.b2 = nn.Parameter(torch.zeros(r))
        self.W3 = nn.Parameter(torch.randn(2*dim, r) * 0.001)
        self.b3 = nn.Parameter(torch.zeros(2*dim))

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
        g = out[:,:self.dim].to(torch.complex64) + 1j*out[:,self.dim:].to(torch.complex64)
        return x + g, a1, a2

    def compute_log_det_J(self, z, a1, a2, detach_correction=False):
        A = self._get_A()
        sign_A, logabs_A = torch.linalg.slogdet(A)
        log_det_A = logabs_A + 1j * torch.angle(sign_A)
        C = self.W3[:self.dim].to(torch.complex64) + 1j*self.W3[self.dim:].to(torch.complex64)
        # BUG FIX: solve(A, C)  not  solve(A.t(), C)
        A_inv_C = torch.linalg.solve(A, C)
        K_mat = self.W1.to(torch.complex64) @ A_inv_C
        bs = z.shape[0]
        r = self.rank
        _a1 = a1.detach() if detach_correction else a1
        _a2 = a2.detach() if detach_correction else a2
        da1 = (1.0 - _a1**2).to(torch.complex64)
        M1 = da1.unsqueeze(-1) * K_mat.unsqueeze(0)
        W2c = self.W2.to(torch.complex64)
        if detach_correction: W2c = W2c.detach()
        M2 = torch.bmm(W2c.unsqueeze(0).expand(bs,-1,-1), M1)
        da2 = (1.0 - _a2**2).to(torch.complex64)
        M3 = da2.unsqueeze(-1) * M2
        I_r = torch.eye(r, dtype=torch.complex64).unsqueeze(0)
        sign_c, logabs_c = torch.linalg.slogdet(I_r + M3)
        log_det_correction = logabs_c + 1j * torch.angle(sign_c)
        if detach_correction: log_det_correction = log_det_correction.detach()
        return log_det_A + log_det_correction


# =====================================================================
#  TRAIN ONE THIMBLE
# =====================================================================
def train_one(lam_val, seed, epochs_val, A_init, label=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    D_val = D
    b_init = torch.linspace(float(x_i_prop), float(x_f_prop), D_val+2)[1:-1]
    model = ThimbleFlowNL(D_val, rank=rank, b_init=b_init, A_init=A_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs_val, eta_min=1e-4)
    loss_history = []
    t0 = time.time()
    for epoch in range(epochs_val):
        optimizer.zero_grad()
        z = torch.randn(batch, D_val)
        x, a1, a2 = model(z)
        S = complex_action(x, x_i_prop, x_f_prop, lam_val, a_c, m, omega)
        log_P = -0.5*torch.sum(z**2, dim=1) - (D_val/2)*np.log(2*np.pi)
        log_det_J = model.compute_log_det_J(z, a1, a2, detach_correction=True)
        log_W = 1j*S + log_det_J - log_P.to(torch.complex64)
        loss = torch.var(log_W.real) + torch.var(log_W.imag)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())
        if (epoch+1) % 5000 == 0 or epoch == 0:
            print(f"  [{label}] Ep {epoch+1:5d}/{epochs_val} "
                  f"loss={loss.item():.4e} ({time.time()-t0:.0f}s)", flush=True)
    elapsed = time.time() - t0
    norm_factor = (m / (2*np.pi*1j*a_c)) ** (N/2)
    with torch.no_grad():
        z_mc = torch.randn(N_mc, D_val)
        x_mc, a1_mc, a2_mc = model(z_mc)
        S_mc = complex_action(x_mc, x_i_prop, x_f_prop, lam_val, a_c, m, omega)
        log_P_mc = -0.5*torch.sum(z_mc**2, dim=1) - (D_val/2)*np.log(2*np.pi)
        log_det_mc = model.compute_log_det_J(z_mc, a1_mc, a2_mc)
        log_W_mc = 1j*S_mc + log_det_mc - log_P_mc.to(torch.complex64)
        log_W_max = torch.max(log_W_mc.real)
        W_shifted = torch.exp(log_W_mc - log_W_max)
        K_mc = norm_factor * torch.exp(log_W_max).item() * torch.mean(W_shifted).numpy()
        W_for_err = norm_factor * torch.exp(log_W_max).item() * W_shifted.numpy()
        K_std = np.std(W_for_err) / np.sqrt(N_mc)
        W_ess = torch.exp(log_W_mc[:10000] - torch.max(log_W_mc[:10000].real)).numpy()
        ess = float(np.abs(np.mean(W_ess))**2 / np.mean(np.abs(W_ess)**2))
    K_re = float(K_mc.real); K_im = float(K_mc.imag)
    print(f"  [{label}] K={K_re:+.8f}{K_im:+.8f}j  |K|={abs(K_mc):.6f}  "
          f"loss={loss_history[-1]:.2e}  ESS={ess:.4f}  ({elapsed:.0f}s)")
    return dict(label=label, lam=lam_val, seed=seed,
                K_re=K_re, K_im=K_im, abs_K=float(abs(K_mc)),
                std_err=float(K_std), final_loss=loss_history[-1],
                ess_n=ess, time_s=elapsed, epochs=epochs_val)


# =====================================================================
#  MEHLER
# =====================================================================
def mehler_kernel():
    wT = omega * T_c
    sinwT = np.sin(wT); coswT = np.cos(wT)
    pf = np.sqrt(m*omega / (2*np.pi*1j*sinwT))
    ex = (1j*m*omega/(2*sinwT))*((x_i_prop**2+x_f_prop**2)*coswT - 2*x_i_prop*x_f_prop)
    return pf * np.exp(ex)


# =====================================================================
#  HELPERS
# =====================================================================
def rel_err(K, exact):
    e1 = abs(K - exact) / abs(exact)
    e2 = abs(K + exact) / abs(exact)
    return min(e1, e2)


def save_results(all_results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)


def find_best_combination(K_list, exact, max_n=2):
    """Try all integer-weight combinations n_k in [-max_n, max_n] and
    return the one closest to exact (or -exact)."""
    n_thimbles = len(K_list)
    best_err = 999.0
    best_weights = None
    best_K = None
    rng = range(-max_n, max_n+1)
    for ns in iproduct(rng, repeat=n_thimbles):
        if all(n == 0 for n in ns):
            continue
        K_comb = sum(n*k for n, k in zip(ns, K_list))
        err = rel_err(K_comb, exact)
        if err < best_err:
            best_err = err
            best_weights = list(ns)
            best_K = K_comb
    return best_weights, best_K, best_err


# =====================================================================
#  REPORT WRITER
# =====================================================================
def write_report(results_by_lam, timestamp):
    lines = []
    lines.append("# Multi-Thimble Overnight Report")
    lines.append(f"\nGenerated: {timestamp}")
    lines.append("\nDet-bug FIXED (A^{-1}C). All runs use multi-thimble branch init.\n")
    lines.append("## Exact Lattice Targets (N=32)\n")
    lines.append("| lam | K_re | K_im | |K| |")
    lines.append("|-----|------|------|-----|")
    for lam in sorted(EXACT_LATTICE):
        e = EXACT_LATTICE[lam]
        lines.append(f"| {lam} | {e.real:+.7f} | {e.imag:+.7f} | {abs(e):.7f} |")

    for lam_val in sorted(results_by_lam):
        exact = EXACT_LATTICE[lam_val]
        runs = results_by_lam[lam_val]
        lines.append(f"\n## lam = {lam_val}\n")
        lines.append(f"Exact: K = {exact:.8f}, |K| = {abs(exact):.8f}\n")
        lines.append("### Individual thimble results\n")
        lines.append("| Label | Flipped modes | Seed | K_re | K_im | |K| | Loss | ESS | Err |")
        lines.append("|-------|---------------|------|------|------|-----|------|-----|-----|")
        for r in runs:
            K = r['K_re'] + 1j*r['K_im']
            err = rel_err(K, exact)
            fm = r.get('flip_modes', '()')
            lines.append(f"| {r['label']} | {fm} | {r['seed']} | "
                         f"{r['K_re']:+.7f} | {r['K_im']:+.7f} | "
                         f"{r['abs_K']:.6f} | {r['final_loss']:.2e} | "
                         f"{r['ess_n']:.4f} | {err:.1%} |")

        K_list = [r['K_re']+1j*r['K_im'] for r in runs]
        if len(K_list) >= 2:
            lines.append("\n### Best weighted combination\n")
            ws, K_c, err_c = find_best_combination(K_list, exact)
            lines.append(f"- Weights: {ws}")
            lines.append(f"- K_combined = {K_c:.8f}")
            lines.append(f"- |K_combined| = {abs(K_c):.8f}")
            lines.append(f"- Error vs exact: {err_c:.2%}")
            ok = err_c < 0.05
            lines.append(f"- **{'MATCH' if ok else 'no match'}**")
        lines.append("\n---\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  -> Report: {REPORT_PATH}")


# =====================================================================
#  MAIN
# =====================================================================
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Multi-Thimble Overnight Run  (det bug FIXED)")
    print(f"Started: {timestamp}")
    print(f"N={N} D={D} T={T} eta={eta} omega={omega} m={m} rank={rank}")
    print(f"a_c = {a_c}\n")

    K_meh = mehler_kernel()
    print(f"Mehler K = {K_meh:.10f}  |K| = {abs(K_meh):.10f}\n")

    all_results = []
    results_by_lam = {}
    lambda_list = [0.0, 0.7, 2.4]

    # Branch configurations to try.
    # Each entry: (flip_modes_tuple, list_of_seeds)
    # flip_modes=() is the standard branch, (0,) flips the lowest mode, etc.
    BRANCH_CONFIGS = [
        ((),      [42, 43, 44]),      # standard branch, 3 seeds
        ((0,),    [42, 43]),           # flip lowest mode
        ((0,1),   [42, 43]),           # flip 2 lowest modes
        ((0,1,2), [42]),              # flip 3 lowest modes
        (tuple(range(D//2)),  [42]),   # flip bottom half of spectrum
        (tuple(range(D)),     [42, 43]),  # flip ALL modes (full parity)
    ]

    epochs_lam0 = 15000
    epochs_other = 25000

    for lam_val in lambda_list:
        exact = EXACT_LATTICE[lam_val]
        ep = epochs_lam0 if lam_val == 0.0 else epochs_other
        print("\n" + "="*70)
        print(f"  lam = {lam_val}   exact = {exact:.8f}   epochs = {ep}")
        print("="*70)

        lam_runs = []

        for flip_modes, seeds in BRANCH_CONFIGS:
            for sd in seeds:
                fm_str = str(flip_modes) if flip_modes else "none"
                lbl = f"lam{lam_val}_flip{fm_str}_s{sd}"
                print(f"\n--- {lbl} ---")
                A_init = compute_hessian_A_branch(N, T, m, omega, D,
                                                  flip_modes=flip_modes)
                r = train_one(lam_val, sd, ep, A_init, label=lbl)
                r['flip_modes'] = str(flip_modes)
                r['strategy'] = 'multi_branch'
                lam_runs.append(r)
                all_results.append(r)
                save_results(all_results)

        results_by_lam[lam_val] = lam_runs

        # quick combination check
        K_list = [r['K_re']+1j*r['K_im'] for r in lam_runs]
        ws, K_c, err_c = find_best_combination(K_list, exact)
        print(f"\n  >> Best combo: weights={ws}, K={K_c:.8f}, err={err_c:.1%}")
        if err_c < 0.05:
            print(f"  >> MATCH for lam={lam_val}!")

    # final report
    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)
    write_report(results_by_lam, timestamp)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for lam_val in lambda_list:
        exact = EXACT_LATTICE[lam_val]
        runs = results_by_lam[lam_val]
        K_list = [r['K_re']+1j*r['K_im'] for r in runs]
        ws, K_c, err_c = find_best_combination(K_list, exact)
        tag = " <<<" if err_c < 0.05 else ""
        print(f"  lam={lam_val:4.1f}  exact={exact:+.8f}  "
              f"best_combo={K_c:+.8f}  err={err_c:.1%}  w={ws}{tag}")
        for r in runs:
            K = r['K_re']+1j*r['K_im']
            e = rel_err(K, exact)
            print(f"    {r['label']:45s}  K={K:+.8f}  err={e:.1%}")

    print(f"\nDone. {len(all_results)} runs total.")
    print(f"Results: {RESULTS_PATH}")
    print(f"Report:  {REPORT_PATH}")
