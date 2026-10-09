"""General propagator evaluation at lambda=0.3 for T=4 and T=5.

Three separate attempts, each written down before any lattice number is read.
The lattice FFT is computed only after each attempt's answer is saved.

Attempt 1  More thimbles. Every saddle found by a wide search is kept or
           dropped by one frozen geometric rule (see `on_real_cycle`), then
           each kept thimble is integrated with the notebook's SaddleFlow.
Attempt 2  Flowed real contour. The original real contour is pushed along the
           holomorphic flow for a short time. No saddle is chosen by hand;
           the intersection numbers are built in. Importance sampling with a
           Gaussian proposal whose widths are exact for the free action.
           Several flow times are evaluated; the integral must not depend on
           the flow time, so their agreement is the internal check.
Attempt 3  Same flowed contour, with the proposal adapted to the sampled
           magnitude of the integrand (defensive Gaussian mixture).

Usage
  python thimble_general_lam0p3.py --T 4.0 5.0 --attempt 1
  python thimble_general_lam0p3.py --T 4.0 5.0 --attempt 2
  python thimble_general_lam0p3.py --T 4.0 5.0 --attempt 3
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import root

import investigate_lambda_up_0p3_to_1 as inv

ROOT = Path(__file__).resolve().parent
inv.NB_PATH = ROOT / "Complex_Time_QHO_Anharmonic_MU_lambda0p3_T4_morethimbles.ipynb"
OUT = ROOT / "runs" / "general_lam0p3"
OUT.mkdir(parents=True, exist_ok=True)

LAM = 0.3
ETA = 0.05
N_SITES = 16

# ---------------------------------------------------------------- frozen rules
# A saddle is on the real cycle when its dual (descending Im S) manifold meets
# the real contour. We shoot rays from the saddle along the dual tangent space
# and record how close any ray comes to a real path. Real-cycle saddles seen so
# far come within 0.03 of real; off-cycle ones never get below about 2.7.
# 0.3 is the geometric middle of that gap and is fixed here.
CYCLE_IM_CUT = 0.3
RAYS_PER_SADDLE = 100
# Thimbles with Im S more than 30 above the trivial saddle carry a factor
# below e^-30 and are recorded as zero instead of being trained.
SUPPRESS_IM_S = 30.0

# Attempt 1 network, same as the T=4 run that preceded this file.
RANK, EPOCHS, BATCH, LR, N_MC = 32, 2000, 1024, 1.0e-3, 200_000

# Attempt 2 and 3 settings.
# Only finite-time blow-up stops a sample; large Im S is kept in the log weight.
FLOW_DT = 0.01
FLOW_X_STOP = 60.0
FINAL_SAMPLES = 200_000
CHUNK = 20_000
# The flowed-contour integral does not depend on the flow time. Attempt 2
# evaluates it at several short flow times with the exact Gaussian widths
# (scale 1) and reports the precision-weighted mean; the spread between flow
# times is the internal consistency check. Long flow times make the integrand
# exponentially thin around each saddle preimage and break fixed-proposal
# importance sampling, so they are not used.
TFLOW_LIST = [0.1, 0.15, 0.2, 0.3, 0.4]
PROPOSAL_SCALE = 1.0
# Attempt 3: adaptive rounds, mixture size, defensive weight on the base proposal.
ADAPT_ROUNDS = 2
ADAPT_COMPONENTS = 4
ADAPT_DEFENSIVE = 0.3
ADAPT_T_FLOW = 0.2

TORCH_DTYPE = torch.complex128


# ------------------------------------------------------------- notebook setup
def configure(g, T):
    g["LAMBDA"] = LAM
    g["T"] = float(T)
    g["A_C"] = (float(T) - 1j * g["ETA"]) / g["N"]
    g["RANK"] = RANK
    g["EPOCHS"] = EPOCHS
    g["BATCH"] = BATCH
    g["LEARNING_RATE"] = LR
    g["N_MC"] = N_MC
    g["LOG_EVERY"] = 500


def ttag(T):
    return f"T{T:g}".replace(".", "p")


def out_dir(T):
    d = OUT / ttag(T)
    d.mkdir(parents=True, exist_ok=True)
    return d


def prefactor(T):
    ac = (T - 1j * ETA) / N_SITES
    return (1.0 / (2.0 * np.pi * 1j * ac)) ** (N_SITES / 2)


# --------------------------------------------------------- saddle bookkeeping
def same(a, b, tol=1e-3):
    return np.max(np.abs(a - b)) < tol or np.max(np.abs(a + b)) < tol


def wide_search(g):
    """Harmonics 1..8 and two-mode sine mixes. Returns distinct saddles."""
    D = g["D"]
    sites = np.arange(1, g["N"])
    shapes = [np.sin(h * np.pi * sites / g["N"]) for h in range(1, 9)]
    for h1, h2 in ((1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (1, 4)):
        s1 = np.sin(h1 * np.pi * sites / g["N"])
        s2 = np.sin(h2 * np.pi * sites / g["N"])
        for mix in (0.5, 1.0):
            shapes.append(s1 + mix * s2)
            shapes.append(s1 - mix * s2)
    found = []
    for shape in shapes:
        for amp in np.linspace(0.4, 12.0, 24):
            guess = np.r_[amp * shape, np.zeros(D)]
            sol = root(g["packed_gradient"], guess, method="lm", options={"maxiter": 4000})
            x = sol.x[:D] + 1j * sol.x[D:]
            if np.max(np.abs(g["gradient_np"](x))) > 1e-8 or np.max(np.abs(x)) < 1e-3:
                continue
            if any(same(x, old) for old in found):
                continue
            found.append(x)
    found.sort(key=lambda x: abs(g["action_np"](x)))
    return found


def dual_descent(g, saddle, start, t_max=8.0):
    D = g["D"]

    def rhs(_t, y):
        x = y[:D] + 1j * y[D:]
        dx = -1j * np.conjugate(g["gradient_np"](x))
        return np.r_[dx.real, dx.imag]

    def far(_t, y):
        return np.max(np.abs(y)) - 80.0

    far.terminal = True
    sol = solve_ivp(rhs, (0.0, t_max), np.r_[start.real, start.imag], events=far,
                    rtol=1e-6, atol=1e-8, max_step=0.05)
    return float(np.min(np.max(np.abs(sol.y[D:, :]), axis=0)))


def on_real_cycle(g, saddle, tangent, seed=0):
    """Frozen rule: closest approach of dual rays to a real path is below CYCLE_IM_CUT."""
    D = g["D"]
    rng = np.random.default_rng(seed)
    closest = float("inf")
    directions = []
    for axis in range(D):
        for sign in (1.0, -1.0):
            z = np.zeros(D)
            z[axis] = sign
            directions.append(z)
    while len(directions) < RAYS_PER_SADDLE:
        z = rng.normal(size=D)
        directions.append(z / np.linalg.norm(z))
    for z in directions:
        start = saddle + 1j * tangent @ (0.03 * z)
        closest = min(closest, dual_descent(g, saddle, start))
        if closest < CYCLE_IM_CUT:
            break
    return closest < CYCLE_IM_CUT, closest


def previous_integrals(T):
    """Rank-32 integrals already computed with this architecture, keyed by saddle."""
    rows = []
    if abs(T - 4.0) < 1e-9:
        res = ROOT / "runs" / "lambda0p3_T4_morethimbles" / "results.json"
        sad = np.load(ROOT / "runs" / "lambda_up_0p3_to_1" / "saddles_lam0p3_T4p0.npz")
        extra = np.load(ROOT / "runs" / "lambda0p3_T4_morethimbles" / "extra_saddles.npz")
        table = {**{k: sad[k] for k in sad.files}, **{k: extra[k] for k in extra.files}}
    elif abs(T - 5.0) < 1e-9:
        res = ROOT / "runs" / "lambda0p3_T5_realcycle" / "results.json"
        sad = np.load(ROOT / "runs" / "lambda0p3_T5_realcycle" / "saddles.npz")
        table = {k: sad[k] for k in sad.files}
    else:
        return rows
    if not res.exists():
        return rows
    data = json.loads(res.read_text(encoding="utf-8"))
    for label, row in data.items():
        if label == "trivial":
            x = np.zeros(15, dtype=np.complex128)
        elif label in table:
            x = table[label]
        else:
            continue
        rows.append((x, complex(row["K_re"], row["K_im"]), float(row["ESS"])))
    return rows


# ------------------------------------------------------------------ attempt 1
def attempt_more_thimbles(g, T):
    d = out_dir(T)
    configure(g, T)
    print(f"\n===== Attempt 1, more thimbles, lambda={LAM} T={T} =====", flush=True)
    saddles = wide_search(g)
    print(f"{len(saddles)} distinct nontrivial saddles", flush=True)
    s0 = g["action_np"](np.zeros(g["D"], dtype=np.complex128))

    kept, report = [], []
    for i, x in enumerate(saddles):
        s = g["action_np"](x)
        re, im = np.max(np.abs(x.real)), np.max(np.abs(x.imag))
        try:
            tangent = g["tangent_map"](g["hessian_np"](x))
        except RuntimeError:
            print(f"  s{i+1}: tangent failed, skipped  S={s:+.3f}", flush=True)
            report.append({"id": i + 1, "status": "tangent_failed", "S": [s.real, s.imag]})
            continue
        on, closest = on_real_cycle(g, x, tangent, seed=i)
        suppressed = (s.imag - s0.imag) > SUPPRESS_IM_S
        status = "drop" if not on else ("zero" if suppressed else "keep")
        print(f"  s{i+1}: max|Re|={re:.3f} max|Im|={im:.3f} dual-closest={closest:.3e} "
              f"S={s.real:+.3f}{s.imag:+.3f}i  {status}", flush=True)
        report.append({"id": i + 1, "status": status, "closest": closest,
                       "max_re": re, "max_im": im, "S": [s.real, s.imag]})
        if status == "keep":
            kept.append((f"s{i+1}", x, tangent))
    (d / "attempt1_selection.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"Kept for training: {[k[0] for k in kept]}", flush=True)

    store_path = d / "attempt1_integrals.json"
    store = json.loads(store_path.read_text(encoding="utf-8")) if store_path.exists() else {}
    old = previous_integrals(T)

    def integrate(label, x, seed):
        if label in store:
            row = store[label]
            k = complex(row["K_re"], row["K_im"])
            print(f"  reuse {label}: K={k:+.6f} ESS/N={row['ESS']:.4g}", flush=True)
            return k
        for ox, ok, oe in old:
            if same(x, ox, 1e-6):
                print(f"  reuse previous rank-32 integral for {label}: K={ok:+.6f} ESS/N={oe:.4g}", flush=True)
                store[label] = {"K_re": ok.real, "K_im": ok.imag, "ESS": oe, "source": "previous"}
                store_path.write_text(json.dumps(store, indent=1), encoding="utf-8")
                return ok
        print(f"\n*** STARTING ML TRAINING: {label} at lambda={LAM}, T={T} ***", flush=True)
        model, _h, secs = g["train_thimble"](label, x, seed)
        ests, esses = [], []
        for j in range(2):
            est, err, ess, _s, _b = g["estimate_thimble"](model, seed + 31 * j)
            ests.append(est)
            esses.append(ess)
            print(f"  draw {j+1}: K={est:+.6f} ESS/N={ess:.5f} stderr={err:.2e}", flush=True)
        k = sum(ests) / 2
        store[label] = {"K_re": k.real, "K_im": k.imag, "ESS": float(np.median(esses)), "seconds": secs}
        store_path.write_text(json.dumps(store, indent=1), encoding="utf-8")
        return k

    total = integrate("trivial", np.zeros(g["D"], dtype=np.complex128), g["SEED"])
    for i, (label, x, _tan) in enumerate(kept):
        total += 2 * integrate(label, x, g["SEED"] + 7 + i)

    summary = {"K_re": total.real, "K_im": total.imag, "kept": ["trivial"] + [k[0] for k in kept]}
    (d / "attempt1_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nAttempt 1 sum written: K={total:+.6f}", flush=True)
    compare(d, "attempt1", total, T)


# ------------------------------------------------------- holomorphic flow core
class Flow:
    """Batched holomorphic flow x' = i conj(dS/dx) with Jacobian transport."""

    def __init__(self, T):
        self.ac = complex((T - 1j * ETA) / N_SITES)
        self.T = T

    def action(self, x):
        pad = torch.nn.functional.pad(x, (1, 1))
        dx = pad[:, 1:] - pad[:, :-1]
        return (0.5 / self.ac) * torch.sum(dx**2, dim=1) \
            - 0.5 * self.ac * torch.sum(x**2, dim=1) \
            - 0.25 * self.ac * LAM * torch.sum(x**4, dim=1)

    def gradient(self, x):
        pad = torch.nn.functional.pad(x, (1, 1))
        return (1.0 / self.ac) * (2 * x - pad[:, :-2] - pad[:, 2:]) - self.ac * x - self.ac * LAM * x**3

    def hess_apply(self, x, J):
        diag = 2.0 / self.ac - self.ac - 3.0 * self.ac * LAM * x**2
        HJ = diag.unsqueeze(-1) * J
        HJ[:, 1:, :] -= J[:, :-1, :] / self.ac
        HJ[:, :-1, :] -= J[:, 1:, :] / self.ac
        return HJ

    def rhs(self, x, J):
        return 1j * torch.conj(self.gradient(x)), 1j * torch.conj(self.hess_apply(x, J))

    def run(self, x0, t_flow):
        """x0 real (B,D). Returns x, J, alive mask (alive = did not blow up)."""
        B, D = x0.shape
        x = x0.to(TORCH_DTYPE)
        J = torch.eye(D, dtype=TORCH_DTYPE).expand(B, D, D).clone()
        alive = torch.ones(B, dtype=torch.bool)
        steps = int(round(t_flow / FLOW_DT))
        for _ in range(steps):
            xa, Ja = x[alive], J[alive]
            k1x, k1J = self.rhs(xa, Ja)
            k2x, k2J = self.rhs(xa + 0.5 * FLOW_DT * k1x, Ja + 0.5 * FLOW_DT * k1J)
            k3x, k3J = self.rhs(xa + 0.5 * FLOW_DT * k2x, Ja + 0.5 * FLOW_DT * k2J)
            k4x, k4J = self.rhs(xa + FLOW_DT * k3x, Ja + FLOW_DT * k3J)
            xa = xa + FLOW_DT / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
            Ja = Ja + FLOW_DT / 6 * (k1J + 2 * k2J + 2 * k3J + k4J)
            x[alive], J[alive] = xa, Ja
            bad = (torch.max(torch.abs(x), dim=1).values > FLOW_X_STOP) | ~torch.isfinite(x).all(dim=1)
            alive = alive & ~bad
            if not alive.any():
                break
        return x, J, alive

    def log_integrand(self, x0, t_flow):
        """log|det J e^{iS}| and its phase on the flowed contour. Blow-ups get -inf."""
        x, J, alive = self.run(x0, t_flow)
        s = self.action(x)
        sign, logabs = torch.linalg.slogdet(J)
        logf = logabs - s.imag
        logf = torch.where(alive & torch.isfinite(logf), logf, torch.full_like(logf, -float("inf")))
        phase = torch.exp(1j * s.real) * sign
        return logf, phase, alive


def mode_basis():
    j = np.arange(1, N_SITES)
    return np.sqrt(2.0 / N_SITES) * np.sin(np.outer(j, j) * np.pi / N_SITES)


def free_flow_widths(T, t_flow):
    """Exact flowed width of each Gaussian mode S = h x^2 / 2 with complex h.

    (u, v)' = [[hi, hr], [hr, -hi]] (u, v),  Im S = hr u v + hi (u^2 - v^2) / 2.
    """
    ac = complex((T - 1j * ETA) / N_SITES)
    j = np.arange(1, N_SITES)
    h = 2.0 / ac - ac - (2.0 / ac) * np.cos(j * np.pi / N_SITES)
    sig = np.empty(N_SITES - 1)
    for k in range(N_SITES - 1):
        hr, hi = h[k].real, h[k].imag
        M = np.array([[hi, hr], [hr, -hi]])
        lam = math.hypot(hr, hi)
        E = np.cosh(lam * t_flow) * np.eye(2) + np.sinh(lam * t_flow) / lam * M
        c, s_ = E @ np.array([1.0, 0.0])
        f = hr * c * s_ + 0.5 * hi * (c * c - s_ * s_)
        sig[k] = 1.0 / math.sqrt(2.0 * f)
    return sig


class Proposal:
    """Gaussian on the real contour in the sine-mode basis, matched to the flowed free action."""

    def __init__(self, T, t_flow, scale):
        D = N_SITES - 1
        self.V = torch.tensor(mode_basis(), dtype=torch.float64)
        self.sig = torch.tensor(scale * free_flow_widths(T, t_flow), dtype=torch.float64)
        self.const = -0.5 * D * math.log(2 * math.pi) - float(torch.sum(torch.log(self.sig)))

    def sample(self, n, gen):
        c = torch.randn(n, N_SITES - 1, dtype=torch.float64, generator=gen) * self.sig
        return c @ self.V.T

    def logq(self, x0):
        c = x0 @ self.V
        return -0.5 * torch.sum((c / self.sig)**2, dim=1) + self.const


class MixtureProposal:
    """Defensive mixture: base Gaussian plus diagonal Gaussians in the mode basis."""

    def __init__(self, base, means, sigs, weights):
        self.base = base
        self.V = base.V
        self.means = torch.tensor(np.asarray(means), dtype=torch.float64)      # (M, D) mode coords
        self.sigs = torch.tensor(np.asarray(sigs), dtype=torch.float64)        # (M, D)
        self.weights = torch.tensor(np.asarray(weights), dtype=torch.float64)  # (M+1,), first = base
        D = N_SITES - 1
        self.consts = -0.5 * D * math.log(2 * math.pi) - torch.sum(torch.log(self.sigs), dim=1)

    def sample(self, n, gen):
        comp = torch.multinomial(self.weights, n, replacement=True, generator=gen)
        out = torch.empty(n, N_SITES - 1, dtype=torch.float64)
        base_mask = comp == 0
        nb = int(base_mask.sum())
        if nb:
            out[base_mask] = self.base.sample(nb, gen)
        for m in range(self.means.shape[0]):
            mask = comp == m + 1
            nm = int(mask.sum())
            if nm:
                c = self.means[m] + torch.randn(nm, N_SITES - 1, dtype=torch.float64, generator=gen) * self.sigs[m]
                out[mask] = c @ self.V.T
        return out

    def logq(self, x0):
        c = x0 @ self.V
        parts = [torch.log(self.weights[0]) + self.base.logq(x0)]
        for m in range(self.means.shape[0]):
            lq = -0.5 * torch.sum(((c - self.means[m]) / self.sigs[m])**2, dim=1) + self.consts[m]
            parts.append(torch.log(self.weights[m + 1]) + lq)
        return torch.logsumexp(torch.stack(parts, dim=0), dim=0)


def estimate_from_logs(logw, phase, pref):
    finite = torch.isfinite(logw)
    if not finite.any():
        return 0j, 0.0, float("inf")
    shift = torch.max(logw[finite])
    w = torch.exp(logw - shift) * phase
    w = torch.where(finite, w, torch.zeros_like(w))
    mean = torch.mean(w)
    scale = pref * math.exp(float(shift))
    k = complex(scale * mean.numpy())
    ess = float(torch.abs(mean)**2 / torch.mean(torch.abs(w)**2))
    stderr = abs(scale) * math.sqrt(float(torch.mean(torch.abs(w - mean)**2)) / len(w))
    return k, ess, stderr


def draw_weights(flow, prop, n, t_flow, seed, keep_x=False):
    gen = torch.Generator().manual_seed(seed)
    logs, phases, xs, alive_n = [], [], [], 0
    done = 0
    while done < n:
        m = min(CHUNK, n - done)
        x0 = prop.sample(m, gen)
        lf, ph, alive = flow.log_integrand(x0, t_flow)
        logs.append(lf - prop.logq(x0))
        phases.append(ph)
        if keep_x:
            xs.append(x0)
        alive_n += int(alive.sum())
        done += m
    out = (torch.cat(logs), torch.cat(phases), alive_n / n)
    return out + (torch.cat(xs),) if keep_x else out


def weighted_mean(rows):
    """Precision-weighted complex mean of {"K_re","K_im","stderr"} rows."""
    w = np.array([1.0 / max(r["stderr"], 1e-12)**2 for r in rows])
    ks = np.array([complex(r["K_re"], r["K_im"]) for r in rows])
    return complex(np.sum(w * ks) / np.sum(w)), float(1.0 / math.sqrt(np.sum(w)))


# ------------------------------------------------------------------ attempt 2
def attempt_flowed_contour(g, T):
    d = out_dir(T)
    print(f"\n===== Attempt 2, flowed real contour, lambda={LAM} T={T} =====", flush=True)
    flow = Flow(T)
    pref = prefactor(T)
    path = d / "attempt2_by_tflow.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    done = {r["t_flow"] for r in rows}
    for t_flow in TFLOW_LIST:
        if t_flow in done:
            continue
        t0 = time.time()
        prop = Proposal(T, t_flow, PROPOSAL_SCALE)
        lw, ph, alive = draw_weights(flow, prop, FINAL_SAMPLES, t_flow, seed=101)
        k, ess, err = estimate_from_logs(lw, ph, pref)
        rows.append({"t_flow": t_flow, "K_re": k.real, "K_im": k.imag, "ESS": ess, "stderr": err, "alive": alive})
        path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"  t_flow={t_flow:.2f}: K={k:+.6f}  ESS/N={ess:.4g}  stderr={err:.2e}  alive={alive:.3f}  "
              f"{time.time()-t0:.0f}s", flush=True)
    k, err = weighted_mean(rows)
    spread = float(np.std([abs(complex(r["K_re"], r["K_im"]) - k) for r in rows]))
    summary = {"K_re": k.real, "K_im": k.imag, "stderr": err, "spread_between_tflow": spread, "rows": rows}
    (d / "attempt2_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nAttempt 2 estimate written: K={k:+.6f}  stderr={err:.2e}  spread={spread:.2e}", flush=True)
    compare(d, "attempt2", k, T)


# ------------------------------------------------------------------ attempt 3
def fit_mixture(c, w, n_comp, rng):
    """Weighted k-means in mode coordinates, then per-cluster mean and std."""
    n = c.shape[0]
    idx = rng.choice(n, size=n_comp, replace=False, p=w / w.sum())
    centers = c[idx].copy()
    for _ in range(25):
        d2 = ((c[:, None, :] - centers[None, :, :])**2).sum(axis=2)
        lab = np.argmin(d2, axis=1)
        for m in range(n_comp):
            mask = lab == m
            if w[mask].sum() > 0:
                centers[m] = np.average(c[mask], axis=0, weights=w[mask])
    means, sigs, weights = [], [], []
    for m in range(n_comp):
        mask = lab == m
        wm = w[mask].sum()
        if wm <= 0 or mask.sum() < 20:
            continue
        mu = np.average(c[mask], axis=0, weights=w[mask])
        var = np.average((c[mask] - mu)**2, axis=0, weights=w[mask])
        means.append(mu)
        sigs.append(1.5 * np.sqrt(np.maximum(var, 1e-12)))  # inflate for safety
        weights.append(wm)
    weights = np.array(weights) / np.sum(weights)
    return means, sigs, weights


def attempt_adaptive(g, T):
    d = out_dir(T)
    print(f"\n===== Attempt 3, flowed contour with adaptive mixture proposal, lambda={LAM} T={T} =====", flush=True)
    flow = Flow(T)
    pref = prefactor(T)
    t_flow = ADAPT_T_FLOW
    base = Proposal(T, t_flow, PROPOSAL_SCALE)
    prop = base
    rng = np.random.default_rng(17)
    V = mode_basis()
    history = []
    for rnd in range(ADAPT_ROUNDS + 1):
        t0 = time.time()
        lw, ph, alive, x0 = draw_weights(flow, prop, FINAL_SAMPLES, t_flow, seed=201 + rnd, keep_x=True)
        k, ess, err = estimate_from_logs(lw, ph, pref)
        history.append({"round": rnd, "K_re": k.real, "K_im": k.imag, "ESS": ess, "stderr": err, "alive": alive})
        print(f"  round {rnd}: K={k:+.6f}  ESS/N={ess:.4g}  stderr={err:.2e}  alive={alive:.3f}  "
              f"{time.time()-t0:.0f}s", flush=True)
        if rnd == ADAPT_ROUNDS:
            break
        finite = torch.isfinite(lw)
        mag = torch.exp(lw[finite] - torch.max(lw[finite])).numpy()
        c = (x0[finite].numpy() @ V)
        means, sigs, weights = fit_mixture(c, mag, ADAPT_COMPONENTS, rng)
        mix_w = np.r_[ADAPT_DEFENSIVE, (1.0 - ADAPT_DEFENSIVE) * weights]
        prop = MixtureProposal(base, means, sigs, mix_w)
        print(f"    fitted {len(means)} components, weights={np.round(mix_w, 3).tolist()}", flush=True)
    final = history[-1]
    k = complex(final["K_re"], final["K_im"])
    summary = {"K_re": k.real, "K_im": k.imag, "stderr": final["stderr"], "ESS": final["ESS"],
               "t_flow": t_flow, "history": history}
    (d / "attempt3_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nAttempt 3 estimate written: K={k:+.6f}", flush=True)
    compare(d, "attempt3", k, T)


# ----------------------------------------------------------------- comparison
def compare(d, tag, k, T):
    """Lattice is read only here, after the estimate has been written to disk."""
    k_lat = inv.lattice_K_fft(LAM, (T - 1j * ETA) / N_SITES)
    err = abs(k - k_lat) / abs(k_lat)
    print(f"[{tag} T={T}] ML={k:+.6f}  lattice={k_lat:+.6f}  error={100*err:.3f}%", flush=True)
    path = d / f"{tag}_vs_lattice.json"
    path.write_text(json.dumps({"K_re": k.real, "K_im": k.imag, "lattice_re": k_lat.real,
                                "lattice_im": k_lat.imag, "rel_err": err}, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, nargs="+", default=[4.0, 5.0])
    ap.add_argument("--attempt", type=int, choices=[1, 2, 3], required=True)
    args = ap.parse_args()
    g = inv.load_notebook_namespace() if args.attempt == 1 else None
    for T in args.T:
        if args.attempt == 1:
            attempt_more_thimbles(g, T)
        elif args.attempt == 2:
            attempt_flowed_contour(g, T)
        else:
            attempt_adaptive(g, T)


if __name__ == "__main__":
    main()
