"""Compare Split-Operator (exact) results with ML results."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── ML data (from user) ──
ml_data = {
    "lambda": [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
               0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
    "Re_ML": [0.307230355, 0.307257564, 0.307613868, 0.308045423,
              0.30853144, 0.309051999, 0.311482275, 0.311483769,
              0.311707652, 0.312276772, 0.313795351, 0.315276192,
              0.318673821, 0.321627281, 0.326363101, 0.329806674,
              0.333883313, 0.335454471],
    "Im_ML": [-0.307623399, -0.307601922, -0.307246798, -0.306801876,
              -0.306316107, -0.305777055, -0.304049745, -0.30367083,
              -0.302996468, -0.302112874, -0.300164734, -0.298170929,
              -0.293095975, -0.287947061, -0.277650885, -0.267631712,
              -0.248951901, -0.232183184],
}
df_ml = pd.DataFrame(ml_data)
df_ml["abs_ML"] = np.sqrt(df_ml["Re_ML"]**2 + df_ml["Im_ML"]**2)

# ── Split-operator data ──
df_so = pd.read_csv("spectral_propagator_results.csv")
df_so.rename(columns={"K_real": "Re_SO", "K_imag": "Im_SO", "abs_K": "abs_SO"},
             inplace=True)

# ── Merge on lambda ──
df = pd.merge(df_ml, df_so, on="lambda", how="inner")

# ── Percentage errors ──
df["pct_err_Re"] = 100.0 * (df["Re_SO"] - df["Re_ML"]) / df["Re_ML"].abs()
df["pct_err_Im"] = 100.0 * (df["Im_SO"] - df["Im_ML"]) / df["Im_ML"].abs()
df["pct_err_abs"] = 100.0 * (df["abs_SO"] - df["abs_ML"]) / df["abs_ML"]

# ── Print table ──
pd.set_option("display.float_format", "{:.6f}".format)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print("=== Comparison: Split-Operator (SO) vs ML ===\n")
print(df[["lambda", "Re_SO", "Re_ML", "pct_err_Re",
          "Im_SO", "Im_ML", "pct_err_Im",
          "abs_SO", "abs_ML", "pct_err_abs"]].to_string(index=False))

# ── Save CSV ──
csv_path = "comparison_splitop_vs_ml.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved to {csv_path}")

# ═══════════════════════════════════════════════════════════════════
#  Plots — lambda on a numeric (not categorical) axis
# ═══════════════════════════════════════════════════════════════════
lam = df["lambda"].values

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ── Panel 1: Real part ──
ax = axes[0, 0]
ax.plot(lam, df["Re_SO"], "o-", color="navy", markersize=5, lw=1.8,
        label="Split-Operator (exact)")
ax.plot(lam, df["Re_ML"], "s--", color="crimson", markersize=5, lw=1.8,
        label="ML (warm-start)")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"Re $K$")
ax.set_title("Real part of propagator")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Panel 2: Imaginary part ──
ax = axes[0, 1]
ax.plot(lam, df["Im_SO"], "o-", color="navy", markersize=5, lw=1.8,
        label="Split-Operator (exact)")
ax.plot(lam, df["Im_ML"], "s--", color="crimson", markersize=5, lw=1.8,
        label="ML (warm-start)")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"Im $K$")
ax.set_title("Imaginary part of propagator")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Panel 3: Magnitude ──
ax = axes[1, 0]
ax.plot(lam, df["abs_SO"], "o-", color="navy", markersize=5, lw=1.8,
        label="Split-Operator (exact)")
ax.plot(lam, df["abs_ML"], "s--", color="crimson", markersize=5, lw=1.8,
        label="ML (warm-start)")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$|K|$")
ax.set_title("Magnitude of propagator")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Panel 4: % error in magnitude ──
ax = axes[1, 1]
ax.plot(lam, df["pct_err_abs"], "D-", color="darkgreen", markersize=5, lw=1.8)
ax.axhline(0, color="grey", lw=0.8)
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel("% error in $|K|$")
ax.set_title(r"Relative error: $\frac{|K|_{\mathrm{SO}} - |K|_{\mathrm{ML}}}{|K|_{\mathrm{ML}}} \times 100$")
ax.grid(True, alpha=0.3)

fig.suptitle(r"Split-Operator (exact) vs ML warm-start  —  $K(0,0;\,T\!=\!1)$",
             fontsize=14, y=1.01)
fig.tight_layout()
plt.savefig("comparison_splitop_vs_ml.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Plot saved to comparison_splitop_vs_ml.png")
