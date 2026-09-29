"""Generate the ROC/PR, review-tier capture, and risk-driver charts.
Run from the repo root after 02_train_model.py: python src/03_generate_figures.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888888", "axes.labelcolor": "#333333",
    "text.color": "#222222", "xtick.color": "#555555", "ytick.color": "#555555",
})
INK = "#1f4e79"       # primary
ACCENT = "#c8511b"    # critical/highlight
GRAY = "#9aa0a6"      # reference/random baseline

scores = pd.read_csv("outputs/test_scores.csv")
drivers = pd.read_csv("outputs/risk_drivers.csv", index_col=0)["coef"]

y = scores.misstate.values
s = scores.rf_score.values  # use Random Forest score (best lift) as the deployed score

# ---------------------------------------------------------------- Figure 1: ROC + PR curves
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

fpr, tpr, _ = roc_curve(y, s)
auc = roc_auc_score(y, s)
axes[0].plot(fpr, tpr, color=INK, lw=2.2, label=f"Model (AUC = {auc:.2f})")
axes[0].plot([0, 1], [0, 1], color=GRAY, lw=1.5, ls="--", label="Random")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC curve — flagging misstatement risk", loc="left", fontsize=11.5, fontweight="bold")
axes[0].legend(frameon=False, loc="lower right")

prec, rec, _ = precision_recall_curve(y, s)
ap = average_precision_score(y, s)
base = y.mean()
axes[1].plot(rec, prec, color=INK, lw=2.2, label=f"Model (AP = {ap:.3f})")
axes[1].axhline(base, color=GRAY, lw=1.5, ls="--", label=f"Random (base rate = {base:.2%})")
axes[1].set_xlabel("Recall (share of frauds caught)")
axes[1].set_ylabel("Precision (share of flags that are real)")
axes[1].set_title("Precision–recall — audit-flag quality", loc="left", fontsize=11.5, fontweight="bold")
axes[1].legend(frameon=False, loc="upper right")

fig.suptitle("Out-of-time test (scored on 2002–2010 filings, model trained on 1991–2001)", x=0.02, ha="left", fontsize=9.5, color="#666")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("figures/fig1_roc_pr.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2: capture / lift by audit-review tier
order = np.argsort(-s)
y_sorted = y[order]
total_fraud = y_sorted.sum()
tiers = [("Critical", 0, 1), ("High", 1, 5), ("Medium", 5, 20), ("Low", 20, 100)]
labels, captures, lifts, shares = [], [], [], []
n = len(y_sorted)
for name, lo, hi in tiers:
    i0, i1 = int(n * lo / 100), int(n * hi / 100)
    seg = y_sorted[i0:i1]
    cap = seg.sum() / total_fraud
    share_pct = (hi - lo)
    lift = (cap / (share_pct / 100)) if share_pct else 0
    labels.append(f"{name}\n(top {lo}-{hi}%)")
    captures.append(cap * 100)
    lifts.append(lift)

fig, ax = plt.subplots(figsize=(8.5, 4.3))
bars = ax.bar(labels, captures, color=[ACCENT, "#e08a4f", "#7fa8c9", GRAY], width=0.6)
for b, lift in zip(bars, lifts):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{lift:.1f}x lift",
            ha="center", fontsize=10, fontweight="bold", color="#333")
ax.set_ylabel("Share of known frauds captured")
ax.set_title("If audit reviewed only the model's top-ranked filings...", loc="left", fontsize=12, fontweight="bold")
ax.set_ylim(0, max(captures) * 1.25)
from matplotlib.ticker import PercentFormatter
ax.yaxis.set_major_formatter(PercentFormatter())
fig.tight_layout()
fig.savefig("figures/fig2_tier_capture.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3: top risk drivers
NICE = {
    "soft_assets": "Soft assets (low tangibility)", "reoa": "Retained earnings / assets (low)",
    "issue": "Issued securities recently", "ch_fcf": "Change in free cash flow (decline)",
    "dch_wc": "Change in working capital accruals", "dch_rec": "Change in receivables (unusual rise)",
    "ch_rsst": "Change in RSST accruals", "EBIT": "EBIT / assets", "dch_inv": "Change in inventory",
    "ch_cs": "Change in common shares", "ch_cm": "Change in cash margin", "bm": "Book-to-market ratio",
    "dpi": "Depreciation index", "ch_roa": "Change in return on assets",
}
top = drivers.reindex(drivers.abs().sort_values(ascending=False).index).head(10)[::-1]
colors = [ACCENT if v > 0 else INK for v in top.values]
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.barh([NICE.get(i, i) for i in top.index], top.values, color=colors, height=0.6)
ax.axvline(0, color="#888", lw=1)
ax.set_xlabel("Standardized coefficient (→ higher fraud risk)")
ax.set_title("Top risk drivers identified by the model", loc="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("figures/fig3_drivers.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("Saved fig1_roc_pr.png, fig2_tier_capture.png, fig3_drivers.png")
print("\nTier table:")
for l, c, lf in zip(labels, captures, lifts):
    print(f"  {l.splitlines()[0]:>10}: captures {c:5.1f}% of frauds, lift {lf:.1f}x")
