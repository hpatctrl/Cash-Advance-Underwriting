"""Static charts for the README and memo (matplotlib, PNG)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical slots 1-4
BLUE_ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # safest -> riskiest

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2,
})


def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def roc_chart(y, preds: dict, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for (name, p), c in zip(preds.items(), SERIES):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=c, label=name)
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="Random")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("ROC, out-of-time cohorts")
    ax.legend(loc="lower right")
    _finish(fig, path)


def calibration_chart(table: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x = np.arange(1, len(table) + 1)
    ax.bar(x - 0.2, table["predicted"] * 100, width=0.38, color=SERIES[0], label="Predicted")
    ax.bar(x + 0.2, table["actual"] * 100, width=0.38, color=SERIES[1], label="Actual")
    ax.set(xlabel="Risk decile (1 = safest)", ylabel="First-advance charge-off rate (%)", xticks=x)
    ax.set_title("Calibration by decile, out-of-time cohorts")
    ax.legend(loc="upper left")
    _finish(fig, path)


def loss_curve_chart(curves: pd.DataFrame, path, title, colors, label_fmt="{}"):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for col, c in zip(curves.columns, colors):
        ax.plot(curves.index, curves[col] * 100, color=c, label=label_fmt.format(col))
    # end labels, nudged apart so they never overlap
    ends = sorted(((curves[col].iloc[-1] * 100, col) for col in curves.columns))
    span = (curves.max().max() - curves.min().min()) * 100
    gap, placed = span * 0.045, []
    for v, col in ends:
        y = max(v, placed[-1] + gap) if placed else v
        placed.append(y)
        ax.annotate(f"{v:.1f}%", (curves.index[-1], v), xytext=(curves.index[-1] + 1.5, y), textcoords="data",
                    va="center", color=INK2, fontsize=9)
    ax.set(xlabel="Days after due date", ylabel="Advance dollars still unpaid (%)", xlim=(0, 66))
    ax.set_title(title)
    ax.legend(loc="upper right")
    _finish(fig, path)


def cutoff_chart(cc: pd.DataFrame, chosen: float, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    x = cc["approval_target"] * 100
    ax = axes[0]
    ax.plot(x, cc["contribution_per_1000_applicants"] / 1000, color=SERIES[0], marker="o", ms=4)
    ax.axhline(0, color=INK2, lw=1)
    ax.axvline(chosen * 100, color=INK2, lw=1, ls="--")
    ax.annotate(f"Chosen: {chosen:.0%} approval", (chosen * 100, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1), xytext=(6, 0),
                textcoords="offset points", color=INK2, fontsize=9)
    ax.set(xlabel="Approval rate (%)", ylabel="12-month contribution per 1,000 applicants ($K)")
    ax.set_title("Profit vs. approval rate")
    ax = axes[1]
    ax.plot(x, cc["net_loss_rate_of_originations"] * 100, color=SERIES[1], marker="o", ms=4)
    ax.axvline(chosen * 100, color=INK2, lw=1, ls="--")
    ax.set(xlabel="Approval rate (%)", ylabel="Net loss rate (% of originations)")
    ax.set_title("Loss rate vs. approval rate")
    _finish(fig, path)


def strategy_chart(sc: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    names = [s.split(". ", 1)[1] for s in sc["strategy"]]
    vals = sc["contribution_per_1000_applicants"].to_numpy() / 1000
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=SERIES[0], height=0.55)
    for yi, v in zip(y, vals):
        ax.annotate(f"{'-' if v < 0 else ''}${abs(v):,.1f}K", (v, yi), xytext=(6 if v >= 0 else -6, 0), textcoords="offset points",
                    va="center", ha="left" if v >= 0 else "right", color=INK, fontsize=9.5)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, names)
    ax.grid(axis="y", visible=False)
    lo, hi = min(vals.min(), 0), max(vals.max(), 0)
    ax.set_xlim(lo - (hi - lo) * 0.25, hi + (hi - lo) * 0.2)
    ax.set_xlabel("12-month contribution per 1,000 applicants ($K)")
    ax.set_title("Strategy comparison, out-of-time cohorts")
    _finish(fig, path)


def iv_chart(iv: dict, path):
    s = pd.Series(iv).sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.barh(s.index, s.values, color=SERIES[0], height=0.6)
    for i, v in enumerate(s.values):
        ax.annotate(f"{v:.2f}", (v, i), xytext=(4, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Information value")
    ax.set_title("Predictive strength of cash-flow features")
    _finish(fig, path)


def sensitivity_chart(sens: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(sens["membership_fee_month"], sens["optimal_approval_rate"] * 100, color=SERIES[0], marker="o", ms=5)
    for _, r in sens.iterrows():
        ax.annotate(f"{r['optimal_approval_rate']:.0%}", (r["membership_fee_month"], r["optimal_approval_rate"] * 100),
                    xytext=(0, 8), textcoords="offset points", ha="center", color=INK2, fontsize=9)
    ax.set_xticks(sens["membership_fee_month"], [f"${v:.2f}" for v in sens["membership_fee_month"]])
    ax.set_ylim(sens["optimal_approval_rate"].min() * 100 - 5, sens["optimal_approval_rate"].max() * 100 + 5)
    ax.set(xlabel="Monthly membership fee", ylabel="Profit-maximizing approval rate (%)")
    ax.set_title("Pricing moves the optimal credit cutoff")
    _finish(fig, path)
