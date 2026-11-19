"""Credit strategy: approval cutoffs, risk-based limits, loss curves and unit economics.

Decisions use the MODEL's predicted risk. Outcomes are drawn from the simulated
ground truth, which lets us compare strategies the way a champion/challenger
test cell would in production, without waiting a year for performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PRODUCT, ProductConfig
from .simulate import pd_at_amount


# ---------------------------------------------------------------------------
# Limit policies
# ---------------------------------------------------------------------------

def flat_limits(pred_pd: np.ndarray, cutoff_pd: float | None, amount: float) -> np.ndarray:
    """Approve everyone below the PD cutoff at a flat amount (0 = declined)."""
    approve = np.ones_like(pred_pd, dtype=bool) if cutoff_pd is None else pred_pd <= cutoff_pd
    return np.where(approve, amount, 0.0)


def risk_based_limits(pred_pd: np.ndarray, cutoff_pd: float, cfg: ProductConfig = PRODUCT) -> np.ndarray:
    """Lower predicted risk earns a bigger starting limit; above the cutoff is declined.

    Approved members are split into equal-sized risk bands, one per limit tier.
    """
    lim = np.zeros_like(pred_pd)
    ok = pred_pd <= cutoff_pd
    if ok.sum() == 0:
        return lim
    grid = np.array(cfg.limit_grid, dtype=float)
    q = np.quantile(pred_pd[ok], np.linspace(0, 1, len(grid) + 1)[1:-1])
    band = np.searchsorted(q, pred_pd[ok])           # 0 = safest
    lim[ok] = grid[::-1][band]
    return lim


# ---------------------------------------------------------------------------
# Member lifetime simulation
# ---------------------------------------------------------------------------

def simulate_lifetime(start_limit: np.ndarray, base_logit: np.ndarray, income: np.ndarray,
                      cfg: ProductConfig = PRODUCT, step_up: bool = False, seed: int = 11,
                      advances_per_month: int = 2, usage_prob: float = 0.7,
                      seasoning_per_clean: float = 0.12, max_seasoning: float = 0.9) -> dict:
    """Monte Carlo of each approved member's next `horizon_months`.

    Assumptions (all in config or arguments): two pay cycles a month, 70% chance
    the member advances in a cycle, each clean repayment lowers true risk a little
    (seasoning) and, with `step_up`, raises the limit by `limit_step_up`.
    A charge-off ends the relationship.
    """
    rng = np.random.default_rng(seed)
    n = len(start_limit)
    approved = start_limit > 0
    limit = start_limit.astype(float).copy()
    active = approved.copy()
    clean = np.zeros(n)
    ledger = {k: np.zeros(n) for k in ("originated", "advances", "membership", "express", "funding",
                                       "processing", "servicing", "net_loss", "first_advance_co")}
    first_done = np.zeros(n, dtype=bool)
    monthly_contrib = np.zeros((cfg.horizon_months, n))
    for m in range(cfg.horizon_months):
        before = {k: v.copy() for k, v in ledger.items()}
        if m > 0:
            active &= rng.random(n) > cfg.monthly_churn
        ledger["membership"] += active * cfg.membership_fee_month
        ledger["servicing"] += active * cfg.servicing_cost_member_month
        for _ in range(advances_per_month):
            takes = active & (rng.random(n) < usage_prob)
            amt = np.where(takes, limit, 0.0)
            seasoning = np.minimum(clean * seasoning_per_clean, max_seasoning)
            p = pd_at_amount(base_logit - seasoning, amt, income)
            co = takes & (rng.random(n) < p)
            express = takes & (rng.random(n) < cfg.express_take_rate)
            fee = np.maximum(amt * cfg.express_fee_rate, cfg.express_fee_min)
            ledger["originated"] += amt
            ledger["advances"] += takes
            ledger["express"] += np.where(express, fee, 0.0)
            ledger["funding"] += amt * cfg.annual_funding_rate * cfg.avg_days_outstanding / 365
            ledger["processing"] += takes * cfg.processing_cost_per_advance
            ledger["net_loss"] += np.where(co, amt * (1 - cfg.post_chargeoff_recovery), 0.0)
            ledger["first_advance_co"] += takes & ~first_done & co
            first_done |= takes
            good = takes & ~co
            clean += good
            if step_up:
                limit = np.where(good, np.minimum(limit + cfg.limit_step_up, cfg.max_limit), limit)
            active &= ~co
        delta = {k: ledger[k] - before[k] for k in ledger}
        monthly_contrib[m] = (delta["membership"] + delta["express"] - delta["funding"] - delta["processing"]
                              - delta["servicing"] - delta["net_loss"])
    monthly_contrib[0] -= approved * cfg.cac_per_funded_member
    return {"ledger": ledger, "approved": approved, "monthly_contrib": monthly_contrib, "first_done": first_done}


def summarize(name: str, sim: dict, n_applicants: int, cfg: ProductConfig = PRODUCT) -> dict:
    L, a = sim["ledger"], sim["approved"]
    na = int(a.sum())
    revenue = L["membership"].sum() + L["express"].sum()
    costs = L["funding"].sum() + L["processing"].sum() + L["servicing"].sum()
    loss = L["net_loss"].sum()
    contrib = revenue - costs - loss - na * cfg.cac_per_funded_member
    cum = sim["monthly_contrib"][:, a].sum(axis=1).cumsum()
    payback = int(np.argmax(cum > 0)) + 1 if (cum > 0).any() else None
    return {
        "strategy": name,
        "approval_rate": round(na / n_applicants, 4),
        "first_advance_co_rate": round(L["first_advance_co"][a].sum() / max(sim["first_done"][a].sum(), 1), 4),
        "avg_advance": round(L["originated"].sum() / max(L["advances"].sum(), 1), 2),
        "originations_per_member": round(L["originated"].sum() / max(na, 1), 2),
        "net_loss_rate_of_originations": round(loss / max(L["originated"].sum(), 1), 4),
        "revenue_per_member": round(revenue / max(na, 1), 2),
        "loss_per_member": round(loss / max(na, 1), 2),
        "contribution_per_member": round(contrib / max(na, 1), 2),
        "contribution_per_1000_applicants": round(contrib / n_applicants * 1000, 0),
        "payback_month": payback,
    }


def compare_strategies(pred_pd: np.ndarray, truth: pd.DataFrame, cutoff_pd: float,
                       cfg: ProductConfig = PRODUCT) -> pd.DataFrame:
    bl, inc = truth["base_logit"].to_numpy(), truth["monthly_income_true"].to_numpy()
    n = len(pred_pd)
    runs = [
        ("A. Approve all, flat $100", flat_limits(pred_pd, None, cfg.flat_limit), False),
        ("B. Model cutoff, flat $100", flat_limits(pred_pd, cutoff_pd, cfg.flat_limit), False),
        ("C. Model cutoff, risk-based limits", risk_based_limits(pred_pd, cutoff_pd, cfg), False),
        ("D. Risk-based limits + increases after clean repayment", risk_based_limits(pred_pd, cutoff_pd, cfg), True),
    ]
    return pd.DataFrame([summarize(name, simulate_lifetime(lim, bl, inc, cfg, step_up=su), n) for name, lim, su in runs])


def cutoff_curve(pred_pd: np.ndarray, truth: pd.DataFrame, cfg: ProductConfig = PRODUCT,
                 approval_rates=np.arange(0.30, 1.001, 0.05)) -> pd.DataFrame:
    bl, inc = truth["base_logit"].to_numpy(), truth["monthly_income_true"].to_numpy()
    rows = []
    for ar in approval_rates:
        cut = np.quantile(pred_pd, min(ar, 1.0))
        sim = simulate_lifetime(risk_based_limits(pred_pd, cut, cfg), bl, inc, cfg, step_up=True)
        s = summarize(f"{ar:.0%}", sim, len(pred_pd))
        s["approval_target"], s["cutoff_pd"] = round(float(ar), 2), round(float(cut), 4)
        rows.append(s)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Loss curves from the historical first-advance book
# ---------------------------------------------------------------------------

def unpaid_curve(dpd: pd.Series, days=range(0, 61)) -> pd.Series:
    """Share of advance dollars still unpaid t days after the due date."""
    d = dpd.to_numpy()
    return pd.Series([(d > t).mean() if t < 60 else (d >= 60).mean() for t in days], index=list(days))


def loss_curves(df: pd.DataFrame, group: str) -> pd.DataFrame:
    return pd.DataFrame({g: unpaid_curve(sub["days_past_due"]) for g, sub in df.groupby(group, observed=True)})
