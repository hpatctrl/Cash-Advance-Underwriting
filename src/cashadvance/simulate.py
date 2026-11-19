"""Simulate applicants, 90 days of bank activity, and first-advance outcomes.

What gets written to data/:
  members.csv          application-time attributes (what an underwriter sees)
  deposits.csv.gz      every deposit in the 90-day window (long format)
  balances.csv.gz      end-of-day balance, one row per member, 90 columns
  outcomes.csv         repayment of the first advance under the historical
                       test policy (everyone approved at a flat $100)
  _truth.csv           simulated ground truth used ONLY to evaluate alternative
                       limit strategies (counterfactuals). Never used to train.

The data-generating process ties a hidden "financial stress" factor to both
observable bank behaviour and repayment, plus noise the model cannot see, so
realistic models land in the 0.70-0.80 AUC range rather than near 1.0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PRODUCT, SIM, SimConfig


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def pd_at_amount(base_logit: np.ndarray, amount: np.ndarray | float, monthly_income: np.ndarray) -> np.ndarray:
    """True probability of charge-off for an advance of `amount`.

    Bigger advances relative to income are harder to repay from one paycheck.
    """
    return _sigmoid(base_logit + 4.0 * (np.asarray(amount) / monthly_income))


def simulate(cfg: SimConfig = SIM) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    n, H = cfg.n_members, cfg.history_days

    cohort = rng.integers(1, cfg.n_cohorts + 1, size=n)
    z = rng.normal(0, 1, n) + cfg.cohort_risk_drift * (cohort - 1)

    labels = np.array([p[0] for p in cfg.pay_mix])
    shares = np.array([p[1] for p in cfg.pay_mix])
    pay_freq = rng.choice(labels, size=n, p=shares)
    gig = pay_freq == "gig"

    monthly_income = cfg.median_monthly_income * np.exp(rng.normal(0, cfg.income_sigma, n) - 0.25 * z)
    monthly_income = np.clip(monthly_income, 700, 15_000)

    spend_ratio = np.clip(0.93 + 0.06 * z + rng.normal(0, 0.04, n), 0.75, 1.25)
    miss_prob = _sigmoid(-3.4 + 0.9 * z)
    account_age = np.clip(rng.exponential(40 * np.exp(-0.3 * z)), 1, 360).round().astype(int)
    fraud = rng.random(n) < np.where(account_age < 3, 0.06, 0.008)

    # ---- 90 days of deposits and balances ---------------------------------
    period = np.select(
        [pay_freq == "weekly", pay_freq == "biweekly", pay_freq == "semimonthly"], [7, 14, 15], default=0
    )
    phase = rng.integers(0, 14, n) % np.maximum(period, 1)
    paycheck = np.where(period > 0, monthly_income * np.maximum(period, 1) / 30.0, 0.0)

    balance = np.empty((n, H))
    bal = monthly_income * 0.25 * np.exp(rng.normal(0, 0.6, n) - 0.6 * z)
    rent_day = rng.integers(0, 30, n)
    dep_rows = []
    for d in range(H):
        # payroll
        is_payday = (period > 0) & ((d - phase) % np.maximum(period, 1) == 0) & (d >= phase)
        paid = is_payday & (rng.random(n) > miss_prob)
        amt = np.where(paid, paycheck * (1 + rng.normal(0, 0.05, n)), 0.0)
        # gig income: frequent irregular deposits
        gig_dep = gig & (rng.random(n) < 0.25)
        gig_amt = np.where(gig_dep, monthly_income / 7.5 * np.exp(rng.normal(-0.12, 0.5, n)), 0.0)
        # small P2P / other transfers for everyone
        p2p = rng.random(n) < 0.05
        p2p_amt = np.where(p2p, np.exp(rng.normal(3.5, 0.8, n)), 0.0)
        for kind, a in (("payroll", amt), ("gig", gig_amt), ("transfer", p2p_amt)):
            idx = np.nonzero(a > 0)[0]
            if idx.size:
                dep_rows.append(pd.DataFrame({"member_id": idx, "day": d, "amount": a[idx].round(2), "source": kind}))
        # spending: rent once a month + noisy daily spend
        rent = np.where(rent_day == d % 30, 0.35 * monthly_income, 0.0)
        daily = monthly_income * (spend_ratio - 0.35) / 30.0 * np.exp(rng.normal(-0.06, 0.35, n))
        bal = bal + amt + gig_amt + p2p_amt - rent - daily
        balance[:, d] = bal

    deposits = pd.concat(dep_rows, ignore_index=True).sort_values(["member_id", "day"]).reset_index(drop=True)

    # ---- requested amount and first-advance outcome -----------------------
    requested = rng.choice([50, 100, 150, 200, 250, 300, 500], size=n,
                           p=[0.10, 0.25, 0.15, 0.15, 0.12, 0.10, 0.13])
    base_logit = -3.35 + 1.25 * z + 0.45 * gig + rng.normal(0, 0.55, n)
    base_logit = np.where(fraud, 2.5, base_logit)

    hist_amount = 100.0  # historical test-and-learn policy: approve all at $100
    p_default = pd_at_amount(base_logit, hist_amount, monthly_income)
    defaulted = rng.random(n) < p_default
    late_prob = _sigmoid(-1.9 + 0.8 * z)
    late = (~defaulted) & (rng.random(n) < late_prob)
    days_late = np.where(late, np.clip(rng.gamma(1.6, 7, n), 1, 55).round(), 0).astype(int)
    days_late = np.where(defaulted, PRODUCT.chargeoff_day, days_late)

    members = pd.DataFrame({
        "member_id": np.arange(n),
        "cohort": cohort,
        "account_age_months": account_age,
        "requested_amount": requested,
    })
    balances = pd.DataFrame(balance.round(2), columns=[f"d{d:02d}" for d in range(H)])
    balances.insert(0, "member_id", np.arange(n))
    outcomes = pd.DataFrame({
        "member_id": np.arange(n),
        "advance_amount": hist_amount,
        "charged_off": defaulted.astype(int),
        "days_past_due": days_late,
    })
    truth = pd.DataFrame({
        "member_id": np.arange(n),
        "base_logit": base_logit,
        "monthly_income_true": monthly_income.round(2),
        "pay_freq_true": pay_freq,
        "fraud": fraud.astype(int),
        "z": z,
    })
    return {"members": members, "deposits": deposits, "balances": balances, "outcomes": outcomes, "_truth": truth}



def write(tables: dict[str, pd.DataFrame], out_dir: str = "data") -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    for name, df in tables.items():
        ext = ".csv.gz" if name in ("deposits", "balances") else ".csv"
        df.to_csv(f"{out_dir}/{name}{ext}", index=False)


if __name__ == "__main__":
    t = simulate()
    write(t)
    print({k: len(v) for k, v in t.items()})
