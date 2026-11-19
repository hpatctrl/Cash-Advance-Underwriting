"""Cash-flow underwriting features built from raw bank data.

Thin-file cash advance applicants often have little or no bureau history, so
underwriting runs on what the linked bank account shows: is there a recurring
paycheck, how regular is it, how close to zero does the balance run, and how
often does the account overdraw.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PAYROLL_MIN = 200.0  # deposits at or above this size are paycheck candidates

FEATURES = [
    "est_monthly_income",
    "paycheck_count",
    "paycheck_gap_cv",
    "detected_pay_period_days",
    "days_since_last_paycheck",
    "income_concentration",
    "avg_daily_balance_to_income",
    "min_balance_to_income",
    "pct_days_negative",
    "overdraft_events",
    "balance_volatility_to_income",
    "ending_balance_to_income",
    "balance_trend_to_income",
    "account_age_months",
    "requested_to_income",
]


def deposit_features(deposits: pd.DataFrame, history_days: int) -> pd.DataFrame:
    g = deposits.groupby("member_id")
    total = g["amount"].sum()
    big = deposits[deposits["amount"] >= PAYROLL_MIN]
    gb = big.groupby("member_id")

    def gap_stats(days: pd.Series) -> pd.Series:
        d = np.sort(days.to_numpy())
        gaps = np.diff(d)
        if gaps.size == 0:
            return pd.Series({"paycheck_gap_cv": 2.0, "detected_pay_period_days": 45.0})
        cv = gaps.std() / gaps.mean() if gaps.mean() > 0 else 2.0
        return pd.Series({"paycheck_gap_cv": min(cv, 2.0), "detected_pay_period_days": float(np.median(gaps))})

    gaps = gb["day"].apply(gap_stats).unstack()
    out = pd.DataFrame({
        "est_monthly_income": total / (history_days / 30.0),
        "paycheck_count": gb.size(),
        "days_since_last_paycheck": (history_days - 1) - gb["day"].max(),
        # share of income from the single largest recurring source size band
        "income_concentration": gb["amount"].sum() / total,
    })
    out = out.join(gaps, how="outer")
    return out


def balance_features(balances: pd.DataFrame, income: pd.Series) -> pd.DataFrame:
    b = balances.set_index("member_id").to_numpy()
    inc = income.reindex(balances["member_id"]).to_numpy()
    inc = np.maximum(np.nan_to_num(inc, nan=1.0), 1.0)
    neg = b < 0
    crossings = (neg[:, 1:] & ~neg[:, :-1]).sum(axis=1)
    t = np.arange(b.shape[1])
    slope = np.polyfit(t, b.T, 1)[0] * 30  # change per 30 days
    return pd.DataFrame({
        "avg_daily_balance_to_income": b.mean(axis=1) / inc,
        "min_balance_to_income": b.min(axis=1) / inc,
        "pct_days_negative": neg.mean(axis=1),
        "overdraft_events": crossings,
        "balance_volatility_to_income": b.std(axis=1) / inc,
        "ending_balance_to_income": b[:, -1] / inc,
        "balance_trend_to_income": slope / inc,
    }, index=balances["member_id"])


def build_features(members: pd.DataFrame, deposits: pd.DataFrame, balances: pd.DataFrame,
                   history_days: int = 90) -> pd.DataFrame:
    dep = deposit_features(deposits, history_days)
    dep = dep.reindex(members["member_id"])
    dep["est_monthly_income"] = dep["est_monthly_income"].fillna(0.0)
    dep["paycheck_count"] = dep["paycheck_count"].fillna(0)
    dep["days_since_last_paycheck"] = dep["days_since_last_paycheck"].fillna(history_days)
    dep["income_concentration"] = dep["income_concentration"].fillna(0.0)
    dep["paycheck_gap_cv"] = dep["paycheck_gap_cv"].fillna(2.0)
    dep["detected_pay_period_days"] = dep["detected_pay_period_days"].fillna(45.0)
    bal = balance_features(balances, dep["est_monthly_income"])
    feats = members.set_index("member_id")[["cohort", "account_age_months", "requested_amount"]].join(dep).join(bal)
    feats["requested_to_income"] = feats["requested_amount"] / np.maximum(feats["est_monthly_income"], 1.0)
    # cap ratio features at sensible bounds so a few extreme accounts don't dominate
    for c in ["avg_daily_balance_to_income", "min_balance_to_income", "balance_volatility_to_income",
              "ending_balance_to_income", "balance_trend_to_income", "requested_to_income"]:
        lo, hi = feats[c].quantile([0.005, 0.995])
        feats[c] = feats[c].clip(lo, hi)
    return feats.reset_index()
