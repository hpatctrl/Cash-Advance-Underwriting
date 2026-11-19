"""Champion (WoE scorecard) vs challenger (monotonic XGBoost) for first-advance charge-off."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from .features import FEATURES

TARGET = "charged_off"

# expected direction of each feature vs. charge-off risk (+1 riskier as it rises)
MONOTONE = {
    "est_monthly_income": -1, "paycheck_count": 0, "paycheck_gap_cv": 1, "detected_pay_period_days": 0,
    "days_since_last_paycheck": 0, "income_concentration": 0, "avg_daily_balance_to_income": -1,
    "min_balance_to_income": -1, "pct_days_negative": 1, "overdraft_events": 1,
    "balance_volatility_to_income": 0, "ending_balance_to_income": -1, "balance_trend_to_income": -1,
    "account_age_months": -1, "requested_to_income": 1,
}

# plain-language reasons for adverse action notices (ECOA / Reg B style)
REASONS = {
    "est_monthly_income": "Income deposits too low",
    "paycheck_count": "Too few recurring deposits",
    "paycheck_gap_cv": "Irregular income deposits",
    "detected_pay_period_days": "Pay frequency could not be established",
    "days_since_last_paycheck": "No recent income deposit",
    "income_concentration": "Income not from a recurring source",
    "avg_daily_balance_to_income": "Low account balance relative to income",
    "min_balance_to_income": "Low account balance relative to income",
    "pct_days_negative": "Account frequently overdrawn",
    "overdraft_events": "Number of overdraft events",
    "balance_volatility_to_income": "Unstable account balance",
    "ending_balance_to_income": "Low account balance relative to income",
    "balance_trend_to_income": "Declining account balance",
    "account_age_months": "Bank account too new",
    "requested_to_income": "Requested amount too high relative to income",
}


def ks_stat(y, p) -> float:
    order = np.argsort(p)
    y = np.asarray(y)[order]
    cum_bad = np.cumsum(y) / max(y.sum(), 1)
    cum_good = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def metrics(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {"auc": round(auc, 4), "gini": round(2 * auc - 1, 4), "ks": round(ks_stat(y, p), 4),
            "bad_rate": round(float(np.mean(y)), 4), "n": int(len(y))}


def psi(expected, actual, bins=10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


class WoEScorecard:
    """Quantile-binned Weight-of-Evidence logistic scorecard with points scaling.

    Score = 600 at 50:1 good:bad odds, 20 points to double the odds.
    """

    def __init__(self, n_bins=6, base_score=600, base_odds=50, pdo=20):
        self.n_bins, self.base_score, self.base_odds, self.pdo = n_bins, base_score, base_odds, pdo

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.bins_, self.woe_, self.iv_ = {}, {}, {}
        good_tot, bad_tot = (y == 0).sum(), (y == 1).sum()
        for c in X.columns:
            edges = np.unique(np.quantile(X[c], np.linspace(0, 1, self.n_bins + 1)))
            edges[0], edges[-1] = -np.inf, np.inf
            b = pd.cut(X[c], edges, include_lowest=True, labels=False)
            tab = pd.crosstab(b, y).reindex(columns=[0, 1], fill_value=0) + 0.5
            dist_good, dist_bad = tab[0] / good_tot, tab[1] / bad_tot
            woe = np.log(dist_good / dist_bad)
            self.bins_[c], self.woe_[c] = edges, woe.to_dict()
            self.iv_[c] = float(((dist_good - dist_bad) * woe).sum())
        W = self.transform(X)
        self.lr_ = LogisticRegression(C=1.0, max_iter=2000).fit(W, y)
        self.factor_ = self.pdo / np.log(2)
        self.offset_ = self.base_score - self.factor_ * np.log(self.base_odds)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in X.columns:
            b = pd.cut(X[c], self.bins_[c], include_lowest=True, labels=False)
            out[c] = b.map(self.woe_[c]).fillna(0.0).astype(float)
        return pd.DataFrame(out, index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr_.predict_proba(self.transform(X))[:, 1]

    def points(self, X: pd.DataFrame) -> pd.DataFrame:
        """Points per feature. Higher = safer. Sum + intercept points = score."""
        W = self.transform(X)
        coef = pd.Series(self.lr_.coef_[0], index=W.columns)
        n = len(W.columns)
        # log-odds of being GOOD = -(intercept + sum(coef * woe))
        pts = -(W * coef) * self.factor_ + (self.offset_ - self.lr_.intercept_[0] * self.factor_) / n
        return pts

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.points(X).sum(axis=1).to_numpy()

    def reason_codes(self, X: pd.DataFrame, k=4) -> pd.Series:
        """Top-k features where the applicant lost the most points vs. the best bin."""
        pts = self.points(X)
        coef = pd.Series(self.lr_.coef_[0], index=pts.columns)
        best = {c: max(-(w * coef[c]) for w in self.woe_[c].values()) * self.factor_ for c in pts.columns}
        n = len(pts.columns)
        const = (self.offset_ - self.lr_.intercept_[0] * self.factor_) / n
        lost = pd.DataFrame({c: best[c] + const - pts[c] for c in pts.columns}, index=pts.index)
        def top(r):
            seen = []
            for c in r.sort_values(ascending=False).index:
                if r[c] <= 0.5:
                    break
                if REASONS[c] not in seen:
                    seen.append(REASONS[c])
                if len(seen) == k:
                    break
            return seen
        return lost.apply(top, axis=1)


def fit_xgb(X: pd.DataFrame, y: pd.Series, seed=7) -> XGBClassifier:
    mono = tuple(MONOTONE[c] for c in X.columns)
    model = XGBClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=20, reg_lambda=2.0, monotone_constraints=mono, eval_metric="auc",
        random_state=seed, n_jobs=4,
    )
    return model.fit(X, y)


def split(df: pd.DataFrame, dev_cohorts=range(1, 9), seed=7):
    dev = df[df["cohort"].isin(list(dev_cohorts))]
    oot = df[~df["cohort"].isin(list(dev_cohorts))]
    rng = np.random.default_rng(seed)
    mask = rng.random(len(dev)) < 0.75
    return dev[mask], dev[~mask], oot


def decile_table(y, p) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["decile"] = pd.qcut(d["p"].rank(method="first"), 10, labels=range(1, 11))
    t = d.groupby("decile", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    return t.round(4)


def train_and_evaluate(df: pd.DataFrame) -> dict:
    tr, va, oot = split(df)
    X, y = tr[FEATURES], tr[TARGET]
    champ = WoEScorecard().fit(X, y)
    chall = fit_xgb(X, y)
    res = {"split": {"train": len(tr), "validation": len(va), "out_of_time": len(oot)}}
    for name, m in (("champion_scorecard", champ), ("challenger_xgboost", chall)):
        pv = m.predict_proba(va[FEATURES]) if name.startswith("champ") else m.predict_proba(va[FEATURES])[:, 1]
        po = m.predict_proba(oot[FEATURES]) if name.startswith("champ") else m.predict_proba(oot[FEATURES])[:, 1]
        pt = m.predict_proba(X) if name.startswith("champ") else m.predict_proba(X)[:, 1]
        res[name] = {"train": metrics(y, pt), "validation": metrics(va[TARGET], pv), "out_of_time": metrics(oot[TARGET], po),
                     "psi_dev_vs_oot": round(psi(pv, po), 4)}
    res["information_value"] = dict(sorted(((k, round(v, 3)) for k, v in champ.iv_.items()), key=lambda kv: -kv[1]))
    res["xgb_importance"] = dict(sorted(((k, round(float(v), 4)) for k, v in zip(FEATURES, chall.feature_importances_)),
                                        key=lambda kv: -kv[1]))
    return {"results": res, "champion": champ, "challenger": chall, "splits": (tr, va, oot)}
