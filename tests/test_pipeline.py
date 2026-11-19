import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cashadvance.config import PRODUCT, SIM  # noqa: E402
from cashadvance.features import FEATURES, build_features  # noqa: E402
from cashadvance.models import WoEScorecard, ks_stat, psi  # noqa: E402
from cashadvance.simulate import simulate  # noqa: E402
from cashadvance.strategy import risk_based_limits  # noqa: E402

SMALL = dataclasses.replace(SIM, n_members=2_000)


def _data():
    t = simulate(SMALL)
    f = build_features(t["members"], t["deposits"], t["balances"]).merge(t["outcomes"], on="member_id")
    return t, f


def test_simulation_is_reproducible():
    a, b = simulate(SMALL), simulate(SMALL)
    assert a["outcomes"]["charged_off"].equals(b["outcomes"]["charged_off"])


def test_features_complete_and_finite():
    _, f = _data()
    assert len(f) == SMALL.n_members
    assert np.isfinite(f[FEATURES].to_numpy()).all()
    assert f["pct_days_negative"].between(0, 1).all()


def test_overdrawn_accounts_are_riskier():
    _, f = _data()
    hi = f[f["pct_days_negative"] > 0.3]["charged_off"].mean()
    lo = f[f["pct_days_negative"] == 0]["charged_off"].mean()
    assert hi > lo


def test_scorecard_ranks_risk_and_scores_are_monotone_in_pd():
    _, f = _data()
    sc = WoEScorecard().fit(f[FEATURES], f["charged_off"])
    p, s = sc.predict_proba(f[FEATURES]), sc.score(f[FEATURES])
    assert ks_stat(f["charged_off"], p) > 0.2
    # score is linear in log-odds, so it must rank exactly opposite to PD
    assert np.corrcoef(np.argsort(np.argsort(p)), np.argsort(np.argsort(s)))[0, 1] < -0.999


def test_psi_zero_on_identical_samples():
    x = np.random.default_rng(0).normal(size=1000)
    assert psi(x, x) < 1e-9


def test_risk_based_limits_decline_above_cutoff_and_favor_safest():
    p = np.linspace(0.01, 0.30, 100)
    lim = risk_based_limits(p, cutoff_pd=0.15, cfg=PRODUCT)
    assert (lim[p > 0.15] == 0).all()
    assert lim[0] == max(PRODUCT.limit_grid)
    assert lim[p <= 0.15].min() == min(PRODUCT.limit_grid)
