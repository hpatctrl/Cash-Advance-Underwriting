"""End-to-end run: simulate -> features -> models -> strategy -> reports.

    python run_pipeline.py            # ~1-2 minutes on a laptop
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from cashadvance import reporting as rp  # noqa: E402
from cashadvance.config import PRODUCT, SIM  # noqa: E402
from cashadvance.features import FEATURES, build_features  # noqa: E402
from cashadvance.models import decile_table, train_and_evaluate  # noqa: E402
from cashadvance.simulate import simulate, write  # noqa: E402
from cashadvance.strategy import compare_strategies, cutoff_curve, loss_curves  # noqa: E402

FIG = "reports/figures"


def main():
    os.makedirs(FIG, exist_ok=True)
    print("1/5 simulating applicants and bank data")
    t = simulate(SIM)
    write(t, "data")
    truth = t["_truth"].set_index("member_id")

    print("2/5 building cash-flow features")
    feats = build_features(t["members"], t["deposits"], t["balances"], SIM.history_days)
    df = feats.merge(t["outcomes"], on="member_id")

    print("3/5 training champion scorecard and XGBoost challenger")
    out = train_and_evaluate(df)
    res, champ, chall = out["results"], out["champion"], out["challenger"]
    tr, va, oot = out["splits"]
    p_champ = champ.predict_proba(oot[FEATURES])
    p_chall = chall.predict_proba(oot[FEATURES])[:, 1]
    rp.roc_chart(oot["charged_off"], {"Champion: WoE scorecard": p_champ, "Challenger: XGBoost": p_chall},
                 f"{FIG}/roc_out_of_time.png")
    cal = decile_table(oot["charged_off"], p_champ)
    rp.calibration_chart(cal, f"{FIG}/calibration_deciles.png")
    rp.iv_chart(res["information_value"], f"{FIG}/information_value.png")

    print("4/5 strategy: cutoffs, limits, economics")
    oot_truth = truth.loc[oot["member_id"]].reset_index()
    cc = cutoff_curve(p_champ, oot_truth, PRODUCT)
    best = cc.loc[cc["contribution_per_1000_applicants"].idxmax()]
    cutoff = float(best["cutoff_pd"])
    rp.cutoff_chart(cc, float(best["approval_target"]), f"{FIG}/cutoff_curve.png")
    sc = compare_strategies(p_champ, oot_truth, cutoff, PRODUCT)
    rp.strategy_chart(sc, f"{FIG}/strategy_comparison.png")

    # value of the challenger's extra lift, at the same approval rate
    cut_chall = float(np.quantile(p_chall, best["approval_target"]))
    sc_chall = compare_strategies(p_chall, oot_truth, cut_chall, PRODUCT)
    challenger_uplift = float(sc_chall.iloc[3]["contribution_per_1000_applicants"]
                              - sc.iloc[3]["contribution_per_1000_applicants"])

    sens = []
    for fee in (4.99, 5.99, 6.99, 7.99, 8.99):
        cfg = dataclasses.replace(PRODUCT, membership_fee_month=fee)
        c = cutoff_curve(p_champ, oot_truth, cfg, approval_rates=np.arange(0.20, 1.001, 0.05))
        b = c.loc[c["contribution_per_1000_applicants"].idxmax()]
        sens.append({"membership_fee_month": fee, "optimal_approval_rate": float(b["approval_target"]),
                     "contribution_per_1000_applicants": float(b["contribution_per_1000_applicants"])})
    sens = pd.DataFrame(sens)
    rp.sensitivity_chart(sens, f"{FIG}/pricing_sensitivity.png")

    print("5/5 loss curves and adverse-action samples")
    hist = df.copy()
    hist["vintage"] = "Q" + ((hist["cohort"] - 1) // 3 + 1).astype(str)
    rp.loss_curve_chart(loss_curves(hist, "vintage"), f"{FIG}/loss_curves_by_vintage.png",
                        "Unpaid dollars after due date, by application quarter", rp.SERIES, "Vintage {}")
    o = oot.copy()
    o["score_band"] = pd.qcut(p_champ, 5, labels=["1 safest", "2", "3", "4", "5 riskiest"])
    rp.loss_curve_chart(loss_curves(o, "score_band"), f"{FIG}/loss_curves_by_score_band.png",
                        "Unpaid dollars after due date, by score band", rp.BLUE_ORDINAL, "Band {}")

    declined = oot.loc[p_champ > cutoff].head(8)
    aa = pd.DataFrame({
        "member_id": declined["member_id"].to_numpy(),
        "score": champ.score(declined[FEATURES]).round(0),
        "predicted_pd": champ.predict_proba(declined[FEATURES]).round(4),
        "principal_reasons": champ.reason_codes(declined[FEATURES]).apply("; ".join).to_numpy(),
    })

    os.makedirs("reports", exist_ok=True)
    cc.to_csv("reports/cutoff_curve.csv", index=False)
    sc.to_csv("reports/strategy_comparison.csv", index=False)
    sens.to_csv("reports/pricing_sensitivity.csv", index=False)
    cal.to_csv("reports/calibration_deciles.csv")
    aa.to_csv("reports/sample_adverse_action_reasons.csv", index=False)
    res.update({
        "chosen_policy": {"approval_rate": float(best["approval_target"]), "cutoff_pd": round(cutoff, 4),
                          "cutoff_score": float(np.round(np.quantile(champ.score(oot[FEATURES]), 1 - best["approval_target"]), 0))},
        "strategy_comparison": sc.to_dict(orient="records"),
        "challenger_uplift_per_1000_applicants": round(challenger_uplift, 0),
        "pricing_sensitivity": sens.to_dict(orient="records"),
        "historical_first_advance_chargeoff_rate": round(float(df["charged_off"].mean()), 4),
    })
    with open("reports/results.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps({k: res[k] for k in ("chosen_policy", "challenger_uplift_per_1000_applicants")}, indent=2))


if __name__ == "__main__":
    main()
