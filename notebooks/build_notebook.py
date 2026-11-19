"""Builds and executes notebooks/cash_advance_underwriting.ipynb from source cells."""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Cash Advance Underwriting: cash-flow scoring, limits and unit economics\n\n"
       "A first-advance credit decision for a direct-to-consumer cash advance / earned-wage-access product, "
       "built the way a lender's credit team would build it: cash-flow features from linked bank data, an "
       "explainable champion scorecard against a machine-learning challenger, risk-based limits, loss curves, "
       "and a profit-maximizing cutoff.\n\n"
       "> **Data note.** No public loan-level data exists for cash advance products, so the data is simulated "
       "from stated assumptions (`src/cashadvance/config.py`, `docs/assumptions.md`). The methods transfer "
       "directly to real bank-transaction data."),
    code("import sys, json\nsys.path.insert(0, '../src')\nimport pandas as pd, numpy as np\n"
         "from cashadvance.simulate import simulate\nfrom cashadvance.features import build_features, FEATURES\n"
         "from cashadvance.models import train_and_evaluate, decile_table\n"
         "from cashadvance.strategy import compare_strategies, cutoff_curve\nfrom cashadvance.config import SIM, PRODUCT\n"
         "pd.set_option('display.width', 160); pd.set_option('display.max_columns', 20)"),
    md("## 1. Applicants and 90 days of bank activity\n\n"
       "30,000 applicants across 12 monthly cohorts. Historical policy was a test-and-learn cell that approved "
       "every applicant at a flat $100, so outcomes are observed for everyone (no reject-inference problem)."),
    code("t = simulate(SIM)\nprint({k: len(v) for k, v in t.items() if not k.startswith('_')})\n"
         "t['deposits'].head()"),
    md("## 2. Cash-flow features\n\nThin-file applicants are underwritten on what the bank account shows: "
       "recurring income, pay regularity, how close to zero the balance runs, and overdraft behaviour."),
    code("df = build_features(t['members'], t['deposits'], t['balances']).merge(t['outcomes'], on='member_id')\n"
         "df[FEATURES].describe().T[['mean', '50%', 'min', 'max']].round(2)"),
    md("## 3. Champion scorecard vs. XGBoost challenger\n\n"
       "Development cohorts 1-8 (75/25 train/validation); cohorts 9-12 held out as an out-of-time test. "
       "The XGBoost challenger uses monotonic constraints so every feature moves risk in a defensible direction."),
    code("out = train_and_evaluate(df)\nres = out['results']\n"
         "rows = []\nfor m in ['champion_scorecard', 'challenger_xgboost']:\n"
         "    for s in ['train', 'validation', 'out_of_time']:\n"
         "        rows.append({'model': m, 'sample': s, **res[m][s]})\n"
         "pd.DataFrame(rows)"),
    code("pd.Series(res['information_value'], name='IV').to_frame()"),
    md("![ROC](../reports/figures/roc_out_of_time.png)\n\n![Calibration](../reports/figures/calibration_deciles.png)"),
    md("## 4. Adverse action reasons\n\nEvery decline maps to plain-language principal reasons, ranked by points "
       "lost against the best bin, as ECOA / Regulation B adverse action notices require."),
    code("champ = out['champion']; tr, va, oot = out['splits']\n"
         "s = oot.head(6)\npd.DataFrame({'score': champ.score(s[FEATURES]).round(0),\n"
         "              'reasons': champ.reason_codes(s[FEATURES]).apply('; '.join)})"),
    md("## 5. Cutoff, limits and unit economics\n\nDecisions use the model; outcomes come from the simulated "
       "ground truth, the way a champion/challenger test cell would read out. Each approved member is followed "
       "for 12 months: membership fees, instant-delivery fees, funding cost, processing, servicing, losses and "
       "acquisition cost."),
    code("p = champ.predict_proba(oot[FEATURES])\ntruth = t['_truth'].set_index('member_id').loc[oot['member_id']].reset_index()\n"
         "cc = cutoff_curve(p, truth)\nbest = cc.loc[cc['contribution_per_1000_applicants'].idxmax()]\n"
         "cc[['approval_target', 'cutoff_pd', 'first_advance_co_rate', 'net_loss_rate_of_originations', "
         "'contribution_per_member', 'contribution_per_1000_applicants', 'payback_month']]"),
    md("![Cutoff](../reports/figures/cutoff_curve.png)"),
    code("compare_strategies(p, truth, float(best['cutoff_pd'])).set_index('strategy').T"),
    md("![Strategies](../reports/figures/strategy_comparison.png)\n\n"
       "![Pricing](../reports/figures/pricing_sensitivity.png)"),
    md("## 6. Loss curves\n\n![Vintage](../reports/figures/loss_curves_by_vintage.png)\n\n"
       "![Score bands](../reports/figures/loss_curves_by_score_band.png)\n\n"
       "Later vintages run slightly worse (built-in macro drift), and risk ranks cleanly across score bands. "
       "See `reports/credit_committee_memo.md` for the recommendation."),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": "notebooks"}})
nbf.write(nb, "notebooks/cash_advance_underwriting.ipynb")
print("notebook written")
