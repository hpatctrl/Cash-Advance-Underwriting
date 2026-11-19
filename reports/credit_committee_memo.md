# Credit Committee Memo: First-Advance Underwriting Policy

**Product:** direct-to-consumer cash advance with monthly membership
**Decision requested:** approve the launch credit policy, limit strategy and challenger test plan
**Prepared by:** Hiren Patel
**Basis:** simulated 12-month test-and-learn book of 30,000 applicants. All figures are model outputs from stated assumptions, not observed production results.

---

## 1. Recommendation

1. **Adopt the WoE scorecard as the launch decision model, with a cutoff score of 558.** About 50% of applicants are approved.
2. **Set risk-based starting limits** of $25 / $50 / $100 / $150 / $250 across five equal risk bands of approved applicants, safest band highest.
3. **Raise the limit by $25 after each on-time repayment**, up to $500. The limit stays flat after any late payment, and a charge-off closes the line.
4. **Run the XGBoost model as a randomized challenger** on 10–20% of applications for two quarters. Promote it only if the out-of-time lift holds and adverse-action reasons can be produced to the same standard.
5. **Keep a small random approve-below-cutoff cell (about 5%)** so the model can be recalibrated and re-developed without a reject-inference problem.

## 2. Why: expected performance

Out-of-time cohorts (months 9–12), 12-month member view:

| Strategy | Approval | First-advance charge-off | Net loss, % of originations | Contribution per approved member | Contribution per 1,000 applicants |
|---|---|---|---|---|---|
| A. Approve all, flat $100 (status quo test) | 100% | 10.4% | 4.0% | –$3.23 | –$3.2K |
| B. Cutoff 558, flat $100 | 50% | 4.7% | 2.0% | $18.82 | $9.4K |
| C. Cutoff 558, risk-based limits | 50% | 4.6% | 1.4% | $30.91 | $15.5K |
| **D. C + increases after clean repayment (recommended)** | **50%** | **4.6%** | **1.7%** | **$36.91** | **$18.5K** |

Acquisition cost is paid back in month 5 under the recommended policy. The approve-all book never pays back.

The net loss rate rises from C to D (1.4% to 1.7%) because dollar exposure more than doubles. Contribution still rises, because step-ups go only to members who have already repaid. Each clean repayment is assumed to lower risk modestly (see Assumptions), so this is the assumption to verify first in production.

## 3. Choosing the cutoff

Under the recommended limit strategy, contribution per 1,000 applicants peaks at $18.5K between 50% and 55% approval, then falls to roughly breakeven at 100% approval. Every applicant approved beyond about 55% loses money on the margin: their losses outweigh the fees they generate. The net loss rate rises steadily with approval, from 1.2% at 30% approval to 3.0% at 100%.

The optimal cutoff depends on pricing. At a $4.99 membership the best approval rate is 40%; at $7.99 it is 55%. Any fee change must come back to this committee together with a new cutoff.

## 4. Model

| | Champion: WoE scorecard | Challenger: monotonic XGBoost |
|---|---|---|
| Out-of-time AUC / KS / Gini | 0.705 / 0.31 / 0.41 | 0.715 / 0.33 / 0.43 |
| Validation AUC | 0.716 | 0.727 |
| Train AUC | 0.727 | 0.775 |
| PSI, development vs. out-of-time scores | 0.004 | 0.005 |
| Explainability | Points by attribute; direct reason codes | Needs SHAP-style attribution |

- **Strongest attributes (Information Value):** balance relative to income (ending, average and minimum), share of days overdrawn, estimated monthly income, and balance trend.
- **Risk ranking:** risk ranks cleanly by score band. 60 days after the due date, 3.2% of dollars remain unpaid in the safest band against 22.5% in the riskiest.
- **Challenger:** at the same approval rate, XGBoost's extra lift is worth about $1.4K per 1,000 applicants. That's material, but not enough to give up the scorecard's transparency before production evidence exists. The gap between its train AUC (0.775) and validation AUC (0.727) is a further reason to prove it in a controlled cell.

## 5. Monitoring and triggers

| Metric | Frequency | Trigger |
|---|---|---|
| First-advance charge-off rate by score band | Weekly (early-read proxy: 15 days past due) | Any band above 1.25× expected for two weeks |
| Score PSI against the development sample | Monthly | Above 0.10 review; above 0.25 redevelop |
| Calibration (predicted vs. actual) | Monthly | Portfolio gap above 1 point; recalibrate intercept |
| Vintage curves (unpaid dollars by days since due date) | Monthly | New vintage above prior quarter at day 30 |
| Step-up performance (loss rate after the 3rd+ increase) | Monthly | Above the first-advance rate of the same band |

The out-of-time charge-off rate (10.9%) ran about 1 point above development (9.9%) because of the built-in macro drift. Rank ordering held (PSI 0.004), but the model under-predicts slightly in the middle deciles. A quarterly intercept recalibration is part of the plan.

## 6. Regulatory and compliance considerations

- **Adverse action:** every decline produces up to four principal reasons in plain language, such as "Account frequently overdrawn" or "Irregular income deposits," ranked by points lost. Sample: `reports/sample_adverse_action_reasons.csv`.
- **Product classification:** EWA and cash advance rules are changing at the state level, and federal treatment has shifted. Legal must confirm, state by state, whether the product is treated as credit, and whether the express fee or any tip counts as a finance charge, before this policy goes live.
- **Fair lending:** real data must be tested for disparate impact, for example with BISG proxies for race and ethnicity and a review of whether attributes like account age act as proxies. The simulated data can't support this test.
- **Model risk management:** documentation, independent validation and ongoing monitoring follow SR 11-7-style practice. This memo, the notebook and the monitoring plan are the core of that package.

## 7. Limitations

- All data is simulated. The structure of the results (which features matter, the shape of the profit curve, the value of risk-based limits) is realistic. The exact levels depend on the assumptions in `docs/assumptions.md`.
- Outcomes under strategies C and D come from a simulated "true" risk function at different advance amounts. In production these are learned only through test cells.
- Fraud is modeled as a small block of near-certain charge-offs. A production stack needs a separate fraud and identity layer in front of credit.
- The economics leave out tips, interchange and cross-sell revenue, so the contribution figures understate them.
