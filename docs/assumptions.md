# Assumptions

No public loan-level data exists for cash advance or EWA products, so this project simulates it. All values below are set in `src/cashadvance/config.py` and `src/cashadvance/simulate.py`. They are modeling assumptions, not claims about any company.

## Applicants

- **Population:** 30,000 applicants across 12 monthly cohorts. Later cohorts carry slightly higher latent risk (macro drift), so the model faces a realistic out-of-time test.
- **Pay frequency:** 20% weekly, 50% biweekly, 15% semimonthly, 15% gig (irregular deposits).
- **Monthly income:** lognormal around a $2,600 median, lower for higher-stress members, bounded at $700–$15,000.
- **Hidden financial stress factor:** drives spending relative to income, the chance of a missed paycheck, starting balance, bank account age, and repayment. A separate unobserved noise term means no model can be perfect.
- **Fraud:** about 1% of applicants, more common with new bank accounts. These are near-certain charge-offs.

## Bank data (90 days before application)

- **Deposits:** payroll on schedule (each paycheck can be missed), frequent irregular gig deposits, and occasional small transfers.
- **Spending:** rent at 35% of income once a month, plus noisy daily spending.
- **Balances:** end-of-day balance for every day of the window.

## Repayment

- **Historical policy:** every applicant was approved at $100 (a test-and-learn cell), so outcomes are observed for all applicants.
- **Charge-off risk:** a logistic function of the stress factor, gig income, fraud and noise. Bigger advances relative to income are riskier: +4.0 on the log-odds per 100% of monthly income.
- **Late payment:** members who don't charge off may repay 1–55 days late; late payment is more likely for higher-stress members.
- **Charge-off timing:** unpaid advances are written off at 60 days past due, and 10% of charged-off principal is recovered later.

## Lifetime economics (12 months)

| Item | Assumption |
|---|---|
| Membership fee | $6.99 a month while active |
| Instant-delivery fee | 5% of the advance, minimum $3.99; 60% of advances |
| Funding cost | 12% a year for an average of 9 days outstanding |
| Processing | $0.75 per advance |
| Servicing | $0.50 per active member per month |
| Acquisition cost | $25 per approved member |
| Usage | 2 pay cycles a month; 70% chance of an advance in each cycle |
| Churn | 8% a month; a charge-off ends the relationship |
| Seasoning | each on-time repayment lowers true log-odds by 0.12, up to 0.9 |
| Limits | flat $100, or $25–$250 by risk band; +$25 per clean repayment up to $500 |

Tips, interchange and cross-sell revenue are excluded.
