"""Single source of truth for every simulation and business assumption.

Nothing in this project uses real customer data. All parameters below are
stated assumptions chosen to look like a direct-to-consumer cash advance /
earned-wage-access (EWA) product. Change them here and re-run the pipeline.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SimConfig:
    seed: int = 7
    n_members: int = 30_000            # applicants across the whole window
    n_cohorts: int = 12                # monthly application cohorts (vintages)
    history_days: int = 90             # bank history pulled at application
    # pay-frequency mix of applicants: (label, share, days between paychecks)
    pay_mix: tuple = (
        ("weekly", 0.20, 7),
        ("biweekly", 0.50, 14),
        ("semimonthly", 0.15, 15),
        ("gig", 0.15, None),           # irregular deposits
    )
    median_monthly_income: float = 2_600.0
    income_sigma: float = 0.45         # lognormal spread
    # macro drift: later cohorts are slightly riskier (tests model stability)
    cohort_risk_drift: float = 0.015   # added to latent risk per cohort


@dataclass(frozen=True)
class ProductConfig:
    # revenue
    membership_fee_month: float = 6.99     # subscription
    express_fee_rate: float = 0.05         # instant-delivery fee, % of advance
    express_fee_min: float = 3.99
    express_take_rate: float = 0.60        # share of advances sent instantly
    # costs
    annual_funding_rate: float = 0.12      # cost of capital on outstanding balance
    avg_days_outstanding: float = 9.0      # paid back on next payday
    processing_cost_per_advance: float = 0.75
    servicing_cost_member_month: float = 0.50
    cac_per_funded_member: float = 25.00
    post_chargeoff_recovery: float = 0.10  # share of charged-off principal recovered
    chargeoff_day: int = 60                # days past due when a missed advance is written off
    # limits
    flat_limit: float = 100.0
    limit_grid: tuple = (25, 50, 100, 150, 250)
    max_limit: float = 500.0
    limit_step_up: float = 25.0            # increase after each clean repayment
    horizon_months: int = 12               # member lifetime simulated for economics
    monthly_churn: float = 0.08


SIM = SimConfig()
PRODUCT = ProductConfig()
