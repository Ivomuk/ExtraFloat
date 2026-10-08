"""
build_capacity_research_dataset.py
=====================================
The data-construction step for the capacity-challenger workstream: one
row per agent, joining the engine's own capacity benchmark against raw,
point-in-time business-activity variables across five independent
economic dimensions (float activity, customer reach, earnings,
liquidity, business persistence), plus actual exposure and forward
outcomes -- everything the next two analyses (redundancy/correlation,
and exposure-intensity-vs-performance) need, built once rather than
re-joined per analysis.

THIS SCRIPT DOES NOT BUILD A CAPACITY MODEL. It only constructs the
dataset. See the module docstring's "What this feeds" note at the
bottom for the planned follow-on analysis script.

FIVE DIMENSIONS (per this session's design discussion, deliberately
narrower than transaction_capacity_features.py's raw schema to avoid
structural redundancy):
  1. Float activity   -- FU_value_1m/3m = cash_in_value + payment_value
                          (confirmed identical to borrower_persona_clustering.py's
                          own "float utilization" definition -- NOT cash_in,
                          payment, AND float-utilization as three separate
                          signals, which would triple-count the same activity).
  2. Customer reach    -- cust_1m/3m/6m (genuine customer COUNTS, confirmed
                          distinct from transaction volume by
                          scripts/check_customers_served_90d.py).
  3. Earnings          -- commission ONLY. Deliberately excludes the
                          engine-derived revenue_1m/3m/6m: the raw mart's
                          OWN revenue_*/rev_* columns are 100% dead/missing
                          (scripts/check_mart_column_completeness.py), while
                          the engine computes a DIFFERENTLY-DEFINED column
                          under the SAME name (extrafloat_limit_engine_features.py:532-533)
                          -- a collision risk this script avoids by simply
                          not using revenue_* at all for now.
  4. Liquidity         -- account_balance, average_balance, LEVELS ONLY.
                          avg_daily_balance_30d/90d are NOT used -- confirmed
                          identical by construction (extrafloat_limit_engine_features.py:544-545,
                          "single snapshot"), so any 30d-vs-90d ratio built
                          from them would contain zero longitudinal information.
  5. Business persistence -- NOT available anywhere in the limit engine's
                          own feature pipeline (everything there named
                          "stability"/"volatility"/"trend" is LOAN REPAYMENT
                          stability, a different concept). Computed fresh
                          here using the IDENTICAL method
                          pd_model/preprocessing/transaction_features.py
                          already uses for volume (lines 180-222: monthly
                          volatility proxy/CV from the 1m/3m/6m cumulative
                          windows; lines 90-97: trend flags; lines 58-65,
                          233-238: dormancy/inactivity flags) -- but applied
                          to the VALUE columns too, since only a volume
                          version exists upstream. That value-side version
                          is a NEW computation using an established method,
                          not a literal pre-existing column -- labelled
                          _derived_here in its column names so this is
                          never mistaken for inherited precedent.

CASH-OUT is retained as a diagnostic column (not folded into float
activity, not treated as a sixth capacity dimension yet) per this
session's decision to let it earn its way in through empirical
validation rather than assuming it belongs.

PAYMENT DOUBLE-WEIGHTING DIAGNOSTIC: the production capacity_cap already
contains payment_value_1m twice -- once directly (capacity_payments_component)
and once inside total_txn_value_1m -> capacity_volume_component
(confirmed: extrafloat_limit_engine_features.py:537-539,
extrafloat_limit_engine_caps.py:505-512,540-542). This script persists
both components plus an estimated `payment_double_count_excess_ugx`
(the portion of capacity_volume_component proportionally attributable
to payment_value_1m, i.e. the part of capacity_payments_component's
signal that is ALSO re-counted inside capacity_volume_component) --
NOT a claim this is a bug (may be intentional policy weighting), just a
quantified diagnostic, as agreed before changing anything live.

Usage:
    python scripts\\build_capacity_research_dataset.py ^
        --engine-output output\\engine_test_output.csv ^
        --transaction-file retail_agents_filtered.csv ^
        --monthly-summary-file monthly_disbursement_summary.csv ^
        --forward-outcomes-file data\\persona_k8_forward_outcomes.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation.borrower_persona_clustering import digits  # noqa: E402

EPS = 1e-9

# Raw mart columns this script needs, organized by what they feed. Anything
# missing degrades gracefully (NOTE, column left NaN) -- never a crash.
RAW_NUMERIC_COLS = [
    "commission", "account_balance", "average_balance",
    "cash_in_value_1m", "cash_in_value_3m", "cash_in_value_6m",
    "cash_in_vol_1m", "cash_in_vol_3m", "cash_in_vol_6m",
    "payment_value_1m", "payment_value_3m", "payment_value_6m",
    "payment_vol_1m", "payment_vol_3m", "payment_vol_6m",
    "cash_out_value_1m", "cash_out_value_3m", "cash_out_value_6m",
    "cash_out_vol_1m", "cash_out_vol_3m", "cash_out_vol_6m",
    "cust_1m", "cust_3m", "cust_6m",
    "vol_1m", "vol_3m", "vol_6m",
]
RAW_DATE_COLS = ["payment_last", "cash_in_last", "cash_out_last"]

ENGINE_COLS = [
    "msisdn", "cal_pd", "assigned_limit", "agent_category", "risk_tier", "combined_cap",
    "capacity_raw", "capacity_structural", "capacity_effective_ceiling",
    "capacity_balance_component", "capacity_revenue_component", "capacity_txn_component",
    "capacity_payments_component", "capacity_customers_component", "capacity_volume_component",
    "capacity_top_driver",
]
ENGINE_REQUIRED = ["msisdn", "cal_pd", "assigned_limit", "combined_cap"]


def _monthly_volatility(df: pd.DataFrame, col_1m: str, col_3m: str, col_6m: str, out_prefix: str) -> None:
    """Identical method to transaction_features.py:211-222, generalized to any
    base column triplet (there, only volume; here, also applied to value).
    """
    if not all(c in df.columns for c in (col_1m, col_3m, col_6m)):
        return
    m1 = df[col_1m]
    m2 = (df[col_3m] - df[col_1m]) / 2.0
    m3 = (df[col_6m] - df[col_3m]) / 3.0
    monthly = np.vstack([m1.to_numpy(dtype=float), m2.to_numpy(dtype=float), m3.to_numpy(dtype=float)]).T
    mean_monthly = monthly.mean(axis=1)
    std_monthly = monthly.std(axis=1)
    df[f"{out_prefix}_monthly_volatility_proxy_derived_here"] = std_monthly
    df[f"{out_prefix}_monthly_volatility_cv_derived_here"] = std_monthly / (mean_monthly + EPS)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine-output", default="output/engine_test_output.csv")
    ap.add_argument("--transaction-file", default="retail_agents_filtered.csv",
                     help="raw point-in-time mart extract, retail-filtered -- same file "
                          "run_retail_filtered.bat/.sh feeds to run_credit_risk_pipeline.py "
                          "as --transaction-file")
    ap.add_argument("--monthly-summary-file", default="monthly_disbursement_summary.csv",
                     help="optional -- built by build_monthly_disbursement_summary.py; "
                          "skipped with a NOTE if not found")
    ap.add_argument("--forward-outcomes-file", default="data/persona_k8_forward_outcomes.csv",
                     help="optional -- skipped with a NOTE if not found")
    ap.add_argument("--out", default="capacity_research_dataset.csv")
    args = ap.parse_args(argv)

    eng_path = Path(args.engine_output)
    txn_path = Path(args.transaction_file)
    if not eng_path.exists():
        sys.exit(f"ERROR: {eng_path} not found.")
    if not txn_path.exists():
        sys.exit(f"ERROR: {txn_path} not found.")

    eng = pd.read_csv(eng_path, low_memory=False)
    missing_required = [c for c in ENGINE_REQUIRED if c not in eng.columns]
    if missing_required:
        sys.exit(f"ERROR: {eng_path} is missing required column(s) {missing_required} -- "
                  f"re-run the pipeline with --keep-intermediate.")
    missing_optional_engine = [c for c in ENGINE_COLS if c not in eng.columns]
    if missing_optional_engine:
        print(f"NOTE: {eng_path.name} is missing optional engine column(s) (needs "
              f"--keep-intermediate): {missing_optional_engine}")

    eng = eng.copy()
    eng["_id"] = digits(eng["msisdn"])
    eng_cols_present = [c for c in ENGINE_COLS if c in eng.columns and c != "msisdn"]
    research = eng[["_id"] + eng_cols_present].drop_duplicates(subset="_id").copy()
    print(f"Engine output: {len(research):,} unique agent(s).")

    # -- Raw mart: point-in-time business-activity variables -----------------
    txn = pd.read_csv(txn_path, low_memory=False)
    msisdn_col = "agent_msisdn" if "agent_msisdn" in txn.columns else "msisdn"
    if msisdn_col not in txn.columns:
        sys.exit(f"ERROR: {txn_path} has neither 'agent_msisdn' nor 'msisdn'. "
                  f"Columns present: {list(txn.columns)}")
    txn = txn.copy()
    txn["_id"] = digits(txn[msisdn_col])

    missing_raw = [c for c in RAW_NUMERIC_COLS if c not in txn.columns]
    if missing_raw:
        print(f"NOTE: {txn_path.name} is missing raw column(s): {missing_raw} -- the "
              f"corresponding research-dataset column(s) will be NaN.")
    missing_dates = [c for c in RAW_DATE_COLS if c not in txn.columns]
    if missing_dates:
        print(f"NOTE: {txn_path.name} is missing recency date column(s): {missing_dates} -- "
              f"business-persistence recency features will be skipped for these.")

    for c in RAW_NUMERIC_COLS:
        if c in txn.columns:
            txn[c] = pd.to_numeric(txn[c], errors="coerce")

    # Point-in-time: one row per agent, the MOST RECENT snapshot_dt available
    # in this extract (the raw mart can carry multiple dated rows per agent).
    if "snapshot_dt" in txn.columns:
        txn["snapshot_dt"] = pd.to_datetime(txn["snapshot_dt"], errors="coerce")
        txn = txn.sort_values("snapshot_dt").drop_duplicates(subset="_id", keep="last")
    else:
        print(f"NOTE: {txn_path.name} has no snapshot_dt -- cannot confirm point-in-time "
              f"de-duplication; keeping the first row per agent as-is.")
        txn = txn.drop_duplicates(subset="_id", keep="first")
    print(f"Raw transaction file: {len(txn):,} unique agent(s) after point-in-time de-duplication.")

    # -- Dimension 1: Float activity (cash-in + payment; NOT a 3rd signal) ---
    if {"cash_in_value_1m", "payment_value_1m"} <= set(txn.columns):
        txn["float_activity_value_1m"] = txn["cash_in_value_1m"].fillna(0) + txn["payment_value_1m"].fillna(0)
    if {"cash_in_value_3m", "payment_value_3m"} <= set(txn.columns):
        txn["float_activity_value_3m"] = txn["cash_in_value_3m"].fillna(0) + txn["payment_value_3m"].fillna(0)
    if {"cash_in_vol_1m", "payment_vol_1m"} <= set(txn.columns):
        txn["float_activity_vol_1m"] = txn["cash_in_vol_1m"].fillna(0) + txn["payment_vol_1m"].fillna(0)
    if {"cash_in_vol_3m", "payment_vol_3m"} <= set(txn.columns):
        txn["float_activity_vol_3m"] = txn["cash_in_vol_3m"].fillna(0) + txn["payment_vol_3m"].fillna(0)
    if {"float_activity_value_1m", "float_activity_value_3m"} <= set(txn.columns):
        txn["float_activity_1m_vs_3m_ratio"] = txn["float_activity_value_1m"] / (
            txn["float_activity_value_3m"] / 3.0 + EPS
        )

    # -- Dimension 2: Customer reach -----------------------------------------
    if {"cust_1m", "cust_3m"} <= set(txn.columns):
        txn["customer_momentum_1m_vs_3m"] = txn["cust_1m"] / (txn["cust_3m"] / 3.0 + EPS)
    if {"float_activity_value_1m", "cust_1m"} <= set(txn.columns):
        txn["float_activity_per_customer_1m"] = txn["float_activity_value_1m"] / (txn["cust_1m"] + EPS)

    # -- Dimension 5: Business persistence (derived here -- see docstring) --
    _monthly_volatility(txn, "cash_in_value_1m", "cash_in_value_3m", "cash_in_value_6m", "cash_in_value")
    _monthly_volatility(txn, "payment_value_1m", "payment_value_3m", "payment_value_6m", "payment_value")
    _monthly_volatility(txn, "vol_1m", "vol_3m", "vol_6m", "vol")  # identical to upstream precedent
    if all(c in txn.columns for c in ("vol_1m", "vol_3m", "vol_6m")):
        txn["is_fully_inactive_6m"] = (
            (txn["vol_1m"].fillna(0) == 0) & (txn["vol_3m"].fillna(0) == 0) & (txn["vol_6m"].fillna(0) == 0)
        ).astype(int)
        txn["consistent_volume_decline_flag"] = (
            (txn["vol_1m"] < txn["vol_3m"] / 3.0) & (txn["vol_3m"] < txn["vol_6m"] / 2.0)
        ).astype(int)
        txn["consistent_volume_growth_flag"] = (
            (txn["vol_1m"] > txn["vol_3m"] / 3.0) & (txn["vol_3m"] > txn["vol_6m"] / 2.0)
        ).astype(int)

    if not missing_dates:
        ref_date = txn["snapshot_dt"].max() if "snapshot_dt" in txn.columns else pd.Timestamp.now()
        for c in RAW_DATE_COLS:
            txn[c] = pd.to_datetime(txn[c], errors="coerce")
            txn[f"days_since_{c}"] = (ref_date - txn[c]).dt.days

    # -- Payment double-weighting diagnostic (needs total_txn_value_1m) -----
    if {"cash_out_value_1m", "cash_in_value_1m", "payment_value_1m"} <= set(txn.columns):
        txn["total_txn_value_1m"] = (
            txn["cash_out_value_1m"].fillna(0) + txn["cash_in_value_1m"].fillna(0) + txn["payment_value_1m"].fillna(0)
        )
        txn["_payment_share_of_total_txn_1m"] = txn["payment_value_1m"].fillna(0) / (txn["total_txn_value_1m"] + EPS)

    derived_cols = [
        "float_activity_value_1m", "float_activity_value_3m", "float_activity_vol_1m",
        "float_activity_vol_3m", "float_activity_1m_vs_3m_ratio",
        "customer_momentum_1m_vs_3m", "float_activity_per_customer_1m",
        "cash_in_value_monthly_volatility_proxy_derived_here", "cash_in_value_monthly_volatility_cv_derived_here",
        "payment_value_monthly_volatility_proxy_derived_here", "payment_value_monthly_volatility_cv_derived_here",
        "vol_monthly_volatility_proxy_derived_here", "vol_monthly_volatility_cv_derived_here",
        "is_fully_inactive_6m", "consistent_volume_decline_flag", "consistent_volume_growth_flag",
        "days_since_payment_last", "days_since_cash_in_last", "days_since_cash_out_last",
        "total_txn_value_1m", "_payment_share_of_total_txn_1m",
    ]
    keep_raw_cols = [c for c in RAW_NUMERIC_COLS + derived_cols if c in txn.columns]
    keep_raw_cols = list(dict.fromkeys(keep_raw_cols))
    research = research.merge(txn[["_id"] + keep_raw_cols], on="_id", how="left")
    n_matched_txn = research[keep_raw_cols[0]].notna().sum() if keep_raw_cols else 0
    print(f"Matched {n_matched_txn:,} / {len(research):,} agents to raw transaction data.\n")

    # -- Payment double-weighting diagnostic, in dollar terms ----------------
    if {"capacity_payments_component", "capacity_volume_component", "_payment_share_of_total_txn_1m"} <= set(research.columns):
        research["payment_double_count_excess_ugx"] = (
            research["capacity_volume_component"] * research["_payment_share_of_total_txn_1m"]
        )
        research["payment_contribution_current_ugx"] = (
            research["capacity_payments_component"].fillna(0) + research["payment_double_count_excess_ugx"].fillna(0)
        )
        research["payment_contribution_if_counted_once_ugx"] = research["capacity_payments_component"]

    # -- Actual exposure (disbursed amount) -----------------------------------
    monthly_path = Path(args.monthly_summary_file)
    if monthly_path.exists():
        monthly = pd.read_csv(monthly_path)
        required_monthly = {"msisdn", "month", "disbursed_amount"}
        if required_monthly <= set(monthly.columns):
            monthly = monthly.copy()
            monthly["_id"] = digits(monthly["msisdn"])
            monthly["month"] = pd.to_datetime(monthly["month"], errors="coerce")
            n_multi_month = int(monthly.groupby("_id")["month"].nunique().gt(1).sum())
            if n_multi_month:
                print(f"NOTE: {n_multi_month:,} agent(s) have disbursement records in more than one "
                      f"month -- using each agent's MOST RECENT month's disbursed_amount as "
                      f"'actual_exposure_ugx' (point-in-time, not a sum across months).")
            latest = monthly.sort_values("month").drop_duplicates(subset="_id", keep="last")
            research = research.merge(
                latest[["_id", "disbursed_amount"]].rename(columns={"disbursed_amount": "actual_exposure_ugx"}),
                on="_id", how="left",
            )
            n_matched_disb = int(research["actual_exposure_ugx"].notna().sum())
            print(f"Matched {n_matched_disb:,} / {len(research):,} agents to an actual disbursed amount.\n")
        else:
            print(f"NOTE: {monthly_path} is missing {required_monthly - set(monthly.columns)} -- "
                  f"skipping actual-exposure enrichment.\n")
    else:
        print(f"NOTE: {monthly_path} not found -- skipping actual-exposure enrichment "
              f"(pass --monthly-summary-file, or run build_monthly_disbursement_summary.py first).\n")

    if {"actual_exposure_ugx", "combined_cap"} <= set(research.columns):
        research["exposure_intensity_vs_combined_cap"] = research["actual_exposure_ugx"] / (
            research["combined_cap"].replace(0, np.nan)
        )

    # -- Forward outcomes ------------------------------------------------------
    fwd_path = Path(args.forward_outcomes_file)
    if fwd_path.exists():
        fwd = pd.read_csv(fwd_path)
        if "customer_msisdn" in fwd.columns:
            fwd = fwd.copy()
            fwd["_id"] = digits(fwd["customer_msisdn"])
            fwd_cols = [c for c in fwd.columns if c.startswith("fwd_")]
            research = research.merge(fwd[["_id"] + fwd_cols], on="_id", how="left")
            n_matched_fwd = int(research[fwd_cols[0]].notna().sum()) if fwd_cols else 0
            print(f"Matched {n_matched_fwd:,} / {len(research):,} agents to forward-outcomes data.\n")
        else:
            print(f"NOTE: {fwd_path} has no 'customer_msisdn' column -- skipping forward-outcomes join.\n")
    else:
        print(f"NOTE: {fwd_path} not found -- skipping forward-outcomes join.\n")

    research = research.drop(columns=["_payment_share_of_total_txn_1m"], errors="ignore")
    research = research.rename(columns={"_id": "msisdn"})
    research.to_csv(args.out, index=False)
    print(f"Capacity research dataset written: {args.out}  ({len(research):,} rows, "
          f"{len(research.columns)} columns)")


if __name__ == "__main__":
    main()
