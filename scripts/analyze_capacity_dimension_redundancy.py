"""
analyze_capacity_dimension_redundancy.py
===========================================
Analysis 1 of 3 in the capacity-challenger workstream (per this session's
agreed sequencing: redundancy/correlation -> capacity validity within PD
bands -> exposure-intensity-vs-performance -- only THEN a formula).

Reads capacity_research_dataset.csv (built by
build_capacity_research_dataset.py) and computes pairwise correlations
across the candidate variables in the five capacity dimensions (float
activity, customer reach, earnings, liquidity, business persistence),
plus the engine's own benchmark/diagnostic columns -- to see which
variables are essentially measuring the same economic phenomenon
before any of them go into a formula.

WHY BOTH RAW AND LOG CORRELATION: these are heavily right-skewed
monetary amounts. A raw Pearson correlation can be dominated by a
handful of extreme-value agents; log1p-transformed correlation gives a
cleaner read of the monotonic relationship across the bulk of the
population. Both are reported -- never just one -- since they can
legitimately disagree (e.g. a pair driven apart only by a few giant
outliers vs. a pair that moves together throughout the distribution).

THREE KNOWN, HYPOTHESIZED REDUNDANCIES ARE CHECKED EXPLICITLY, not left
to be found by chance in a big matrix:
  1. float_activity_value_1m vs. cash_in_value_1m / payment_value_1m
     -- float_activity IS their sum by construction; expect very high r.
  2. capacity_payments_component vs. capacity_volume_component
     -- production's payment_value_1m double-weighting diagnostic;
     expect a non-trivial positive correlation if the double-count is
     material.
  3. commission vs. capacity_revenue_component
     -- the earnings-family candidate vs. the engine's own revenue-based
     component; checks whether using commission alone loses information
     capacity_revenue_component already captures, or whether they
     diverge meaningfully.

This script does NOT decide which variables to keep -- it only reports
the correlation structure so that decision is made from evidence.

Usage:
    python scripts\\analyze_capacity_dimension_redundancy.py ^
        --research-dataset capacity_research_dataset.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -- Candidate variables per dimension (matches build_capacity_research_dataset.py) --
DIMENSION_COLUMNS = {
    "float_activity": [
        "float_activity_value_1m", "float_activity_value_3m",
        "float_activity_vol_1m", "float_activity_vol_3m", "float_activity_1m_vs_3m_ratio",
    ],
    "customer_reach": [
        "cust_1m", "cust_3m", "cust_6m", "customer_momentum_1m_vs_3m", "float_activity_per_customer_1m",
    ],
    "earnings": ["commission"],
    "liquidity": ["account_balance", "average_balance"],
    "business_persistence": [
        "cash_in_value_monthly_volatility_cv_derived_here", "payment_value_monthly_volatility_cv_derived_here",
        "vol_monthly_volatility_cv_derived_here", "is_fully_inactive_6m",
        "consistent_volume_decline_flag", "consistent_volume_growth_flag",
        "days_since_payment_last", "days_since_cash_in_last", "days_since_cash_out_last",
    ],
    "diagnostic_not_a_dimension": [
        "cash_out_value_1m", "cash_out_value_3m",
    ],
    "engine_benchmark": [
        "combined_cap", "capacity_raw", "capacity_structural", "capacity_effective_ceiling",
        "capacity_balance_component", "capacity_revenue_component", "capacity_txn_component",
        "capacity_payments_component", "capacity_customers_component", "capacity_volume_component",
        "payment_contribution_current_ugx", "payment_contribution_if_counted_once_ugx",
    ],
}

# Monetary/count columns where a log1p transform is meaningful (strictly non-negative,
# right-skewed). Ratios, flags, and already-bounded measures are excluded.
LOG_ELIGIBLE_COLUMNS = {
    "float_activity_value_1m", "float_activity_value_3m", "float_activity_vol_1m", "float_activity_vol_3m",
    "cash_in_value_1m", "payment_value_1m",  # not dimension candidates themselves, but referenced by
                                              # KNOWN_REDUNDANCY_CHECKS and equally monetary/skewed.
    "cust_1m", "cust_3m", "cust_6m", "float_activity_per_customer_1m",
    "commission", "account_balance", "average_balance",
    "days_since_payment_last", "days_since_cash_in_last", "days_since_cash_out_last",
    "cash_out_value_1m", "cash_out_value_3m",
    "combined_cap", "capacity_raw", "capacity_structural", "capacity_effective_ceiling",
    "capacity_balance_component", "capacity_revenue_component", "capacity_txn_component",
    "capacity_payments_component", "capacity_customers_component", "capacity_volume_component",
    "payment_contribution_current_ugx", "payment_contribution_if_counted_once_ugx",
}

KNOWN_REDUNDANCY_CHECKS = [
    ("float_activity_value_1m", "cash_in_value_1m",
     "float_activity IS cash_in + payment by construction -- expect very high r"),
    ("float_activity_value_1m", "payment_value_1m",
     "float_activity IS cash_in + payment by construction -- expect very high r"),
    ("capacity_payments_component", "capacity_volume_component",
     "production's confirmed payment_value_1m double-weighting -- how material is it?"),
    ("commission", "capacity_revenue_component",
     "earnings-family candidate vs. the engine's own revenue-based component"),
]


def _log_corr(df: pd.DataFrame, col_a: str, col_b: str) -> float:
    a = pd.to_numeric(df[col_a], errors="coerce")
    b = pd.to_numeric(df[col_b], errors="coerce")
    valid = a.notna() & b.notna()
    if valid.sum() < 3:
        return float("nan")
    log_a = np.log1p(a[valid].clip(lower=0))
    log_b = np.log1p(b[valid].clip(lower=0))
    if log_a.std() == 0 or log_b.std() == 0:
        return float("nan")
    return float(np.corrcoef(log_a, log_b)[0, 1])


def _raw_corr(df: pd.DataFrame, col_a: str, col_b: str) -> float:
    a = pd.to_numeric(df[col_a], errors="coerce")
    b = pd.to_numeric(df[col_b], errors="coerce")
    valid = a.notna() & b.notna()
    if valid.sum() < 3 or a[valid].std() == 0 or b[valid].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--research-dataset", default="capacity_research_dataset.csv")
    ap.add_argument("--high-correlation-threshold", type=float, default=0.8,
                     help="|r| at or above this is flagged as a likely redundant pair")
    ap.add_argument("--out", default="capacity_dimension_redundancy.csv")
    args = ap.parse_args(argv)

    path = Path(args.research_dataset)
    if not path.exists():
        sys.exit(f"ERROR: {path} not found -- run build_capacity_research_dataset.py first.")

    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} agent(s) from {path}.\n")

    all_candidate_cols = [c for cols in DIMENSION_COLUMNS.values() for c in cols]
    present_cols = [c for c in all_candidate_cols if c in df.columns]
    missing_cols = [c for c in all_candidate_cols if c not in df.columns]
    if missing_cols:
        print(f"NOTE: column(s) not found in {path.name}, excluded from this analysis: {missing_cols}\n")

    col_to_dimension = {c: dim for dim, cols in DIMENSION_COLUMNS.items() for c in cols}

    rows = []
    n = len(present_cols)
    for i in range(n):
        for j in range(i + 1, n):
            col_a, col_b = present_cols[i], present_cols[j]
            raw_r = _raw_corr(df, col_a, col_b)
            log_r = (
                _log_corr(df, col_a, col_b)
                if col_a in LOG_ELIGIBLE_COLUMNS and col_b in LOG_ELIGIBLE_COLUMNS
                else float("nan")
            )
            n_valid = int((pd.to_numeric(df[col_a], errors="coerce").notna()
                           & pd.to_numeric(df[col_b], errors="coerce").notna()).sum())
            rows.append({
                "dimension_a": col_to_dimension[col_a], "variable_a": col_a,
                "dimension_b": col_to_dimension[col_b], "variable_b": col_b,
                "same_dimension": col_to_dimension[col_a] == col_to_dimension[col_b],
                "n_valid_pairs": n_valid,
                "pearson_r_raw": raw_r,
                "pearson_r_log1p": log_r,
            })

    pairs = pd.DataFrame(rows)
    pairs.to_csv(args.out, index=False)
    print(f"Full pairwise correlation table written: {args.out}  ({len(pairs):,} pairs)\n")

    print("=" * 100)
    print(f"Known, hypothesized redundancy checks")
    print("=" * 100)
    for col_a, col_b, note in KNOWN_REDUNDANCY_CHECKS:
        if col_a not in df.columns or col_b not in df.columns:
            print(f"  {col_a} vs {col_b}: SKIPPED -- column(s) not present")
            continue
        raw_r = _raw_corr(df, col_a, col_b)
        log_r = _log_corr(df, col_a, col_b) if col_a in LOG_ELIGIBLE_COLUMNS and col_b in LOG_ELIGIBLE_COLUMNS else float("nan")
        print(f"  {col_a}  vs.  {col_b}")
        print(f"    raw r = {raw_r:.3f}   log1p r = {log_r:.3f}   ({note})")

    print(f"\n{'=' * 100}")
    print(f"High-correlation pairs (|r| >= {args.high_correlation_threshold}, raw OR log1p)")
    print("=" * 100)
    flagged = pairs[
        (pairs["pearson_r_raw"].abs() >= args.high_correlation_threshold)
        | (pairs["pearson_r_log1p"].abs() >= args.high_correlation_threshold)
    ].copy()
    flagged["_max_abs_r"] = flagged[["pearson_r_raw", "pearson_r_log1p"]].abs().max(axis=1)
    flagged = flagged.sort_values("_max_abs_r", ascending=False).drop(columns=["_max_abs_r"])
    if flagged.empty:
        print("  (none)")
    else:
        with pd.option_context("display.float_format", "{:.3f}".format, "display.max_columns", None, "display.width", 200):
            print(flagged.drop(columns=["n_valid_pairs"]).to_string(index=False))

    n_same_dim_flagged = int((flagged["same_dimension"]).sum())
    n_cross_dim_flagged = int((~flagged["same_dimension"]).sum())
    print(f"\n{len(flagged)} pair(s) flagged: {n_same_dim_flagged} within the same dimension "
          f"(expected/acceptable -- these are alternative candidates for the SAME dimension), "
          f"{n_cross_dim_flagged} ACROSS different dimensions (worth scrutiny -- two supposedly "
          f"independent dimensions that move together this strongly may not be independent in "
          f"this population).")

    print(f"\n{'=' * 100}")
    print("What this does and does not establish")
    print("=" * 100)
    print("A high correlation between two variables means they carry similar INFORMATION in this\n"
          "population -- it does not by itself say which one (if either) belongs in a capacity\n"
          "formula. That question is Analysis 2 (capacity validity within PD bands) and Analysis 3\n"
          "(exposure-intensity-vs-performance), not this one. This script only tells you where\n"
          "redundancy exists so those later analyses aren't double-counting the same signal.")


if __name__ == "__main__":
    main()
