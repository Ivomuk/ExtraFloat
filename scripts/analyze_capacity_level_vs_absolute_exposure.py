"""
analyze_capacity_level_vs_absolute_exposure.py
=================================================
Analysis 2, refined: the exposure-intensity grid
(analyze_capacity_exposure_performance_matrix.py) can be fooled by a
ratio effect -- within a fixed capacity band, moving to a higher
exposure-INTENSITY band can mean the DENOMINATOR (capacity) fell while
absolute exposure stayed flat, not that the agent was extended more
credit. Confirmed in the first real run: <5% PD, high float-activity,
median actual exposure was a constant UGX 750K across all three EI
bands while median float activity fell from 62.2M to 41.4M to 33.6M.

This script replaces the EI axis with ABSOLUTE exposure instead:

    PD band  x  capacity band (within PD band)  x  absolute exposure
    (the agent's actual disbursed amount, grouped by its own discrete
    value if actual_exposure_ugx takes a small number of distinct
    values -- confirmed to be the case: loan product sizes, not a
    continuum)
    -> forward performance

answering the more direct, actionable question: for an agent with a
given level of float activity / commission at a given PD, what
happens to performance as ACTUAL exposure itself increases from one
real loan size to the next?

Restricted to FLOAT ACTIVITY and COMMISSION only, per this session's
Analysis-2 findings: customer reach showed material deterioration at
higher PD and average_balance's exposure/balance ratio was judged not
economically meaningful as a monetary capacity denominator -- both
retained as supporting/diagnostic signals, neither as a primary
capacity anchor warranting this finer grid yet.

exposure_intensity is still reported per cell, for reference only --
never as the grouping axis here, precisely to avoid re-introducing the
ratio effect this script exists to correct.

Usage:
    python scripts\\analyze_capacity_level_vs_absolute_exposure.py ^
        --research-dataset capacity_research_dataset.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PD_BAND_EDGES = [0.0, 0.05, 0.15, 0.30, 1.0]
PD_BAND_LABELS = ["<5%", "5-15%", "15-30%", "30%+"]

CAPACITY_VARIABLES = {
    "float_activity_value_1m": "Float activity (cash-in + payment, 1m)",
    "commission": "Earnings (commission)",
}
N_CAPACITY_BANDS = 3
MIN_CELL_N = 10
MAX_DISCRETE_EXPOSURE_VALUES = 15  # above this, fall back to quantile exposure bands


def _qcut_safe(s: pd.Series, q: int, labels_hint: list[str]) -> tuple[pd.Series, int]:
    """qcut with duplicate bin edges dropped; returns (string-labeled band
    series, n bins actually produced -- may be < q)."""
    try:
        codes, bins = pd.qcut(s, q, duplicates="drop", retbins=True, labels=False)
    except ValueError:
        return pd.Series(np.nan, index=s.index, dtype=object), 0
    n_bins = len(bins) - 1
    if n_bins <= 0:
        return pd.Series(np.nan, index=s.index, dtype=object), 0
    labels = labels_hint[:n_bins] if n_bins <= len(labels_hint) else [f"Q{i + 1}" for i in range(n_bins)]
    return codes.map(dict(enumerate(labels))), n_bins


def build_matrix(df: pd.DataFrame, capacity_col: str, label: str) -> pd.DataFrame:
    working = df[
        df["cal_pd"].notna() & df[capacity_col].notna() & (df[capacity_col] > 0)
        & df["actual_exposure_ugx"].notna() & (df["actual_exposure_ugx"] > 0)
    ].copy()
    if working.empty:
        print(f"  (no agents with valid cal_pd, {capacity_col}, and actual_exposure_ugx -- skipped)")
        return pd.DataFrame()

    working["_pd_band"] = pd.cut(working["cal_pd"], bins=PD_BAND_EDGES, labels=PD_BAND_LABELS)
    working["_ei_reference_only"] = working["actual_exposure_ugx"] / working[capacity_col]
    if "fwd_new_loan_count" in working.columns:
        working["_took_new_loan"] = working["fwd_new_loan_count"] > 0
    else:
        working["_took_new_loan"] = False

    exposure_vals = sorted(working["actual_exposure_ugx"].dropna().unique())
    is_discrete = len(exposure_vals) <= MAX_DISCRETE_EXPOSURE_VALUES
    print(f"  Exposure buckets: {'discrete loan-size values ' + str(exposure_vals) if is_discrete else f'{len(exposure_vals)} distinct values -- too many, using quantile bands instead'}")

    cap_labels_hint = ["Low", "Medium", "High"]

    rows = []
    for pd_band in PD_BAND_LABELS:
        band_df = working[working["_pd_band"] == pd_band]
        if len(band_df) < MIN_CELL_N:
            if len(band_df):
                rows.append({"pd_band": pd_band, "capacity_band": "(all)", "exposure_bucket": "(all)",
                             "n_agents": len(band_df), "_low_n_warning": True})
            continue
        cap_band, n_cap_bins = _qcut_safe(band_df[capacity_col], N_CAPACITY_BANDS, cap_labels_hint)
        if n_cap_bins == 0:
            continue
        band_df = band_df.assign(_cap_band=cap_band)

        if is_discrete:
            band_df = band_df.assign(_exposure_bucket=band_df["actual_exposure_ugx"])
        else:
            exp_band, n_exp_bins = _qcut_safe(band_df["actual_exposure_ugx"], 5,
                                               ["Lowest", "Low", "Mid", "High", "Highest"])
            if n_exp_bins == 0:
                continue
            band_df = band_df.assign(_exposure_bucket=exp_band)

        for (cb, eb), cell in band_df.groupby(["_cap_band", "_exposure_bucket"], observed=True):
            n = len(cell)
            row = {
                "pd_band": pd_band, "capacity_band": cb, "exposure_bucket": eb,
                "n_agents": n, "_low_n_warning": n < MIN_CELL_N,
                f"median_{capacity_col}": cell[capacity_col].median(),
                "median_actual_exposure_ugx": cell["actual_exposure_ugx"].median(),
                "median_exposure_intensity_reference_only": cell["_ei_reference_only"].median(),
                "median_combined_cap_benchmark": cell["combined_cap"].median() if "combined_cap" in cell.columns else np.nan,
                "median_cash_out_value_1m_diagnostic": cell["cash_out_value_1m"].median() if "cash_out_value_1m" in cell.columns else np.nan,
                "pct_took_new_loan": round(cell["_took_new_loan"].mean() * 100, 1),
            }
            borrowers = cell[cell["_took_new_loan"]]
            if "fwd_any_bad_3dpd" in cell.columns:
                row["fwd_any_bad_3dpd_rate_among_borrowers_pct"] = (
                    round(borrowers["fwd_any_bad_3dpd"].mean() * 100, 1) if len(borrowers) else float("nan")
                )
            if {"fwd_new_loans_closed_good_count", "fwd_new_loans_closed_bad_count"} <= set(cell.columns):
                n_good = int(cell["fwd_new_loans_closed_good_count"].sum())
                n_bad = int(cell["fwd_new_loans_closed_bad_count"].sum())
                n_closed = n_good + n_bad
                row["n_closed_new_loans"] = n_closed
                row["bad_closure_rate_pct"] = round(n_bad / n_closed * 100, 1) if n_closed else float("nan")
            if {"fwd_new_loans_disbursed_ugx", "fwd_new_loans_repaid_ugx"} <= set(cell.columns):
                disbursed = cell["fwd_new_loans_disbursed_ugx"].sum()
                repaid = cell["fwd_new_loans_repaid_ugx"].sum()
                row["shortfall_pct_of_forward_disbursed"] = (
                    round((disbursed - repaid) / disbursed * 100, 2) if disbursed else float("nan")
                )
            rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    cat_order = pd.CategoricalDtype(PD_BAND_LABELS, ordered=True)
    result["pd_band"] = result["pd_band"].astype(cat_order)
    sort_cols = ["pd_band", "capacity_band"] + (["exposure_bucket"] if is_discrete else [])
    try:
        result = result.sort_values(sort_cols).reset_index(drop=True)
    except TypeError:
        result = result.sort_values(["pd_band", "capacity_band"]).reset_index(drop=True)
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--research-dataset", default="capacity_research_dataset.csv")
    ap.add_argument("--diamond-high-risk-cal-pd", type=float, default=0.30)
    ap.add_argument("--out-prefix", default="capacity_level_absolute_exposure")
    args = ap.parse_args(argv)

    path = Path(args.research_dataset)
    if not path.exists():
        sys.exit(f"ERROR: {path} not found -- run build_capacity_research_dataset.py first.")
    df = pd.read_csv(path, low_memory=False)

    missing_cap_vars = [c for c in CAPACITY_VARIABLES if c not in df.columns]
    if missing_cap_vars:
        print(f"NOTE: capacity variable(s) not found, skipped: {missing_cap_vars}")
    required_other = ["cal_pd", "actual_exposure_ugx"]
    missing_other = [c for c in required_other if c not in df.columns]
    if missing_other:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing_other}")

    def run_population(pop_df: pd.DataFrame, pop_label: str, out_suffix: str) -> None:
        print(f"\n{'#' * 100}")
        print(f"# Population: {pop_label}  (n={len(pop_df):,})")
        print(f"{'#' * 100}")
        for col, label in CAPACITY_VARIABLES.items():
            if col not in pop_df.columns:
                continue
            print(f"\n{'=' * 100}")
            print(f"{label}  [{col}]")
            print("=" * 100)
            matrix = build_matrix(pop_df, col, label)
            if matrix.empty:
                continue
            with pd.option_context("display.float_format", "{:,.2f}".format, "display.max_columns", None, "display.width", 240):
                print(matrix.to_string(index=False))
            out_path = f"{args.out_prefix}_{col}_{out_suffix}.csv"
            matrix.to_csv(out_path, index=False)
            print(f"  -> written: {out_path}")
            n_low_n = int(matrix.get("_low_n_warning", pd.Series(dtype=bool)).sum())
            if n_low_n:
                print(f"  NOTE: {n_low_n} cell(s)/row(s) have n_agents < {MIN_CELL_N} -- treat with caution.")

    run_population(df, "All agents", "all")

    diamond_mask = (
        df["agent_category"].astype(str).str.strip().str.lower() == "diamond"
        if "agent_category" in df.columns else pd.Series(False, index=df.index)
    )
    low_risk_mask = df["cal_pd"] < args.diamond_high_risk_cal_pd
    diamond_df = df[diamond_mask & low_risk_mask]
    run_population(
        diamond_df,
        f"Diamond, cal_pd < {args.diamond_high_risk_cal_pd * 100:.0f}% (the 'Diamond A' population)",
        "diamond_below_high_risk",
    )

    print(f"\n{'#' * 100}")
    print("What this does and does not establish")
    print(f"{'#' * 100}")
    print("Within a fixed pd_band and capacity_band, comparing across exposure_bucket answers: for an\n"
          "agent with THIS level of float activity / commission at THIS risk, what happens as actual\n"
          "exposure itself rises from one real loan size to the next? That is a cleaner question than the\n"
          "exposure-intensity grid's, which can be driven by the denominator moving rather than the loan\n"
          "size. median_exposure_intensity_reference_only is reported for context only -- it is never the\n"
          "grouping axis here. As before: no capacity formula is computed or implied by this script.")


if __name__ == "__main__":
    main()
