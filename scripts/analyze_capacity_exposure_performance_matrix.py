"""
analyze_capacity_exposure_performance_matrix.py
===================================================
Analysis 2 of 3 in the capacity-challenger workstream: NOT a model --
a matrix. Tests whether, holding credit risk approximately constant,
stronger business-capacity fundamentals are associated with agents
supporting larger actual exposure WITHOUT materially worse subsequent
repayment performance.

Central relationship tested, per capacity variable X:

    PD band  x  capacity band (within PD band)  x  exposure-intensity
    band (EI_X = actual_exposure_ugx / X, within PD band)
    -> forward performance

This deliberately does NOT compute a capacity decile -> bad-rate table
on its own (that would miss whether EXPOSURE relative to capacity is
sustainable -- a large business with modest exposure tells us nothing
about how much MORE exposure it could safely support). Exposure
intensity is the point of this analysis.

FOUR capacity variables are tested independently (never averaged into
one score yet -- that is explicitly deferred to a later stage once a
stable exposure-performance frontier is actually found):
  - float_activity_value_1m   (float activity)
  - commission                (earnings)
  - cust_1m                   (customer/activity scale -- chosen over
    the transaction-count-based capacity_txn_component/vol_1m, since
    the redundancy analysis + code inspection already established
    those two carry near-identical information in this population;
    cust_1m is the more conceptually distinct "business breadth"
    signal, not another volume/value measure like float activity).
  - average_balance            (liquidity)

PERSISTENCE is NOT given its own exposure-intensity ratio -- "dollars
of exposure per day of recency" isn't economically meaningful the way
"dollars of exposure per shilling of commission" is. Instead,
days_since_any_activity (= min of days_since_payment_last/cash_in_last/
cash_out_last -- whichever channel was most recently active) is
reported as a per-cell QUALIFIER column, alongside combined_cap (the
current-engine BENCHMARK, explicitly not a capacity feature here) and
cash_out_value_1m (DIAGNOSTIC only, per this session's decision not to
fold cash-out into any capacity dimension yet).

WHAT THIS DOES NOT DO: compute, fit, or imply a capacity formula. If a
stable region emerges where higher exposure intensity shows no
material performance deterioration, THAT is the evidence a capacity
challenger would eventually be built from -- not this script's output
directly.

Usage:
    python scripts\\analyze_capacity_exposure_performance_matrix.py ^
        --research-dataset capacity_research_dataset.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Broad PD bands deliberately -- first pass, avoid sparse cells. Can be
# refined around interesting regions later.
PD_BAND_EDGES = [0.0, 0.05, 0.15, 0.30, 1.0]
PD_BAND_LABELS = ["<5%", "5-15%", "15-30%", "30%+"]

CAPACITY_VARIABLES = {
    "float_activity_value_1m": "Float activity (cash-in + payment, 1m)",
    "commission": "Earnings (commission)",
    "cust_1m": "Customer/activity scale (distinct customers, 1m)",
    "average_balance": "Liquidity (average balance)",
}
N_CAPACITY_BANDS = 3
N_EI_BANDS = 3
MIN_CELL_N = 10  # cells smaller than this are still reported but flagged as low-N

RECENCY_COLS = ["days_since_payment_last", "days_since_cash_in_last", "days_since_cash_out_last"]


def _band_label(n_actual: int, n_requested: int) -> str:
    return "" if n_actual == n_requested else f" (collapsed to {n_actual} due to ties)"


def _qcut_safe(s: pd.Series, q: int, labels_hint: list[str]) -> tuple[pd.Series, int]:
    """qcut with duplicate bin edges dropped (common when a variable has many ties,
    e.g. small integer customer counts); returns (string-labeled band series, n
    bins actually produced -- may be < q)."""
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
    working["_ei"] = working["actual_exposure_ugx"] / working[capacity_col]
    if "fwd_new_loan_count" in working.columns:
        working["_took_new_loan"] = working["fwd_new_loan_count"] > 0
    else:
        working["_took_new_loan"] = False
    if RECENCY_COLS and all(c in working.columns for c in RECENCY_COLS):
        working["_days_since_any_activity"] = working[RECENCY_COLS].min(axis=1)
    else:
        working["_days_since_any_activity"] = np.nan

    cap_labels_hint = ["Low", "Medium", "High"]
    ei_labels_hint = ["Low", "Medium", "High"]

    rows = []
    for pd_band in PD_BAND_LABELS:
        band_df = working[working["_pd_band"] == pd_band]
        if len(band_df) < MIN_CELL_N:
            if len(band_df):
                rows.append({"pd_band": pd_band, "capacity_band": "(all)", "ei_band": "(all)",
                             "n_agents": len(band_df), "_low_n_warning": True})
            continue
        cap_band, n_cap_bins = _qcut_safe(band_df[capacity_col], N_CAPACITY_BANDS, cap_labels_hint)
        ei_band, n_ei_bins = _qcut_safe(band_df["_ei"], N_EI_BANDS, ei_labels_hint)
        if n_cap_bins == 0 or n_ei_bins == 0:
            continue
        band_df = band_df.assign(_cap_band=cap_band, _ei_band=ei_band)

        for (cb, eb), cell in band_df.groupby(["_cap_band", "_ei_band"], observed=True):
            n = len(cell)
            row = {
                "pd_band": pd_band, "capacity_band": cb, "ei_band": eb,
                "n_agents": n, "_low_n_warning": n < MIN_CELL_N,
                f"median_{capacity_col}": cell[capacity_col].median(),
                "median_actual_exposure_ugx": cell["actual_exposure_ugx"].median(),
                "median_exposure_intensity": cell["_ei"].median(),
                "median_combined_cap_benchmark": cell["combined_cap"].median() if "combined_cap" in cell.columns else np.nan,
                "median_cash_out_value_1m_diagnostic": cell["cash_out_value_1m"].median() if "cash_out_value_1m" in cell.columns else np.nan,
                "median_days_since_any_activity": cell["_days_since_any_activity"].median(),
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
    result = result.sort_values(["pd_band", "capacity_band", "ei_band"]).reset_index(drop=True)
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--research-dataset", default="capacity_research_dataset.csv")
    ap.add_argument("--diamond-high-risk-cal-pd", type=float, default=0.30,
                     help="threshold for the separate 'diamond below high-risk' repeat analysis")
    ap.add_argument("--out-prefix", default="capacity_exposure_performance")
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
    print("A cell where higher exposure-intensity bands show no material increase in\n"
          "fwd_any_bad_3dpd_rate_among_borrowers_pct, bad_closure_rate_pct, or shortfall_pct_of_forward_disbursed\n"
          "-- within the SAME pd_band and capacity_band -- is evidence that agents with that capacity profile\n"
          "may be able to sustain more exposure than they currently carry, relative to the engine's own\n"
          "combined_cap benchmark (reported alongside for context, not as a feature). A cell where performance\n"
          "deteriorates sharply as exposure intensity rises is evidence for where a capacity boundary should sit.\n"
          "This script does NOT compute that boundary or any capacity formula -- it only locates where a stable\n"
          "or deteriorating region appears, per the agreed sequencing (raw fundamentals -> this PD-controlled\n"
          "exposure/performance relationship -> an empirically supported exposure-intensity region -> only then\n"
          "L_capacity_challenger).")


if __name__ == "__main__":
    main()
