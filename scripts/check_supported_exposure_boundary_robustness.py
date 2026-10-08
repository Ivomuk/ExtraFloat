"""
check_supported_exposure_boundary_robustness.py
===================================================
Robustness check for the supported-exposure boundary computed by
derive_capacity_risk_supported_exposure_surface.py's Part A, raised by
this session's real-data run: the Diamond A, 15-30% PD, High
float-activity boundary (250K) rested on a baseline of only 19 agents at
UGX 100K, and the All-agents, 5-15% PD, High float-activity boundary
collapsed to 50K on a 10-agent baseline, producing a since-flagged
"99.3% of agents exceed by a median 765%" result that is almost
certainly a baseline-noise artifact, not a real finding. Per this
session's explicit instruction: before revisiting C3's calibration on
the strength of that comparison, check whether the boundary itself is an
artifact of comparing every bucket against one small reference cell.

THREE BASELINE CONSTRUCTIONS, same exposure-bucket grid as before:
  "current" -- the original method: the smallest exposure bucket with at
      least --min-cell-n agents (default 10). Reproduced here unchanged,
      NOT to claim it's correct, but as the reference point the other
      two methods are being checked against.
  "pooled"  -- pools the smallest exposure buckets, in increasing order,
      until their COMBINED agent count reaches --pool-min-n (default
      100), and computes the baseline bad-closure/shortfall rate from
      the POOLED sums (not any single bucket). The frontier then walks
      forward only through buckets above the pooled range.
  "smoothed" -- fits an isotonic (monotonic, non-decreasing) regression
      of bad-closure rate and of shortfall, separately, against exposure
      across EVERY bucket in the cell (weighted by each bucket's own
      n_closed_new_loans / disbursed total respectively) -- the same
      IsotonicRegression(increasing=True, out_of_bounds="clip") already
      used by fit_shadow_risk_calibration.py for cal_pd -> bad rate. The
      baseline is the SMOOTHED curve's value at the lowest bucket, and
      every subsequent bucket is compared against the smoothed curve
      rather than its own single raw rate -- removing the dependence on
      any one bucket's sampling noise, at both ends of the comparison.

TOLERANCE SWEEP: each method is run at every tolerance in
--tolerances (default 2,3,4,5 percentage points), applied SYMMETRICALLALLY
to both the bad-closure delta and the shortfall delta for this sweep
(a deliberate simplification for sensitivity-checking purposes -- this
does NOT reproduce derive_capacity_risk_supported_exposure_surface.py's
asymmetric 3.0/2.0pp production default row-for-row at tolerance=3).

MINIMUM-N ROBUSTNESS LABEL (separate from the tolerance sweep): every
resulting boundary is labelled "robust" only when BOTH the baseline's
own agent count and the final highest-supported bucket's agent count
are >= --min-robust-n (default 100); otherwise "exploratory_insufficient_
evidence". For "smoothed", the baseline n used for this label is the
WHOLE CELL's agent count (every bucket contributes to the fit), not just
the lowest bucket's -- smoothing is specifically meant to fix the
small-single-cell problem, so penalizing it for the lowest bucket's own
size would defeat the point.

This script does NOT change C3, does NOT compute a capacity formula, and
does NOT decide which baseline method is "right" -- it reports whether
the boundary survives across all three constructions and four
tolerances, which is the question that needs answering before any
calibration decision.

Usage:
    python scripts\\check_supported_exposure_boundary_robustness.py ^
        --research-dataset capacity_research_dataset.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

PD_BAND_EDGES = [0.0, 0.05, 0.15, 0.30, 1.0]
PD_BAND_LABELS = ["<5%", "5-15%", "15-30%", "30%+"]

CAPACITY_VARIABLES = {
    "float_activity_value_1m": "Float activity (cash-in + payment, 1m)",
    "commission": "Earnings (commission)",
}
N_CAPACITY_BANDS = 3
CAP_LABELS_HINT = ["Low", "Medium", "High"]
MIN_CELL_N = 10
MAX_DISCRETE_EXPOSURE_VALUES = 15

DEFAULT_TOLERANCES_PP = [2.0, 3.0, 4.0, 5.0]
DEFAULT_POOL_MIN_N = 100
DEFAULT_MIN_ROBUST_N = 100
BASELINE_METHODS = ["current", "pooled", "smoothed"]


def _qcut_safe(s: pd.Series, q: int, labels_hint: list[str]):
    try:
        codes, bins = pd.qcut(s, q, duplicates="drop", retbins=True, labels=False)
    except ValueError:
        return pd.Series(np.nan, index=s.index, dtype=object), 0
    n_bins = len(bins) - 1
    if n_bins <= 0:
        return pd.Series(np.nan, index=s.index, dtype=object), 0
    labels = labels_hint[:n_bins] if n_bins <= len(labels_hint) else [f"Q{i + 1}" for i in range(n_bins)]
    return codes.map(dict(enumerate(labels))), n_bins


def build_exposure_grid_detailed(df: pd.DataFrame, capacity_col: str) -> pd.DataFrame:
    """Same PD band x capacity tercile x absolute-exposure-bucket grid as
    derive_capacity_risk_supported_exposure_surface.py (restated, not
    imported), but keeping the RAW counts/sums (not just the already-
    computed rates) so the pooled and smoothed methods can recombine them
    correctly instead of averaging rates across buckets."""
    working = df[
        df["cal_pd"].notna() & df[capacity_col].notna() & (df[capacity_col] > 0)
        & df["actual_exposure_ugx"].notna() & (df["actual_exposure_ugx"] > 0)
    ].copy()
    if working.empty:
        return pd.DataFrame()

    working["_pd_band"] = pd.cut(working["cal_pd"], bins=PD_BAND_EDGES, labels=PD_BAND_LABELS)
    exposure_vals = sorted(working["actual_exposure_ugx"].dropna().unique())
    is_discrete = len(exposure_vals) <= MAX_DISCRETE_EXPOSURE_VALUES

    rows = []
    for pd_band in PD_BAND_LABELS:
        band_df = working[working["_pd_band"] == pd_band]
        if len(band_df) < MIN_CELL_N:
            continue
        cap_band, n_cap_bins = _qcut_safe(band_df[capacity_col], N_CAPACITY_BANDS, CAP_LABELS_HINT)
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
            n_good = int(cell["fwd_new_loans_closed_good_count"].sum()) if "fwd_new_loans_closed_good_count" in cell.columns else 0
            n_bad = int(cell["fwd_new_loans_closed_bad_count"].sum()) if "fwd_new_loans_closed_bad_count" in cell.columns else 0
            n_closed = n_good + n_bad
            disbursed_sum = cell["fwd_new_loans_disbursed_ugx"].sum() if "fwd_new_loans_disbursed_ugx" in cell.columns else np.nan
            repaid_sum = cell["fwd_new_loans_repaid_ugx"].sum() if "fwd_new_loans_repaid_ugx" in cell.columns else np.nan
            rows.append({
                "pd_band": pd_band, "capacity_band": cb, "exposure_bucket": eb,
                "exposure_bucket_numeric": cell["actual_exposure_ugx"].median(),
                "n_agents": len(cell),
                "n_good": n_good, "n_bad": n_bad, "n_closed": n_closed,
                "disbursed_sum": disbursed_sum, "repaid_sum": repaid_sum,
                "bad_closure_rate_pct": round(n_bad / n_closed * 100, 4) if n_closed else float("nan"),
                "shortfall_pct": round((disbursed_sum - repaid_sum) / disbursed_sum * 100, 4) if disbursed_sum else float("nan"),
            })
    grid = pd.DataFrame(rows)
    if grid.empty:
        return grid
    cat_order = pd.CategoricalDtype(PD_BAND_LABELS, ordered=True)
    grid["pd_band"] = grid["pd_band"].astype(cat_order)
    return grid.sort_values(["pd_band", "capacity_band", "exposure_bucket_numeric"]).reset_index(drop=True)


def _fit_smoothed(cell_rows: pd.DataFrame) -> tuple:
    """Isotonic fit of bad-closure rate and shortfall against exposure,
    across every bucket in this (pd_band, capacity_band) cell -- same
    sklearn convention as fit_shadow_risk_calibration.py. Returns
    (smoothed_bad_pct array, smoothed_shortfall_pct array) aligned to
    cell_rows' row order, or (None, None) if there isn't enough data to fit."""
    x = cell_rows["exposure_bucket_numeric"].to_numpy(dtype=float)
    if len(np.unique(x)) < 2:
        return None, None

    smoothed_bad = None
    bad_mask = cell_rows["bad_closure_rate_pct"].notna() & (cell_rows["n_closed"] > 0)
    if bad_mask.sum() >= 2 and cell_rows.loc[bad_mask, "exposure_bucket_numeric"].nunique() >= 2:
        iso_bad = IsotonicRegression(y_min=0.0, y_max=100.0, increasing=True, out_of_bounds="clip")
        iso_bad.fit(cell_rows.loc[bad_mask, "exposure_bucket_numeric"], cell_rows.loc[bad_mask, "bad_closure_rate_pct"],
                    sample_weight=cell_rows.loc[bad_mask, "n_closed"])
        smoothed_bad = iso_bad.predict(x)

    smoothed_shortfall = None
    sf_mask = cell_rows["shortfall_pct"].notna() & cell_rows["disbursed_sum"].notna() & (cell_rows["disbursed_sum"] > 0)
    if sf_mask.sum() >= 2 and cell_rows.loc[sf_mask, "exposure_bucket_numeric"].nunique() >= 2:
        iso_sf = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso_sf.fit(cell_rows.loc[sf_mask, "exposure_bucket_numeric"], cell_rows.loc[sf_mask, "shortfall_pct"],
                   sample_weight=cell_rows.loc[sf_mask, "disbursed_sum"])
        smoothed_shortfall = iso_sf.predict(x)

    return smoothed_bad, smoothed_shortfall


def _walk_frontier(buckets_sorted: list, baseline_n: float, baseline_bad_pct: float, baseline_shortfall_pct: float,
                    start_after_numeric: float, max_pp: float, min_cell_n: int) -> tuple:
    """Shared frontier walker for all three methods: starting strictly
    above start_after_numeric, extend highest_supported through buckets
    whose EFFECTIVE bad/shortfall rate (raw for current/pooled, smoothed
    prediction for the smoothed method -- the caller decides which values
    went into each row's 'bad_closure_rate_pct'/'shortfall_pct' fields)
    stays within max_pp of the baseline, and which have >= min_cell_n raw
    agents. Stops at the first breach or insufficient-n bucket."""
    highest_supported = {"exposure_bucket_numeric": start_after_numeric, "exposure_bucket": start_after_numeric,
                          "n_agents": baseline_n, "bad_closure_rate_pct": baseline_bad_pct,
                          "shortfall_pct": baseline_shortfall_pct}
    first_unsupported = None
    for row in buckets_sorted:
        if row["exposure_bucket_numeric"] <= start_after_numeric:
            continue
        if row["n_agents"] < min_cell_n:
            first_unsupported = {"exposure_bucket": row["exposure_bucket"], "n_agents": row["n_agents"], "reason": "insufficient_n"}
            break
        bad_delta = (row["bad_closure_rate_pct"] - baseline_bad_pct
                     if pd.notna(row["bad_closure_rate_pct"]) and pd.notna(baseline_bad_pct) else np.nan)
        shortfall_delta = (row["shortfall_pct"] - baseline_shortfall_pct
                           if pd.notna(row["shortfall_pct"]) and pd.notna(baseline_shortfall_pct) else np.nan)
        bad_ok = pd.isna(bad_delta) or bad_delta <= max_pp
        shortfall_ok = pd.isna(shortfall_delta) or shortfall_delta <= max_pp
        if bad_ok and shortfall_ok:
            highest_supported = row
        else:
            first_unsupported = {"exposure_bucket": row["exposure_bucket"], "n_agents": row["n_agents"], "reason": "breach",
                                  "bad_closure_delta_pp": bad_delta, "shortfall_delta_pp": shortfall_delta}
            break
    return highest_supported, first_unsupported


def compute_boundary(cell_rows: pd.DataFrame, method: str, tolerance_pp: float,
                      min_cell_n: int, pool_min_n: int) -> dict:
    """One (pd_band, capacity_band) cell, one method, one tolerance ->
    one boundary result dict. cell_rows must already be sorted ascending
    by exposure_bucket_numeric."""
    rows = cell_rows.to_dict("records")
    if not rows:
        return {"status": "no_buckets"}

    if method == "current":
        sufficient = [r for r in rows if r["n_agents"] >= min_cell_n]
        if not sufficient:
            return {"status": "no_bucket_with_sufficient_n"}
        baseline = sufficient[0]
        highest_supported, first_unsupported = _walk_frontier(
            rows, baseline["n_agents"], baseline["bad_closure_rate_pct"], baseline["shortfall_pct"],
            baseline["exposure_bucket_numeric"], tolerance_pp, min_cell_n)
        baseline_n_for_label = baseline["n_agents"]
        baseline_exposure_bucket = baseline["exposure_bucket"]

    elif method == "pooled":
        cum_n = 0
        pooled_good = pooled_bad = pooled_closed = 0
        pooled_disbursed = pooled_repaid = 0.0
        top_numeric = None
        for r in rows:
            cum_n += r["n_agents"]
            pooled_good += r["n_good"]; pooled_bad += r["n_bad"]; pooled_closed += r["n_closed"]
            if pd.notna(r["disbursed_sum"]):
                pooled_disbursed += r["disbursed_sum"]
                pooled_repaid += r["repaid_sum"] if pd.notna(r["repaid_sum"]) else 0.0
            top_numeric = r["exposure_bucket_numeric"]
            if cum_n >= pool_min_n:
                break
        if cum_n < pool_min_n:
            return {"status": "insufficient_data_for_pooled_baseline"}
        baseline_bad_pct = round(pooled_bad / pooled_closed * 100, 4) if pooled_closed else float("nan")
        baseline_shortfall_pct = round((pooled_disbursed - pooled_repaid) / pooled_disbursed * 100, 4) if pooled_disbursed else float("nan")
        highest_supported, first_unsupported = _walk_frontier(
            rows, cum_n, baseline_bad_pct, baseline_shortfall_pct, top_numeric, tolerance_pp, min_cell_n)
        baseline_n_for_label = cum_n
        baseline_exposure_bucket = f"pooled up to {top_numeric:,.0f}"

    elif method == "smoothed":
        smoothed_bad, smoothed_shortfall = _fit_smoothed(cell_rows)
        if smoothed_bad is None and smoothed_shortfall is None:
            return {"status": "insufficient_data_for_smoothed_fit"}
        effective_rows = []
        for i, r in enumerate(rows):
            effective_rows.append({
                **r,
                "bad_closure_rate_pct": smoothed_bad[i] if smoothed_bad is not None else r["bad_closure_rate_pct"],
                "shortfall_pct": smoothed_shortfall[i] if smoothed_shortfall is not None else r["shortfall_pct"],
            })
        baseline = effective_rows[0]
        total_n = int(cell_rows["n_agents"].sum())
        highest_supported, first_unsupported = _walk_frontier(
            effective_rows, baseline["n_agents"], baseline["bad_closure_rate_pct"], baseline["shortfall_pct"],
            baseline["exposure_bucket_numeric"], tolerance_pp, min_cell_n)
        baseline_n_for_label = total_n  # smoothing draws on the whole cell, not just the lowest bucket
        baseline_exposure_bucket = f"smoothed @ {baseline['exposure_bucket']}"
    else:
        raise ValueError(method)

    highest_n_for_label = highest_supported["n_agents"]
    status = "robust" if (baseline_n_for_label >= DEFAULT_MIN_ROBUST_N and highest_n_for_label >= DEFAULT_MIN_ROBUST_N) \
        else "exploratory_insufficient_evidence"

    return {
        "status": "ok", "method": method, "tolerance_pp": tolerance_pp,
        "baseline_exposure_bucket": baseline_exposure_bucket, "baseline_n_for_label": baseline_n_for_label,
        "highest_supported_exposure_numeric": highest_supported["exposure_bucket_numeric"],
        "highest_supported_exposure_bucket": highest_supported["exposure_bucket"],
        "highest_supported_n_agents": highest_n_for_label,
        "first_unsupported_exposure_bucket": first_unsupported["exposure_bucket"] if first_unsupported else np.nan,
        "first_unsupported_reason": first_unsupported["reason"] if first_unsupported else "",
        "robustness_status": status,
    }


def run_sensitivity(grid: pd.DataFrame, tolerances: list, min_cell_n: int, pool_min_n: int) -> pd.DataFrame:
    rows = []
    if grid.empty:
        return pd.DataFrame()
    for (pd_band, cap_band), cell_rows in grid.groupby(["pd_band", "capacity_band"], observed=True):
        cell_rows = cell_rows.sort_values("exposure_bucket_numeric").reset_index(drop=True)
        for method in BASELINE_METHODS:
            for tol in tolerances:
                result = compute_boundary(cell_rows, method, tol, min_cell_n, pool_min_n)
                rows.append({"pd_band": pd_band, "capacity_band": cap_band, **result})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    cat_order = pd.CategoricalDtype(PD_BAND_LABELS, ordered=True)
    result["pd_band"] = result["pd_band"].astype(cat_order)
    return result.sort_values(["pd_band", "capacity_band", "method", "tolerance_pp"]).reset_index(drop=True)


def summarize_stability(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """One row per (pd_band, capacity_band): across all method x tolerance
    combinations that actually produced a boundary, how much does it move,
    and separately, restricted to combinations labelled 'robust'."""
    rows = []
    ok = sensitivity[sensitivity["status"] == "ok"]
    for (pd_band, cap_band), g in ok.groupby(["pd_band", "capacity_band"], observed=True):
        robust = g[g["robustness_status"] == "robust"]
        rows.append({
            "pd_band": pd_band, "capacity_band": cap_band,
            "n_combinations": len(g), "n_robust_combinations": len(robust),
            "min_boundary_all": g["highest_supported_exposure_numeric"].min(),
            "max_boundary_all": g["highest_supported_exposure_numeric"].max(),
            "min_boundary_robust_only": robust["highest_supported_exposure_numeric"].min() if len(robust) else np.nan,
            "max_boundary_robust_only": robust["highest_supported_exposure_numeric"].max() if len(robust) else np.nan,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    cat_order = pd.CategoricalDtype(PD_BAND_LABELS, ordered=True)
    out["pd_band"] = out["pd_band"].astype(cat_order)
    return out.sort_values(["pd_band", "capacity_band"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--research-dataset", default="capacity_research_dataset.csv")
    ap.add_argument("--diamond-high-risk-cal-pd", type=float, default=0.30)
    ap.add_argument("--tolerances", default=",".join(str(t) for t in DEFAULT_TOLERANCES_PP),
                     help="comma-separated percentage-point tolerances, applied symmetrically to both "
                          "bad-closure and shortfall deltas")
    ap.add_argument("--pool-min-n", type=int, default=DEFAULT_POOL_MIN_N)
    ap.add_argument("--min-cell-n", type=int, default=MIN_CELL_N)
    ap.add_argument("--out-prefix", default="supported_exposure_boundary_robustness")
    args = ap.parse_args(argv)
    tolerances = [float(t) for t in args.tolerances.split(",")]

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

    print(f"Tolerance sweep: {tolerances} percentage points (symmetric on bad-closure and shortfall deltas).")
    print(f"Pooled-baseline target: >= {args.pool_min_n} agents. Robustness label requires baseline AND "
          f"highest-supported bucket n >= {DEFAULT_MIN_ROBUST_N} each.")

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
            grid = build_exposure_grid_detailed(pop_df, col)
            if grid.empty:
                print("  (no agents with valid cal_pd, capacity, and actual exposure -- skipped)")
                continue
            sensitivity = run_sensitivity(grid, tolerances, args.min_cell_n, args.pool_min_n)
            sens_out = f"{args.out_prefix}_sensitivity_{col}_{out_suffix}.csv"
            sensitivity.to_csv(sens_out, index=False)
            print(f"  -> full sensitivity table written: {sens_out}")

            print("\n-- Boundary (highest_supported_exposure_numeric) by method x tolerance --")
            pivot = sensitivity[sensitivity["status"] == "ok"].pivot_table(
                index=["pd_band", "capacity_band"], columns=["method", "tolerance_pp"],
                values="highest_supported_exposure_numeric", observed=True)
            with pd.option_context("display.float_format", "{:,.0f}".format, "display.max_columns", None, "display.width", 240):
                print(pivot.to_string())

            print("\n-- Stability summary (across all method x tolerance combinations) --")
            stability = summarize_stability(sensitivity)
            with pd.option_context("display.float_format", "{:,.0f}".format, "display.max_columns", None, "display.width", 240):
                print(stability.to_string(index=False))
            stability_out = f"{args.out_prefix}_stability_{col}_{out_suffix}.csv"
            stability.to_csv(stability_out, index=False)
            print(f"  -> stability summary written: {stability_out}")

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
    print("This checks whether derive_capacity_risk_supported_exposure_surface.py's Part A boundary survives\n"
          "under alternative baseline constructions (pooled, smoothed) and a range of tolerances -- it does\n"
          "not itself decide a capacity formula, and it does not change C3. A cell whose boundary stays in a\n"
          "narrow range across method x tolerance combinations, AND whose robust-only range is similarly\n"
          "narrow, is evidence the original boundary wasn't a small-reference-cell artifact. A cell whose\n"
          "boundary swings widely, or has zero robust combinations, means the original single-method/single-\n"
          "tolerance number should not be treated as a finding on its own.")


if __name__ == "__main__":
    main()
