"""
derive_capacity_risk_supported_exposure_surface.py
======================================================
Joint capacity x risk surface -- the next step after
analyze_capacity_level_vs_absolute_exposure.py's finding that, within a
fixed capacity band, a monotonic absolute-exposure deterioration appears
at 15-30% PD (independently confirmed via Float Activity and Commission)
but NOT at <5%/5-15% PD. Per this session's explicit architectural
decision: capacity and risk stay separate (L_recommended = L_capacity x
M_C3) -- the capacity side should not be made PD-specific to reproduce
that 15-30% deterioration; that is M_C3's job. This script instead:

  PART A -- derive an EMPIRICAL SUPPORTED-EXPOSURE SURFACE: for each
  (capacity variable x PD band x capacity tercile) cell, walk the same
  discrete loan-size / exposure buckets used in
  analyze_capacity_level_vs_absolute_exposure.py, in increasing order,
  starting from a baseline (the smallest sufficiently-populated bucket in
  that cell). The "highest supported exposure" is the largest bucket for
  which EVERY bucket from the baseline up to and including it stays
  within a configurable tolerance of the baseline's bad-closure and
  shortfall rates (both must hold) AND has sufficient sample size. This
  is NOT "maximum observed exposure" -- it is where the evidence stops
  supporting a further increase, walking contiguously from a known-safe
  floor, per this session's explicit framing.

  PART B -- compare the EXISTING system against that surface. No new
  capacity formula is invented here. L_capacity_candidate is simply the
  engine's own current combined_cap (already computed, already the
  "capacity before the live tier multiplier and before C3" quantity),
  and M_C3 is the already-calibrated shadow_multiplier_base/conservative
  -- so L_capacity_candidate x M_C3 is just shadow_limit_post_transition_
  base/conservative, both already present in the research dataset. For
  every agent (not only those currently carrying exposure), this checks
  whether that existing C3-implied limit already sits inside or outside
  the Part A boundary for their own PD band x capacity tercile, and by
  how much headroom. This directly answers the question this session's
  analysis raised: before enlarging the capacity base, does C3 already
  push close to (or past) the empirically supported region at the
  CURRENT base -- telling us whether a higher future capacity base could
  be transplanted onto the SAME calibrated C3 without recalibration, or
  not.

TOLERANCE: the "acceptable performance envelope" is defined RELATIVE TO
EACH CELL'S OWN BASELINE (its own PD band x capacity tier), not a single
portfolio-wide cutoff -- deliberately, because the baseline level of risk
already differs by PD band and that is expected (that's what C3 is for).
What this surface tests is whether, AT a given PD and a given capacity
scale, going to a bigger absolute loan size adds MORE deterioration on
top of that baseline than the configured tolerance allows.
MAX_BAD_CLOSURE_DELTA_PP / MAX_SHORTFALL_DELTA_PP below are reasoned
defaults (both are percentage-point deltas, not ratios), exposed as CLI
flags specifically so they can be revisited rather than silently baked
in.

Restricted to Float Activity and Commission only, matching
analyze_capacity_level_vs_absolute_exposure.py's narrowing. All bucket-
construction and capacity-tercile logic is independently restated here
(this session's one-way scripts/ layering convention: scripts/ never
imports from other scripts/ files), not imported.

Usage:
    python scripts\\derive_capacity_risk_supported_exposure_surface.py ^
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
CAP_LABELS_HINT = ["Low", "Medium", "High"]
MIN_CELL_N = 10
MAX_DISCRETE_EXPOSURE_VALUES = 15  # above this, fall back to quantile exposure bands

# Reasoned defaults -- see module docstring's TOLERANCE note. A bucket is
# "supported" only while BOTH deltas from its cell's own baseline stay
# within these margins; exposed as CLI flags so they are never a silently
# fixed judgment call.
DEFAULT_MAX_BAD_CLOSURE_DELTA_PP = 3.0
DEFAULT_MAX_SHORTFALL_DELTA_PP = 2.0


def _qcut_safe(s: pd.Series, q: int, labels_hint: list[str]):
    """qcut with duplicate bin edges dropped. Returns (string-labeled band
    series, n bins actually produced -- may be < q, bin edges array)."""
    try:
        codes, bins = pd.qcut(s, q, duplicates="drop", retbins=True, labels=False)
    except ValueError:
        return pd.Series(np.nan, index=s.index, dtype=object), 0, None
    n_bins = len(bins) - 1
    if n_bins <= 0:
        return pd.Series(np.nan, index=s.index, dtype=object), 0, None
    labels = labels_hint[:n_bins] if n_bins <= len(labels_hint) else [f"Q{i + 1}" for i in range(n_bins)]
    return codes.map(dict(enumerate(labels))), n_bins, bins


def _cell_perf(cell: pd.DataFrame) -> dict:
    """Bad-closure and shortfall rates for one exposure-bucket cell -- the
    two performance-envelope metrics this session's analysis has used
    throughout. NaN (not 0) when the underlying columns are absent, so a
    missing metric never silently reads as "no deterioration"."""
    out = {"n_agents": len(cell)}
    if {"fwd_new_loans_closed_good_count", "fwd_new_loans_closed_bad_count"} <= set(cell.columns):
        n_good = int(cell["fwd_new_loans_closed_good_count"].sum())
        n_bad = int(cell["fwd_new_loans_closed_bad_count"].sum())
        n_closed = n_good + n_bad
        out["n_closed_new_loans"] = n_closed
        out["bad_closure_rate_pct"] = round(n_bad / n_closed * 100, 2) if n_closed else float("nan")
    else:
        out["n_closed_new_loans"] = 0
        out["bad_closure_rate_pct"] = float("nan")
    if {"fwd_new_loans_disbursed_ugx", "fwd_new_loans_repaid_ugx"} <= set(cell.columns):
        disbursed = cell["fwd_new_loans_disbursed_ugx"].sum()
        repaid = cell["fwd_new_loans_repaid_ugx"].sum()
        out["shortfall_pct_of_forward_disbursed"] = round((disbursed - repaid) / disbursed * 100, 2) if disbursed else float("nan")
    else:
        out["shortfall_pct_of_forward_disbursed"] = float("nan")
    return out


def build_exposure_grid(df: pd.DataFrame, capacity_col: str):
    """Restated (not imported) from analyze_capacity_level_vs_absolute_exposure.py:
    PD band x capacity tercile (within PD band) x absolute-exposure bucket.
    Returns (grid DataFrame, cap_edges dict keyed by pd_band)."""
    working = df[
        df["cal_pd"].notna() & df[capacity_col].notna() & (df[capacity_col] > 0)
        & df["actual_exposure_ugx"].notna() & (df["actual_exposure_ugx"] > 0)
    ].copy()
    if working.empty:
        return pd.DataFrame(), {}

    working["_pd_band"] = pd.cut(working["cal_pd"], bins=PD_BAND_EDGES, labels=PD_BAND_LABELS)

    exposure_vals = sorted(working["actual_exposure_ugx"].dropna().unique())
    is_discrete = len(exposure_vals) <= MAX_DISCRETE_EXPOSURE_VALUES

    rows = []
    cap_edges_by_pd_band: dict = {}
    for pd_band in PD_BAND_LABELS:
        band_df = working[working["_pd_band"] == pd_band]
        if len(band_df) < MIN_CELL_N:
            continue
        cap_band, n_cap_bins, cap_edges = _qcut_safe(band_df[capacity_col], N_CAPACITY_BANDS, CAP_LABELS_HINT)
        if n_cap_bins == 0:
            continue
        cap_edges_by_pd_band[pd_band] = (cap_edges, CAP_LABELS_HINT[:n_cap_bins])
        band_df = band_df.assign(_cap_band=cap_band)

        if is_discrete:
            band_df = band_df.assign(_exposure_bucket=band_df["actual_exposure_ugx"])
        else:
            exp_band, n_exp_bins, _ = _qcut_safe(band_df["actual_exposure_ugx"], 5,
                                                  ["Lowest", "Low", "Mid", "High", "Highest"])
            if n_exp_bins == 0:
                continue
            band_df = band_df.assign(_exposure_bucket=exp_band)

        for (cb, eb), cell in band_df.groupby(["_cap_band", "_exposure_bucket"], observed=True):
            perf = _cell_perf(cell)
            rows.append({
                "pd_band": pd_band, "capacity_band": cb, "exposure_bucket": eb,
                "exposure_bucket_numeric": cell["actual_exposure_ugx"].median(),
                "median_actual_exposure_ugx": cell["actual_exposure_ugx"].median(),
                f"median_{capacity_col}": cell[capacity_col].median(),
                **perf,
            })

    grid = pd.DataFrame(rows)
    return grid, cap_edges_by_pd_band


def derive_supported_frontier(grid: pd.DataFrame, max_bad_closure_delta_pp: float,
                               max_shortfall_delta_pp: float, min_cell_n: int) -> pd.DataFrame:
    """PART A: for each (pd_band, capacity_band), walk exposure buckets in
    increasing order from a sufficiently-populated baseline; the frontier
    stops at the first bucket that either lacks sufficient n or breaches
    tolerance on bad-closure or shortfall relative to the baseline."""
    rows = []
    if grid.empty:
        return pd.DataFrame()
    for (pd_band, cap_band), g in grid.groupby(["pd_band", "capacity_band"], observed=True):
        g = g.sort_values("exposure_bucket_numeric")
        sufficient = g[g["n_agents"] >= min_cell_n]
        if sufficient.empty:
            rows.append({"pd_band": pd_band, "capacity_band": cap_band,
                         "status": "no_bucket_with_sufficient_n"})
            continue
        baseline = sufficient.iloc[0]
        highest_supported = baseline
        first_unsupported = None
        for _, row in g.iterrows():
            if row["exposure_bucket_numeric"] <= baseline["exposure_bucket_numeric"]:
                continue
            if row["n_agents"] < min_cell_n:
                first_unsupported = {"exposure_bucket": row["exposure_bucket"], "n_agents": row["n_agents"],
                                      "reason": "insufficient_n"}
                break
            bad_delta = (row["bad_closure_rate_pct"] - baseline["bad_closure_rate_pct"]
                         if pd.notna(row["bad_closure_rate_pct"]) and pd.notna(baseline["bad_closure_rate_pct"]) else np.nan)
            shortfall_delta = (row["shortfall_pct_of_forward_disbursed"] - baseline["shortfall_pct_of_forward_disbursed"]
                               if pd.notna(row["shortfall_pct_of_forward_disbursed"]) and pd.notna(baseline["shortfall_pct_of_forward_disbursed"]) else np.nan)
            bad_ok = pd.isna(bad_delta) or bad_delta <= max_bad_closure_delta_pp
            shortfall_ok = pd.isna(shortfall_delta) or shortfall_delta <= max_shortfall_delta_pp
            if bad_ok and shortfall_ok:
                highest_supported = row
            else:
                first_unsupported = {"exposure_bucket": row["exposure_bucket"], "n_agents": row["n_agents"],
                                      "reason": "breach", "bad_closure_delta_pp": bad_delta,
                                      "shortfall_delta_pp": shortfall_delta}
                break
        rows.append({
            "pd_band": pd_band, "capacity_band": cap_band, "status": "ok",
            "baseline_exposure_bucket": baseline["exposure_bucket"], "baseline_n_agents": baseline["n_agents"],
            "baseline_bad_closure_rate_pct": baseline["bad_closure_rate_pct"],
            "baseline_shortfall_pct": baseline["shortfall_pct_of_forward_disbursed"],
            "highest_supported_exposure_bucket": highest_supported["exposure_bucket"],
            "highest_supported_exposure_numeric": highest_supported["exposure_bucket_numeric"],
            "highest_supported_n_agents": highest_supported["n_agents"],
            "highest_supported_bad_closure_rate_pct": highest_supported["bad_closure_rate_pct"],
            "highest_supported_shortfall_pct": highest_supported["shortfall_pct_of_forward_disbursed"],
            "first_unsupported_exposure_bucket": first_unsupported["exposure_bucket"] if first_unsupported else np.nan,
            "first_unsupported_reason": first_unsupported["reason"] if first_unsupported else "",
            "first_unsupported_bad_closure_delta_pp": first_unsupported.get("bad_closure_delta_pp", np.nan) if first_unsupported else np.nan,
            "first_unsupported_shortfall_delta_pp": first_unsupported.get("shortfall_delta_pp", np.nan) if first_unsupported else np.nan,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    cat_order = pd.CategoricalDtype(PD_BAND_LABELS, ordered=True)
    result["pd_band"] = result["pd_band"].astype(cat_order)
    return result.sort_values(["pd_band", "capacity_band"]).reset_index(drop=True)


def compare_c3_to_surface(df: pd.DataFrame, capacity_col: str, surface: pd.DataFrame,
                           cap_edges_by_pd_band: dict) -> pd.DataFrame:
    """PART B: classify EVERY agent (not only those currently carrying
    exposure) into the SAME PD band x capacity tercile used to build the
    surface (reusing the exact bin edges from Part A, so tiers are defined
    identically), then compare their EXISTING shadow_limit_post_transition_
    base/conservative (combined_cap x already-calibrated C3 -- no new
    capacity formula) against that cell's highest_supported_exposure_numeric."""
    required = ["shadow_limit_post_transition_base", "shadow_limit_post_transition_conservative"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()
    ok_surface = surface[surface.get("status", "") == "ok"] if not surface.empty else surface
    if ok_surface.empty:
        return pd.DataFrame()

    working = df[
        df["cal_pd"].notna() & df[capacity_col].notna() & (df[capacity_col] > 0)
        & df["shadow_limit_post_transition_base"].notna() & df["shadow_limit_post_transition_conservative"].notna()
    ].copy()
    if "shadow_status" in working.columns:
        working = working[working["shadow_status"].astype(str).str.strip().str.lower() == "ok"]
    working["_pd_band"] = pd.cut(working["cal_pd"], bins=PD_BAND_EDGES, labels=PD_BAND_LABELS)

    rows = []
    for pd_band, band_df in working.groupby("_pd_band", observed=True):
        if pd_band not in cap_edges_by_pd_band or band_df.empty:
            continue
        cap_edges, cap_labels = cap_edges_by_pd_band[pd_band]
        band_df = band_df.assign(
            _cap_band=pd.cut(band_df[capacity_col], bins=cap_edges, labels=cap_labels, include_lowest=True)
        )
        for cap_band, cell in band_df.groupby("_cap_band", observed=True):
            boundary_row = ok_surface[(ok_surface["pd_band"] == pd_band) & (ok_surface["capacity_band"] == cap_band)]
            if boundary_row.empty or cell.empty:
                continue
            boundary = boundary_row.iloc[0]["highest_supported_exposure_numeric"]
            if pd.isna(boundary) or boundary <= 0:
                continue
            c3_base = cell["shadow_limit_post_transition_base"]
            c3_cons = cell["shadow_limit_post_transition_conservative"]
            headroom_base_pct = (boundary - c3_base) / boundary * 100
            headroom_cons_pct = (boundary - c3_cons) / boundary * 100
            rows.append({
                "pd_band": pd_band, "capacity_band": cap_band, "n_agents": len(cell),
                "highest_supported_exposure_numeric": boundary,
                "median_shadow_limit_post_transition_base": c3_base.median(),
                "pct_agents_base_exceeds_boundary": round((c3_base > boundary).mean() * 100, 1),
                "median_headroom_base_pct": round(headroom_base_pct.median(), 1),
                "median_shadow_limit_post_transition_conservative": c3_cons.median(),
                "pct_agents_conservative_exceeds_boundary": round((c3_cons > boundary).mean() * 100, 1),
                "median_headroom_conservative_pct": round(headroom_cons_pct.median(), 1),
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    cat_order = pd.CategoricalDtype(PD_BAND_LABELS, ordered=True)
    result["pd_band"] = result["pd_band"].astype(cat_order)
    return result.sort_values(["pd_band", "capacity_band"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--research-dataset", default="capacity_research_dataset.csv")
    ap.add_argument("--diamond-high-risk-cal-pd", type=float, default=0.30)
    ap.add_argument("--max-bad-closure-delta-pp", type=float, default=DEFAULT_MAX_BAD_CLOSURE_DELTA_PP,
                     help="max allowed increase (percentage points) in bad_closure_rate_pct over a cell's "
                          "own baseline bucket before an exposure bucket is called unsupported")
    ap.add_argument("--max-shortfall-delta-pp", type=float, default=DEFAULT_MAX_SHORTFALL_DELTA_PP,
                     help="same, for shortfall_pct_of_forward_disbursed")
    ap.add_argument("--min-cell-n", type=int, default=MIN_CELL_N)
    ap.add_argument("--out-prefix", default="capacity_risk_supported_exposure_surface")
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

    c3_cols = ["shadow_limit_post_transition_base", "shadow_limit_post_transition_conservative"]
    missing_c3 = [c for c in c3_cols if c not in df.columns]
    if missing_c3:
        print(f"WARNING: {path} is missing C3 shadow column(s) {missing_c3} -- Part B (comparing the "
              f"existing system against the empirical surface) will be SKIPPED. Re-run "
              f"build_capacity_research_dataset.py (schema now includes these columns) to enable it. "
              f"Part A (the empirical surface itself) does not need them and will still run.")

    print(f"Tolerance: a bucket is 'supported' only while it stays within +{args.max_bad_closure_delta_pp} "
          f"percentage points of bad_closure_rate_pct AND +{args.max_shortfall_delta_pp} percentage points "
          f"of shortfall_pct_of_forward_disbursed relative to its own (pd_band, capacity_band) baseline bucket. "
          f"min_cell_n={args.min_cell_n}.")

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
            grid, cap_edges_by_pd_band = build_exposure_grid(pop_df, col)
            if grid.empty:
                print("  (no agents with valid cal_pd, capacity, and actual exposure -- skipped)")
                continue
            surface = derive_supported_frontier(grid, args.max_bad_closure_delta_pp, args.max_shortfall_delta_pp,
                                                 args.min_cell_n)
            print("\n-- PART A: empirical supported-exposure surface --")
            with pd.option_context("display.float_format", "{:,.2f}".format, "display.max_columns", None, "display.width", 240):
                print(surface.to_string(index=False))
            surface_out = f"{args.out_prefix}_surface_{col}_{out_suffix}.csv"
            surface.to_csv(surface_out, index=False)
            print(f"  -> written: {surface_out}")

            if not missing_c3:
                comparison = compare_c3_to_surface(pop_df, col, surface, cap_edges_by_pd_band)
                print("\n-- PART B: current combined_cap x C3 vs. the empirical surface --")
                if comparison.empty:
                    print("  (no comparable cells -- skipped)")
                else:
                    with pd.option_context("display.float_format", "{:,.2f}".format, "display.max_columns", None, "display.width", 240):
                        print(comparison.to_string(index=False))
                    comparison_out = f"{args.out_prefix}_c3_comparison_{col}_{out_suffix}.csv"
                    comparison.to_csv(comparison_out, index=False)
                    print(f"  -> written: {comparison_out}")

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
    print("Part A derives, for each PD band x capacity tercile, the highest absolute exposure the\n"
          "historical record supports within a tolerance of that cell's own baseline -- not a capacity\n"
          "formula, and not a claim this is causal (shadow C3 has not yet been live-piloted). Part B asks\n"
          "whether the EXISTING combined_cap x already-calibrated C3 already sits inside or outside that\n"
          "boundary -- no new capacity base is proposed or computed here. A high pct_agents_..._exceeds_\n"
          "boundary or a negative median_headroom_..._pct means the current system is already at or past\n"
          "the empirically supported region for that cell; a large positive headroom means there is room\n"
          "before a higher future capacity base would need C3 to be recalibrated or additionally controlled.")


if __name__ == "__main__":
    main()
