"""
compare_capacity_f_vs_incumbent_cap_layers.py
================================================
Compares the frozen Capacity(F; k=0.05) challenger directly against the
INCUMBENT ENGINE'S OWN internal cap layers -- capacity_cap, combined_cap,
and assigned_limit -- rather than against assigned_limit alone.

Rationale (per reviewer): assigned_limit has already absorbed recent-
usage, prior-exposure, risk-tier multiplier, prior-limit smoothing, and
policy floors on top of capacity_cap (see extrafloat_limit_engine_caps.py:
compute_capacity_cap -> combine_caps -> apply_policy_adjustments ->
finalize_limits). Comparing Capacity(F) only against assigned_limit
(already done via r_capacity_shadow_k0_05_base in the Decision Log, and
via live_shadow_vs_category_limit.csv's pct_of_capacity_disbursed column)
cannot distinguish "the fundamental capacity model itself disagrees with
the incumbent's own fundamental-capacity assessment" from "the
disagreement only emerges downstream, in usage/history/risk/smoothing/
policy." This script isolates the three stages explicitly:
    capacity_cap    -- incumbent's own fundamentals-driven capacity cap
                       (balance/revenue/txn/payments/customers/volume,
                       log-scaled, activity/seasonality-adjusted)
    combined_cap    -- after blending capacity_cap with recent_usage_cap/
                       prior_exposure_cap/risk_cap, risk guardrail, and
                       prior-limit smoothing
    assigned_limit  -- after the risk-tier multiplier, proven-good/
                       active-borrower floors, ceiling clip, and rounding

Also optionally triangulates against actual historical disbursement
(--monthly-summary-file), giving three independent perspectives: the
fundamentals-only challenger, the incumbent's richer internal
calculation, and real lending practice.

NEVER recomputes capacity_shadow_k0_05, capacity_cap, combined_cap, or
assigned_limit -- reads every value directly from its own frozen/live
source (the Stage 4C Decision Log for the former, engine_output.csv for
the latter three). Purely observational/diagnostic. Does not touch k,
Capacity(F), or C3 -- no result here is grounds for retuning any of them.

IMPORTANT PRACTICAL NOTE: capacity_cap and combined_cap are intermediate
engine columns, dropped from engine_output.csv by run_credit_risk_
pipeline.py UNLESS that run was invoked with --keep-intermediate. The
real frozen Stage 4C baseline run (run_retail_filtered_stage4c_shadow.bat,
Step 6) DID pass --keep-intermediate, writing to
output/engine_test_output_stage4c_shadow.csv -- check that file's header
for capacity_cap/combined_cap before assuming a re-run is needed. If
re-running is genuinely necessary, it MUST be the exact same invocation
(same input files/snapshot/config) that produced the frozen Decision Log
run_id, or capacity_cap will not correspond to the same agent states
already logged as capacity_shadow_k0_05 -- this script cannot detect a
mismatched re-run and will produce numbers that look fine but compare
different snapshots.

Usage:
    python scripts\\compare_capacity_f_vs_incumbent_cap_layers.py ^
        --engine-output output\\engine_test_output_stage4c_shadow.csv ^
        --decision-log-file output\\stage_4c_shadow\\stage_4c_decision_log_<run_id>.csv ^
        --monthly-summary-file monthly_disbursement_summary.csv ^
        --out-prefix output\\stage_4c_shadow\\capacity_f_vs_incumbent_layers
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAYERS = ["capacity_cap", "combined_cap", "assigned_limit"]

# Same "90-110% = agreement" convention used throughout this repo
# (check_monthly_summary_vs_assigned_limit.py, restated independently
# each time), right=False per the boundary-value fix already found in
# analyze_diamond_a_capacity_f_reconciliation.py (pd.cut's default
# right=True would misclassify R==0.90/1.10 at the boundaries).
R_BAND_EDGES = [-0.01, 0.90, 1.10, np.inf]
R_BAND_LABELS = ["BELOW", "AT", "ABOVE"]

N_DECILES_DEFAULT = 10


def _norm_msisdn(s: pd.Series) -> pd.Series:
    """Strips a trailing '.0' (pandas float-dtype read of a numeric ID
    column) BEFORE stripping non-digits -- restated from
    stage_4c_decision_logger.py's _norm_msisdn, not
    check_live_shadow_vs_category_limit.py's simpler digit-strip (which
    was found, elsewhere in this session, to mis-normalize a float-typed
    msisdn column by leaving a spurious trailing 0 from '.0')."""
    out = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    out = out.str.replace(r"[^0-9]", "", regex=True)
    return out.mask(out.str.len() == 0)


def load_engine_output(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"msisdn", "agent_category", "assigned_limit"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s) {missing}.")
    missing_layers = {"capacity_cap", "combined_cap"} - set(df.columns)
    if missing_layers:
        sys.exit(
            f"ERROR: {path} is missing {sorted(missing_layers)} -- these are intermediate engine "
            f"columns, only present if run_credit_risk_pipeline.py was invoked with --keep-intermediate. "
            f"Check output/engine_test_output_stage4c_shadow.csv (the frozen baseline run already used "
            f"--keep-intermediate, per run_retail_filtered_stage4c_shadow.bat Step 6) before re-running "
            f"anything -- a re-run must use the EXACT SAME inputs/snapshot as the frozen Decision Log run, "
            f"or capacity_cap will not correspond to the same agent states already logged there."
        )
    df["_key"] = _norm_msisdn(df["msisdn"])
    return df


def load_decision_log(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"agent_msisdn", "capacity_shadow_k0_05", "capacity_estimable",
                "float_activity_value_1m", "commission"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {path} is missing required Decision Log column(s) {missing}.")
    df["_key"] = _norm_msisdn(df["agent_msisdn"])
    return df


def load_monthly_summary(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"msisdn", "disbursed_amount"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s) {missing}.")
    df["_key"] = _norm_msisdn(df["msisdn"])
    return df


def build_joined(eng_df: pd.DataFrame, dlog_df: pd.DataFrame) -> pd.DataFrame:
    """One row per agent in engine_output, left-joined onto the Decision
    Log by normalized msisdn. Reports the match rate explicitly rather
    than silently dropping unmatched agents."""
    dlog_cols = ["capacity_shadow_k0_05", "capacity_estimable", "float_activity_value_1m", "commission"]
    keep_dlog = dlog_df[["_key"] + dlog_cols].drop_duplicates(subset="_key", keep="first")

    # engine_output.csv (with --keep-intermediate) can ALSO carry a raw "commission"
    # column of its own -- with suffixes=("", "_dlog") that would silently leave
    # the engine's own value under the plain "commission" name (used below for the
    # decile banding) instead of the Decision Log's, rather than raising -- found
    # as a real KeyError in overlay_capacity_f_vs_observed_exposure_frontier.py's
    # sibling join, which has no suffixes to mask it. Drop the overlap first so the
    # Decision Log's own value -- what Capacity(F) was actually computed from -- is
    # unambiguously the one used here too.
    overlapping = [c for c in dlog_cols if c in eng_df.columns]
    if overlapping:
        print(f"NOTE: engine output also had column(s) {overlapping} -- dropped before the Decision "
              f"Log merge so the Decision Log's own values (what Capacity(F) was actually computed "
              f"from) are used unambiguously.")
        eng_df = eng_df.drop(columns=overlapping)

    n_before = len(eng_df)
    joined = eng_df.merge(keep_dlog, on="_key", how="left")
    n_matched = int(joined["_key"].isin(keep_dlog["_key"]).sum())
    print(f"Joined engine output ({n_before:,} agents) to Decision Log: "
          f"{n_matched:,} / {n_before:,} matched a Decision Log row.")
    return joined


def compute_layer_ratios(df: pd.DataFrame, layers: list[str] = LAYERS) -> pd.DataFrame:
    """R_<layer> = capacity_shadow_k0_05 / df[layer]. Valid only where
    capacity_estimable=True, df[layer] is non-null and > 0 -- NaN
    (never Inf) elsewhere, same guard discipline as R_full/R_capacity
    elsewhere in this workstream."""
    out = df.copy()
    estimable = out["capacity_estimable"] == True  # noqa: E712
    cap = pd.to_numeric(out["capacity_shadow_k0_05"], errors="coerce")
    for layer in layers:
        layer_val = pd.to_numeric(out[layer], errors="coerce")
        valid = estimable & layer_val.notna() & (layer_val > 0) & cap.notna()
        r = pd.Series(np.nan, index=out.index, dtype="float64")
        r.loc[valid] = cap.loc[valid] / layer_val.loc[valid]
        out[f"R_vs_{layer}"] = r
    return out


def classify_band(r: pd.Series) -> pd.Series:
    return pd.cut(r, bins=R_BAND_EDGES, labels=R_BAND_LABELS, right=False)


def summarize_layer_comparison(df: pd.DataFrame, layers: list[str] = LAYERS) -> pd.DataFrame:
    """Headline table: for each incumbent layer, what fraction of
    evaluable agents have Capacity(F) strictly below / in the 90-110%
    agreement band / at or above that layer."""
    rows = []
    for layer in layers:
        r = df[f"R_vs_{layer}"].dropna()
        n = len(r)
        band = classify_band(r)
        band_counts = band.value_counts().reindex(R_BAND_LABELS, fill_value=0)
        rows.append({
            "layer": layer,
            "n_valid": n,
            "pct_capacity_f_below_layer (R<1.0)": round(float((r < 1.0).mean() * 100), 2) if n else float("nan"),
            "pct_BELOW (R<0.90)": round(float(band_counts["BELOW"] / n * 100), 2) if n else float("nan"),
            "pct_AT (0.90<=R<1.10)": round(float(band_counts["AT"] / n * 100), 2) if n else float("nan"),
            "pct_ABOVE (R>=1.10)": round(float(band_counts["ABOVE"] / n * 100), 2) if n else float("nan"),
            "median_R": round(float(r.median()), 4) if n else float("nan"),
            "p25_R": round(float(r.quantile(0.25)), 4) if n else float("nan"),
            "p75_R": round(float(r.quantile(0.75)), 4) if n else float("nan"),
        })
    return pd.DataFrame(rows)


def summarize_by_group(df: pd.DataFrame, layer: str, group_col: str) -> pd.DataFrame:
    """Per group (e.g. agent_category), the same headline metrics for ONE
    layer -- called once per layer so the caller can print layer-by-layer
    tables, mirroring the printed-table style used throughout this repo's
    scripts/ rather than a wide multi-layer table that gets hard to read."""
    rcol = f"R_vs_{layer}"
    rows = []
    for grp, sub in df.groupby(group_col, observed=True):
        r = sub[rcol].dropna()
        n = len(r)
        rows.append({
            group_col: grp,
            "n_valid": n,
            "pct_capacity_f_below_layer": round(float((r < 1.0).mean() * 100), 2) if n else float("nan"),
            "median_R": round(float(r.median()), 4) if n else float("nan"),
        })
    return pd.DataFrame(rows)


def _qcut_safe(s: pd.Series, q: int = N_DECILES_DEFAULT, prefix: str = "D") -> pd.Series:
    """Restated (not imported) from analyze_diamond_a_capacity_f_reconciliation.py
    -- degrades the bin count downward on a duplicate-edge ValueError
    rather than failing the whole diagnostic over a sparse/skewed tail."""
    for n_bins in range(q, 1, -1):
        try:
            return pd.qcut(s, n_bins, labels=[f"{prefix}{i}" for i in range(1, n_bins + 1)], duplicates="drop")
        except ValueError:
            continue
    return pd.Series([f"{prefix}1"] * len(s), index=s.index)


def summarize_by_scale_decile(df: pd.DataFrame, layer: str, scale_col: str, q: int = N_DECILES_DEFAULT) -> pd.DataFrame:
    """Does Capacity(F)'s agreement with `layer` trend toward/above 1.0
    as business scale (float_activity_value_1m or commission) increases?
    Deciled on the full `df` (not restricted to rows with a defined R, to
    avoid letting the decile boundaries themselves shift with layer
    validity) -- same convention as r_capacity_by_scale_decile()."""
    rcol = f"R_vs_{layer}"
    work = df.copy()
    work["_decile"] = _qcut_safe(work[scale_col], q=q)
    rows = []
    for dec, sub in work.groupby("_decile", observed=True):
        r = sub[rcol].dropna()
        n = len(r)
        rows.append({
            "decile": dec,
            "n_valid": n,
            "pct_capacity_f_below_layer": round(float((r < 1.0).mean() * 100), 2) if n else float("nan"),
            "median_R": round(float(r.median()), 4) if n else float("nan"),
        })
    out = pd.DataFrame(rows)
    out["_sort"] = out["decile"].astype(str).str[1:].astype(int)
    return out.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


def build_triangulation_crosstab(df: pd.DataFrame, disbursed_col: str = "disbursed_amount") -> pd.DataFrame:
    """Joint distribution of band(Capacity(F)/capacity_cap) x
    band(Capacity(F)/disbursed_amount) -- gives the raw counts needed to
    read off the reviewer's three candidate scenarios (both capacity_cap
    and actual borrowing sit above Capacity(F); capacity_cap agrees with
    Capacity(F) but borrowing is higher; Capacity(F) tracks borrowing
    better than capacity_cap does) WITHOUT this script pre-deciding which
    scenario holds -- that judgment is left to the reader."""
    cap = pd.to_numeric(df["capacity_shadow_k0_05"], errors="coerce")
    disb = pd.to_numeric(df[disbursed_col], errors="coerce")
    valid = df["capacity_estimable"] == True  # noqa: E712
    valid &= disb.notna() & (disb > 0)
    r_vs_disb = pd.Series(np.nan, index=df.index, dtype="float64")
    r_vs_disb.loc[valid & cap.notna()] = cap.loc[valid & cap.notna()] / disb.loc[valid & cap.notna()]

    r_vs_cap_cap = df["R_vs_capacity_cap"]
    both_valid = r_vs_disb.notna() & r_vs_cap_cap.notna()
    band_cap_cap = classify_band(r_vs_cap_cap[both_valid])
    band_disb = classify_band(r_vs_disb[both_valid])
    ct = pd.crosstab(band_cap_cap, band_disb, rownames=["Capacity(F) vs capacity_cap"],
                      colnames=["Capacity(F) vs disbursed_amount"])
    ct = ct.reindex(index=R_BAND_LABELS, columns=R_BAND_LABELS, fill_value=0)
    return ct


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine-output", required=True,
                     help="engine output CSV with capacity_cap/combined_cap retained -- see module "
                          "docstring's IMPORTANT PRACTICAL NOTE before pointing this at a file that may "
                          "have been produced without --keep-intermediate")
    ap.add_argument("--decision-log-file", required=True)
    ap.add_argument("--monthly-summary-file", default=None,
                     help="optional -- enables the Capacity(F) vs capacity_cap vs actual-disbursement "
                          "triangulation crosstab; skipped with a NOTE if not given")
    ap.add_argument("--n-deciles", type=int, default=N_DECILES_DEFAULT)
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args(argv)

    print("=" * 78)
    print("Capacity(F) vs. the incumbent engine's OWN internal cap layers")
    print("(capacity_cap -> combined_cap -> assigned_limit), not just assigned_limit")
    print("=" * 78)
    print("Diagnostic of the already-frozen k=0.05 Capacity(F) challenger. Purely "
          "observational -- does not change k, Capacity(F), or C3, and does not by "
          "itself establish that either the incumbent or the challenger is correct.\n")

    eng = load_engine_output(args.engine_output)
    dlog = load_decision_log(args.decision_log_file)
    df = build_joined(eng, dlog)
    df = compute_layer_ratios(df, LAYERS)

    print(f"\n{'=' * 78}")
    print("Headline: Capacity(F) vs. capacity_cap / combined_cap / assigned_limit")
    print("=" * 78)
    headline = summarize_layer_comparison(df, LAYERS)
    print(headline.to_string(index=False))

    for layer in LAYERS:
        print(f"\n{'=' * 78}")
        print(f"By agent_category -- Capacity(F) vs. {layer}")
        print("=" * 78)
        print(summarize_by_group(df, layer, "agent_category").to_string(index=False))

    for layer in LAYERS:
        print(f"\n{'=' * 78}")
        print(f"By float_activity_value_1m decile -- Capacity(F) vs. {layer}")
        print("=" * 78)
        print(summarize_by_scale_decile(df, layer, "float_activity_value_1m", args.n_deciles).to_string(index=False))

        print(f"\n{'=' * 78}")
        print(f"By commission decile -- Capacity(F) vs. {layer}")
        print("=" * 78)
        print(summarize_by_scale_decile(df, layer, "commission", args.n_deciles).to_string(index=False))

    if args.monthly_summary_file is None:
        print("\nNOTE: --monthly-summary-file not given -- skipping the Capacity(F) vs. "
              "capacity_cap vs. actual-disbursement triangulation.")
    else:
        monthly_path = Path(args.monthly_summary_file)
        if not monthly_path.exists():
            print(f"\nNOTE: {monthly_path} not found -- skipping triangulation.")
        else:
            monthly = load_monthly_summary(monthly_path)
            keep_monthly = monthly[["_key", "disbursed_amount"]]
            n_before = len(df)
            tri = df.merge(keep_monthly, on="_key", how="left")
            n_with_disb = int(tri["disbursed_amount"].notna().sum())
            print(f"\nMerged {monthly_path.name}: {n_with_disb:,} / {len(tri):,} agent-row(s) "
                  f"matched a disbursement record ({len(tri) - n_before:+,} rows vs. before this merge -- "
                  f"an agent with records in more than one month appears once per month).")
            print(f"\n{'=' * 78}")
            print("Triangulation: Capacity(F) vs. capacity_cap  x  Capacity(F) vs. actual disbursed_amount")
            print("(reads off the three candidate scenarios -- this script does not pre-classify them)")
            print("=" * 78)
            ct = build_triangulation_crosstab(tri)
            print(ct.to_string())
            if args.out_prefix:
                ct.to_csv(f"{args.out_prefix}_triangulation_crosstab.csv")

    if args.out_prefix:
        headline.to_csv(f"{args.out_prefix}_headline.csv", index=False)
        for layer in LAYERS:
            summarize_by_group(df, layer, "agent_category").to_csv(
                f"{args.out_prefix}_by_agent_category_{layer}.csv", index=False)
            summarize_by_scale_decile(df, layer, "float_activity_value_1m", args.n_deciles).to_csv(
                f"{args.out_prefix}_by_float_decile_{layer}.csv", index=False)
            summarize_by_scale_decile(df, layer, "commission", args.n_deciles).to_csv(
                f"{args.out_prefix}_by_commission_decile_{layer}.csv", index=False)
        print(f"\nCSV outputs written with prefix: {args.out_prefix}")


if __name__ == "__main__":
    main()
