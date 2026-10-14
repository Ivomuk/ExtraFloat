"""
overlay_capacity_f_vs_observed_exposure_frontier.py
======================================================
Overlays THREE independent curves against business scale (float activity
and commission, evaluated independently per Analysis 4's convention):

    1. Observed historically-supported exposure frontier
       (contiguous_supported_frontier_tier_ugx, already derived and
       frozen by scripts/derive_capacity_frontier_from_business_state.py
       -- optionally the PD-comparable variant from
       scripts/compute_pd_comparable_capacity_frontier.py, which is a
       documented drop-in replacement for the same CSV schema and is the
       closer match to "controlling for risk").
    2. Capacity(F) at the frozen k=0.05 (capacity_shadow_k0_05, from the
       live Stage 4C Decision Log).
    3. The incumbent engine's own fundamentals-driven capacity_cap (from
       engine_output.csv, requires --keep-intermediate).

Per reviewer: this is the targeted follow-up to Analysis 4 and the
PD-comparable frontier work, now asking specifically whether Capacity(F)
undershoots the empirical frontier at low/mid scale, tracks it at high
scale, or whether historical lending itself exceeded the empirically
safe frontier. NEVER regenerates the frontier CSVs -- they are read-only
inputs, exactly like compute_pd_comparable_capacity_frontier.py's own
"REQUIRES the real Deliverable 1 CSVs, never regenerates them" rule.
NEVER recomputes capacity_shadow_k0_05 or capacity_cap. Purely
observational. Does not touch k, Capacity(F), capacity_cap, or C3 -- no
result here is grounds for retuning any of them.

METHODOLOGICAL NOTE (load-bearing, read before interpreting output): the
frontier's bands (D1..Dn) were computed on the historical loan-episode
population, over a different time window than the live Decision Log/
engine snapshot. Decile RANK does not necessarily correspond across the
two populations -- "D3" in 2026 H1 loan history is not guaranteed to
cover the same UGX range as "D3" of today's live snapshot. This script
therefore does NOT match live agents to historical bands by rank. It
re-derives each historical band's own UGX RANGE (the qcut bin edges,
captured from the same episode dataset that produced the frozen
frontier) and places live agents into that UGX range directly --
comparing like-for-like business scale, not like-for-like decile rank.
Live agents whose fundamental falls outside every historical band's
observed range are counted and reported separately, never silently
included in the nearest band or dropped.

REPRODUCIBILITY CHECK (lighter than compute_pd_comparable_capacity_
frontier.py's hard-exit reproduction gate, since this script re-derives
only the qcut BAND ASSIGNMENT, not the frontier's own classification
arithmetic): if the number of bands this script derives from
--episode-dataset does not match the number of distinct `band` values in
the provided frontier summary CSV, this is a loud WARNING (not a hard
exit) that --episode-dataset is probably not the exact file version that
produced the frozen frontier -- the overlay can still be read, but the
band-to-UGX-range mapping should not be trusted without checking this.

Usage:
    python scripts\\overlay_capacity_f_vs_observed_exposure_frontier.py ^
        --episode-dataset loan_episode_capacity_dataset.csv ^
        --float-frontier-summary-csv output\\capacity_frontier_float_frontier_summary.csv ^
        --commission-frontier-summary-csv output\\capacity_frontier_commission_frontier_summary.csv ^
        --engine-output output\\engine_test_output_stage4c_shadow.csv ^
        --decision-log-file output\\stage_4c_shadow\\stage_4c_decision_log_<run_id>.csv ^
        --out-prefix output\\stage_4c_shadow\\capacity_f_vs_observed_frontier
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_DECILES_DEFAULT = 10

# Same "90-110% = agreement" convention used throughout this repo, applied
# here to ratio = (candidate estimate) / (observed frontier tier).
R_BAND_EDGES = [-0.01, 0.90, 1.10, np.inf]
R_BAND_LABELS = ["UNDERSHOOTS", "TRACKS", "EXCEEDS"]


def _norm_msisdn(s: pd.Series) -> pd.Series:
    """Restated from compare_capacity_f_vs_incumbent_cap_layers.py (strips
    a trailing '.0' before stripping non-digits -- avoids the float-dtype
    mis-normalization found elsewhere in this session)."""
    out = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    out = out.str.replace(r"[^0-9]", "", regex=True)
    return out.mask(out.str.len() == 0)


def _qcut_safe_with_edges(s: pd.Series, q: int, prefix: str = "D") -> tuple:
    """Restated from derive_capacity_frontier_from_business_state.py's
    _qcut_safe, EXTENDED to also return the numeric bin edges (retbins) --
    the one piece of information the original function computes via
    pd.qcut but never persists anywhere downstream. Deterministic: same
    column + same q on the same data reproduces the same bands as the
    original frontier-deriving run, with no risk of "drift" the way a
    re-derived outcome statistic could drift -- this is why the
    reproducibility check above is a warning, not a hard exit."""
    try:
        codes, bins = pd.qcut(s, q, duplicates="drop", retbins=True, labels=False)
    except ValueError:
        codes, bins = None, None
    n_bins = (len(bins) - 1) if bins is not None else 0
    if n_bins <= 0:
        return pd.Series([f"{prefix}1"] * len(s), index=s.index), [f"{prefix}1"], np.array([-np.inf, np.inf])
    labels = [f"{prefix}{i + 1}" for i in range(n_bins)]
    band_series = codes.map(dict(enumerate(labels)))
    return band_series, labels, np.asarray(bins, dtype="float64")


def build_agent_period_summary(episode_df: pd.DataFrame) -> pd.DataFrame:
    """Restated verbatim (subset) from derive_capacity_frontier_from_
    business_state.py: one row per (agent_msisdn, fundamentals_snapshot_date)
    unit, float_activity_value_1m/commission via .first() (constant within
    a unit by construction)."""
    required = {"agent_msisdn", "fundamentals_snapshot_date", "float_activity_value_1m", "commission"}
    missing = required - set(episode_df.columns)
    if missing:
        sys.exit(f"ERROR: --episode-dataset is missing required column(s) {missing}.")
    key = ["agent_msisdn", "fundamentals_snapshot_date"]
    gb = episode_df.groupby(key, sort=False)
    return pd.concat([
        gb["float_activity_value_1m"].first().rename("float_activity_value_1m"),
        gb["commission"].first().rename("commission"),
    ], axis=1).reset_index()


def derive_band_edges(agent_period_summary: pd.DataFrame, fund_col: str, n_bands: int = N_DECILES_DEFAULT) -> tuple:
    """Restated assign_bands(), extended to also return the qcut bin edges.
    Returns (band_labels, edges_array) -- edges_array has len(band_labels)+1
    entries, bin i covers (edges[i], edges[i+1]] (edges[0] inclusive via
    include_lowest), matching pd.qcut's own right-closed default."""
    valid_mask = agent_period_summary[fund_col].notna() & (agent_period_summary[fund_col] > 0)
    if not valid_mask.any():
        return [], np.array([])
    _, labels, edges = _qcut_safe_with_edges(agent_period_summary.loc[valid_mask, fund_col], n_bands)
    return labels, edges


def load_frontier_summary(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"status", "band", "tolerance_pp", "contiguous_supported_frontier_tier_ugx",
                "highest_tier_with_any_supported_evidence_ugx"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s) {missing} -- is this a "
                  f"*_frontier_summary.csv from derive_capacity_frontier_from_business_state.py "
                  f"or compute_pd_comparable_capacity_frontier.py?")
    return df


def load_engine_output(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"msisdn", "agent_category", "capacity_cap"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(
            f"ERROR: {path} is missing required column(s) {missing}. capacity_cap is an intermediate "
            f"engine column, only present if run_credit_risk_pipeline.py was invoked with "
            f"--keep-intermediate (see compare_capacity_f_vs_incumbent_cap_layers.py's own note on this)."
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


def build_live_population(eng_df: pd.DataFrame, dlog_df: pd.DataFrame) -> pd.DataFrame:
    dlog_cols = ["capacity_shadow_k0_05", "capacity_estimable", "float_activity_value_1m", "commission"]
    keep_dlog = dlog_df[["_key"] + dlog_cols].drop_duplicates(subset="_key", keep="first")

    # engine_output.csv (with --keep-intermediate) can ALSO carry a raw "commission"
    # column of its own (used elsewhere in live feature construction) -- if both
    # sides of the merge have a same-named column, pandas silently renames BOTH to
    # commission_x/commission_y rather than keeping a plain "commission", which
    # breaks every downstream df["commission"] lookup with a KeyError (found on
    # real data). Drop any such overlapping column from eng_df first so the
    # Decision Log's own value -- what Capacity(F) was actually computed from --
    # is unambiguously the one used, never silently shadowed or suffixed away.
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


def classify_ratio(r: pd.Series) -> pd.Series:
    return pd.cut(r, bins=R_BAND_EDGES, labels=R_BAND_LABELS, right=False)


def build_overlay_table(fund_col: str, episode_df: pd.DataFrame, frontier_summary: pd.DataFrame,
                         live_df: pd.DataFrame, tolerance_pp: float, n_bands: int = N_DECILES_DEFAULT) -> pd.DataFrame:
    agent_period_summary = build_agent_period_summary(episode_df)
    band_labels, edges = derive_band_edges(agent_period_summary, fund_col, n_bands)
    if not band_labels:
        print(f"NOTE: no valid {fund_col} values in --episode-dataset -- cannot derive bands.")
        return pd.DataFrame()

    n_bands_frontier = frontier_summary["band"].nunique()
    if len(band_labels) != n_bands_frontier:
        print(f"WARNING: derived {len(band_labels)} bands from --episode-dataset for {fund_col}, but the "
              f"frontier summary CSV has {n_bands_frontier} distinct band(s) -- --episode-dataset is "
              f"probably not the exact file version that produced that frontier. The band-to-UGX-range "
              f"mapping below should not be trusted without checking this.")

    live_vals = pd.to_numeric(live_df[fund_col], errors="coerce")
    live_band = pd.cut(live_vals, bins=edges, labels=band_labels, right=True, include_lowest=True)
    n_outside = int((live_band.isna() & live_vals.notna()).sum())
    n_missing_fund = int(live_vals.isna().sum())
    print(f"{fund_col}: {n_outside:,} live agent(s) fall outside every historically-observed band range "
          f"(not assigned to any band); {n_missing_fund:,} have no {fund_col} value at all.")

    tol_rows = frontier_summary[(frontier_summary["tolerance_pp"] == tolerance_pp)
                                 & (frontier_summary["status"] == "ok")].set_index("band")

    cap = pd.to_numeric(live_df["capacity_shadow_k0_05"], errors="coerce")
    cap_cap = pd.to_numeric(live_df["capacity_cap"], errors="coerce")
    estimable = live_df["capacity_estimable"] == True  # noqa: E712

    rows = []
    for i, label in enumerate(band_labels):
        lo = edges[i] if i > 0 else np.nan
        hi = edges[i + 1]
        in_band = live_band == label
        n_live = int(in_band.sum())
        cap_in_band = cap[in_band & estimable].dropna()
        cap_cap_in_band = cap_cap[in_band].dropna()

        if label not in tol_rows.index:
            rows.append({
                "band": label, "ugx_range_lo": lo, "ugx_range_hi": hi, "n_live_agents": n_live,
                "observed_frontier_tier_ugx": np.nan, "highest_any_supported_ugx": np.nan,
                "median_capacity_f": np.nan, "median_capacity_cap": np.nan,
                "ratio_capacity_f_vs_frontier": np.nan, "ratio_capacity_cap_vs_frontier": np.nan,
                "capacity_f_classification": "no_frontier_evidence_at_this_tolerance",
                "capacity_cap_classification": "no_frontier_evidence_at_this_tolerance",
            })
            continue

        frontier_ugx = float(tol_rows.loc[label, "contiguous_supported_frontier_tier_ugx"])
        highest_any_ugx = float(tol_rows.loc[label, "highest_tier_with_any_supported_evidence_ugx"])
        median_cap = float(cap_in_band.median()) if len(cap_in_band) else np.nan
        median_cap_cap = float(cap_cap_in_band.median()) if len(cap_cap_in_band) else np.nan
        ratio_cap = median_cap / frontier_ugx if frontier_ugx and not np.isnan(median_cap) else np.nan
        ratio_cap_cap = median_cap_cap / frontier_ugx if frontier_ugx and not np.isnan(median_cap_cap) else np.nan

        rows.append({
            "band": label, "ugx_range_lo": lo, "ugx_range_hi": hi, "n_live_agents": n_live,
            "observed_frontier_tier_ugx": frontier_ugx, "highest_any_supported_ugx": highest_any_ugx,
            "median_capacity_f": median_cap, "median_capacity_cap": median_cap_cap,
            "ratio_capacity_f_vs_frontier": round(ratio_cap, 4) if not np.isnan(ratio_cap) else np.nan,
            "ratio_capacity_cap_vs_frontier": round(ratio_cap_cap, 4) if not np.isnan(ratio_cap_cap) else np.nan,
            "capacity_f_classification": str(classify_ratio(pd.Series([ratio_cap])).iloc[0])
            if not np.isnan(ratio_cap) else "undefined",
            "capacity_cap_classification": str(classify_ratio(pd.Series([ratio_cap_cap])).iloc[0])
            if not np.isnan(ratio_cap_cap) else "undefined",
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dataset", required=True,
                     help="loan_episode_capacity_dataset.csv -- MUST be the exact file version that "
                          "produced the frontier summary CSV(s) below, or the band-to-UGX mapping is wrong")
    ap.add_argument("--float-frontier-summary-csv", default=None)
    ap.add_argument("--commission-frontier-summary-csv", default=None)
    ap.add_argument("--engine-output", required=True)
    ap.add_argument("--decision-log-file", required=True)
    ap.add_argument("--tolerances", default=None,
                     help="comma-separated tolerance_pp values to overlay (default: every distinct value "
                          "found in the frontier summary CSV -- never collapsed to a single 'the' tolerance)")
    ap.add_argument("--n-bands", type=int, default=N_DECILES_DEFAULT)
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args(argv)

    if not args.float_frontier_summary_csv and not args.commission_frontier_summary_csv:
        sys.exit("ERROR: provide at least one of --float-frontier-summary-csv / --commission-frontier-summary-csv.")

    print("=" * 78)
    print("Capacity(F) vs. incumbent capacity_cap vs. the OBSERVED historically-")
    print("supported exposure frontier -- three curves over business scale")
    print("=" * 78)
    print("Diagnostic of the already-frozen k=0.05 Capacity(F) challenger and Analysis 4's "
          "already-frozen frontier. Purely observational -- does not change k, Capacity(F), "
          "capacity_cap, or C3. Does not establish causal safe capacity; the frontier itself "
          "identifies historical support, not a causal boundary (see docs/analysis_4_findings.md).\n")

    episode_df = pd.read_csv(args.episode_dataset)
    eng = load_engine_output(args.engine_output)
    dlog = load_decision_log(args.decision_log_file)
    live = build_live_population(eng, dlog)

    for fund_col, frontier_path in [("float_activity_value_1m", args.float_frontier_summary_csv),
                                     ("commission", args.commission_frontier_summary_csv)]:
        if frontier_path is None:
            continue
        frontier_summary = load_frontier_summary(frontier_path)
        tolerances = (
            [float(t) for t in args.tolerances.split(",")]
            if args.tolerances else sorted(frontier_summary["tolerance_pp"].unique())
        )
        for tol in tolerances:
            print(f"\n{'=' * 78}")
            print(f"Overlay -- {fund_col}, tolerance={tol}pp")
            print("=" * 78)
            table = build_overlay_table(fund_col, episode_df, frontier_summary, live, tol, args.n_bands)
            if table.empty:
                print("  (no overlay rows -- see NOTE/WARNING above)")
                continue
            with pd.option_context("display.max_rows", None, "display.width", 200,
                                    "display.float_format", "{:.4f}".format):
                print(table.to_string(index=False))
            if args.out_prefix:
                table.to_csv(f"{args.out_prefix}_{fund_col}_tol{tol}.csv", index=False)

    if args.out_prefix:
        print(f"\nCSV outputs written with prefix: {args.out_prefix}")


if __name__ == "__main__":
    main()
