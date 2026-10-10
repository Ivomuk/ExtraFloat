"""
derive_capacity_function_and_backtest_frontier.py
======================================================
Deliverable 2, Stages 1-3 (part 2 of 2 new scripts). Computes a
fundamentals-only diagnostic score S(F) and a fundamentals-only UGX
Capacity(F) estimate per agent-period, then backtests Capacity(F) against
Analysis 4's historically-supported exposure frontiers -- as a GUARDRAIL,
never as a fitting target.

NOT A CONTINUATION of scripts/fit_capacity_challenger_model.py or
scripts/analyze_capacity_dimension_redundancy.py (an earlier, abandoned
lineage that fits log(actual_exposure_ugx) -- exactly the endogenous
historical-assignment information this script is designed to avoid).

REQUIRED_COLS below is the ENTIRE usecols whitelist for the episode-
dataset read -- no outcome, exposure, assigned_limit, or cal_pd column is
EVER named anywhere this script reads from loan_episode_capacity_
dataset.csv. The backtest instead joins against Analysis 4's already-
aggregated, band-level `*_frontier_summary.csv` outputs (produced by
scripts/derive_capacity_frontier_from_business_state.py) -- a stronger
structural guarantee than "promises not to use column X from a wider
dataframe": the forbidden information is never in the schema this script
can read at all.

THE FORMULA:
    Capacity(F) = k * ( exp(w_F*log1p(Float) + w_C*log1p(Commission)) - 1 )
                = k * ( (1+Float)^w_F * (1+Commission)^w_C - 1 )
i.e. k times a weighted GEOMETRIC MEAN of (1+Float) and (1+Commission),
minus 1 (log-space implementation for numerical stability at real UGX
scale). A constant-returns-to-scale ("Cobb-Douglas") combination -- a
standard functional family for combining partially-substitutable,
comonotonic inputs. Properties, stated precisely (not oversold):
  - APPROXIMATELY scale-homogeneous in the currency unit at economically
    relevant UGX magnitudes (float activity and commission routinely in
    the tens-of-thousands-to-millions range) -- NOT exactly homogeneous
    of degree 1, because of the +1 regularizer: (1+aF)^w_F*(1+aC)^w_C-1
    is not exactly a*((1+F)^w_F*(1+C)^w_C-1) for a general rescale
    factor a. The +1 regularization is load-bearing, not cosmetic: it
    prevents a zero value in EITHER fundamental from mechanically
    forcing Capacity to zero (an unwanted "AND gate" at the origin),
    the same role log1p already plays elsewhere in this workstream.
  - Monotonic by construction, in a STRONGER form than required: jointly
    monotonic in each raw fundamental (d Capacity/d Float > 0 for any
    w_F>0, k>0, symmetric for Commission), not merely in a derived index.
  - The only free parameter is k -- a single conservatism constant,
    chosen by POLICY JUDGMENT informed by (never fit to) the backtest
    below, exactly analogous to C3's r_floor/m_min/m_max.
  - Linear in k: backtest ratios at any k equal k times the ratios at
    k=1 -- this script always reports capacity_ugx_at_k1 so every
    candidate k can be evaluated by arithmetic on the output CSVs,
    never by rerunning this script.

A REAL MODELING CHOICE, flagged rather than buried: the geometric mean
lets a large Float compensate for a small Commission (and vice versa)
smoothly in log-space (elasticity of substitution = 1). Analysis 4 found
8.4% of agent-periods separated by >=3 deciles between float and
commission (docs/analysis_4_findings.md, Section 5) -- this script
reports a concordant-vs-discordant split specifically so this
substitution assumption's behavior on that tail is visible, not averaged
away. If that split shows concordant ratios comfortable but discordant
ratios persistently high across every candidate k, that is evidence to
revisit the geometric-mean substitution assumption itself -- not to pick
a smaller k and move on.

GUARDRAIL DISCIPLINE, stated loudly because it is the whole point of this
script: Capacity(F) is NEVER refit to reproduce the Analysis 4 frontier.
That would reintroduce the historical-assignment endogeneity Analysis 3/4
exist to exclude. The backtest (including the k-sensitivity table) is a
DECISION AID for a human policy choice of k, informed by risk appetite
and business judgment -- this script never selects a k itself.

BAND CAVEAT: qcut decile boundaries are population-dependent, not fixed
UGX thresholds. This script's band assignment is only valid against a
frontier summary CSV derived from the IDENTICAL --episode-dataset file
used to produce it -- mixing snapshots silently misattributes
agent-periods to the wrong historical band.

Usage:
    python scripts\\derive_capacity_function_and_backtest_frontier.py ^
        --episode-dataset loan_episode_capacity_dataset.csv ^
        --weights-file capacity_artifacts\\capacity_score_weights.json ^
        --float-frontier-summary-csv capacity_frontier_float_frontier_summary.csv ^
        --commission-frontier-summary-csv capacity_frontier_commission_frontier_summary.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPOSURE_TIERS_UGX = [50_000, 100_000, 250_000, 350_000, 500_000, 750_000, 1_000_000]
N_DECILES_DEFAULT = 10
DEFAULT_TOLERANCES_PP = [2.0, 3.0, 4.0, 5.0]
FUNDAMENTALS_FOR_SCORE = ["float_activity_value_1m", "commission"]
DEFAULT_K = 1.0
DEFAULT_WEIGHTS = {"float_activity_value_1m": 0.5, "commission": 0.5}
DEFAULT_K_SENSITIVITY_GRID = [0.10, 0.15, 0.20, 0.25, 0.30]
DISCORDANCE_DECILE_THRESHOLD = 3

REQUIRED_COLS = ["agent_msisdn", "fundamentals_snapshot_date", "float_activity_value_1m", "commission"]


def _qcut_safe(s: pd.Series, q: int, prefix: str = "D") -> tuple:
    """Restated verbatim from derive_capacity_frontier_from_business_state.py
    / check_float_commission_band_overlap.py (one-way scripts/ layering
    convention). Used ONLY as backtest join machinery below -- Capacity(F)
    itself stays a continuous function of the raw fundamentals throughout;
    bands exist solely to look up the right row of the frontier CSV. This
    is NOT turning Analysis 4's descriptive bands into a hand-built policy
    table -- the production formula never reads a band label."""
    try:
        codes, bins = pd.qcut(s, q, duplicates="drop", retbins=True, labels=False)
    except ValueError:
        codes, bins = None, None
    n_bins = (len(bins) - 1) if bins is not None else 0
    if n_bins <= 0:
        return pd.Series(f"{prefix}1", index=s.index), 1
    labels = [f"{prefix}{i + 1}" for i in range(n_bins)]
    return codes.map(dict(enumerate(labels))), n_bins


def load_agent_period_fundamentals(episode_dataset_path: Path) -> pd.DataFrame:
    """usecols-whitelisted read (REQUIRED_COLS only), deduped to one row
    per (agent_msisdn, fundamentals_snapshot_date) agent-period unit."""
    key = ["agent_msisdn", "fundamentals_snapshot_date"]
    df = pd.read_csv(episode_dataset_path, usecols=lambda c: c in set(REQUIRED_COLS))
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: episode dataset is missing required column(s): {missing}")
    df = df[df["fundamentals_snapshot_date"].notna()]
    return df.groupby(key, sort=False)[["float_activity_value_1m", "commission"]].first().reset_index()


def assign_bands(df: pd.DataFrame, fund_col: str, n_bands: int = N_DECILES_DEFAULT) -> tuple:
    """Restated verbatim (join machinery only, see module docstring)."""
    valid_mask = df[fund_col].notna() & (df[fund_col] > 0)
    out = pd.Series(np.nan, index=df.index, dtype=object)
    if not valid_mask.any():
        return out, []
    band_series, n_bins = _qcut_safe(df.loc[valid_mask, fund_col], n_bands)
    labels = [f"D{i + 1}" for i in range(n_bins)]
    out.loc[valid_mask] = band_series.values
    return out, labels


def load_weights(weights_file: Path | None, fundamentals: list[str]) -> dict:
    """Reads derive_capacity_score_weights.py's recommended_weights if
    given, else falls back to DEFAULT_WEIGHTS with a printed warning.
    Validates sum-to-1, renormalizes defensively with a loud warning if
    not -- never silently wrong."""
    if weights_file is None:
        print(f"WARNING: no --weights-file given -- using DEFAULT_WEIGHTS {DEFAULT_WEIGHTS}. "
              f"Run derive_capacity_score_weights.py first to produce a governed artifact.")
        weights = dict(DEFAULT_WEIGHTS)
    else:
        import json
        meta = json.loads(Path(weights_file).read_text())
        weights = dict(meta["recommended_weights"])
        missing = [c for c in fundamentals if c not in weights]
        if missing:
            sys.exit(f"ERROR: --weights-file has no weight for {missing}.")
        weights = {c: weights[c] for c in fundamentals}

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        print(f"WARNING: weights sum to {total}, not 1.0 -- renormalizing.")
        weights = {c: w / total for c, w in weights.items()}
    return weights


def compute_diagnostic_score(df: pd.DataFrame, cols: list[str], weights: dict) -> pd.Series:
    """S(F) = sum_i w_i * z(log1p(x_i)) -- the Stage-1 diagnostic score,
    reported in its own right, NEVER fed back into compute_capacity_ugx."""
    valid_mask = pd.Series(True, index=df.index)
    for c in cols:
        valid_mask &= df[c].notna() & (df[c] > 0)
    score = pd.Series(np.nan, index=df.index)
    log1p_df = np.log1p(df.loc[valid_mask, cols])
    z = (log1p_df - log1p_df.mean()) / log1p_df.std().replace(0, np.nan)
    score.loc[valid_mask] = sum(weights[c] * z[c].fillna(0) for c in cols)
    return score


def compute_capacity_ugx(df: pd.DataFrame, cols: list[str], weights: dict, k: float) -> pd.Series:
    """Capacity(F) = k * (exp(sum_i w_i*log1p(x_i)) - 1). Implemented in
    log-space for numerical stability at real UGX scale rather than
    chaining literal (1+x)**w products."""
    valid_mask = pd.Series(True, index=df.index)
    for c in cols:
        valid_mask &= df[c].notna() & (df[c] >= 0)
    out = pd.Series(np.nan, index=df.index)
    log1p_df = np.log1p(df.loc[valid_mask, cols])
    s_log = sum(weights[c] * log1p_df[c] for c in cols)
    out.loc[valid_mask] = k * (np.exp(s_log) - 1.0)
    return out


def monotonicity_sweep_check(cols: list[str], weights: dict, k: float, grid: np.ndarray) -> pd.DataFrame:
    """Explicit synthetic sweep: for each axis, hold the other fixed at
    several representative grid values and vary the swept axis over
    `grid`; used both as a unit-test fixture and an optional real-data
    diagnostic over the observed min/max of each fundamental."""
    rows = []
    for fixed_col in cols:
        swept_col = [c for c in cols if c != fixed_col][0]
        for fixed_val in grid:
            df = pd.DataFrame({fixed_col: fixed_val, swept_col: grid})
            df = df[cols]
            capacity = compute_capacity_ugx(df, cols, weights, k)
            rows.append({"fixed_col": fixed_col, "fixed_val": fixed_val, "swept_col": swept_col,
                         "capacity_values": capacity.tolist(), "is_nondecreasing": bool(np.all(np.diff(capacity) >= -1e-6))})
    return pd.DataFrame(rows)


def build_band_frontier_lookup(frontier_summary_csv: Path) -> pd.DataFrame:
    """Reads one of Analysis 4's {out_prefix}_{fund_name}_frontier_
    summary.csv files, keeps status=='ok' rows."""
    df = pd.read_csv(frontier_summary_csv)
    df = df[df["status"] == "ok"]
    return df[["band", "tolerance_pp", "contiguous_supported_frontier_tier_ugx",
               "highest_tier_with_any_supported_evidence_ugx"]]


def backtest_against_frontier(scored_df: pd.DataFrame, band_col: str,
                               frontier_lookup: pd.DataFrame, tolerances: list[float]) -> pd.DataFrame:
    """Long format: one row per (agent-period, tolerance_pp). Computes
    BOTH ratios every time, never just one: ratio_vs_contiguous_frontier
    (the PRIMARY governance guardrail) and ratio_vs_highest_any_frontier
    (a secondary diagnostic). Never collapsed across tolerances, matching
    Deliverable 1's sweep discipline."""
    base = scored_df[["agent_msisdn", "fundamentals_snapshot_date", band_col, "capacity_ugx_at_k1",
                       "float_band_rank", "commission_band_rank"]].dropna(subset=[band_col])
    rows = []
    for tol in tolerances:
        tol_lookup = frontier_lookup[frontier_lookup["tolerance_pp"] == tol]
        merged = base.merge(tol_lookup, left_on=band_col, right_on="band", how="inner")
        merged["tolerance_pp"] = tol
        merged["ratio_vs_contiguous_frontier"] = (
            merged["capacity_ugx_at_k1"] / merged["contiguous_supported_frontier_tier_ugx"])
        merged["ratio_vs_highest_any_frontier"] = (
            merged["capacity_ugx_at_k1"] / merged["highest_tier_with_any_supported_evidence_ugx"])
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out[["agent_msisdn", "fundamentals_snapshot_date", band_col, "tolerance_pp", "capacity_ugx_at_k1",
                "ratio_vs_contiguous_frontier", "ratio_vs_highest_any_frontier",
                "float_band_rank", "commission_band_rank"]]


def _discordance_label(row) -> str:
    if pd.isna(row["float_band_rank"]) or pd.isna(row["commission_band_rank"]):
        return "unknown"
    diff = abs(row["float_band_rank"] - row["commission_band_rank"])
    if diff <= 1:
        return "concordant"
    if diff >= DISCORDANCE_DECILE_THRESHOLD:
        return "discordant"
    return "middle"


def summarize_backtest_by_band(backtest_df: pd.DataFrame, band_col: str) -> pd.DataFrame:
    """Per band x tolerance: BOTH ratios reported side by side, contiguous
    first/primary."""
    if backtest_df.empty:
        return pd.DataFrame()
    rows = []
    for (band, tol), g in backtest_df.groupby([band_col, "tolerance_pp"]):
        rc, ra = g["ratio_vs_contiguous_frontier"], g["ratio_vs_highest_any_frontier"]
        rows.append({
            "band": band, "tolerance_pp": tol, "n_units": len(g),
            "median_ratio_vs_contiguous": rc.median(), "p25_vs_contiguous": rc.quantile(0.25),
            "p75_vs_contiguous": rc.quantile(0.75), "pct_aggressive_vs_contiguous": (rc > 1).mean() * 100,
            "median_ratio_vs_highest_any": ra.median(), "p25_vs_highest_any": ra.quantile(0.25),
            "p75_vs_highest_any": ra.quantile(0.75), "pct_aggressive_vs_highest_any": (ra > 1).mean() * 100,
        })
    return pd.DataFrame(rows).sort_values(["band", "tolerance_pp"]).reset_index(drop=True)


def summarize_backtest_by_concordance(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """Concordant (|float_rank-commission_rank|<=1) vs. discordant (>=3)
    split -- the Cobb-Douglas substitution assumption's stress test."""
    if backtest_df.empty:
        return pd.DataFrame()
    df = backtest_df.copy()
    df["concordance"] = df.apply(_discordance_label, axis=1)
    rows = []
    for (concordance, tol), g in df.groupby(["concordance", "tolerance_pp"]):
        rc, ra = g["ratio_vs_contiguous_frontier"], g["ratio_vs_highest_any_frontier"]
        rows.append({
            "concordance": concordance, "tolerance_pp": tol, "n_units": len(g),
            "median_ratio_vs_contiguous": rc.median(), "pct_aggressive_vs_contiguous": (rc > 1).mean() * 100,
            "median_ratio_vs_highest_any": ra.median(), "pct_aggressive_vs_highest_any": (ra > 1).mean() * 100,
        })
    return pd.DataFrame(rows).sort_values(["concordance", "tolerance_pp"]).reset_index(drop=True)


def build_k_sensitivity_table(backtest_df: pd.DataFrame, k_grid: list[float]) -> pd.DataFrame:
    """Pure arithmetic on the already-computed k=1 ratios (R(k) = k*R(1)):
    NEVER a re-run, NEVER a search for an 'optimal' k. This is a decision
    aid for a human policy choice, informed by risk appetite and business
    judgment -- the code never selects a k itself."""
    if backtest_df.empty:
        return pd.DataFrame()
    df = backtest_df.copy()
    df["concordance"] = df.apply(_discordance_label, axis=1)
    rows = []
    for k in k_grid:
        rc_k, ra_k = df["ratio_vs_contiguous_frontier"] * k, df["ratio_vs_highest_any_frontier"] * k
        concordant = df["concordance"] == "concordant"
        discordant = df["concordance"] == "discordant"
        rows.append({
            "k": k,
            "pct_exceeding_contiguous": (rc_k > 1).mean() * 100,
            "pct_exceeding_highest_any": (ra_k > 1).mean() * 100,
            "pct_exceeding_contiguous_concordant": (rc_k[concordant] > 1).mean() * 100 if concordant.any() else np.nan,
            "pct_exceeding_contiguous_discordant": (rc_k[discordant] > 1).mean() * 100 if discordant.any() else np.nan,
            "pct_exceeding_highest_any_concordant": (ra_k[concordant] > 1).mean() * 100 if concordant.any() else np.nan,
            "pct_exceeding_highest_any_discordant": (ra_k[discordant] > 1).mean() * 100 if discordant.any() else np.nan,
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dataset", type=Path, default=Path("loan_episode_capacity_dataset.csv"))
    ap.add_argument("--weights-file", type=Path, default=None)
    ap.add_argument("--k", type=float, default=DEFAULT_K)
    ap.add_argument("--k-sensitivity-grid", type=str, default=",".join(str(k) for k in DEFAULT_K_SENSITIVITY_GRID))
    ap.add_argument("--float-frontier-summary-csv", type=Path, required=True)
    ap.add_argument("--commission-frontier-summary-csv", type=Path, required=True)
    ap.add_argument("--tolerances", type=str, default=",".join(str(t) for t in DEFAULT_TOLERANCES_PP))
    ap.add_argument("--n-bands", type=int, default=N_DECILES_DEFAULT)
    ap.add_argument("--out-prefix", type=str, default="capacity_function_backtest")
    args = ap.parse_args(argv)

    if not args.episode_dataset.exists():
        sys.exit(f"ERROR: {args.episode_dataset} not found.")
    if not args.float_frontier_summary_csv.exists():
        sys.exit(f"ERROR: {args.float_frontier_summary_csv} not found -- run "
                  f"derive_capacity_frontier_from_business_state.py first.")
    if not args.commission_frontier_summary_csv.exists():
        sys.exit(f"ERROR: {args.commission_frontier_summary_csv} not found -- run "
                  f"derive_capacity_frontier_from_business_state.py first.")

    tolerances = sorted({float(x) for x in args.tolerances.split(",") if x.strip()})
    k_grid = sorted({float(x) for x in args.k_sensitivity_grid.split(",") if x.strip()})

    print(f"\n{'#' * 100}\nBAND CAVEAT (read before interpreting any band-level number below)\n{'#' * 100}")
    print(
        "qcut decile boundaries are population-dependent, not fixed UGX thresholds. This script's\n"
        "band assignment is only valid against a frontier summary CSV derived from the IDENTICAL\n"
        "--episode-dataset file used here. Mixing snapshots silently misattributes agent-periods to\n"
        "the wrong historical band."
    )
    print(f"\n{'#' * 100}\nGUARDRAIL (read before interpreting any ratio below)\n{'#' * 100}")
    print(
        "This is a diagnostic backtest. Capacity(F) is NEVER refit to reproduce the historical\n"
        "frontier -- doing so would reintroduce the endogeneity Analysis 3/4 excluded Capacity(F)\n"
        "from using. The k-sensitivity table below is a policy decision aid, not an optimization --\n"
        "this script never selects a k itself."
    )

    weights = load_weights(args.weights_file, FUNDAMENTALS_FOR_SCORE)
    print(f"\nWeights in use: {weights}")

    df = load_agent_period_fundamentals(args.episode_dataset)
    print(f"{len(df):,} agent-period unit(s) across {df['agent_msisdn'].nunique():,} agent(s).")

    df["float_band"], float_labels = assign_bands(df, "float_activity_value_1m", args.n_bands)
    df["commission_band"], commission_labels = assign_bands(df, "commission", args.n_bands)
    df["float_band_rank"] = df["float_band"].map(lambda b: np.nan if pd.isna(b) else int(str(b)[1:]))
    df["commission_band_rank"] = df["commission_band"].map(lambda b: np.nan if pd.isna(b) else int(str(b)[1:]))

    df["score_S_diagnostic"] = compute_diagnostic_score(df, FUNDAMENTALS_FOR_SCORE, weights)
    df["capacity_ugx_at_k1"] = compute_capacity_ugx(df, FUNDAMENTALS_FOR_SCORE, weights, 1.0)
    df["capacity_ugx_at_k"] = df["capacity_ugx_at_k1"] * args.k

    print("\n-- Monotonicity sweep (real-data diagnostic over observed min/max) --")
    obs_grid = np.linspace(0, float(df[FUNDAMENTALS_FOR_SCORE].max().max()), 6)
    sweep = monotonicity_sweep_check(FUNDAMENTALS_FOR_SCORE, weights, 1.0, obs_grid)
    print(f"  all sweeps non-decreasing: {bool(sweep['is_nondecreasing'].all())}")

    scores_path = f"{args.out_prefix}_scores.csv"
    df[["agent_msisdn", "fundamentals_snapshot_date", "float_activity_value_1m", "commission",
        "float_band", "commission_band", "score_S_diagnostic", "capacity_ugx_at_k1",
        "capacity_ugx_at_k"]].to_csv(scores_path, index=False)
    print(f"\nWrote {scores_path}")

    for band_col, frontier_csv, fund_name in [
        ("float_band", args.float_frontier_summary_csv, "float"),
        ("commission_band", args.commission_frontier_summary_csv, "commission"),
    ]:
        print(f"\n{'#' * 100}\n# Backtest against the {fund_name} frontier\n{'#' * 100}")
        frontier_lookup = build_band_frontier_lookup(frontier_csv)
        backtest_df = backtest_against_frontier(df, band_col, frontier_lookup, tolerances)
        summary_df = summarize_backtest_by_band(backtest_df, band_col)
        concordance_df = summarize_backtest_by_concordance(backtest_df)
        k_sensitivity_df = build_k_sensitivity_table(backtest_df, k_grid)

        with pd.option_context("display.width", 180, "display.float_format", "{:.3f}".format):
            print("-- Summary by band x tolerance --")
            print(summary_df.to_string(index=False))
            print("\n-- Summary by concordance x tolerance --")
            print(concordance_df.to_string(index=False))
            print("\n-- k-sensitivity table (policy decision aid, not an optimization) --")
            print(k_sensitivity_df.to_string(index=False))

        backtest_df.to_csv(f"{args.out_prefix}_backtest_{fund_name}.csv", index=False)
        summary_df.to_csv(f"{args.out_prefix}_backtest_summary_{fund_name}.csv", index=False)
        concordance_df.to_csv(f"{args.out_prefix}_backtest_by_concordance_{fund_name}.csv", index=False)
        k_sensitivity_df.to_csv(f"{args.out_prefix}_k_sensitivity_{fund_name}.csv", index=False)

    print(f"\n{'#' * 100}\nCausal caveat (printed every run)\n{'#' * 100}")
    print(
        "This is observational, not causal evidence. Capacity(F) is a fundamentals-only estimate;\n"
        "the backtest ratios describe how it compares to historically-supported exposure, not\n"
        "whether raising a real agent's limit to that level would be safe. Selection into exposure\n"
        "tier, lender information, and policy changes all remain live explanations throughout."
    )


if __name__ == "__main__":
    main()
