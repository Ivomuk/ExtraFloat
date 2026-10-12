"""
diagnose_capacity_score_tail_composition.py
=============================================
Stage 4B specification diagnostic. Neither a backtest nor another
outcome/consequence analysis -- a pure inspection of the RAW Float and
Commission fundamentals behind the two tails characterize_capacity_
candidate_k_distribution.py already surfaced:
  - LOW tail: capacity_ugx_at_k < --low-tail-threshold-ugx (default
    50,000, the lowest exposure tier) for each candidate k. Real-data
    finding this follows up on: Float D1's p75 stays below 50,000 across
    the ENTIRE 0.03-0.06 candidate range.
  - HIGH tail: capacity_ugx_at_k > --high-tail-threshold-ugx (default
    1,000,000, the highest exposure tier) for each candidate k, plus the
    k-INDEPENDENT raw score (capacity_ugx_at_k1) at p99/p99.5/p99.9.
    Real-data finding this follows up on: the population max reaches
    37.3m-74.6m UGX across the candidate range -- at k=.06 that implies
    an unscaled score of roughly 75m/.06 ~= 1.25 BILLION UGX, large
    enough to want to see the underlying Float/Commission before treating
    it as something to cap.

THE QUESTION: are these tails legitimate business fundamentals, or
artifacts of measurement/unit/skew? This script answers that by showing,
for both tails: Float, Commission, their Float/Commission ratio, each
variable's OWN empirical percentile rank within the population, and
whether one variable or both are driving the extreme value -- never by
computing an outcome, a PD, a C3 multiplier, or fitting anything.

NO OUTCOMES, PD, C3, OR ASSIGNED_LIMIT: REQUIRED_COLS below is the entire
usecols whitelist for the episode-dataset read -- restates (one-way
scripts/ layering convention, never imports) the same usecols-whitelisted
read, weight loading, and Capacity(F) formula as the sibling scripts.

CAPACITY_NOT_ESTIMABLE rows (float_activity_value_1m<=0 or commission<=0)
are excluded from BOTH tails entirely -- they are a categorically
different population (missing usable fundamentals, not a measured low
capacity) and are reported separately, never folded into the low tail.

TAIL-DRIVER CLASSIFICATION, explicit and documented (never hidden inside
a single opaque score): each row's float_percentile_rank and commission_
percentile_rank are its own empirical rank (0-100) within the ESTIMABLE
population. For the low tail: "both_low" if both ranks <=50;
"float_dominates_low" if float_percentile_rank<=LOW_DOMINANCE_PCT while
commission_percentile_rank>LOW_OTHER_MIN_PCT (float is pulling the
geometric mean down despite a decent Commission); "commission_dominates_
low" symmetric; else "other_combination". For the high tail, symmetric
with HIGH_DOMINANCE_PCT/HIGH_OTHER_MAX_PCT. These thresholds are
descriptive bucketing choices, not policy thresholds -- the raw percentile
ranks are ALSO reported on every extreme row so a reader is never limited
to this script's own bucketing.

OUTPUT, per --out-prefix (six tables):
    {out_prefix}_capacity_not_estimable_summary.csv
        One row: n, pct_of_total for the excluded not-estimable population.
    {out_prefix}_raw_score_percentiles.csv
        One row per percentile (p99/p99.5/p99.9/p99.99/max) of the
        k-INDEPENDENT capacity_ugx_at_k1, plus the implied capacity_ugx_at_k
        for every --candidate-ks value (linear: R(k) = k*R(1), never
        recomputed from scratch).
    {out_prefix}_low_tail_composition_by_k.csv
        One row per k: n, pct_of_estimable_population, Float/Commission/
        ratio median-p25-p75, and the four driver-category counts.
    {out_prefix}_high_tail_composition_by_k.csv
        Symmetric, for capacity_ugx_at_k > --high-tail-threshold-ugx.
    {out_prefix}_low_tail_extreme_rows.csv / _high_tail_extreme_rows.csv
        Row-level: the --top-n-extreme (default 50) most extreme
        agent-periods by capacity_ugx_at_k1 (ascending for low, descending
        for high), each with agent_msisdn, fundamentals_snapshot_date,
        float_activity_value_1m, commission, float_commission_ratio,
        float_percentile_rank, commission_percentile_rank,
        capacity_ugx_at_k1, and the driver-category label.

Usage:
    python scripts\\diagnose_capacity_score_tail_composition.py ^
        --episode-dataset loan_episode_capacity_dataset.csv ^
        --weights-file capacity_artifacts\\capacity_score_weights.json ^
        --candidate-ks 0.03,0.04,0.05,0.06 ^
        --out-prefix capacity_tail_composition
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FUNDAMENTALS_FOR_SCORE = ["float_activity_value_1m", "commission"]
DEFAULT_WEIGHTS = {"float_activity_value_1m": 0.5, "commission": 0.5}
DEFAULT_CANDIDATE_KS = [0.03, 0.04, 0.05, 0.06]
DEFAULT_LOW_TAIL_THRESHOLD_UGX = 50_000
DEFAULT_HIGH_TAIL_THRESHOLD_UGX = 1_000_000
DEFAULT_TOP_N_EXTREME = 50
RAW_SCORE_PERCENTILES = [0.99, 0.995, 0.999, 0.9999]

LOW_DOMINANCE_PCT = 25.0
LOW_OTHER_MIN_PCT = 50.0
HIGH_DOMINANCE_PCT = 99.0
HIGH_OTHER_MAX_PCT = 75.0

REQUIRED_COLS = ["agent_msisdn", "fundamentals_snapshot_date", "float_activity_value_1m", "commission"]


# ============================================================================
# Restated verbatim from the sibling capacity-function scripts (one-way
# scripts/ layering convention -- never imported).
# ============================================================================


def load_agent_period_fundamentals(episode_dataset_path: Path) -> pd.DataFrame:
    """usecols-whitelisted read (REQUIRED_COLS only -- no outcome,
    exposure, assigned_limit, or cal_pd column is ever named here),
    deduped to one row per (agent_msisdn, fundamentals_snapshot_date)
    agent-period unit."""
    key = ["agent_msisdn", "fundamentals_snapshot_date"]
    df = pd.read_csv(episode_dataset_path, usecols=lambda c: c in set(REQUIRED_COLS))
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: episode dataset is missing required column(s): {missing}")
    df = df[df["fundamentals_snapshot_date"].notna()]
    return df.groupby(key, sort=False)[["float_activity_value_1m", "commission"]].first().reset_index()


def load_weights(weights_file: Path | None, fundamentals: list[str]) -> dict:
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


def compute_capacity_ugx(df: pd.DataFrame, cols: list[str], weights: dict, k: float) -> pd.Series:
    """Capacity(F) = k * (exp(sum_i w_i*log1p(x_i)) - 1)."""
    valid_mask = pd.Series(True, index=df.index)
    for c in cols:
        valid_mask &= df[c].notna() & (df[c] >= 0)
    out = pd.Series(np.nan, index=df.index)
    log1p_df = np.log1p(df.loc[valid_mask, cols])
    s_log = sum(weights[c] * log1p_df[c] for c in cols)
    out.loc[valid_mask] = k * (np.exp(s_log) - 1.0)
    return out


# ============================================================================
# New: tail composition diagnostics.
# ============================================================================


def compute_percentile_ranks(df: pd.DataFrame, cols: list[str], estimable_mask: pd.Series) -> pd.DataFrame:
    """Adds '{col}_percentile_rank' (0-100) computed via empirical rank,
    over the ESTIMABLE population only -- a percentile rank computed
    against a population that includes not-estimable rows would be
    meaningless (those rows have no genuine fundamental to rank)."""
    out = df.copy()
    for c in cols:
        out[f"{c}_percentile_rank"] = np.nan
        out.loc[estimable_mask, f"{c}_percentile_rank"] = (
            out.loc[estimable_mask, c].rank(pct=True) * 100.0
        )
    return out


def classify_low_tail_driver(float_pct: float, commission_pct: float) -> str:
    if float_pct <= LOW_DOMINANCE_PCT and commission_pct > LOW_OTHER_MIN_PCT:
        return "float_dominates_low"
    if commission_pct <= LOW_DOMINANCE_PCT and float_pct > LOW_OTHER_MIN_PCT:
        return "commission_dominates_low"
    if float_pct <= 50.0 and commission_pct <= 50.0:
        return "both_low"
    return "other_combination"


def classify_high_tail_driver(float_pct: float, commission_pct: float) -> str:
    if float_pct >= HIGH_DOMINANCE_PCT and commission_pct < HIGH_OTHER_MAX_PCT:
        return "float_dominates_high"
    if commission_pct >= HIGH_DOMINANCE_PCT and float_pct < HIGH_OTHER_MAX_PCT:
        return "commission_dominates_high"
    if float_pct >= 50.0 and commission_pct >= 50.0:
        return "both_high"
    return "other_combination"


def compute_raw_score_percentile_table(capacity_ugx_at_k1: pd.Series, percentiles: list[float],
                                        candidate_ks: list[float]) -> pd.DataFrame:
    s = capacity_ugx_at_k1.dropna()
    rows = []
    for p in percentiles:
        v1 = float(s.quantile(p))
        row = {"percentile": f"p{p * 100:.2f}".rstrip("0").rstrip("."), "capacity_ugx_at_k1": v1}
        for k in candidate_ks:
            row[f"capacity_ugx_at_k{k:.2f}"] = v1 * k
        rows.append(row)
    max_row = {"percentile": "max", "capacity_ugx_at_k1": float(s.max())}
    for k in candidate_ks:
        max_row[f"capacity_ugx_at_k{k:.2f}"] = float(s.max()) * k
    rows.append(max_row)
    return pd.DataFrame(rows)


def build_tail_composition_row(k: float, n_total_estimable: int, g: pd.DataFrame, classifier, driver_labels: list[str]) -> dict:
    n = len(g)
    driver_counts = g.apply(
        lambda r: classifier(r["float_activity_value_1m_percentile_rank"], r["commission_percentile_rank"]), axis=1
    ).value_counts().to_dict() if n else {}
    ratio = (g["float_activity_value_1m"] / g["commission"]) if n else pd.Series([], dtype=float)
    row = {
        "k": k, "n": n, "pct_of_estimable_population": (100.0 * n / n_total_estimable) if n_total_estimable else np.nan,
        "float_median": g["float_activity_value_1m"].median() if n else np.nan,
        "float_p25": g["float_activity_value_1m"].quantile(0.25) if n else np.nan,
        "float_p75": g["float_activity_value_1m"].quantile(0.75) if n else np.nan,
        "commission_median": g["commission"].median() if n else np.nan,
        "commission_p25": g["commission"].quantile(0.25) if n else np.nan,
        "commission_p75": g["commission"].quantile(0.75) if n else np.nan,
        "ratio_median": ratio.median() if n else np.nan,
        "ratio_p25": ratio.quantile(0.25) if n else np.nan,
        "ratio_p75": ratio.quantile(0.75) if n else np.nan,
    }
    for label in driver_labels:
        row[f"n_{label}"] = driver_counts.get(label, 0)
    return row


LOW_TAIL_DRIVER_LABELS = ["both_low", "float_dominates_low", "commission_dominates_low", "other_combination"]
HIGH_TAIL_DRIVER_LABELS = ["both_high", "float_dominates_high", "commission_dominates_high", "other_combination"]


def build_extreme_rows_table(df: pd.DataFrame, top_n: int, ascending: bool, classifier) -> pd.DataFrame:
    sorted_df = df.sort_values("capacity_ugx_at_k1", ascending=ascending).head(top_n).copy()
    sorted_df["float_commission_ratio"] = sorted_df["float_activity_value_1m"] / sorted_df["commission"]
    sorted_df["driver"] = sorted_df.apply(
        lambda r: classifier(r["float_activity_value_1m_percentile_rank"], r["commission_percentile_rank"]), axis=1)
    return sorted_df[[
        "agent_msisdn", "fundamentals_snapshot_date", "float_activity_value_1m", "commission",
        "float_commission_ratio", "float_activity_value_1m_percentile_rank", "commission_percentile_rank",
        "capacity_ugx_at_k1", "driver",
    ]].rename(columns={"float_activity_value_1m_percentile_rank": "float_percentile_rank"})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dataset", type=Path, default=Path("loan_episode_capacity_dataset.csv"))
    ap.add_argument("--weights-file", type=Path, default=None)
    ap.add_argument("--candidate-ks", type=str, default=",".join(str(k) for k in DEFAULT_CANDIDATE_KS))
    ap.add_argument("--low-tail-threshold-ugx", type=float, default=DEFAULT_LOW_TAIL_THRESHOLD_UGX)
    ap.add_argument("--high-tail-threshold-ugx", type=float, default=DEFAULT_HIGH_TAIL_THRESHOLD_UGX)
    ap.add_argument("--top-n-extreme", type=int, default=DEFAULT_TOP_N_EXTREME)
    ap.add_argument("--out-prefix", type=str, default="capacity_tail_composition")
    args = ap.parse_args(argv)

    if not args.episode_dataset.exists():
        sys.exit(f"ERROR: {args.episode_dataset} not found.")

    candidate_ks = sorted({float(x) for x in args.candidate_ks.split(",") if x.strip()})

    print(f"\n{'#' * 100}\nNOT AN OUTCOME ANALYSIS\n{'#' * 100}")
    print(
        "No outcomes, PD, C3, or assigned_limit are read anywhere in this script -- REQUIRED_COLS is\n"
        "the ENTIRE usecols whitelist. This is a pure inspection of the RAW Float/Commission fundamentals\n"
        "behind Capacity(F)'s low and high tails: are they legitimate business fundamentals, or\n"
        "artifacts of measurement/unit/skew? It does not select a k, apply a floor/cap, or touch C3."
    )

    weights = load_weights(args.weights_file, FUNDAMENTALS_FOR_SCORE)
    print(f"\nWeights in use: {weights}")

    df = load_agent_period_fundamentals(args.episode_dataset)
    n_total = len(df)
    print(f"{n_total:,} agent-period unit(s) across {df['agent_msisdn'].nunique():,} agent(s).")

    df["capacity_ugx_at_k1"] = compute_capacity_ugx(df, FUNDAMENTALS_FOR_SCORE, weights, 1.0)
    capacity_estimable = (df["float_activity_value_1m"] > 0) & (df["commission"] > 0)
    n_not_estimable = int((~capacity_estimable).sum())
    df.loc[~capacity_estimable, "capacity_ugx_at_k1"] = np.nan

    not_estimable_df = pd.DataFrame([{
        "n_not_estimable": n_not_estimable, "n_total": n_total,
        "pct_of_total": 100.0 * n_not_estimable / n_total if n_total else np.nan,
    }])
    print(f"\n{n_not_estimable:,} of {n_total:,} agent-period(s) "
          f"({100.0 * n_not_estimable / n_total:.2f}%) are CAPACITY_NOT_ESTIMABLE (float_activity_value_1m<=0 "
          f"or commission<=0) -- excluded entirely from both tails below, never folded into the low tail.")

    df_estimable = df[capacity_estimable].copy()
    n_estimable = len(df_estimable)
    df_estimable = compute_percentile_ranks(df_estimable, FUNDAMENTALS_FOR_SCORE, pd.Series(True, index=df_estimable.index))

    percentile_df = compute_raw_score_percentile_table(df_estimable["capacity_ugx_at_k1"], RAW_SCORE_PERCENTILES, candidate_ks)

    low_rows, high_rows = [], []
    for k in candidate_ks:
        df_estimable["capacity_ugx_at_k"] = df_estimable["capacity_ugx_at_k1"] * k
        low_g = df_estimable[df_estimable["capacity_ugx_at_k"] < args.low_tail_threshold_ugx]
        high_g = df_estimable[df_estimable["capacity_ugx_at_k"] > args.high_tail_threshold_ugx]
        low_rows.append(build_tail_composition_row(k, n_estimable, low_g, classify_low_tail_driver, LOW_TAIL_DRIVER_LABELS))
        high_rows.append(build_tail_composition_row(k, n_estimable, high_g, classify_high_tail_driver, HIGH_TAIL_DRIVER_LABELS))

    low_df = pd.DataFrame(low_rows)
    high_df = pd.DataFrame(high_rows)

    low_extreme_df = build_extreme_rows_table(df_estimable, args.top_n_extreme, ascending=True,
                                               classifier=classify_low_tail_driver)
    high_extreme_df = build_extreme_rows_table(df_estimable, args.top_n_extreme, ascending=False,
                                                classifier=classify_high_tail_driver)

    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("\n-- CAPACITY_NOT_ESTIMABLE summary --")
        print(not_estimable_df.to_string(index=False))
        print("\n-- Raw score (capacity_ugx_at_k1) percentiles, with implied capacity_ugx_at_k per candidate k --")
        print(percentile_df.to_string(index=False))
        print(f"\n-- Low tail composition (capacity_ugx_at_k < {args.low_tail_threshold_ugx:,.0f}), by k --")
        print(low_df.to_string(index=False))
        print(f"\n-- High tail composition (capacity_ugx_at_k > {args.high_tail_threshold_ugx:,.0f}), by k --")
        print(high_df.to_string(index=False))
        print(f"\n-- {args.top_n_extreme} most extreme LOW capacity_ugx_at_k1 rows --")
        print(low_extreme_df.to_string(index=False))
        print(f"\n-- {args.top_n_extreme} most extreme HIGH capacity_ugx_at_k1 rows --")
        print(high_extreme_df.to_string(index=False))

    not_estimable_df.to_csv(f"{args.out_prefix}_capacity_not_estimable_summary.csv", index=False)
    percentile_df.to_csv(f"{args.out_prefix}_raw_score_percentiles.csv", index=False)
    low_df.to_csv(f"{args.out_prefix}_low_tail_composition_by_k.csv", index=False)
    high_df.to_csv(f"{args.out_prefix}_high_tail_composition_by_k.csv", index=False)
    low_extreme_df.to_csv(f"{args.out_prefix}_low_tail_extreme_rows.csv", index=False)
    high_extreme_df.to_csv(f"{args.out_prefix}_high_tail_extreme_rows.csv", index=False)
    print(f"\nWrote {args.out_prefix}_capacity_not_estimable_summary.csv, _raw_score_percentiles.csv, "
          f"_low_tail_composition_by_k.csv, _high_tail_composition_by_k.csv, _low_tail_extreme_rows.csv, "
          f"_high_tail_extreme_rows.csv")


if __name__ == "__main__":
    main()
