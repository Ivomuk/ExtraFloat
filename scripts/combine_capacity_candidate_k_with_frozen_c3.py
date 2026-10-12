"""
combine_capacity_candidate_k_with_frozen_c3.py
=================================================
The step deliberately postponed throughout Stage 4B: combine each
candidate economic-capacity scale with the FROZEN C3 risk multiplier --

    L_full(k, scenario) = Capacity_k(F) x M_C3^retro(PD, scenario)

-- for k in {0.03 (pre-Stage-4 benchmark), 0.04, 0.05, 0.06} (the region
the capacity-side evidence narrowed to) and scenario in {base,
conservative} (Stage 4B-R4's two frozen policy scenarios). Raw
Capacity_k(F) is kept alongside L_full at every step -- this script
never collapses the two into one number, so a reader can see what C3
alone does to each candidate k's distribution.

C3 STAYS FROZEN: this script does NOT recompute, refit, or re-derive the
C3 multiplier -- it REQUIRES Stage 4B-R4's own `_scores.csv` output
(retrospective_current_c3_multiplier_base/_conservative, keyed by
disbursement_fid) as input, exactly as compute_pd_comparable_capacity_
frontier.py requires Analysis 4 Deliverable 1's output rather than
regenerating it. If that file is missing, this script hard-exits and
names apply_retrospective_shadow_c3_multiplier.py as the separate,
already-verified script to run first. k is never tuned against C3, and
C3 is never tuned against k -- each was independently frozen before this
step, which only multiplies them together.

GRAIN: Capacity_k(F) is a per-agent-period (agent_msisdn, fundamentals_
snapshot_date) quantity, by construction (it depends only on Float/
Commission, which are period-level). The C3 multiplier is a per-loan
(disbursement_fid) quantity. This script therefore operates at the LOAN
grain: every loan in a period shares that period's own Capacity_k(F)
value, broadcast via the SAME (agent_msisdn, fundamentals_snapshot_date)
key the sibling scripts already use. Loans are included only if BOTH
hurdles clear: capacity is CAPACITY_ESTIMABLE for that loan's period
(float_activity_value_1m>0 and commission>0) AND R4 produced a valid C3
multiplier for that loan (thick-file, valid retrospective_current_
cal_pd). Both exclusions are reported explicitly, never silently.

NO re-litigating R1-R4: cal_pd, route_status, and the multiplier values
themselves are taken as-is from R4's output -- this script does not
re-verify R4's own mechanical-consistency checks (already done and
frozen in that script's own validation report).

OUTPUT, per --out-prefix:
    {out_prefix}_overall_stats_by_k_scenario.csv
        One row per (k, scenario): n, raw Capacity_k(F) median/p25/p75,
        and L_full median/p25/p75, side by side.
    {out_prefix}_tier_distribution_by_k_scenario.csv
        One row per (k, scenario, stage in {raw, l_full}): n,
        pct_below_floor (< --low-tail-threshold-ugx), pct_in_range,
        pct_above_ceiling (> --high-tail-threshold-ugx) -- answers "how
        many move below 50k, remain within the product range, or remain
        above 1m after risk adjustment," with raw and L_full reported
        side by side so the C3 EFFECT is directly readable.
    {out_prefix}_transition_matrix_by_k_scenario.csv
        One row per (k, scenario, raw_tier, l_full_tier): n,
        pct_of_total -- the full 3x3 crosstab of where each loan started
        (on raw Capacity_k(F)) versus where it ended up (after C3),
        making visible exactly how many loans C3 pushes out of the
        product-supportable 50k-1m range (in either direction), not
        just the before/after marginal percentages.

Usage:
    python scripts\\combine_capacity_candidate_k_with_frozen_c3.py ^
        --episode-dataset loan_episode_capacity_dataset.csv ^
        --r4-c3-multiplier-csv stage4br4_retrospective_c3_multiplier_scores.csv ^
        --weights-file capacity_artifacts\\capacity_score_weights.json ^
        --candidate-ks 0.03,0.04,0.05,0.06 ^
        --out-prefix capacity_times_c3
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
SCENARIO_NAMES = ("base", "conservative")
CAL_COL = "retrospective_current_cal_pd"

REQUIRED_EPISODE_COLS = ["disbursement_fid", "agent_msisdn", "fundamentals_snapshot_date",
                          "float_activity_value_1m", "commission"]
REQUIRED_R4_COLS = ["disbursement_fid", CAL_COL] + [f"retrospective_current_c3_multiplier_{s}" for s in SCENARIO_NAMES]


# ============================================================================
# Restated verbatim from the sibling capacity-function scripts (one-way
# scripts/ layering convention -- never imported).
# ============================================================================


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
# New: loan-grain join of Capacity_k(F) with R4's frozen C3 multiplier.
# ============================================================================


def load_loan_level_fundamentals(episode_dataset_path: Path) -> pd.DataFrame:
    """usecols-whitelisted read (REQUIRED_EPISODE_COLS only), one row per
    LOAN (disbursement_fid) -- deliberately NOT deduped to the agent-
    period grain here, since every loan needs its own disbursement_fid
    to join against R4's output."""
    df = pd.read_csv(episode_dataset_path, usecols=lambda c: c in set(REQUIRED_EPISODE_COLS))
    missing = [c for c in REQUIRED_EPISODE_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: episode dataset is missing required column(s): {missing}")
    df = df[df["fundamentals_snapshot_date"].notna()].copy()
    dup = int(df["disbursement_fid"].duplicated().sum())
    if dup:
        sys.exit(f"ERROR: {dup} duplicate disbursement_fid value(s) in the episode dataset -- "
                  f"disbursement_fid is not behaving as a loan-level key.")
    return df


def load_r4_c3_multipliers(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: c in set(REQUIRED_R4_COLS))
    missing = [c for c in REQUIRED_R4_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing} -- this must be Stage "
                  f"4B-R4's own *_scores.csv output (run apply_retrospective_shadow_c3_multiplier.py "
                  f"first if it does not exist).")
    dup = int(df["disbursement_fid"].duplicated().sum())
    if dup:
        sys.exit(f"ERROR: {dup} duplicate disbursement_fid value(s) in {path}.")
    return df


def build_agent_period_capacity(df_loan: pd.DataFrame, weights: dict, candidate_ks: list[float]) -> pd.DataFrame:
    """Dedups to one row per (agent_msisdn, fundamentals_snapshot_date)
    agent-period, computes capacity_ugx_at_k1 and masks it to NaN for
    CAPACITY_NOT_ESTIMABLE periods (float_activity_value_1m<=0 or
    commission<=0 -- not a measured zero capacity, just unusable
    fundamentals), then derives capacity_ugx_at_k for every candidate k."""
    key = ["agent_msisdn", "fundamentals_snapshot_date"]
    period_df = df_loan.groupby(key, sort=False)[["float_activity_value_1m", "commission"]].first().reset_index()
    period_df["capacity_ugx_at_k1"] = compute_capacity_ugx(period_df, FUNDAMENTALS_FOR_SCORE, weights, 1.0)
    capacity_estimable = (period_df["float_activity_value_1m"] > 0) & (period_df["commission"] > 0)
    period_df.loc[~capacity_estimable, "capacity_ugx_at_k1"] = np.nan
    for k in candidate_ks:
        period_df[f"capacity_ugx_at_k{k:.2f}"] = period_df["capacity_ugx_at_k1"] * k
    return period_df


def classify_tier(series: pd.Series, low_threshold: float, high_threshold: float) -> pd.Series:
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    valid = series.notna()
    out.loc[valid & (series <= low_threshold)] = "below_floor"
    out.loc[valid & (series > low_threshold) & (series <= high_threshold)] = "in_range"
    out.loc[valid & (series > high_threshold)] = "above_ceiling"
    return out


def build_overall_stats_row(k: float, scenario: str, raw: pd.Series, l_full: pd.Series) -> dict:
    return {
        "k": k, "scenario": scenario, "n": int(raw.notna().sum()),
        "raw_median": raw.median(), "raw_p25": raw.quantile(0.25), "raw_p75": raw.quantile(0.75),
        "l_full_median": l_full.median(), "l_full_p25": l_full.quantile(0.25), "l_full_p75": l_full.quantile(0.75),
    }


def build_tier_distribution_rows(k: float, scenario: str, stage: str, tiers: pd.Series) -> list[dict]:
    n = int(tiers.notna().sum())
    counts = tiers.value_counts()
    row = {
        "k": k, "scenario": scenario, "stage": stage, "n": n,
        "pct_below_floor": 100.0 * counts.get("below_floor", 0) / n if n else np.nan,
        "pct_in_range": 100.0 * counts.get("in_range", 0) / n if n else np.nan,
        "pct_above_ceiling": 100.0 * counts.get("above_ceiling", 0) / n if n else np.nan,
    }
    return [row]


def build_transition_matrix_rows(k: float, scenario: str, raw_tier: pd.Series, l_full_tier: pd.Series) -> list[dict]:
    both_valid = raw_tier.notna() & l_full_tier.notna()
    n_total = int(both_valid.sum())
    rows = []
    tier_order = ["below_floor", "in_range", "above_ceiling"]
    for rt in tier_order:
        for lt in tier_order:
            n = int(((raw_tier == rt) & (l_full_tier == lt) & both_valid).sum())
            rows.append({
                "k": k, "scenario": scenario, "raw_tier": rt, "l_full_tier": lt,
                "n": n, "pct_of_total": 100.0 * n / n_total if n_total else np.nan,
            })
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dataset", type=Path, default=Path("loan_episode_capacity_dataset.csv"))
    ap.add_argument("--r4-c3-multiplier-csv", type=Path, required=True,
                     help="Stage 4B-R4's own *_scores.csv output. This script never regenerates "
                          "it -- run apply_retrospective_shadow_c3_multiplier.py separately first "
                          "if it does not exist.")
    ap.add_argument("--weights-file", type=Path, default=None)
    ap.add_argument("--candidate-ks", type=str, default=",".join(str(k) for k in DEFAULT_CANDIDATE_KS))
    ap.add_argument("--low-tail-threshold-ugx", type=float, default=DEFAULT_LOW_TAIL_THRESHOLD_UGX)
    ap.add_argument("--high-tail-threshold-ugx", type=float, default=DEFAULT_HIGH_TAIL_THRESHOLD_UGX)
    ap.add_argument("--out-prefix", type=str, default="capacity_times_c3")
    args = ap.parse_args(argv)

    if not args.episode_dataset.exists():
        sys.exit(f"ERROR: {args.episode_dataset} not found.")
    if not args.r4_c3_multiplier_csv.exists():
        sys.exit(f"ERROR: {args.r4_c3_multiplier_csv} not found -- run "
                  f"apply_retrospective_shadow_c3_multiplier.py first.")

    candidate_ks = sorted({float(x) for x in args.candidate_ks.split(",") if x.strip()})

    print(f"\n{'#' * 100}\nL_full(k) = Capacity_k(F) x FROZEN C3 -- neither side is tuned to the other\n{'#' * 100}")
    print(
        "C3 is taken AS-IS from Stage 4B-R4's own output -- never recomputed, refit, or re-derived\n"
        "here. k is never selected to make C3 look good, and C3 is never adjusted to make a given k\n"
        "look good; both were independently frozen before this step. Raw Capacity_k(F) is reported\n"
        "alongside L_full at every step -- this script never collapses the two into one number."
    )

    weights = load_weights(args.weights_file, FUNDAMENTALS_FOR_SCORE)
    print(f"\nWeights in use: {weights}")

    df_loan = load_loan_level_fundamentals(args.episode_dataset)
    n_loans_total = len(df_loan)
    print(f"{n_loans_total:,} loan(s) with a valid measured business-state anchor.")

    period_df = build_agent_period_capacity(df_loan, weights, candidate_ks)
    capacity_cols = ["agent_msisdn", "fundamentals_snapshot_date", "capacity_ugx_at_k1"] + \
        [f"capacity_ugx_at_k{k:.2f}" for k in candidate_ks]
    df_loan = df_loan.merge(period_df[capacity_cols], on=["agent_msisdn", "fundamentals_snapshot_date"], how="left")
    n_capacity_not_estimable = int(df_loan["capacity_ugx_at_k1"].isna().sum())

    r4_df = load_r4_c3_multipliers(args.r4_c3_multiplier_csv)
    df = df_loan.merge(r4_df, on="disbursement_fid", how="left")

    scenario_valid = {s: df[f"retrospective_current_c3_multiplier_{s}"].notna() for s in SCENARIO_NAMES}
    n_no_c3_multiplier = int((~scenario_valid[SCENARIO_NAMES[0]]).sum())
    print(f"\n{n_capacity_not_estimable:,} of {n_loans_total:,} loan(s) are CAPACITY_NOT_ESTIMABLE "
          f"(excluded from every table below). {n_no_c3_multiplier:,} loan(s) have no valid R4 C3 "
          f"multiplier (thin-file or missing cal_pd -- also excluded). Both exclusions apply "
          f"independently per loan.")

    overall_rows, tier_dist_rows, transition_rows = [], [], []
    for k in candidate_ks:
        raw_col = f"capacity_ugx_at_k{k:.2f}"
        for scenario in SCENARIO_NAMES:
            mult_col = f"retrospective_current_c3_multiplier_{scenario}"
            valid = df[raw_col].notna() & df[mult_col].notna()
            raw = df.loc[valid, raw_col]
            l_full = raw * df.loc[valid, mult_col]

            overall_rows.append(build_overall_stats_row(k, scenario, raw, l_full))

            raw_tier = classify_tier(raw, args.low_tail_threshold_ugx, args.high_tail_threshold_ugx)
            l_full_tier = classify_tier(l_full, args.low_tail_threshold_ugx, args.high_tail_threshold_ugx)
            tier_dist_rows.extend(build_tier_distribution_rows(k, scenario, "raw", raw_tier))
            tier_dist_rows.extend(build_tier_distribution_rows(k, scenario, "l_full", l_full_tier))
            transition_rows.extend(build_transition_matrix_rows(k, scenario, raw_tier, l_full_tier))

    overall_df = pd.DataFrame(overall_rows)
    tier_dist_df = pd.DataFrame(tier_dist_rows)
    transition_df = pd.DataFrame(transition_rows)

    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("\n-- Overall stats, raw Capacity_k(F) vs L_full, by (k, scenario) --")
        print(overall_df.to_string(index=False))
        print(f"\n-- Tier distribution (below {args.low_tail_threshold_ugx:,.0f} / in range / "
              f"above {args.high_tail_threshold_ugx:,.0f}), raw vs l_full, by (k, scenario) --")
        print(tier_dist_df.to_string(index=False))
        print("\n-- Transition matrix: raw tier -> l_full tier, by (k, scenario) --")
        print(transition_df.to_string(index=False))

    overall_df.to_csv(f"{args.out_prefix}_overall_stats_by_k_scenario.csv", index=False)
    tier_dist_df.to_csv(f"{args.out_prefix}_tier_distribution_by_k_scenario.csv", index=False)
    transition_df.to_csv(f"{args.out_prefix}_transition_matrix_by_k_scenario.csv", index=False)
    print(f"\nWrote {args.out_prefix}_overall_stats_by_k_scenario.csv, "
          f"_tier_distribution_by_k_scenario.csv, _transition_matrix_by_k_scenario.csv")


if __name__ == "__main__":
    main()
