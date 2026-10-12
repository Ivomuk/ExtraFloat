"""
characterize_capacity_times_c3_dollar_consequence.py
=======================================================
The targeted dollar-weighted follow-up to combine_capacity_candidate_k_
with_frozen_c3.py's population-count transition matrix. That analysis
showed k-up moves many loans back above the 50,000 floor while pushing
more loans above the 1,000,000 ceiling -- a real trade-off C3 alone
cannot resolve, since the frozen multiplier is <=1 everywhere (it can
only shrink L_full relative to raw Capacity_k(F), never grow it). This
script asks what that trade-off is worth in UGX, not just in loan counts,
and -- deliberately -- NOT by simply re-running the same population
table with dollar weights. Four distinct things are computed instead:

1. FLOOR DETAIL (per k, scenario) -- quantified, never floored to 50,000.
   For the below-floor population: n, pct, total L_full, median/p25/p75.
   Separately, for the SUBSET that crossed into below-floor BECAUSE of C3
   (in_range on raw Capacity_k(F), below_floor on L_full): n, total raw
   capacity of just those loans, total post-C3 L_full of just those
   loans, and median/p25/p75 of each -- distinguishing a near-miss
   (49,000 -> 47,000) from a deep cut (20,000 -> 15,000), which the
   population table alone cannot.

2. CEILING DETAIL (per k, scenario) -- for the above-ceiling population:
   n, pct, total (uncapped) L_full, median/p25/p75, and total excess
   exposure beyond the ceiling (sum of max(L_full - 1,000,000, 0)).
   Every above-ceiling loan on L_full was ALSO above-ceiling on raw
   Capacity_k(F) (L_full <= raw always, since the multiplier is <=1), so
   there is no "downcrossed-into-ceiling" population to report -- only
   "survived above the ceiling despite C3."

3. GLOBAL DOLLAR AGGREGATES (per k, scenario), over the WHOLE valid
   population: E_full = sum(L_full); E_capped = sum(min(L_full,
   1,000,000)) (the ceiling-only hypothetical cap); E_excess = E_full -
   E_capped. Also E_product_test = sum(L_product_test), where
   L_product_test is the FULL hypothetical product-ladder value (0 if
   L_full<50,000 -- not eligible from capacity evidence alone; L_full if
   in [50,000, 1,000,000]; 1,000,000 if above) -- a DIFFERENT, stricter
   quantity than E_capped, since E_capped does not zero out the floor.
   Both are reported, never collapsed into one number.

4. INCREMENTAL DECOMPOSITION (the primary deliverable) -- for each
   (k_lo, k_hi) pair in --incremental-k-pairs (default 0.04:0.05 and
   0.05:0.06), per scenario: delta_E_full = E_full(k_hi) - E_full(k_lo),
   split by each loan's OWN tier at k_hi (the destination k) into
   delta_E_below_floor / delta_E_in_range / delta_E_above_ceiling (these
   three sum exactly to delta_E_full, a partition of the same total, not
   three separate estimates) -- this answers whether raising k mostly
   creates USABLE incremental capacity or mostly accumulates in a tier
   that cannot be used anyway. Separately, delta_E_capped = E_capped(k_hi)
   - E_capped(k_lo) and delta_E_excess = delta_E_full - delta_E_capped --
   the literal "additional exposure after a hypothetical 1,000,000 cap"
   the review asked for.

NEITHER k NOR C3 IS TUNED HERE. C3 is taken as-is from Stage 4B-R4's own
output (never recomputed). The cap/floor used throughout is explicitly
HYPOTHETICAL (L_product_test, a separately-named, separately-computed
quantity) -- Capacity(F) itself is never redefined as min(Capacity(F),
1,000,000) or max(Capacity(F), 50,000) anywhere in this script or any
upstream one. assigned_limit or any historical exposure column is NEVER
read anywhere -- this stays a consequence analysis of the independently
derived candidate capacity function, never a fit against the old
assignment policy.

Usage:
    python scripts\\characterize_capacity_times_c3_dollar_consequence.py ^
        --episode-dataset loan_episode_capacity_dataset.csv ^
        --r4-c3-multiplier-csv stage4br4_retrospective_c3_multiplier_scores.csv ^
        --weights-file capacity_artifacts\\capacity_score_weights.json ^
        --candidate-ks 0.04,0.05,0.06 ^
        --incremental-k-pairs 0.04:0.05,0.05:0.06 ^
        --out-prefix capacity_times_c3_dollar
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FUNDAMENTALS_FOR_SCORE = ["float_activity_value_1m", "commission"]
DEFAULT_WEIGHTS = {"float_activity_value_1m": 0.5, "commission": 0.5}
DEFAULT_CANDIDATE_KS = [0.04, 0.05, 0.06]
DEFAULT_LOW_TAIL_THRESHOLD_UGX = 50_000
DEFAULT_HIGH_TAIL_THRESHOLD_UGX = 1_000_000
DEFAULT_INCREMENTAL_K_PAIRS = [(0.04, 0.05), (0.05, 0.06)]
SCENARIO_NAMES = ("base", "conservative")
CAL_COL = "retrospective_current_cal_pd"

REQUIRED_EPISODE_COLS = ["disbursement_fid", "agent_msisdn", "fundamentals_snapshot_date",
                          "float_activity_value_1m", "commission"]
REQUIRED_R4_COLS = ["disbursement_fid", CAL_COL] + [f"retrospective_current_c3_multiplier_{s}" for s in SCENARIO_NAMES]

QUANTILE_POINTS = [0.25, 0.5, 0.75]


# ============================================================================
# Restated verbatim from combine_capacity_candidate_k_with_frozen_c3.py
# (one-way scripts/ layering convention -- never imported).
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


def load_loan_level_fundamentals(episode_dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(episode_dataset_path, usecols=lambda c: c in set(REQUIRED_EPISODE_COLS))
    missing = [c for c in REQUIRED_EPISODE_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: episode dataset is missing required column(s): {missing}")
    df = df[df["fundamentals_snapshot_date"].notna()].copy()
    dup = int(df["disbursement_fid"].duplicated().sum())
    if dup:
        sys.exit(f"ERROR: {dup} duplicate disbursement_fid value(s) in the episode dataset.")
    return df


def load_r4_c3_multipliers(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: c in set(REQUIRED_R4_COLS))
    missing = [c for c in REQUIRED_R4_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing} -- this must be Stage "
                  f"4B-R4's own *_scores.csv output.")
    dup = int(df["disbursement_fid"].duplicated().sum())
    if dup:
        sys.exit(f"ERROR: {dup} duplicate disbursement_fid value(s) in {path}.")
    return df


def build_agent_period_capacity(df_loan: pd.DataFrame, weights: dict, candidate_ks: list[float]) -> pd.DataFrame:
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


# ============================================================================
# New: dollar-weighted detail, global aggregates, incremental decomposition.
# ============================================================================


def compute_l_product_test(l_full: pd.Series, low_threshold: float, high_threshold: float) -> pd.Series:
    """The counterfactual, hypothetical product-controlled exposure --
    NEVER written back into Capacity(F) or L_full. 0 (not an eligibility
    flag -- the literal $0 a loan not supported by capacity evidence
    alone would receive) if L_full<low; L_full if in [low,high]; high if
    L_full>high."""
    out = pd.Series(np.nan, index=l_full.index)
    valid = l_full.notna()
    out.loc[valid & (l_full <= low_threshold)] = 0.0
    mid = valid & (l_full > low_threshold) & (l_full <= high_threshold)
    out.loc[mid] = l_full.loc[mid]
    out.loc[valid & (l_full > high_threshold)] = high_threshold
    return out


def quantiles_dict(s: pd.Series, prefix: str) -> dict:
    if s.empty:
        return {f"{prefix}_{int(q * 100)}": np.nan for q in QUANTILE_POINTS}
    return {f"{prefix}_{int(q * 100)}": float(s.quantile(q)) for q in QUANTILE_POINTS}


def build_floor_detail_row(k: float, scenario: str, raw: pd.Series, l_full: pd.Series,
                            raw_tier: pd.Series, l_full_tier: pd.Series, n_total: int, low: float) -> dict:
    below = l_full_tier == "below_floor"
    n_below = int(below.sum())
    downcrossed = below & (raw_tier == "in_range")
    n_downcrossed = int(downcrossed.sum())
    row = {
        "k": k, "scenario": scenario,
        "n_below_floor": n_below, "pct_below_floor": 100.0 * n_below / n_total if n_total else np.nan,
        "total_l_full_below_floor": float(l_full.loc[below].sum()) if n_below else 0.0,
        **quantiles_dict(l_full.loc[below], "l_full_below_floor_p"),
        "n_downcrossed_by_c3": n_downcrossed,
        "total_raw_of_downcrossed": float(raw.loc[downcrossed].sum()) if n_downcrossed else 0.0,
        "total_l_full_of_downcrossed": float(l_full.loc[downcrossed].sum()) if n_downcrossed else 0.0,
        **quantiles_dict(raw.loc[downcrossed], "raw_of_downcrossed_p"),
        **quantiles_dict(l_full.loc[downcrossed], "l_full_of_downcrossed_p"),
    }
    return row


def build_ceiling_detail_row(k: float, scenario: str, l_full: pd.Series, l_full_tier: pd.Series,
                              n_total: int, high: float) -> dict:
    above = l_full_tier == "above_ceiling"
    n_above = int(above.sum())
    excess = (l_full.loc[above] - high).clip(lower=0.0)
    row = {
        "k": k, "scenario": scenario,
        "n_above_ceiling": n_above, "pct_above_ceiling": 100.0 * n_above / n_total if n_total else np.nan,
        "total_l_full_above_ceiling_uncapped": float(l_full.loc[above].sum()) if n_above else 0.0,
        **quantiles_dict(l_full.loc[above], "l_full_above_ceiling_p"),
        "total_excess_above_ceiling": float(excess.sum()) if n_above else 0.0,
    }
    return row


def build_global_aggregates_row(k: float, scenario: str, l_full: pd.Series, l_product_test: pd.Series,
                                 n_total: int, high: float) -> dict:
    e_full = float(l_full.sum())
    e_capped = float(l_full.clip(upper=high).sum())
    e_excess = e_full - e_capped
    e_product_test = float(l_product_test.sum())
    return {
        "k": k, "scenario": scenario, "n": n_total,
        "e_full": e_full, "e_capped_ceiling_only": e_capped, "e_excess_ceiling_only": e_excess,
        "e_product_test_floor_and_ceiling": e_product_test,
    }


def build_incremental_row(k_lo: float, k_hi: float, scenario: str,
                           l_full_lo: pd.Series, l_full_hi: pd.Series, high: float) -> dict:
    """delta_E_full split by each loan's OWN tier AT k_hi -- a partition
    of the same total, never three independent estimates. delta_E_capped
    is the literal 'additional exposure after a hypothetical 1,000,000
    cap' -- computed from the SAME capped quantity at each k, not
    re-derived from the tier split."""
    delta = l_full_hi - l_full_lo
    tier_hi = classify_tier(l_full_hi, DEFAULT_LOW_TAIL_THRESHOLD_UGX, high)

    delta_e_full = float(delta.sum())
    delta_by_tier = {}
    for tier_name in ("below_floor", "in_range", "above_ceiling"):
        mask = tier_hi == tier_name
        delta_by_tier[tier_name] = float(delta.loc[mask].sum()) if mask.any() else 0.0

    e_capped_lo = float(l_full_lo.clip(upper=high).sum())
    e_capped_hi = float(l_full_hi.clip(upper=high).sum())
    delta_e_capped = e_capped_hi - e_capped_lo
    delta_e_excess = delta_e_full - delta_e_capped

    partition_check = delta_by_tier["below_floor"] + delta_by_tier["in_range"] + delta_by_tier["above_ceiling"]
    return {
        "k_lo": k_lo, "k_hi": k_hi, "scenario": scenario,
        "delta_e_full": delta_e_full,
        "delta_e_below_floor_at_k_hi": delta_by_tier["below_floor"],
        "delta_e_in_range_at_k_hi": delta_by_tier["in_range"],
        "delta_e_above_ceiling_at_k_hi": delta_by_tier["above_ceiling"],
        "delta_e_capped_ceiling_only": delta_e_capped,
        "delta_e_excess_ceiling_only": delta_e_excess,
        "partition_matches_delta_e_full": bool(abs(partition_check - delta_e_full) < 1e-3),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dataset", type=Path, default=Path("loan_episode_capacity_dataset.csv"))
    ap.add_argument("--r4-c3-multiplier-csv", type=Path, required=True,
                     help="Stage 4B-R4's own *_scores.csv output. Never regenerated here.")
    ap.add_argument("--weights-file", type=Path, default=None)
    ap.add_argument("--candidate-ks", type=str, default=",".join(str(k) for k in DEFAULT_CANDIDATE_KS))
    ap.add_argument("--incremental-k-pairs", type=str,
                     default=",".join(f"{lo}:{hi}" for lo, hi in DEFAULT_INCREMENTAL_K_PAIRS),
                     help="Comma-separated lo:hi pairs, e.g. '0.04:0.05,0.05:0.06'. Each k named "
                          "here must also appear in --candidate-ks.")
    ap.add_argument("--low-tail-threshold-ugx", type=float, default=DEFAULT_LOW_TAIL_THRESHOLD_UGX)
    ap.add_argument("--high-tail-threshold-ugx", type=float, default=DEFAULT_HIGH_TAIL_THRESHOLD_UGX)
    ap.add_argument("--out-prefix", type=str, default="capacity_times_c3_dollar")
    args = ap.parse_args(argv)

    if not args.episode_dataset.exists():
        sys.exit(f"ERROR: {args.episode_dataset} not found.")
    if not args.r4_c3_multiplier_csv.exists():
        sys.exit(f"ERROR: {args.r4_c3_multiplier_csv} not found -- run "
                  f"apply_retrospective_shadow_c3_multiplier.py first.")

    candidate_ks = sorted({float(x) for x in args.candidate_ks.split(",") if x.strip()})
    incremental_pairs = []
    for pair in args.incremental_k_pairs.split(","):
        lo_str, hi_str = pair.split(":")
        lo, hi = float(lo_str), float(hi_str)
        if lo not in candidate_ks or hi not in candidate_ks:
            sys.exit(f"ERROR: incremental pair {lo}:{hi} references a k not in --candidate-ks {candidate_ks}.")
        incremental_pairs.append((lo, hi))

    print(f"\n{'#' * 100}\nDOLLAR-WEIGHTED CONSEQUENCE ANALYSIS -- neither k nor C3 is tuned here\n{'#' * 100}")
    print(
        "C3 is taken AS-IS from Stage 4B-R4's own output. The 50,000/1,000,000 product boundaries\n"
        "used throughout are a HYPOTHETICAL counterfactual (L_product_test) -- Capacity(F) itself is\n"
        "never redefined as min/max'd against them. assigned_limit or any historical exposure column\n"
        "is never read -- this is a consequence analysis of the independently derived capacity\n"
        "function, never a fit against the old assignment policy."
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
    n_no_c3_multiplier = int(df[f"retrospective_current_c3_multiplier_{SCENARIO_NAMES[0]}"].isna().sum())
    print(f"\n{n_capacity_not_estimable:,} of {n_loans_total:,} loan(s) are CAPACITY_NOT_ESTIMABLE "
          f"(excluded). {n_no_c3_multiplier:,} loan(s) have no valid R4 C3 multiplier (excluded). "
          f"Both exclusions apply independently per loan.")

    low, high = args.low_tail_threshold_ugx, args.high_tail_threshold_ugx

    # Precompute, per k and scenario: raw, l_full, tiers, l_product_test -- reused across
    # the floor/ceiling/global tables AND the incremental decomposition.
    cache: dict[tuple[float, str], dict] = {}
    for k in candidate_ks:
        raw_col = f"capacity_ugx_at_k{k:.2f}"
        for scenario in SCENARIO_NAMES:
            mult_col = f"retrospective_current_c3_multiplier_{scenario}"
            valid = df[raw_col].notna() & df[mult_col].notna()
            raw = df.loc[valid, raw_col]
            l_full = raw * df.loc[valid, mult_col]
            raw_tier = classify_tier(raw, low, high)
            l_full_tier = classify_tier(l_full, low, high)
            l_product_test = compute_l_product_test(l_full, low, high)
            cache[(k, scenario)] = {
                "n": int(valid.sum()), "raw": raw, "l_full": l_full,
                "raw_tier": raw_tier, "l_full_tier": l_full_tier, "l_product_test": l_product_test,
            }

    floor_rows, ceiling_rows, global_rows = [], [], []
    for k in candidate_ks:
        for scenario in SCENARIO_NAMES:
            c = cache[(k, scenario)]
            floor_rows.append(build_floor_detail_row(
                k, scenario, c["raw"], c["l_full"], c["raw_tier"], c["l_full_tier"], c["n"], low))
            ceiling_rows.append(build_ceiling_detail_row(
                k, scenario, c["l_full"], c["l_full_tier"], c["n"], high))
            global_rows.append(build_global_aggregates_row(
                k, scenario, c["l_full"], c["l_product_test"], c["n"], high))

    incremental_rows = []
    for k_lo, k_hi in incremental_pairs:
        for scenario in SCENARIO_NAMES:
            lo_c, hi_c = cache[(k_lo, scenario)], cache[(k_hi, scenario)]
            # Both sides must be reindexed to a common, aligned loan population: inner-align on index.
            common_idx = lo_c["l_full"].index.intersection(hi_c["l_full"].index)
            incremental_rows.append(build_incremental_row(
                k_lo, k_hi, scenario, lo_c["l_full"].loc[common_idx], hi_c["l_full"].loc[common_idx], high))

    floor_df = pd.DataFrame(floor_rows)
    ceiling_df = pd.DataFrame(ceiling_rows)
    global_df = pd.DataFrame(global_rows)
    incremental_df = pd.DataFrame(incremental_rows)

    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print("\n-- Global dollar aggregates, by (k, scenario) --")
        print(global_df.to_string(index=False))
        print("\n-- Floor detail (quantified, never floored), by (k, scenario) --")
        print(floor_df.to_string(index=False))
        print("\n-- Ceiling detail, by (k, scenario) --")
        print(ceiling_df.to_string(index=False))
        print("\n-- Incremental decomposition, by (k_lo->k_hi, scenario) --")
        print(incremental_df.to_string(index=False))

    global_df.to_csv(f"{args.out_prefix}_global_aggregates.csv", index=False)
    floor_df.to_csv(f"{args.out_prefix}_floor_detail.csv", index=False)
    ceiling_df.to_csv(f"{args.out_prefix}_ceiling_detail.csv", index=False)
    incremental_df.to_csv(f"{args.out_prefix}_incremental_decomposition.csv", index=False)
    print(f"\nWrote {args.out_prefix}_global_aggregates.csv, _floor_detail.csv, _ceiling_detail.csv, "
          f"_incremental_decomposition.csv")

    if not incremental_df.empty and not incremental_df["partition_matches_delta_e_full"].all():
        print("\nWARNING: at least one incremental row's tier-at-k_hi partition did not reconcile "
              "to delta_e_full within tolerance -- inspect before trusting the decomposition.")


if __name__ == "__main__":
    main()
