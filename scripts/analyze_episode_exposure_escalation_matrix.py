"""
analyze_episode_exposure_escalation_matrix.py
=================================================
Deliverable 5 of the episode-grain rebuild of Analysis 3 ("Level 3:
same-agent cross-tier" evidence -- the strongest of the three evidence
levels in this rebuild, since it compares the SAME agent against itself
across tiers rather than across different agents or merely consecutive
pairs).

For every agent, for every UNORDERED pair of distinct exposure tiers they
were EVER observed at across their full episode history (an agent seen at
250K, 500K, and 750K contributes pairs (250K,500K), (250K,750K),
(500K,750K)) -- this answers: among agents who were, at some point,
observed at BOTH tiers, how did their business fundamentals and subsequent
loan performance differ between the lower-tier episodes and the
higher-tier episodes?

ASSIGNMENT vs OUTCOME SEPARATION (same discipline as Deliverable 3, now
applied per-agent-per-tier):

    F_{i,L}^(c) = median(fundamental over ALL of agent i's episodes at
                  tier L with a valid fundamental satisfying freshness cut
                  c), regardless of label_eligible_30d.
    BR_{i,L}    = (bad eligible episodes at L) / (eligible episodes at L),
                  using ONLY label_eligible_30d==1 episodes at L --
                  independent of which episodes fed F_{i,L}.

BR_{i,L} is undefined (NaN) if agent i has zero eligible episodes at L.
This does NOT also blank out F_{i,L}, which has its own, independent
validity condition. Per the plan review, this separation is deliberate:
conflating the two would bias fundamentals-change statistics toward
whichever episodes happened to also be outcome-eligible.

POOLED vs PAIRED evidence (kept distinct, never blended into one number):
  - POOLED: episode-level aggregate bad rate at each tier, summed across
    every qualifying agent's episodes at that tier.
  - PAIRED: within-agent delta (BR_{i,L_high} - BR_{i,L_low}), the
    stronger panel-structure comparison, reported as its own distribution
    (median delta, pct positive/negative/zero).

RELATIVE EXPOSURE GROWTH (REG): REG_i^(c) = (L_high/L_low) /
FundamentalChangeRatio_i^(c) -- financing growth relative to business
growth. <1 means exposure grew LESS than the fundamental; ~1 proportional;
>1 exposure grew faster. Reported descriptively via distribution bands
(<0.8 / 0.8-1.2 / >1.2), never as a policy threshold.

Every fundamentals-based ratio (change ratio, REG) is computed separately
under THREE snapshot-freshness cuts (all / <=30d / <=60d -- same cuts used
throughout this rebuild) and carries its OWN per-cut N (an agent can have a
valid 'all'-cut ratio but a missing '<=30d'-cut ratio if their qualifying
episodes at one tier are stale; that must show up as a missing value, not
a silently reused stale number). Pooled N/bad-rate/temporal columns are
freshness-independent (tier membership and outcome eligibility do not
depend on fundamentals freshness at all).

TEMPORAL DIRECTION diagnostics: this matrix is explicitly UNORDERED (tier
pairs are labelled "250K vs 500K", never "->"), so these columns supply the
directionality that Deliverable 4 already covers for consecutive pairs --
which tier did the agent reach first, and were both orders ever observed
(a genuinely interleaved history, not just "first observed" order).

CAUSAL CAVEAT (printed explicitly, every run, same as Deliverable 4): a
within-agent performance difference across tiers is observational, not
causal. Report as "same agents exhibited X difference in observed
performance when historically at the higher vs. lower tier", never as the
higher tier CAUSING X -- later exposure often coincides with more lender
information, calendar effects, and selection that remain live
explanations.

Restated independently (one-way scripts/ layering convention): the 7-tier
exposure set and the freshness cuts, consistent with every other script in
this rebuild.

Usage:
    python scripts\\analyze_episode_exposure_escalation_matrix.py ^
        --episode-dataset loan_episode_capacity_dataset.csv
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPOSURE_TIERS_UGX = [50_000, 100_000, 250_000, 350_000, 500_000, 750_000, 1_000_000]
FRESHNESS_CUTS = {"all": None, "le30d": 30, "le60d": 60}
FUNDAMENTALS = {
    "float_activity_value_1m": "float",
    "commission": "commission",
    "cust_1m": "customer",
    "average_balance": "balance",
}
REG_FUNDAMENTALS = ["float", "commission"]  # REG headline computed for these two only.

REQUIRED_COLS = [
    "agent_msisdn", "loan_date", "disbursement_ts", "target_loan_seq", "disbursement_amount_ugx",
    "fundamentals_age_days", "float_activity_value_1m", "commission", "cust_1m", "average_balance",
    "bad_state_3dpd_30d", "label_eligible_30d",
]


def _assign_sort_ts(df: pd.DataFrame) -> pd.DataFrame:
    """disbursement_ts preferred, loan_date fallback -- restated independently,
    same ordering priority used throughout this rebuild."""
    df = df.copy()
    df["_sort_ts"] = pd.to_datetime(df["disbursement_ts"], errors="coerce")
    fallback = pd.to_datetime(df["loan_date"], errors="coerce")
    df["_sort_ts"] = df["_sort_ts"].fillna(fallback)
    return df


def _reg_band(x) -> object:
    if pd.isna(x):
        return np.nan
    if x < 0.8:
        return "<0.8"
    if x > 1.2:
        return ">1.2"
    return "0.8-1.2"


def agent_tier_fundamental_median(df: pd.DataFrame, fund_col: str, max_age_days) -> dict:
    """F_{i,L}^(c): median fundamental over ALL of agent i's episodes at
    tier L with a valid fundamental satisfying freshness cut c, regardless
    of label_eligible_30d. Returns {(agent, tier): median_value}."""
    valid = df[df[fund_col].notna() & (df[fund_col] > 0)]
    if max_age_days is not None:
        valid = valid[valid["fundamentals_age_days"].notna() & (valid["fundamentals_age_days"] <= max_age_days)]
    if valid.empty:
        return {}
    return valid.groupby(["agent_msisdn", "disbursement_amount_ugx"])[fund_col].median().to_dict()


def agent_tier_badrate(df: pd.DataFrame) -> dict:
    """BR_{i,L}: bad-eligible / eligible episodes at tier L, label_eligible_30d==1
    only -- independent of fundamentals validity/freshness. Returns
    {(agent, tier): (n_eligible, bad_rate)}."""
    elig = df[df["label_eligible_30d"] == 1]
    if elig.empty:
        return {}
    grouped = elig.groupby(["agent_msisdn", "disbursement_amount_ugx"])["bad_state_3dpd_30d"]
    n = grouped.count()
    rate = grouped.mean()
    return {k: (n[k], rate[k]) for k in n.index}


def agent_tier_episode_counts(df: pd.DataFrame) -> dict:
    """Per (agent, tier): (n_total episodes, n_eligible episodes) regardless
    of fundamentals validity."""
    total = df.groupby(["agent_msisdn", "disbursement_amount_ugx"]).size()
    elig = df[df["label_eligible_30d"] == 1].groupby(["agent_msisdn", "disbursement_amount_ugx"]).size()
    out = {}
    for k in total.index:
        out[k] = (int(total[k]), int(elig.get(k, 0)))
    return out


def agent_tier_timestamps(df: pd.DataFrame) -> dict:
    """Per (agent, tier): sorted list of valid (non-NaT) _sort_ts values."""
    out = {}
    for (agent, tier), g in df.groupby(["agent_msisdn", "disbursement_amount_ugx"]):
        ts_list = sorted(t for t in g["_sort_ts"] if pd.notna(t))
        out[(agent, tier)] = ts_list
    return out


def temporal_relation(ts_low: list, ts_high: list) -> tuple:
    """Returns (temporal_direction, observed_both_orders). temporal_direction
    compares the EARLIEST timestamp at each tier: 'lower_first' /
    'higher_first' / 'same_timestamp' / NaN (no valid timestamp on one
    side). observed_both_orders is True only if a low-tier episode
    precedes SOME high-tier episode AND a high-tier episode precedes SOME
    low-tier episode for this agent -- a genuinely interleaved history,
    not merely "first observed" order."""
    if not ts_low or not ts_high:
        return np.nan, False
    min_low, min_high = min(ts_low), min(ts_high)
    if min_low < min_high:
        direction = "lower_first"
    elif min_low > min_high:
        direction = "higher_first"
    else:
        direction = "same_timestamp"
    low_before_high = any(l < h for l in ts_low for h in ts_high)
    high_before_low = any(h < l for l in ts_low for h in ts_high)
    return direction, bool(low_before_high and high_before_low)


def build_agent_pair_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (agent, tier_low, tier_high) for every unordered pair of
    tiers the agent was ever observed at. All downstream aggregation reads
    from this intermediate table."""
    episode_counts = agent_tier_episode_counts(df)
    badrate = agent_tier_badrate(df)
    timestamps = agent_tier_timestamps(df)

    fund_median_by_cut = {
        fund_col: {cut_label: agent_tier_fundamental_median(df, fund_col, max_age) for cut_label, max_age in FRESHNESS_CUTS.items()}
        for fund_col in FUNDAMENTALS
    }

    agent_tiers = df.groupby("agent_msisdn")["disbursement_amount_ugx"].unique().apply(lambda arr: sorted(set(arr)))

    rows = []
    for agent, tiers in agent_tiers.items():
        if len(tiers) < 2:
            continue
        for tier_low, tier_high in itertools.combinations(tiers, 2):
            n_low_total, n_low_elig = episode_counts.get((agent, tier_low), (0, 0))
            n_high_total, n_high_elig = episode_counts.get((agent, tier_high), (0, 0))

            br_low = badrate.get((agent, tier_low))
            br_high = badrate.get((agent, tier_high))
            bad_rate_lower_i = br_low[1] if br_low is not None else np.nan
            bad_rate_higher_i = br_high[1] if br_high is not None else np.nan
            delta_bad_rate_i = (
                bad_rate_higher_i - bad_rate_lower_i
                if pd.notna(bad_rate_lower_i) and pd.notna(bad_rate_higher_i)
                else np.nan
            )

            direction, both_orders = temporal_relation(
                timestamps.get((agent, tier_low), []), timestamps.get((agent, tier_high), [])
            )

            row = {
                "agent_msisdn": agent, "tier_low": tier_low, "tier_high": tier_high,
                "n_lower_episodes_total_i": n_low_total, "n_lower_episodes_eligible_i": n_low_elig,
                "n_higher_episodes_total_i": n_high_total, "n_higher_episodes_eligible_i": n_high_elig,
                "bad_rate_lower_i": bad_rate_lower_i, "bad_rate_higher_i": bad_rate_higher_i,
                "delta_bad_rate_i": delta_bad_rate_i,
                "temporal_direction_i": direction, "observed_both_orders_i": both_orders,
            }

            for fund_col, fund_name in FUNDAMENTALS.items():
                for cut_label in FRESHNESS_CUTS:
                    f_low = fund_median_by_cut[fund_col][cut_label].get((agent, tier_low), np.nan)
                    f_high = fund_median_by_cut[fund_col][cut_label].get((agent, tier_high), np.nan)
                    if pd.notna(f_low) and f_low > 0 and pd.notna(f_high):
                        ratio = f_high / f_low
                    else:
                        ratio = np.nan
                    row[f"{fund_name}_change_ratio_{cut_label}"] = ratio
                    if fund_name in REG_FUNDAMENTALS:
                        reg = (tier_high / tier_low) / ratio if pd.notna(ratio) and ratio > 0 else np.nan
                        row[f"reg_{fund_name}_{cut_label}"] = reg

            rows.append(row)
    return pd.DataFrame(rows)


def _pooled_tier_stats(df: pd.DataFrame, agents: set, tier) -> tuple:
    sub = df[df["agent_msisdn"].isin(agents) & (df["disbursement_amount_ugx"] == tier)]
    n_total = len(sub)
    elig = sub[sub["label_eligible_30d"] == 1]
    n_elig = len(elig)
    bad_rate = elig["bad_state_3dpd_30d"].mean() if n_elig else np.nan
    return n_total, n_elig, bad_rate


def aggregate_pair_table(df: pd.DataFrame, agent_pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per unordered tier pair, aggregating agent_pairs across every
    agent observed at both tiers."""
    if agent_pairs.empty:
        return pd.DataFrame()
    rows = []
    for (tier_low, tier_high), g in agent_pairs.groupby(["tier_low", "tier_high"]):
        agents = set(g["agent_msisdn"])
        n_agents_pair = len(agents)

        n_lower_total, n_lower_elig, bad_rate_lower = _pooled_tier_stats(df, agents, tier_low)
        n_higher_total, n_higher_elig, bad_rate_higher = _pooled_tier_stats(df, agents, tier_high)

        both_defined = g[g["bad_rate_lower_i"].notna() & g["bad_rate_higher_i"].notna()]
        n_agents_outcome_both = len(both_defined)
        if n_agents_outcome_both:
            deltas = both_defined["delta_bad_rate_i"]
            median_delta_bad_rate = deltas.median()
            pct_delta_bad_positive = (deltas > 0).mean() * 100
            pct_delta_bad_negative = (deltas < 0).mean() * 100
            pct_delta_bad_zero = (deltas == 0).mean() * 100
        else:
            median_delta_bad_rate = np.nan
            pct_delta_bad_positive = pct_delta_bad_negative = pct_delta_bad_zero = np.nan

        temporal_defined = g[g["temporal_direction_i"].notna()]
        n_temporal_defined = len(temporal_defined)
        if n_temporal_defined:
            pct_lower_first = (temporal_defined["temporal_direction_i"] == "lower_first").mean() * 100
            pct_higher_first = (temporal_defined["temporal_direction_i"] == "higher_first").mean() * 100
            pct_same_ts = (temporal_defined["temporal_direction_i"] == "same_timestamp").mean() * 100
            pct_both_directions = temporal_defined["observed_both_orders_i"].mean() * 100
        else:
            pct_lower_first = pct_higher_first = pct_same_ts = pct_both_directions = np.nan

        row = {
            "tier_low": tier_low, "tier_high": tier_high, "pair_label": f"{tier_low:,} vs {tier_high:,}",
            "n_agents_pair": n_agents_pair,
            "n_lower_episodes_total": n_lower_total, "n_lower_episodes_eligible": n_lower_elig,
            "bad_rate_lower": bad_rate_lower,
            "n_higher_episodes_total": n_higher_total, "n_higher_episodes_eligible": n_higher_elig,
            "bad_rate_higher": bad_rate_higher,
            "n_agents_outcome_both": n_agents_outcome_both,
            "median_delta_bad_rate": median_delta_bad_rate,
            "pct_delta_bad_positive": pct_delta_bad_positive,
            "pct_delta_bad_negative": pct_delta_bad_negative,
            "pct_delta_bad_zero": pct_delta_bad_zero,
            "exposure_ratio": tier_high / tier_low,
            "n_agents_temporal_defined": n_temporal_defined,
            "pct_lower_observed_first": pct_lower_first,
            "pct_higher_observed_first": pct_higher_first,
            "pct_same_timestamp": pct_same_ts,
            "pct_both_directions_observed": pct_both_directions,
        }

        for fund_name in FUNDAMENTALS.values():
            for cut_label in FRESHNESS_CUTS:
                col = f"{fund_name}_change_ratio_{cut_label}"
                vals = g[col].dropna()
                row[f"median_{col}"] = vals.median() if len(vals) else np.nan
                row[f"n_agents_{col}"] = len(vals)

        for fund_name in REG_FUNDAMENTALS:
            for cut_label in FRESHNESS_CUTS:
                col = f"reg_{fund_name}_{cut_label}"
                vals = g[col].dropna()
                row[f"median_{col}"] = vals.median() if len(vals) else np.nan
                row[f"n_agents_{col}"] = len(vals)
                if len(vals):
                    bands = vals.map(_reg_band)
                    row[f"pct_{col}_below_0.8"] = (bands == "<0.8").mean() * 100
                    row[f"pct_{col}_0.8_to_1.2"] = (bands == "0.8-1.2").mean() * 100
                    row[f"pct_{col}_above_1.2"] = (bands == ">1.2").mean() * 100
                else:
                    row[f"pct_{col}_below_0.8"] = np.nan
                    row[f"pct_{col}_0.8_to_1.2"] = np.nan
                    row[f"pct_{col}_above_1.2"] = np.nan

        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["tier_low", "tier_high"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dataset", default="loan_episode_capacity_dataset.csv")
    ap.add_argument("--out-prefix", default="episode_exposure_escalation_matrix")
    args = ap.parse_args(argv)

    path = Path(args.episode_dataset)
    if not path.exists():
        sys.exit(f"ERROR: {path} not found -- run build_loan_episode_capacity_dataset.py first.")
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing}")
    df = _assign_sort_ts(df)

    agent_pairs = build_agent_pair_rows(df)
    n_agents_any_pair = agent_pairs["agent_msisdn"].nunique() if not agent_pairs.empty else 0
    print(f"Built {len(agent_pairs):,} agent-tier-pair row(s) across {n_agents_any_pair:,} agent(s) "
          f"observed at >=2 distinct exposure tiers.")
    if agent_pairs.empty:
        sys.exit("ERROR: no agent was observed at >=2 distinct exposure tiers -- nothing to analyze.")
    agent_pairs.to_csv(f"{args.out_prefix}_agent_pairs.csv", index=False)

    pair_table = aggregate_pair_table(df, agent_pairs)
    pair_table.to_csv(f"{args.out_prefix}_pair_summary.csv", index=False)

    print(f"\n{'#' * 100}")
    print("# Unordered exposure-tier-pair escalation matrix (pooled + paired evidence)")
    print(f"{'#' * 100}")
    display_cols = [
        "pair_label", "n_agents_pair", "n_lower_episodes_total", "n_lower_episodes_eligible", "bad_rate_lower",
        "n_higher_episodes_total", "n_higher_episodes_eligible", "bad_rate_higher",
        "n_agents_outcome_both", "median_delta_bad_rate", "pct_delta_bad_positive", "pct_delta_bad_negative",
        "exposure_ratio",
    ]
    with pd.option_context("display.float_format", "{:,.3f}".format, "display.max_columns", None, "display.width", 240):
        print(pair_table[display_cols].to_string(index=False))

    print(f"\n{'=' * 100}\nRelative Exposure Growth (REG) -- headline float/commission columns, 'all' freshness cut\n{'=' * 100}")
    reg_cols = ["pair_label", "n_agents_pair"]
    for fund_name in REG_FUNDAMENTALS:
        reg_cols += [
            f"median_reg_{fund_name}_all", f"n_agents_reg_{fund_name}_all",
            f"pct_reg_{fund_name}_all_below_0.8", f"pct_reg_{fund_name}_all_0.8_to_1.2", f"pct_reg_{fund_name}_all_above_1.2",
        ]
    with pd.option_context("display.float_format", "{:,.3f}".format, "display.max_columns", None, "display.width", 240):
        print(pair_table[reg_cols].to_string(index=False))

    print(f"\n{'=' * 100}\nTemporal direction (this matrix is UNORDERED; these columns supply directionality)\n{'=' * 100}")
    temporal_cols = ["pair_label", "n_agents_temporal_defined", "pct_lower_observed_first",
                     "pct_higher_observed_first", "pct_same_timestamp", "pct_both_directions_observed"]
    with pd.option_context("display.float_format", "{:,.1f}".format, "display.max_columns", None, "display.width", 240):
        print(pair_table[temporal_cols].to_string(index=False))

    pct_same_ts_overall = agent_pairs["temporal_direction_i"].eq("same_timestamp").mean() * 100
    print(f"\nSame-timestamp agent-tier-pairs (both tiers' earliest episode land on the identical sort "
          f"timestamp): {pct_same_ts_overall:.1f}% of all agent-tier-pair rows.")
    if pct_same_ts_overall > 5.0:
        print("  NOTE: this is frequent enough that ordering granularity may be insufficient for some agents.")

    print(f"\n{'#' * 100}")
    print("Causal caveat (printed every run)")
    print(f"{'#' * 100}")
    print("A within-agent performance difference across exposure tiers is OBSERVATIONAL, not causal.\n"
          "Report as 'same agents exhibited X difference in observed performance when historically at the\n"
          "higher vs. lower tier' -- never as the higher tier CAUSING X. Later/higher exposure often\n"
          "coincides with more lender information, calendar effects, and selection that remain live\n"
          "explanations.")


if __name__ == "__main__":
    main()
