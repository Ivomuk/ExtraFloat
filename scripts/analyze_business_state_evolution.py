"""
analyze_business_state_evolution.py
=======================================
Level 3 of the business-state-period evidence hierarchy (cadence-gate
pivot). The question: did the agent's exposure profile move together with
their measured business-state change, ACROSS periods? This is directional
(State_earlier -> State_later), never unordered -- chronology is the whole
point here, unlike analyze_business_state_exposure_variation.py's (Level
2) within-period pairs.

UNIT: one row per (agent_msisdn, fundamentals_snapshot_date), reusing
Level 2's agent-period summary fields (median_exposure, p75_exposure,
max_exposure, n_loans_total/eligible, bad_rate_period, the period's own
float_activity_value_1m/commission), restated independently per the
one-way scripts/ layering convention.

TWO ANALYSES, built from the SAME underlying (agent, period_a, period_b)
pair table so the k=1 stratum of the Secondary table is a genuine
cross-check on the Primary analysis's own numbers, not a separately
re-implemented path that could silently drift:

  PRIMARY -- consecutive period transitions (k=1 only): float_growth,
  commission_growth, exposure_growth_median, exposure_growth_p75, and
  four REG (Relative Exposure Growth) variants: reg_median_float =
  exposure_growth_median / float_growth, reg_median_commission,
  reg_p75_float, reg_p75_commission -- each guarded (denominator must be
  > 0, else missing). Two independent direction grids (reusing archive/
  analyze_episode_agent_transitions.py's up/flat/down banding and 3x3-grid
  machinery, restated independently): ExposureGrowthMedianDirection x
  FloatDirection, and x CommissionDirection.

  SECONDARY -- all chronological pairs, stratified by k = the ORDINAL
  state gap (index(S_b) - index(S_a) among that agent's own periods,
  1-indexed, NOT elapsed time or a raw "periods between" count -- S_1->S_2
  is k=1, S_1->S_3 is k=2, S_1->S_4 is k=3). Aggregated grouped BY k, same
  ratio/REG computation.

WEIGHTING DISCIPLINE (per review correction) -- every period-level bad-
rate figure in this script reports BOTH, clearly named, never collapsed:
  bad_rate_t_loan_weighted      = sum(bad eligible loans across
                                   contributing agent-periods) /
                                   sum(eligible loans across the same).
                                   Answers "what fraction of subsequent
                                   loans went bad."
  median_bad_rate_t_agent_period = median(each contributing agent-period's
                                   OWN bad rate). Answers "what was the
                                   typical subsequent agent-period's bad
                                   rate" -- prevents one very-high-
                                   frequency agent's loan count from
                                   silently dominating the loan-weighted
                                   figure without that being visible.
In both cases "t" / "the later period" is the thing being evaluated,
mirroring the archived script's Y_j-only discipline at period grain.

Causal caveat printed every run (observational, not causal).

Restated independently (one-way scripts/ layering convention): the
agent-period summary, direction bands, and REG bands, consistent with
every other script in this rebuild.

Usage:
    python scripts\\analyze_business_state_evolution.py ^
        --episode-dataset loan_episode_capacity_dataset.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DIRECTION_UP_THRESHOLD = 1.10
DIRECTION_DOWN_THRESHOLD = 0.90
DIRECTION_LABELS = ["down", "flat", "up"]

REQUIRED_COLS = [
    "agent_msisdn", "fundamentals_snapshot_date", "disbursement_amount_ugx",
    "float_activity_value_1m", "commission", "bad_state_3dpd_30d", "label_eligible_30d",
]

REG_VARIANTS = ["reg_median_float", "reg_median_commission", "reg_p75_float", "reg_p75_commission"]


def _ratio(cur, prev) -> float:
    if pd.isna(prev) or prev <= 0 or pd.isna(cur):
        return np.nan
    return cur / prev


def _reg(exposure_growth, fund_growth) -> float:
    if pd.isna(exposure_growth) or pd.isna(fund_growth) or fund_growth <= 0:
        return np.nan
    return exposure_growth / fund_growth


def _direction(ratio) -> object:
    if pd.isna(ratio):
        return np.nan
    if ratio > DIRECTION_UP_THRESHOLD:
        return "up"
    if ratio < DIRECTION_DOWN_THRESHOLD:
        return "down"
    return "flat"


def _reg_band(x) -> object:
    if pd.isna(x):
        return np.nan
    if x < 0.8:
        return "<0.8"
    if x > 1.2:
        return ">1.2"
    return "0.8-1.2"


def build_agent_period_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (agent_msisdn, fundamentals_snapshot_date) unit.
    Restated independently from analyze_business_state_exposure_
    variation.py's identically-behaving function (one-way scripts/
    layering convention), with n_bad_eligible added -- needed here for
    the loan-weighted aggregation across multiple agent-periods."""
    rows = []
    for (agent, period), g in df.groupby(["agent_msisdn", "fundamentals_snapshot_date"]):
        elig = g[g["label_eligible_30d"] == 1]
        n_eligible = len(elig)
        n_bad = int(elig["bad_state_3dpd_30d"].sum())
        bad_rate_period = n_bad / n_eligible if n_eligible else np.nan
        rows.append({
            "agent_msisdn": agent, "fundamentals_snapshot_date": period,
            "n_loans_total": len(g), "n_loans_eligible": n_eligible, "n_bad_eligible": n_bad,
            "median_exposure": g["disbursement_amount_ugx"].median(),
            "p75_exposure": g["disbursement_amount_ugx"].quantile(0.75),
            "max_exposure": g["disbursement_amount_ugx"].max(),
            "bad_rate_period": bad_rate_period,
            "float_activity_value_1m": g["float_activity_value_1m"].iloc[0],
            "commission": g["commission"].iloc[0],
        })
    return pd.DataFrame(rows)


def order_periods(summary: pd.DataFrame) -> pd.DataFrame:
    """Assigns each agent's own periods a 1-indexed chronological ordinal
    (period_index) -- the basis for k = index(S_b) - index(S_a)."""
    out = summary.copy()
    out["fundamentals_snapshot_date"] = pd.to_datetime(out["fundamentals_snapshot_date"])
    out = out.sort_values(["agent_msisdn", "fundamentals_snapshot_date"]).reset_index(drop=True)
    out["period_index"] = out.groupby("agent_msisdn").cumcount() + 1
    return out


def _build_pair_row(agent: str, prev: pd.Series, cur: pd.Series, k: int) -> dict:
    days_between = (cur["fundamentals_snapshot_date"] - prev["fundamentals_snapshot_date"]).days
    float_growth = _ratio(cur["float_activity_value_1m"], prev["float_activity_value_1m"])
    commission_growth = _ratio(cur["commission"], prev["commission"])
    exposure_growth_median = _ratio(cur["median_exposure"], prev["median_exposure"])
    exposure_growth_p75 = _ratio(cur["p75_exposure"], prev["p75_exposure"])
    return {
        "agent_msisdn": agent, "k": k,
        "period_a": prev["fundamentals_snapshot_date"], "period_b": cur["fundamentals_snapshot_date"],
        "days_between": days_between,
        "float_growth": float_growth, "float_direction": _direction(float_growth),
        "commission_growth": commission_growth, "commission_direction": _direction(commission_growth),
        "exposure_growth_median": exposure_growth_median,
        "exposure_growth_median_direction": _direction(exposure_growth_median),
        "exposure_growth_p75": exposure_growth_p75,
        "reg_median_float": _reg(exposure_growth_median, float_growth),
        "reg_median_commission": _reg(exposure_growth_median, commission_growth),
        "reg_p75_float": _reg(exposure_growth_p75, float_growth),
        "reg_p75_commission": _reg(exposure_growth_p75, commission_growth),
        "n_eligible_b": cur["n_loans_eligible"], "n_bad_eligible_b": cur["n_bad_eligible"],
        "bad_rate_b": cur["bad_rate_period"],
    }


def build_all_chronological_pairs(summary_ordered: pd.DataFrame) -> pd.DataFrame:
    """One row per (agent, period_a, period_b) for EVERY chronological pair
    t_a < t_b (never unordered). k = index(S_b) - index(S_a), the ORDINAL
    state gap -- S_1->S_2 is k=1 (adjacent), S_1->S_3 is k=2, etc. The
    Primary analysis below reads its transitions directly from this
    table's k==1 subset, so the two are a genuine cross-check, never
    separately re-implemented paths that could silently drift apart."""
    rows = []
    for agent, g in summary_ordered.groupby("agent_msisdn"):
        g = g.sort_values("period_index").reset_index(drop=True)
        n = len(g)
        for ia in range(n):
            for ib in range(ia + 1, n):
                prev, cur = g.iloc[ia], g.iloc[ib]
                k = int(cur["period_index"] - prev["period_index"])
                rows.append(_build_pair_row(agent, prev, cur, k))
    return pd.DataFrame(rows)


def direction_grid(transitions: pd.DataFrame, fund_name: str) -> pd.DataFrame:
    """ExposureGrowthMedianDirection x <fund_name>Direction, 9 cells. Each
    cell reports n_transitions, n_agents, and BOTH bad-rate weightings for
    period t_b (the later period -- the thing being evaluated)."""
    fund_dir_col = f"{fund_name}_direction"
    rows = []
    for exp_d in DIRECTION_LABELS:
        for fund_d in DIRECTION_LABELS:
            cell = transitions[(transitions["exposure_growth_median_direction"] == exp_d)
                                & (transitions[fund_dir_col] == fund_d)]
            n_transitions = len(cell)
            n_agents = cell["agent_msisdn"].nunique()
            n_eligible_sum = cell["n_eligible_b"].sum()
            n_bad_sum = cell["n_bad_eligible_b"].sum()
            bad_rate_loan_weighted = n_bad_sum / n_eligible_sum if n_eligible_sum else np.nan
            per_period_rates = cell.loc[cell["bad_rate_b"].notna(), "bad_rate_b"]
            median_bad_rate_agent_period = per_period_rates.median() if len(per_period_rates) else np.nan
            rows.append({
                "cell_name": f"exposuregrowth_{exp_d}_{fund_name}_{fund_d}",
                "exposure_growth_median_direction": exp_d, f"{fund_name}_direction": fund_d,
                "n_transitions": n_transitions, "n_agents": n_agents,
                "bad_rate_t_loan_weighted": bad_rate_loan_weighted,
                "median_bad_rate_t_agent_period": median_bad_rate_agent_period,
                "median_days_between": cell["days_between"].median() if n_transitions else np.nan,
            })
    return pd.DataFrame(rows)


def aggregate_by_k(all_pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per ordinal state gap k = 1, 2, 3, ... -- the Secondary
    analysis. Same weighting discipline as direction_grid()."""
    if all_pairs.empty:
        return pd.DataFrame()
    rows = []
    for k, g in all_pairs.groupby("k"):
        n_eligible_sum = g["n_eligible_b"].sum()
        n_bad_sum = g["n_bad_eligible_b"].sum()
        bad_rate_loan_weighted = n_bad_sum / n_eligible_sum if n_eligible_sum else np.nan
        per_period_rates = g.loc[g["bad_rate_b"].notna(), "bad_rate_b"]
        median_bad_rate_agent_period = per_period_rates.median() if len(per_period_rates) else np.nan
        rows.append({
            "k": int(k), "n_agents": g["agent_msisdn"].nunique(), "n_transitions": len(g),
            "median_float_growth": g["float_growth"].median(),
            "median_commission_growth": g["commission_growth"].median(),
            "median_exposure_growth_median": g["exposure_growth_median"].median(),
            "median_exposure_growth_p75": g["exposure_growth_p75"].median(),
            "median_reg_median_float": g["reg_median_float"].median(),
            "median_reg_median_commission": g["reg_median_commission"].median(),
            "median_reg_p75_float": g["reg_p75_float"].median(),
            "median_reg_p75_commission": g["reg_p75_commission"].median(),
            "bad_rate_t_loan_weighted": bad_rate_loan_weighted,
            "median_bad_rate_t_agent_period": median_bad_rate_agent_period,
        })
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def reg_band_distribution(transitions: pd.DataFrame) -> pd.DataFrame:
    """Descriptive distribution bands (<0.8 / 0.8-1.2 / >1.2) for each REG
    variant, over the Primary (k=1) transition population -- turns
    'median REG=0.85' into '62% of consecutive transitions show business
    scale growing materially faster than exposure.' Purely descriptive,
    not a policy threshold."""
    rows = []
    for col in REG_VARIANTS:
        vals = transitions[col].dropna()
        n = len(vals)
        if n == 0:
            rows.append({"reg_variant": col, "n_defined": 0, "pct_below_0.8": np.nan,
                         "pct_0.8_to_1.2": np.nan, "pct_above_1.2": np.nan})
            continue
        bands = vals.map(_reg_band)
        rows.append({
            "reg_variant": col, "n_defined": n,
            "pct_below_0.8": (bands == "<0.8").mean() * 100,
            "pct_0.8_to_1.2": (bands == "0.8-1.2").mean() * 100,
            "pct_above_1.2": (bands == ">1.2").mean() * 100,
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dataset", default="loan_episode_capacity_dataset.csv")
    ap.add_argument("--out-prefix", default="business_state_evolution")
    args = ap.parse_args(argv)

    path = Path(args.episode_dataset)
    if not path.exists():
        sys.exit(f"ERROR: {path} not found -- run build_loan_episode_capacity_dataset.py first.")
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing}")

    df_valid = df[df["fundamentals_snapshot_date"].notna()].copy()
    if df_valid.empty:
        sys.exit("ERROR: no loan has a valid measured business-state anchor -- nothing to analyze.")

    summary = build_agent_period_summary(df_valid)
    summary_ordered = order_periods(summary)
    n_multi_period_agents = (summary_ordered.groupby("agent_msisdn")["period_index"].max() >= 2).sum()
    print(f"Built {len(summary_ordered):,} agent-period unit(s) across {summary_ordered['agent_msisdn'].nunique():,} "
          f"agent(s); {n_multi_period_agents:,} agent(s) have >=2 periods (orderable for evolution).")

    all_pairs = build_all_chronological_pairs(summary_ordered)
    if all_pairs.empty:
        sys.exit("ERROR: no agent has >=2 business-state periods -- nothing to analyze.")
    all_pairs.to_csv(f"{args.out_prefix}_all_pairs.csv", index=False)

    transitions = all_pairs[all_pairs["k"] == 1].reset_index(drop=True)
    print(f"Primary analysis: {len(transitions):,} consecutive (k=1) period transition(s) across "
          f"{transitions['agent_msisdn'].nunique():,} agent(s). Median days between consecutive "
          f"periods: {transitions['days_between'].median():.1f}.")

    print(f"\n{'#' * 100}\n# PRIMARY: consecutive period transitions -- direction grids\n{'#' * 100}")
    for fund_name in ["float", "commission"]:
        print(f"\n{'=' * 100}\nExposureGrowthMedianDirection x {fund_name.capitalize()}Direction grid\n{'=' * 100}")
        grid = direction_grid(transitions, fund_name)
        with pd.option_context("display.float_format", "{:,.3f}".format, "display.max_columns", None, "display.width", 240):
            print(grid.to_string(index=False))
        grid.to_csv(f"{args.out_prefix}_primary_grid_{fund_name}.csv", index=False)

    print(f"\n{'=' * 100}\nREG distribution bands (Primary, k=1 population)\n{'=' * 100}")
    reg_dist = reg_band_distribution(transitions)
    with pd.option_context("display.float_format", "{:,.1f}".format):
        print(reg_dist.to_string(index=False))
    reg_dist.to_csv(f"{args.out_prefix}_primary_reg_bands.csv", index=False)

    print(f"\n{'#' * 100}\n# SECONDARY: all chronological pairs, stratified by k (ordinal state gap)\n{'#' * 100}")
    by_k = aggregate_by_k(all_pairs)
    with pd.option_context("display.float_format", "{:,.3f}".format, "display.max_columns", None, "display.width", 240):
        print(by_k.to_string(index=False))
    by_k.to_csv(f"{args.out_prefix}_secondary_by_k.csv", index=False)

    k1_row = by_k[by_k["k"] == 1].iloc[0] if (by_k["k"] == 1).any() else None
    if k1_row is not None:
        manual_median_exposure_growth = transitions["exposure_growth_median"].median()
        print(f"\nk=1 cross-check: Secondary table's median_exposure_growth_median at k=1 is "
              f"{k1_row['median_exposure_growth_median']:.4f}; recomputed directly from the Primary "
              f"transitions is {manual_median_exposure_growth:.4f} (must match -- both read the same "
              f"underlying pair rows).")

    print(f"\n{'#' * 100}")
    print("Causal caveat (printed every run)")
    print(f"{'#' * 100}")
    print("This is observational, not causal evidence. A within-agent performance difference across\n"
          "business-state transitions must be reported as 'agents who experienced this transition\n"
          "showed X difference in subsequent performance' -- never as the transition CAUSING X.\n"
          "Later/higher exposure often coincides with more lender information, calendar effects, and\n"
          "selection that remain live explanations.")


if __name__ == "__main__":
    main()
