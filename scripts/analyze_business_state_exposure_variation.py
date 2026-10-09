"""
analyze_business_state_exposure_variation.py
================================================
Level 2 of the business-state-period evidence hierarchy (cadence-gate
pivot). The question this script answers, stated first: for the same
agent, under the same *measured business-state anchor*, what happened
across the different exposure tiers they actually experienced? This
exploits the daily-loan frequency (median consecutive-loan spacing of 1
day, per scripts/analyze_loan_frequency_for_snapshot_cadence.py) rather
than fighting it -- a daily working-capital agent can easily be observed
at several exposure tiers within one measured business-state period,
giving a genuinely paired, within-unit comparison that the retired
loan-to-loan design never had.

TERMINOLOGY (say this, not "same business state"): every loan sharing
(agent_msisdn, fundamentals_snapshot_date) shares the identical snapshot
row's fundamental values -- but a May-2 loan and a May-29 loan anchored to
the same April-30 snapshot have the same MEASURED PRE-PERIOD business
state, not necessarily the same ACTUAL business state at the moment of
each loan. This script's design is: same agent + same pre-period
measurement + different exposure -- not "everything except exposure held
constant." Still useful, genuinely observational evidence.

UNIT OF ANALYSIS: Agent x BusinessStatePeriod = (agent_msisdn,
fundamentals_snapshot_date) -- the measured business-state anchor. Rows
with fundamentals_snapshot_date NaN (no valid prior snapshot) are excluded
(no anchor to measure against); build_loan_episode_capacity_dataset.py and
check_loan_episode_dataset_integrity.py already report on them separately.

FOUR OUTPUTS:
  Step 1 -- agent-period summary (one row per unit).
  Step 2 -- Table A: descriptive decile(of unit's fundamental) x
            exposure-tier cross-tab (n_loans, n_agents, bad_rate).
  Step 3 -- Table B: Exposure Intensity (EI = exposure / fundamental,
            guarded) -> performance, within each business-state band.
            Operationalizes "EI -> P(3+DPD) within business-state bands,"
            the metric that replaces the retired loan-to-loan
            delta-fundamentals design.
  Step 4 -- Table C: paired same-agent-same-anchor multi-tier comparison.
            Reuses, restated independently, archive/analyze_episode_
            exposure_escalation_matrix.py's pair-aggregation mechanic
            (build_agent_pair_rows/aggregate_pair_table), re-keyed from
            "agent" to "agent-period" -- this fixes that script's weakest
            point (fundamentals compared via a median across an agent's
            whole, uncontrolled history) by construction: within one unit
            the fundamental is the SAME real value for every loan, not an
            approximation. Tier pairs stay UNORDERED here (no temporal
            "before/after" claim within one period -- that is explicitly
            Level 3's job, in analyze_business_state_evolution.py).

POOLED AND PAIRED, kept distinct in Table C (never blended into one
number): pooled = portfolio-level, loan-weighted, across all qualifying
units' loans at each tier (bad_rate_*_loan_weighted). Paired = within-unit
BR_{u,L} = bad eligible / eligible loans at tier L for unit u, and
delta_BR_u = BR_{u,L_high} - BR_{u,L_low}, defined only when both sides
are defined for that unit -- reported as median_delta_bad_rate and
pct_delta_bad_positive/zero/negative, with its own n_units_outcome_both
denominator, distinct from n_units_pair (the assignment-side count).

Level hierarchy this script sits in (see the plan's "Cadence-gate pivot"
section): Level 1 (analyze_business_state_exposure_performance.py)
compares comparable measured business states at different exposures,
cross-sectionally. Level 2 (this script) tightens to the same agent + same
measured business-state anchor. Level 3 (analyze_business_state_
evolution.py) adds genuine temporal movement in the agent's measured
business state. None of the three establish a causal effect of increasing
a limit -- progressively tighter observational evidence, not a causal
estimate.

Restated independently (one-way scripts/ layering convention): the 7-tier
exposure set, _qcut_safe-style graceful degradation, and the pair-
aggregation mechanic, consistent with every other script in this rebuild.

Usage:
    python scripts\\analyze_business_state_exposure_variation.py ^
        --episode-dataset loan_episode_capacity_dataset.csv
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPOSURE_TIERS_UGX = [50_000, 100_000, 250_000, 350_000, 500_000, 750_000, 1_000_000]
FUNDAMENTALS_FOR_BANDS = {"float_activity_value_1m": "float", "commission": "commission"}
N_DECILES = 10
MIN_CELL_N = 10

REQUIRED_COLS = [
    "agent_msisdn", "fundamentals_snapshot_date", "disbursement_amount_ugx",
    "float_activity_value_1m", "commission", "bad_state_3dpd_30d", "label_eligible_30d",
]


def _qcut_safe(s: pd.Series, q: int) -> tuple:
    """qcut with duplicate bin edges dropped; falls back to as many bins as
    the data supports (minimum 1) rather than silently dropping every row
    the way a bare pd.qcut would on too-few-rows/too-little-variation.
    Restated independently from analyze_business_state_exposure_
    performance.py's identically-behaving helper (one-way scripts/
    layering convention). Returns (labeled band series, n_bins)."""
    try:
        codes, bins = pd.qcut(s, q, duplicates="drop", retbins=True, labels=False)
    except ValueError:
        codes, bins = None, None
    n_bins = (len(bins) - 1) if bins is not None else 0
    if n_bins <= 0:
        return pd.Series("D1", index=s.index), 1
    labels = [f"D{i + 1}" for i in range(n_bins)]
    return codes.map(dict(enumerate(labels))), n_bins


def build_agent_period_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (agent_msisdn, fundamentals_snapshot_date) unit. The
    period's own float/commission values are carried through via .iloc[0]
    -- constant within the group by construction (every loan in a unit
    shares the identical matched snapshot row), no aggregation ambiguity."""
    rows = []
    for (agent, period), g in df.groupby(["agent_msisdn", "fundamentals_snapshot_date"]):
        elig = g[g["label_eligible_30d"] == 1]
        n_eligible = len(elig)
        bad_rate_period = elig["bad_state_3dpd_30d"].mean() if n_eligible else np.nan
        tiers = sorted(g["disbursement_amount_ugx"].unique())
        rows.append({
            "agent_msisdn": agent, "fundamentals_snapshot_date": period,
            "n_loans_total": len(g), "n_loans_eligible": n_eligible,
            "distinct_tiers_experienced": len(tiers),
            "median_exposure": g["disbursement_amount_ugx"].median(),
            "p75_exposure": g["disbursement_amount_ugx"].quantile(0.75),
            "max_exposure": g["disbursement_amount_ugx"].max(),
            "bad_rate_period": bad_rate_period,
            "float_activity_value_1m": g["float_activity_value_1m"].iloc[0],
            "commission": g["commission"].iloc[0],
        })
    return pd.DataFrame(rows)


def assign_bands(agent_period_summary: pd.DataFrame, fund_col: str) -> tuple:
    """Decile-of-unit's-own-fundamental assignment, computed once per unit
    (not per loan -- the fundamental is shared by every loan in a unit)."""
    valid_mask = agent_period_summary[fund_col].notna() & (agent_period_summary[fund_col] > 0)
    out = pd.Series(np.nan, index=agent_period_summary.index, dtype=object)
    if not valid_mask.any():
        return out, []
    band_series, n_bins = _qcut_safe(agent_period_summary.loc[valid_mask, fund_col], N_DECILES)
    labels = [f"D{i + 1}" for i in range(n_bins)]
    out.loc[valid_mask] = band_series.values
    return out, labels


def table_a(df_with_bands: pd.DataFrame, band_col: str, labels: list) -> tuple:
    """Descriptive cross-tab: decile-of-unit's-fundamental x exposure-tier
    -> n_loans, n_agents (distinct), bad_rate (NaN below MIN_CELL_N
    eligible loans). Context for Table C, not itself a paired comparison."""
    working = df_with_bands[df_with_bands[band_col].notna()]
    if working.empty:
        empty = pd.DataFrame(index=labels, columns=EXPOSURE_TIERS_UGX)
        return empty, empty, empty
    n_loans = pd.crosstab(working[band_col], working["disbursement_amount_ugx"]).reindex(
        index=labels, columns=EXPOSURE_TIERS_UGX, fill_value=0)
    n_agents = working.groupby([band_col, "disbursement_amount_ugx"])["agent_msisdn"].nunique().unstack().reindex(
        index=labels, columns=EXPOSURE_TIERS_UGX, fill_value=0)
    elig = working[working["label_eligible_30d"] == 1]
    if elig.empty:
        bad_rate = pd.DataFrame(np.nan, index=labels, columns=EXPOSURE_TIERS_UGX)
    else:
        bad_count = elig.groupby([band_col, "disbursement_amount_ugx"])["bad_state_3dpd_30d"].sum().unstack()
        elig_count = elig.groupby([band_col, "disbursement_amount_ugx"])["bad_state_3dpd_30d"].count().unstack()
        bad_rate = (bad_count / elig_count).reindex(index=labels, columns=EXPOSURE_TIERS_UGX)
        elig_count = elig_count.reindex(index=labels, columns=EXPOSURE_TIERS_UGX, fill_value=0)
        bad_rate = bad_rate.where(elig_count >= MIN_CELL_N)
    return n_loans, n_agents, bad_rate


def compute_ei(df: pd.DataFrame, exposure_col: str, fund_col: str) -> pd.Series:
    """Exposure Intensity = exposure / fundamental, with a denominator
    guard: fundamental must be > 0, else missing -- never a silent
    inf/nan propagated."""
    valid = df[fund_col].notna() & (df[fund_col] > 0)
    ei = pd.Series(np.nan, index=df.index)
    ei.loc[valid] = df.loc[valid, exposure_col] / df.loc[valid, fund_col]
    return ei


def table_b(df_with_ei: pd.DataFrame, band_col: str, ei_col: str, band_labels: list) -> pd.DataFrame:
    """EI -> performance within each business-state band. EI deciles are
    computed SEPARATELY within each band's own loans (not globally) --
    operationalizes 'EI -> P(3+DPD) within business-state bands'."""
    rows = []
    for band in band_labels:
        sub = df_with_ei[(df_with_ei[band_col] == band) & df_with_ei[ei_col].notna()]
        if sub.empty:
            continue
        ei_decile, n_bins = _qcut_safe(sub[ei_col], N_DECILES)
        sub = sub.copy()
        sub["_ei_decile"] = ei_decile
        for i in range(n_bins):
            ei_lab = f"EI{i + 1}"
            cell = sub[sub["_ei_decile"] == ei_lab]
            n_loans = len(cell)
            elig = cell[cell["label_eligible_30d"] == 1]
            n_eligible = len(elig)
            bad_rate = elig["bad_state_3dpd_30d"].mean() if n_eligible >= MIN_CELL_N else np.nan
            rows.append({"business_state_band": band, "ei_decile": ei_lab,
                         "n_loans": n_loans, "n_eligible": n_eligible, "bad_rate": bad_rate})
    return pd.DataFrame(rows)


def build_unit_pair_rows(df_with_ei: pd.DataFrame) -> pd.DataFrame:
    """One row per (agent_msisdn, fundamentals_snapshot_date, tier_low,
    tier_high) for every unordered pair of tiers that unit's loans
    actually hit. Units with only 1 tier contribute zero rows (they
    feed Table A/B only)."""
    rows = []
    for (agent, period), g in df_with_ei.groupby(["agent_msisdn", "fundamentals_snapshot_date"]):
        tiers = sorted(g["disbursement_amount_ugx"].unique())
        if len(tiers) < 2:
            continue
        for tier_low, tier_high in itertools.combinations(tiers, 2):
            low = g[g["disbursement_amount_ugx"] == tier_low]
            high = g[g["disbursement_amount_ugx"] == tier_high]
            low_elig = low[low["label_eligible_30d"] == 1]
            high_elig = high[high["label_eligible_30d"] == 1]
            n_eligible_low, n_eligible_high = len(low_elig), len(high_elig)
            n_bad_low = int(low_elig["bad_state_3dpd_30d"].sum())
            n_bad_high = int(high_elig["bad_state_3dpd_30d"].sum())
            br_low = low_elig["bad_state_3dpd_30d"].mean() if n_eligible_low else np.nan
            br_high = high_elig["bad_state_3dpd_30d"].mean() if n_eligible_high else np.nan
            delta_br = (br_high - br_low) if pd.notna(br_low) and pd.notna(br_high) else np.nan
            rows.append({
                "agent_msisdn": agent, "fundamentals_snapshot_date": period,
                "tier_low": tier_low, "tier_high": tier_high,
                "n_loans_low": len(low), "n_loans_high": len(high),
                "n_eligible_low": n_eligible_low, "n_eligible_high": n_eligible_high,
                "n_bad_low": n_bad_low, "n_bad_high": n_bad_high,
                "br_low": br_low, "br_high": br_high, "delta_br": delta_br,
                "median_ei_float_low": low["ei_float"].median(), "median_ei_float_high": high["ei_float"].median(),
                "median_ei_commission_low": low["ei_commission"].median(),
                "median_ei_commission_high": high["ei_commission"].median(),
            })
    return pd.DataFrame(rows)


def aggregate_unit_pairs(unit_pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per unordered tier pair, aggregating across every qualifying
    (agent, period) unit. Pooled (loan-weighted) and paired (within-unit
    delta_BR) statistics are kept distinct -- never blended."""
    if unit_pairs.empty:
        return pd.DataFrame()
    rows = []
    for (tl, th), g in unit_pairs.groupby(["tier_low", "tier_high"]):
        n_units_pair = len(g)
        n_loans_lower, n_loans_higher = int(g["n_loans_low"].sum()), int(g["n_loans_high"].sum())
        n_eligible_lower, n_eligible_higher = int(g["n_eligible_low"].sum()), int(g["n_eligible_high"].sum())
        n_bad_lower, n_bad_higher = int(g["n_bad_low"].sum()), int(g["n_bad_high"].sum())
        bad_rate_lower_loan_weighted = n_bad_lower / n_eligible_lower if n_eligible_lower else np.nan
        bad_rate_higher_loan_weighted = n_bad_higher / n_eligible_higher if n_eligible_higher else np.nan

        both = g[g["br_low"].notna() & g["br_high"].notna()]
        n_units_outcome_both = len(both)
        if n_units_outcome_both:
            deltas = both["delta_br"]
            median_delta_bad_rate = deltas.median()
            pct_delta_bad_positive = (deltas > 0).mean() * 100
            pct_delta_bad_zero = (deltas == 0).mean() * 100
            pct_delta_bad_negative = (deltas < 0).mean() * 100
        else:
            median_delta_bad_rate = pct_delta_bad_positive = pct_delta_bad_zero = pct_delta_bad_negative = np.nan

        rows.append({
            "tier_low": tl, "tier_high": th, "pair_label": f"{tl:,} vs {th:,}",
            "n_units_pair": n_units_pair,
            "n_loans_lower": n_loans_lower, "n_eligible_lower": n_eligible_lower,
            "bad_rate_lower_loan_weighted": bad_rate_lower_loan_weighted,
            "n_loans_higher": n_loans_higher, "n_eligible_higher": n_eligible_higher,
            "bad_rate_higher_loan_weighted": bad_rate_higher_loan_weighted,
            "n_units_outcome_both": n_units_outcome_both,
            "median_delta_bad_rate": median_delta_bad_rate,
            "pct_delta_bad_positive": pct_delta_bad_positive,
            "pct_delta_bad_zero": pct_delta_bad_zero,
            "pct_delta_bad_negative": pct_delta_bad_negative,
            "median_ei_float_lower": g["median_ei_float_low"].median(),
            "median_ei_float_higher": g["median_ei_float_high"].median(),
            "median_ei_commission_lower": g["median_ei_commission_low"].median(),
            "median_ei_commission_higher": g["median_ei_commission_high"].median(),
        })
    return pd.DataFrame(rows).sort_values(["tier_low", "tier_high"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dataset", default="loan_episode_capacity_dataset.csv")
    ap.add_argument("--out-prefix", default="business_state_exposure_variation")
    args = ap.parse_args(argv)

    path = Path(args.episode_dataset)
    if not path.exists():
        sys.exit(f"ERROR: {path} not found -- run build_loan_episode_capacity_dataset.py first.")
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing}")

    df_valid = df[df["fundamentals_snapshot_date"].notna()].copy()
    n_excluded = len(df) - len(df_valid)
    print(f"Loans with a valid measured business-state anchor: {len(df_valid):,} of {len(df):,} "
          f"({n_excluded:,} excluded -- no matched prior snapshot).")
    if df_valid.empty:
        sys.exit("ERROR: no loan has a valid measured business-state anchor -- nothing to analyze.")

    agent_period_summary = build_agent_period_summary(df_valid)
    n_units = len(agent_period_summary)
    n_agents = agent_period_summary["agent_msisdn"].nunique()
    print(f"Built {n_units:,} agent-period unit(s) across {n_agents:,} agent(s).")

    agent_period_summary["float_band"], float_labels = assign_bands(agent_period_summary, "float_activity_value_1m")
    agent_period_summary["commission_band"], commission_labels = assign_bands(agent_period_summary, "commission")
    agent_period_summary.to_csv(f"{args.out_prefix}_agent_period_summary.csv", index=False)

    df_with_bands = df_valid.merge(
        agent_period_summary[["agent_msisdn", "fundamentals_snapshot_date", "float_band", "commission_band"]],
        on=["agent_msisdn", "fundamentals_snapshot_date"], how="left",
    )
    df_with_bands["ei_float"] = compute_ei(df_with_bands, "disbursement_amount_ugx", "float_activity_value_1m")
    df_with_bands["ei_commission"] = compute_ei(df_with_bands, "disbursement_amount_ugx", "commission")

    band_cols = {"float_activity_value_1m": ("float_band", float_labels), "commission": ("commission_band", commission_labels)}
    for fund_col, fund_name in FUNDAMENTALS_FOR_BANDS.items():
        band_col, labels = band_cols[fund_col]
        print(f"\n{'#' * 100}\n# Table A ({fund_name}-banded): decile(measured business state) x exposure tier\n{'#' * 100}")
        n_loans, n_agents_tbl, bad_rate = table_a(df_with_bands, band_col, labels)
        print("-- N (loans) --")
        print(n_loans.to_string())
        print("-- N (distinct agents) --")
        print(n_agents_tbl.to_string())
        print(f"-- Bad rate (NaN where N eligible < {MIN_CELL_N}) --")
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(bad_rate.to_string())
        n_loans.to_csv(f"{args.out_prefix}_table_a_{fund_name}_n_loans.csv")
        n_agents_tbl.to_csv(f"{args.out_prefix}_table_a_{fund_name}_n_agents.csv")
        bad_rate.to_csv(f"{args.out_prefix}_table_a_{fund_name}_bad_rate.csv")

        ei_col = "ei_float" if fund_name == "float" else "ei_commission"
        print(f"\n{'=' * 100}\nTable B ({fund_name}-EI): Exposure Intensity -> performance within measured business-state bands\n{'=' * 100}")
        tb = table_b(df_with_bands, band_col, ei_col, labels)
        if tb.empty:
            print("  (no qualifying loans)")
        else:
            with pd.option_context("display.float_format", "{:.3f}".format, "display.max_rows", None):
                print(tb.to_string(index=False))
            tb.to_csv(f"{args.out_prefix}_table_b_{fund_name}.csv", index=False)

    unit_pairs = build_unit_pair_rows(df_with_bands)
    n_units_multi_tier = unit_pairs[["agent_msisdn", "fundamentals_snapshot_date"]].drop_duplicates().shape[0] if not unit_pairs.empty else 0
    print(f"\n{'#' * 100}\n# Table C: paired same-agent-same-anchor multi-tier comparison\n{'#' * 100}")
    print(f"{n_units_multi_tier:,} of {n_units:,} agent-period unit(s) experienced >=2 distinct exposure tiers.")
    pair_table = aggregate_unit_pairs(unit_pairs)
    if pair_table.empty:
        print("  (no unit experienced >=2 distinct exposure tiers -- nothing to report)")
    else:
        display_cols = [
            "pair_label", "n_units_pair", "n_loans_lower", "n_eligible_lower", "bad_rate_lower_loan_weighted",
            "n_loans_higher", "n_eligible_higher", "bad_rate_higher_loan_weighted",
            "n_units_outcome_both", "median_delta_bad_rate", "pct_delta_bad_positive",
            "pct_delta_bad_zero", "pct_delta_bad_negative",
        ]
        with pd.option_context("display.float_format", "{:,.3f}".format, "display.max_columns", None, "display.width", 240):
            print(pair_table[display_cols].to_string(index=False))
        pair_table.to_csv(f"{args.out_prefix}_table_c_pairs.csv", index=False)
        unit_pairs.to_csv(f"{args.out_prefix}_table_c_unit_pairs.csv", index=False)

    print(f"\n{'#' * 100}")
    print("Causal caveat (printed every run)")
    print(f"{'#' * 100}")
    print("This is observational, not causal evidence. A within-unit (same agent, same measured\n"
          "business-state anchor) performance difference across exposure tiers must be reported as\n"
          "'the same agent-periods exhibited X difference in observed performance at the higher vs.\n"
          "lower tier' -- never as the higher tier CAUSING X. Selection into exposure tier, lender\n"
          "information, and policy all remain live explanations.")


if __name__ == "__main__":
    main()
