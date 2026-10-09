"""
check_float_commission_band_overlap.py
==========================================
Lightweight, outcome-free diagnostic, motivated directly by Analysis 4's
float-vs-commission heterogeneity finding: the "persistently locked"
first-step bands (delta_br_loan_weighted_pp > 5pp through every swept
tolerance) are a DIFFERENT set of band labels on each axis -- float D2/D3/
D7 vs. commission D4/D5/D10, zero overlap. Because each axis's deciles are
constructed independently, that zero overlap does not by itself mean the
two fundamentals are ranking agent-periods differently -- D7-float and
D7-commission have no necessary population equivalence. This script is
what actually answers that, before building any joint float x commission
frontier (which would fragment the same-agent/same-anchor paired evidence
into up to 100 cells, destroying exactly the common support that makes
the current one-dimensional results persuasive).

QUESTION: are float-based and commission-based business-state bands
classifying the same agent-periods into the same relative-scale rank
(genuine redundancy), or materially reclassifying them (an omitted
dimension partly explaining the one-dimensional frontier irregularities)?

GRAIN: agent-period UNITS, not loans -- so a handful of high-frequency
borrowers cannot dominate the joint population structure the way loan-grain
counting would let them.

Reports, at agent-period grain:
  1. The 10x10 (or fewer, per _qcut_safe's graceful degradation) count
     matrix N(float=Di, commission=Dj).
  2. The float-row-normalized matrix P(commission=Dj | float=Di) -- each
     float band's own distribution across commission bands.
  3. The commission-column-normalized matrix P(float=Di | commission=Dj)
     -- the reverse conditional, each commission band's own distribution
     across float bands.
  4. Global rank-agreement diagnostics: Spearman correlation of the raw
     underlying fundamentals; Spearman correlation of the decile-rank
     integers (coarser, since many units share a decile); exact-decile-
     match %; within-+/-1-decile %; >=3-decile-separation %.
  5. Four focused extracts (just readable slices of #2/#3, not new
     computation) for the specific bands Analysis 4's frontier flagged as
     notable: P(commission | float=D7), P(commission | float=D8),
     P(float | commission=D5), P(float | commission=D10).

NO OUTCOMES. NO EXPOSURE TIERS. NO FRONTIER CLASSIFICATION. NO MODELING.
Adding any of those would turn a diagnostic about feature structure into
another frontier search, re-subdividing the paired population until the
evidence goes sparse -- exactly what Analysis 4's independent float/
commission segmentation was designed to avoid.

Restated (never imported, one-way scripts/ layering convention):
build_agent_period_summary/assign_bands/_qcut_safe, minimal subset of
scripts/derive_capacity_frontier_from_business_state.py's versions (no
exposure/outcome columns needed for this diagnostic at all).

Usage:
    python scripts\\check_float_commission_band_overlap.py ^
        --episode-dataset loan_episode_capacity_dataset.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

N_DECILES_DEFAULT = 10
FOCUS_FLOAT_BANDS = ["D7", "D8"]
FOCUS_COMMISSION_BANDS = ["D5", "D10"]

REQUIRED_COLS = [
    "agent_msisdn", "fundamentals_snapshot_date", "float_activity_value_1m", "commission",
]


def _qcut_safe(s: pd.Series, q: int, prefix: str = "D") -> tuple:
    """qcut with duplicate bin edges dropped; falls back to as many bins as
    the data supports. Restated verbatim from derive_capacity_frontier_
    from_business_state.py (one-way scripts/ layering convention)."""
    try:
        codes, bins = pd.qcut(s, q, duplicates="drop", retbins=True, labels=False)
    except ValueError:
        codes, bins = None, None
    n_bins = (len(bins) - 1) if bins is not None else 0
    if n_bins <= 0:
        return pd.Series(f"{prefix}1", index=s.index), 1
    labels = [f"{prefix}{i + 1}" for i in range(n_bins)]
    return codes.map(dict(enumerate(labels))), n_bins


def build_agent_period_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (agent_msisdn, fundamentals_snapshot_date) unit --
    minimal subset carrying only float_activity_value_1m and commission
    (no exposure/outcome columns needed for a support-structure check)."""
    key = ["agent_msisdn", "fundamentals_snapshot_date"]
    gb = df.groupby(key, sort=False)
    summary = pd.concat([
        gb["float_activity_value_1m"].first().rename("float_activity_value_1m"),
        gb["commission"].first().rename("commission"),
    ], axis=1).reset_index()
    return summary


def assign_bands(agent_period_summary: pd.DataFrame, fund_col: str, n_bands: int = N_DECILES_DEFAULT) -> tuple:
    """Restated verbatim from derive_capacity_frontier_from_business_state.py."""
    valid_mask = agent_period_summary[fund_col].notna() & (agent_period_summary[fund_col] > 0)
    out = pd.Series(np.nan, index=agent_period_summary.index, dtype=object)
    if not valid_mask.any():
        return out, []
    band_series, n_bins = _qcut_safe(agent_period_summary.loc[valid_mask, fund_col], n_bands)
    labels = [f"D{i + 1}" for i in range(n_bins)]
    out.loc[valid_mask] = band_series.values
    return out, labels


def _band_rank(label) -> float:
    """'D7' -> 7.0. NaN-safe (returns NaN for a missing/invalid band)."""
    if pd.isna(label):
        return np.nan
    return float(str(label)[1:])


def build_overlap_tables(agent_period_summary: pd.DataFrame, n_bands: int) -> tuple:
    """Returns (counts, float_row_pct, commission_col_pct, float_labels,
    commission_labels, valid_units_df).

    counts: N(float=Di, commission=Dj), agent-period grain.
    float_row_pct: P(commission=Dj | float=Di) -- each row sums to 100.
    commission_col_pct: P(float=Di | commission=Dj) -- each column sums to 100.
    valid_units_df: the subset with both bands defined, plus numeric rank
        columns, for the rank-agreement diagnostics.
    """
    summary = agent_period_summary.copy()
    summary["float_band"], float_labels = assign_bands(summary, "float_activity_value_1m", n_bands)
    summary["commission_band"], commission_labels = assign_bands(summary, "commission", n_bands)

    valid = summary[summary["float_band"].notna() & summary["commission_band"].notna()].copy()
    counts = pd.crosstab(valid["float_band"], valid["commission_band"]).reindex(
        index=float_labels, columns=commission_labels, fill_value=0)

    row_totals = counts.sum(axis=1)
    float_row_pct = counts.div(row_totals.replace(0, np.nan), axis=0) * 100

    col_totals = counts.sum(axis=0)
    commission_col_pct = counts.div(col_totals.replace(0, np.nan), axis=1) * 100

    valid["float_band_rank"] = valid["float_band"].map(_band_rank)
    valid["commission_band_rank"] = valid["commission_band"].map(_band_rank)

    return counts, float_row_pct, commission_col_pct, float_labels, commission_labels, valid


def compute_rank_agreement_stats(valid_units_df: pd.DataFrame) -> dict:
    """Global diagnostics on the agent-period units with both bands
    defined: Spearman correlation of the RAW underlying fundamentals (full
    resolution), Spearman correlation of the DECILE-RANK integers (coarser
    -- many units tie within a decile), exact-decile-match %,
    within-+/-1-decile %, and >=3-decile-separation %."""
    n = len(valid_units_df)
    if n < 2:
        return {"n_units": n, "spearman_raw_fundamentals": np.nan, "spearman_decile_ranks": np.nan,
                "pct_exact_decile_match": np.nan, "pct_within_1_decile": np.nan,
                "pct_3plus_decile_separation": np.nan}

    raw_corr, _ = spearmanr(valid_units_df["float_activity_value_1m"], valid_units_df["commission"])
    decile_corr, _ = spearmanr(valid_units_df["float_band_rank"], valid_units_df["commission_band_rank"])
    diff = (valid_units_df["float_band_rank"] - valid_units_df["commission_band_rank"]).abs()

    return {
        "n_units": n,
        "spearman_raw_fundamentals": raw_corr,
        "spearman_decile_ranks": decile_corr,
        "pct_exact_decile_match": (diff == 0).mean() * 100,
        "pct_within_1_decile": (diff <= 1).mean() * 100,
        "pct_3plus_decile_separation": (diff >= 3).mean() * 100,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dataset", default="loan_episode_capacity_dataset.csv")
    ap.add_argument("--out-prefix", default="float_commission_overlap")
    ap.add_argument("--n-bands", type=int, default=N_DECILES_DEFAULT)
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

    print(f"\n{'#' * 100}\nNO OUTCOMES, NO EXPOSURE TIERS, NO FRONTIER CLASSIFICATION, NO MODELING -- "
          f"purely a feature-structure check\n{'#' * 100}")
    print(
        "Question: are float-based and commission-based business-state bands classifying the\n"
        "same agent-periods into the same relative-scale rank, or materially reclassifying them?\n"
        "This does not itself decide whether a joint float x commission frontier is warranted --\n"
        "it only checks whether the two fundamentals are redundant or genuinely different rankings.\n"
        "Grain is agent-period UNITS, not loans, so high-frequency borrowers cannot dominate this."
    )

    agent_period_summary = build_agent_period_summary(df_valid)
    print(f"\n{len(agent_period_summary):,} agent-period unit(s) across "
          f"{agent_period_summary['agent_msisdn'].nunique():,} agent(s).")

    counts, float_row_pct, commission_col_pct, float_labels, commission_labels, valid = build_overlap_tables(
        agent_period_summary, args.n_bands)

    print("\n-- 1. Agent-period counts: float band (rows) x commission band (columns) --")
    with pd.option_context("display.width", 180):
        print(counts.to_string())

    print("\n-- 2. P(commission band | float band): each float band's own distribution "
          "across commission bands (rows sum to 100) --")
    with pd.option_context("display.width", 180, "display.float_format", "{:.1f}".format):
        print(float_row_pct.to_string())

    print("\n-- 3. P(float band | commission band): each commission band's own distribution "
          "across float bands (columns sum to 100) --")
    with pd.option_context("display.width", 180, "display.float_format", "{:.1f}".format):
        print(commission_col_pct.to_string())

    stats = compute_rank_agreement_stats(valid)
    print("\n-- 4. Global rank-agreement diagnostics --")
    print(f"  n (units with both bands defined):     {stats['n_units']:,}")
    print(f"  Spearman corr, RAW fundamentals:        {stats['spearman_raw_fundamentals']:.3f}")
    print(f"  Spearman corr, DECILE-RANK integers:    {stats['spearman_decile_ranks']:.3f} "
          f"(coarser -- many units tie within a decile)")
    print(f"  Exact decile match:                     {stats['pct_exact_decile_match']:.1f}%")
    print(f"  Within +/-1 decile:                     {stats['pct_within_1_decile']:.1f}%")
    print(f"  >=3-decile separation:                  {stats['pct_3plus_decile_separation']:.1f}%")

    print("\n-- 5. Focused extracts (readable slices of #2/#3 above, not new computation) --")
    for band in FOCUS_FLOAT_BANDS:
        if band in float_row_pct.index:
            print(f"\n  P(commission band | float = {band}):")
            with pd.option_context("display.float_format", "{:.1f}".format):
                print("    " + float_row_pct.loc[band].to_string().replace("\n", "\n    "))
        else:
            print(f"\n  P(commission band | float = {band}): band not present (fewer bins produced)")
    for band in FOCUS_COMMISSION_BANDS:
        if band in commission_col_pct.columns:
            print(f"\n  P(float band | commission = {band}):")
            with pd.option_context("display.float_format", "{:.1f}".format):
                print("    " + commission_col_pct[band].to_string().replace("\n", "\n    "))
        else:
            print(f"\n  P(float band | commission = {band}): band not present (fewer bins produced)")

    counts.to_csv(f"{args.out_prefix}_counts.csv")
    float_row_pct.to_csv(f"{args.out_prefix}_float_row_pct.csv")
    commission_col_pct.to_csv(f"{args.out_prefix}_commission_col_pct.csv")
    print(f"\nWritten: {args.out_prefix}_counts.csv, {args.out_prefix}_float_row_pct.csv, "
          f"{args.out_prefix}_commission_col_pct.csv")


if __name__ == "__main__":
    main()
