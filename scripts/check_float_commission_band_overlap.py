"""
check_float_commission_band_overlap.py
==========================================
Lightweight, outcome-free diagnostic, motivated directly by Analysis 4's
float-vs-commission heterogeneity finding (float D7 behaves like the
"locked" float D2/D3 bands, not like float D8 -- breaking any monotonic
float-decile story). Before building any joint float x commission
frontier -- which would fragment the same-agent/same-anchor paired
evidence into up to 100 cells, destroying exactly the common support that
makes the current one-dimensional results persuasive -- first check
whether the two fundamentals are even putting different agent-periods
into different relative-scale ranks at all.

QUESTION: are float-based and commission-based business-state bands
classifying the same agent-periods into the same relative-scale rank, or
materially reclassifying them? If most mass sits near the diagonal, the
two fundamentals are largely redundant and a joint frontier may add
little. If there is substantial off-diagonal mass, that is a plausible
explanation for some of the one-dimensional heterogeneity already found
(e.g. float D7 landing in a different, weaker commission band than float
D8's agent-periods).

NO OUTCOMES. NO EXPOSURE TIERS. NO MODELING. Just:
    float decile x commission decile -> N agent-period units,
plus each float decile's row-percentage distribution across commission
deciles.

Restated (never imported, one-way scripts/ layering convention):
build_agent_period_summary/assign_bands, minimal subset of
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

N_DECILES_DEFAULT = 10

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


def build_overlap_table(agent_period_summary: pd.DataFrame, n_bands: int) -> tuple:
    """float decile (rows) x commission decile (columns) -> N agent-period
    units, plus row percentages (each float band's own distribution
    across commission bands)."""
    summary = agent_period_summary.copy()
    summary["float_band"], float_labels = assign_bands(summary, "float_activity_value_1m", n_bands)
    summary["commission_band"], commission_labels = assign_bands(summary, "commission", n_bands)

    valid = summary[summary["float_band"].notna() & summary["commission_band"].notna()]
    counts = pd.crosstab(valid["float_band"], valid["commission_band"]).reindex(
        index=float_labels, columns=commission_labels, fill_value=0)
    row_totals = counts.sum(axis=1)
    row_pct = counts.div(row_totals.replace(0, np.nan), axis=0) * 100
    return counts, row_pct, float_labels, commission_labels


def diagonal_mass_pct(counts: pd.DataFrame, float_labels: list, commission_labels: list) -> float:
    """Share of agent-periods where the float-band label equals the
    commission-band label (e.g. float D7 AND commission D7) -- a rough,
    label-based proxy for 'same relative rank on both axes'. Only sums
    labels present on both axes (band counts can differ if one
    fundamental's values support fewer distinct deciles than the other)."""
    shared = sorted(set(float_labels) & set(commission_labels))
    total = counts.values.sum()
    if not shared or total == 0:
        return float("nan")
    diag = sum(counts.loc[lbl, lbl] for lbl in shared if lbl in counts.index and lbl in counts.columns)
    return diag / total * 100


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

    print(f"\n{'#' * 100}\nNO OUTCOMES, NO EXPOSURE TIERS, NO MODELING -- purely a support-structure check\n{'#' * 100}")
    print(
        "Question: are float-based and commission-based business-state bands classifying the\n"
        "same agent-periods into the same relative-scale rank, or materially reclassifying them?\n"
        "This does not itself decide whether a joint float x commission frontier is warranted --\n"
        "it only checks whether the two fundamentals are redundant or genuinely different rankings."
    )

    agent_period_summary = build_agent_period_summary(df_valid)
    print(f"\n{len(agent_period_summary):,} agent-period unit(s) across "
          f"{agent_period_summary['agent_msisdn'].nunique():,} agent(s).")

    counts, row_pct, float_labels, commission_labels = build_overlap_table(agent_period_summary, args.n_bands)

    print("\n-- Agent-period counts: float band (rows) x commission band (columns) --")
    with pd.option_context("display.width", 180):
        print(counts.to_string())

    print("\n-- Row percentages (each float band's own distribution across commission bands) --")
    with pd.option_context("display.width", 180, "display.float_format", "{:.1f}".format):
        print(row_pct.to_string())

    diag_pct = diagonal_mass_pct(counts, float_labels, commission_labels)
    print(f"\nShare of agent-periods where the float-band label equals the commission-band label "
          f"(rough same-relative-rank proxy): {diag_pct:.1f}% of {counts.values.sum():,} unit(s). "
          f"Low diagonal mass does not by itself prove the two fundamentals carry different "
          f"information -- read the full row-percentage table above, especially for any band "
          f"already flagged as notable in the frontier analysis (e.g. float D7, D8, D10).")

    counts.to_csv(f"{args.out_prefix}_counts.csv")
    row_pct.to_csv(f"{args.out_prefix}_row_pct.csv")
    print(f"\nWritten: {args.out_prefix}_counts.csv, {args.out_prefix}_row_pct.csv")


if __name__ == "__main__":
    main()
