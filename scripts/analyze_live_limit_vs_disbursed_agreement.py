"""
analyze_live_limit_vs_disbursed_agreement.py
===============================================
Thorough, standalone analysis of how closely the incumbent engine's
LIVE assigned_limit tracks actual historical disbursed_amount, and how
that agreement varies by agent_category. Reads live_shadow_vs_category_
limit.csv directly (the file built by check_live_shadow_vs_category_
limit.py) -- no new merges, no engine re-run, no Decision Log needed.

Recomputes pct_of_live_disbursed = disbursed_amount / assigned_limit and
both band classifications FRESH from disbursed_amount/assigned_limit
(never trusts pre-computed agreement_band_live/pct_of_live_disbursed
columns that may or may not be present depending on which version of
check_live_shadow_vs_category_limit.py produced the file) -- self-
contained and independently auditable.

UNIT OF ANALYSIS: one row per (agent, month-with-a-disbursement-record)
-- an agent with disbursements in more than one month appears once per
month, same convention as live_shadow_vs_category_limit.csv's own
monthly-disbursement enrichment. Both row-count (agent-months) and
unique-agent-count are reported, never conflated.

Band conventions restated (not imported) from check_live_shadow_vs_
category_limit.py -- the same "90-110% = agreement" house convention
used everywhere else in this repo:
    AGREEMENT_EDGES/LABELS -- the coarse 4-band split (well under /
        moderately under / agreement / over limit).
    BUCKET_EDGES/LABELS -- the finer 9-band split, used to see HOW far
        over/under, not just whether.

Also reports a few named "near vs. far" groupings (symmetric bands
around 100%, and an explicit 2x+ severity cut) as a concrete point of
comparison against an earlier, informally-recalled "~77% agreement /
~33% off by up to 2x" figure -- without assuming any single grouping is
what that recollection meant. The real distribution below is the
source of truth going forward, not the recollection.

Purely descriptive. Does not touch assigned_limit, Capacity(F),
capacity_cap, or C3.

Usage:
    python scripts\\analyze_live_limit_vs_disbursed_agreement.py ^
        --live-shadow-vs-category-file live_shadow_vs_category_limit.csv ^
        --out-prefix output\\live_limit_vs_disbursed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Restated verbatim from check_live_shadow_vs_category_limit.py.
AGREEMENT_EDGES = [-0.01, 0.50, 0.90, 1.10, 100]
AGREEMENT_LABELS = [
    "Well under limit (0-50%)", "Moderately under (50-90%)",
    "Agreement band (90-110%)", "Over limit (110%+)",
]
BUCKET_EDGES = [-0.01, 0.25, 0.50, 0.75, 0.90, 1.10, 1.25, 1.50, 2.00, 100]
BUCKET_LABELS = [
    "0-25% (well under)", "25-50% (under)", "50-75% (under)",
    "75-90% (near, under)", "90-110% (AGREEMENT band)", "110-125% (near, over)",
    "125-150% (over)", "150-200% (well over)", "200%+ (far over)",
]


def load(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"agent_category", "assigned_limit", "disbursed_amount"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s) {missing}. Columns present: {list(df.columns)}")
    return df


def compute_pct(df: pd.DataFrame) -> pd.DataFrame:
    """pct_of_live_disbursed = disbursed_amount / assigned_limit, valid
    only where disbursed_amount is present and assigned_limit > 0 --
    recomputed fresh here, never read from a pre-existing column, so
    this analysis is self-contained regardless of which generation of
    live_shadow_vs_category_limit.csv is passed in."""
    out = df.copy()
    valid = out["disbursed_amount"].notna() & out["assigned_limit"].notna() & (out["assigned_limit"] > 0)
    pct = pd.Series(np.nan, index=out.index, dtype="float64")
    pct.loc[valid] = out.loc[valid, "disbursed_amount"] / out.loc[valid, "assigned_limit"]
    out["_pct"] = pct
    out["_agreement_band"] = pd.cut(pct, bins=AGREEMENT_EDGES, labels=AGREEMENT_LABELS)
    out["_fine_band"] = pd.cut(pct, bins=BUCKET_EDGES, labels=BUCKET_LABELS)
    return out


def band_distribution(df: pd.DataFrame, band_col: str, labels: list) -> pd.DataFrame:
    valid = df[band_col].notna()
    n = int(valid.sum())
    counts = df.loc[valid, band_col].value_counts().reindex(labels, fill_value=0)
    pct = (counts / n * 100).round(2) if n else counts.astype(float)
    return pd.DataFrame({"n": counts, "pct": pct})


def overall_summary(df: pd.DataFrame) -> dict:
    valid = df["_pct"].notna()
    n_rows = int(valid.sum())
    n_agents = int(df.loc[valid, "msisdn"].nunique()) if "msisdn" in df.columns else np.nan
    vals = df.loc[valid, "_pct"]
    return {
        "n_agent_month_rows": n_rows,
        "n_unique_agents": n_agents,
        "median_pct": round(float(vals.median()) * 100, 1) if n_rows else np.nan,
        "p25_pct": round(float(vals.quantile(0.25)) * 100, 1) if n_rows else np.nan,
        "p75_pct": round(float(vals.quantile(0.75)) * 100, 1) if n_rows else np.nan,
    }


def near_vs_far_groupings(df: pd.DataFrame) -> pd.DataFrame:
    """A few explicit, named symmetric/severity groupings of the fine
    9-band distribution -- offered as candidate reference points against
    an earlier informally-recalled '~77%/~33%' split, not as a claim that
    any one of these IS that split."""
    valid = df["_pct"].notna()
    n = int(valid.sum())
    vals = df.loc[valid, "_pct"]
    if n == 0:
        return pd.DataFrame()

    def pct_in(lo, hi_inclusive_exclusive):
        lo_ok = vals >= lo if lo is not None else pd.Series(True, index=vals.index)
        hi_ok = vals < hi_inclusive_exclusive if hi_inclusive_exclusive is not None else pd.Series(True, index=vals.index)
        return round(float((lo_ok & hi_ok).mean() * 100), 2)

    rows = [
        {"grouping": "AGREEMENT band only (90-110%)", "pct": pct_in(0.90, 1.10)},
        {"grouping": "Near the limit, symmetric (75-125%)", "pct": pct_in(0.75, 1.25)},
        {"grouping": "Near the limit, wider (50-150%)", "pct": pct_in(0.50, 1.50)},
        {"grouping": "At or under the limit (0-110%)", "pct": pct_in(None, 1.10)},
        {"grouping": "Over by less than 2x (110-200%)", "pct": pct_in(1.10, 2.00)},
        {"grouping": "At or over 2x the limit (200%+) -- the 'Diamond A' severity cut", "pct": pct_in(2.00, None)},
        {"grouping": "Under half the limit (0-50%)", "pct": pct_in(None, 0.50)},
    ]
    return pd.DataFrame(rows)


def by_category(df: pd.DataFrame, band_col: str, labels: list) -> pd.DataFrame:
    valid = df[band_col].notna()
    sub = df.loc[valid]
    rows = []
    for cat, grp in sub.groupby("agent_category", observed=True):
        n = len(grp)
        counts = grp[band_col].value_counts().reindex(labels, fill_value=0)
        row = {"agent_category": cat, "n": n,
               "median_pct": round(float(grp["_pct"].median()) * 100, 1)}
        for lbl in labels:
            row[lbl] = round(float(counts[lbl] / n * 100), 2) if n else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-shadow-vs-category-file", default="live_shadow_vs_category_limit.csv")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args(argv)

    path = Path(args.live_shadow_vs_category_file)
    if not path.exists():
        sys.exit(f"ERROR: {path} not found.")

    df = load(path)
    df = compute_pct(df)

    print("=" * 78)
    print("Live assigned_limit vs. actual disbursed_amount -- agreement analysis")
    print("=" * 78)
    summary = overall_summary(df)
    print(f"Agent-month rows with a valid comparison: {summary['n_agent_month_rows']:,}")
    print(f"Unique agents represented: {summary['n_unique_agents']:,}")
    print(f"Median disbursed/assigned_limit: {summary['median_pct']}%  "
          f"(p25={summary['p25_pct']}%, p75={summary['p75_pct']}%)\n")

    print(f"{'=' * 78}")
    print("Coarse 4-band agreement distribution (90-110% = house 'agreement' convention)")
    print("=" * 78)
    print(band_distribution(df, "_agreement_band", AGREEMENT_LABELS).to_string())

    print(f"\n{'=' * 78}")
    print("Fine 9-band distribution -- how far under/over, not just whether")
    print("=" * 78)
    print(band_distribution(df, "_fine_band", BUCKET_LABELS).to_string())

    print(f"\n{'=' * 78}")
    print("Named near-vs-far groupings (candidate reference points -- see module docstring)")
    print("=" * 78)
    print(near_vs_far_groupings(df).to_string(index=False))

    print(f"\n{'=' * 78}")
    print("By agent_category -- coarse 4-band distribution (%) + median pct + n")
    print("=" * 78)
    cat_coarse = by_category(df, "_agreement_band", AGREEMENT_LABELS)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(cat_coarse.to_string(index=False))

    print(f"\n{'=' * 78}")
    print("By agent_category -- fine 9-band distribution (%)")
    print("=" * 78)
    cat_fine = by_category(df, "_fine_band", BUCKET_LABELS)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(cat_fine.to_string(index=False))

    if args.out_prefix:
        band_distribution(df, "_agreement_band", AGREEMENT_LABELS).to_csv(f"{args.out_prefix}_coarse_overall.csv")
        band_distribution(df, "_fine_band", BUCKET_LABELS).to_csv(f"{args.out_prefix}_fine_overall.csv")
        near_vs_far_groupings(df).to_csv(f"{args.out_prefix}_near_vs_far.csv", index=False)
        cat_coarse.to_csv(f"{args.out_prefix}_by_category_coarse.csv", index=False)
        cat_fine.to_csv(f"{args.out_prefix}_by_category_fine.csv", index=False)
        print(f"\nCSV outputs written with prefix: {args.out_prefix}")


if __name__ == "__main__":
    main()
