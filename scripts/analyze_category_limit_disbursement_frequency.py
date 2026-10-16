"""
analyze_category_limit_disbursement_frequency.py
====================================================
How many agents borrow below their fixed agent-CATEGORY limit (the flat
business-rule ceiling -- new bronze/bronze/silver/gold/platinum/titanium/
diamond -- NOT the engine's per-agent assigned_limit), broken down by
agent_category, and accounting for agents who disburse MULTIPLE TIMES.

Reads live_shadow_vs_category_limit.csv directly (category_limit,
disbursed_amount, n_transactions, agent_category, msisdn already present
from check_live_shadow_vs_category_limit.py's own monthly-disbursement
enrichment). No new merges, no engine re-run.

CRITICAL GRAIN CAVEAT (read before interpreting any number below): per
build_monthly_disbursement_summary.py's own documented design,
disbursed_amount in this file is the SINGLE constant per-transaction
amount for an agent/month where EVERY transaction that month disbursed
the exact same amount under the exact same profile -- n_transactions is
how many times that same amount was disbursed. It is NOT a monthly sum,
and it is NOT available at all for agent/months where the amount varied
across transactions -- those are written to a SEPARATE file
(variable_disbursement_agents.csv) and are excluded from
live_shadow_vs_category_limit.csv entirely. So "how many agents borrow
multiple times" is only directly answerable here for the constant-amount
case (via n_transactions); an agent who borrowed different amounts in
the same month is invisible to this analysis by construction -- pass
--variable-disbursement-file to at least COUNT that excluded population
by category (no per-transaction amount comparison is possible for them,
since a single "disbursed_amount" isn't meaningful there either).

Grain is MONTHLY. n_transactions counts occurrences within one calendar
month; this file cannot distinguish same-day/same-week repeat borrowing
from borrowing spread across the month.

Two weightings are reported for every band breakdown, never collapsed
into one number:
    row-weighted          -- % of agent-month ROWS in each band (one
                              agent/month = one vote, regardless of how
                              many times that amount was disbursed).
    transaction-weighted   -- % of actual DISBURSEMENT EVENTS (n_transactions
                              summed) in each band -- answers "of all the
                              times money actually went out, what share
                              were below/at/over the category limit,"
                              which a high-frequency agent dominates more
                              than the row-weighted view does.

Band convention restated (not imported) from check_live_shadow_vs_
category_limit.py -- the same house "90-110% = agreement" split, applied
here to disbursed_amount / category_limit rather than / assigned_limit.

Purely descriptive. Does not touch assigned_limit, category_limit,
Capacity(F), capacity_cap, or C3.

Usage:
    python scripts\\analyze_category_limit_disbursement_frequency.py ^
        --live-shadow-vs-category-file live_shadow_vs_category_limit.csv ^
        --variable-disbursement-file variable_disbursement_agents.csv ^
        --out-prefix output\\category_limit_disbursement_frequency
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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


def _normalize_msisdn(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    out = out.str.replace(r"[^0-9]", "", regex=True)
    return out.mask(out.str.len() == 0)


def load(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"agent_category", "category_limit", "disbursed_amount", "n_transactions", "msisdn"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s) {missing}. Columns present: {list(df.columns)}")
    return df


def compute_pct(df: pd.DataFrame) -> pd.DataFrame:
    """pct = disbursed_amount / category_limit, recomputed fresh (never
    trusts a pre-existing pct_of_category_limit_disbursed column, so this
    is self-contained regardless of which generation of the CSV is
    passed in). Valid only where disbursed_amount is present and
    category_limit > 0."""
    out = df.copy()
    valid = out["disbursed_amount"].notna() & out["category_limit"].notna() & (out["category_limit"] > 0)
    pct = pd.Series(np.nan, index=out.index, dtype="float64")
    pct.loc[valid] = out.loc[valid, "disbursed_amount"] / out.loc[valid, "category_limit"]
    out["_pct"] = pct
    out["_agreement_band"] = pd.cut(pct, bins=AGREEMENT_EDGES, labels=AGREEMENT_LABELS)
    out["_fine_band"] = pd.cut(pct, bins=BUCKET_EDGES, labels=BUCKET_LABELS)
    return out


def band_distribution(df: pd.DataFrame, band_col: str, labels: list, weight_col: str | None = None) -> pd.DataFrame:
    valid = df[band_col].notna()
    if weight_col is None:
        n = int(valid.sum())
        counts = df.loc[valid, band_col].value_counts().reindex(labels, fill_value=0)
        pct = (counts / n * 100).round(2) if n else counts.astype(float)
        return pd.DataFrame({"n_rows": counts, "pct_of_rows": pct})
    w = df.loc[valid, weight_col].fillna(0)
    total_w = float(w.sum())
    grp = df.loc[valid].groupby(band_col, observed=True)[weight_col].sum().reindex(labels, fill_value=0)
    pct = (grp / total_w * 100).round(2) if total_w else grp.astype(float)
    return pd.DataFrame({"n_transactions": grp, "pct_of_transactions": pct})


def overall_summary(df: pd.DataFrame) -> dict:
    valid = df["_pct"].notna()
    n_rows = int(valid.sum())
    n_agents = int(df.loc[valid, "msisdn"].nunique())
    n_txn = float(df.loc[valid, "n_transactions"].fillna(0).sum())
    multi = df.loc[valid, "n_transactions"] > 1
    return {
        "n_agent_month_rows": n_rows,
        "n_unique_agents": n_agents,
        "n_total_transactions": int(n_txn),
        "pct_rows_with_multiple_transactions": round(float(multi.mean() * 100), 2) if n_rows else np.nan,
        "median_n_transactions": float(df.loc[valid, "n_transactions"].median()) if n_rows else np.nan,
    }


def by_category(df: pd.DataFrame) -> pd.DataFrame:
    valid = df["_pct"].notna()
    sub = df.loc[valid]
    rows = []
    for cat, grp in sub.groupby("agent_category", observed=True):
        n = len(grp)
        n_agents = int(grp["msisdn"].nunique())
        n_txn = float(grp["n_transactions"].fillna(0).sum())
        coarse_rows = grp["_agreement_band"].value_counts().reindex(AGREEMENT_LABELS, fill_value=0)
        coarse_txn = grp.groupby("_agreement_band", observed=True)["n_transactions"].sum().reindex(
            AGREEMENT_LABELS, fill_value=0)
        row = {
            "agent_category": cat,
            "n_agent_month_rows": n,
            "n_unique_agents": n_agents,
            "n_transactions": int(n_txn),
            "pct_rows_multi_transaction": round(float((grp["n_transactions"] > 1).mean() * 100), 2),
            "median_n_transactions": float(grp["n_transactions"].median()),
        }
        for lbl in AGREEMENT_LABELS:
            row[f"rows_pct__{lbl}"] = round(float(coarse_rows[lbl] / n * 100), 2) if n else np.nan
        for lbl in AGREEMENT_LABELS:
            row[f"txn_pct__{lbl}"] = round(float(coarse_txn[lbl] / n_txn * 100), 2) if n_txn else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("n_agent_month_rows", ascending=False).reset_index(drop=True)


def report_excluded_variable_amount_agents(variable_path, live_df: pd.DataFrame) -> pd.DataFrame | None:
    """variable_disbursement_agents.csv has no agent_category of its own
    -- join by normalized msisdn onto live_df's agent_category so the
    excluded (varying-amount) population can at least be COUNTED by
    category, even though no single disbursed_amount exists for them to
    compare against category_limit."""
    vdf = pd.read_csv(variable_path)
    required = {"msisdn", "month"}
    missing = required - set(vdf.columns)
    if missing:
        print(f"NOTE: {variable_path} is missing {missing} -- skipping the excluded-population count.")
        return None
    vdf["_key"] = _normalize_msisdn(vdf["msisdn"])
    cat_lookup = live_df[["msisdn", "agent_category"]].copy()
    cat_lookup["_key"] = _normalize_msisdn(cat_lookup["msisdn"])
    cat_lookup = cat_lookup.drop_duplicates(subset="_key")[["_key", "agent_category"]]
    joined = vdf.merge(cat_lookup, on="_key", how="left")
    n_matched = int(joined["agent_category"].notna().sum())
    print(f"\n{'=' * 78}")
    print("Agent-months EXCLUDED from this analysis -- amount varied within the month")
    print("=" * 78)
    print(f"{len(joined):,} agent-month row(s) in {Path(variable_path).name} "
          f"({n_matched:,} matched to a known agent_category); "
          f"{int(joined['_key'].nunique()):,} unique agent(s). No single disbursed_amount "
          f"exists for these -- they cannot be compared to category_limit by this analysis.")
    by_cat = joined.dropna(subset=["agent_category"]).groupby("agent_category", observed=True).agg(
        n_excluded_agent_month_rows=("_key", "size"),
        n_excluded_unique_agents=("_key", "nunique"),
    ).reset_index().sort_values("n_excluded_agent_month_rows", ascending=False)
    print(by_cat.to_string(index=False))
    return by_cat


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-shadow-vs-category-file", default="live_shadow_vs_category_limit.csv")
    ap.add_argument("--variable-disbursement-file", default=None,
                     help="optional -- build_monthly_disbursement_summary.py's variable_disbursement_agents.csv; "
                          "counts (never compares) the population this analysis structurally excludes")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args(argv)

    path = Path(args.live_shadow_vs_category_file)
    if not path.exists():
        sys.exit(f"ERROR: {path} not found.")

    df = load(path)
    df = compute_pct(df)

    print("=" * 78)
    print("Disbursed amount vs. fixed agent-CATEGORY limit -- how many agents borrow below it,")
    print("by agent_category, accounting for agents who disburse the same amount multiple times")
    print("=" * 78)
    print("GRAIN CAVEAT: disbursed_amount here is a per-transaction constant amount (not a monthly")
    print("sum), valid only for agent/months where every transaction disbursed the same amount.")
    print("Agent/months with varying amounts are excluded entirely -- see --variable-disbursement-file.\n")

    summary = overall_summary(df)
    print(f"Agent-month rows: {summary['n_agent_month_rows']:,}  |  "
          f"Unique agents: {summary['n_unique_agents']:,}  |  "
          f"Total transactions (disbursement events): {summary['n_total_transactions']:,}")
    print(f"Rows with more than one transaction that month (same constant amount, repeated): "
          f"{summary['pct_rows_with_multiple_transactions']}%  |  "
          f"median n_transactions per row: {summary['median_n_transactions']}\n")

    print(f"{'=' * 78}")
    print("ROW-weighted (one agent-month = one vote) -- coarse 4-band vs. category_limit")
    print("=" * 78)
    print(band_distribution(df, "_agreement_band", AGREEMENT_LABELS).to_string())

    print(f"\n{'=' * 78}")
    print("TRANSACTION-weighted (one disbursement event = one vote) -- coarse 4-band vs. category_limit")
    print("=" * 78)
    print(band_distribution(df, "_agreement_band", AGREEMENT_LABELS, weight_col="n_transactions").to_string())

    print(f"\n{'=' * 78}")
    print("ROW-weighted fine 9-band vs. category_limit")
    print("=" * 78)
    print(band_distribution(df, "_fine_band", BUCKET_LABELS).to_string())

    print(f"\n{'=' * 78}")
    print("By agent_category -- row-weighted AND transaction-weighted 4-band distribution, "
          "multi-transaction rate")
    print("=" * 78)
    cat_table = by_category(df)
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(cat_table.to_string(index=False))

    excluded_by_cat = None
    if args.variable_disbursement_file:
        vpath = Path(args.variable_disbursement_file)
        if not vpath.exists():
            print(f"\nNOTE: {vpath} not found -- skipping excluded-population count.")
        else:
            excluded_by_cat = report_excluded_variable_amount_agents(vpath, df)
    else:
        print("\nNOTE: --variable-disbursement-file not given -- the excluded (varying-amount) "
              "population is not counted (pass it to see how many agent-months/agents by category "
              "this analysis structurally cannot cover).")

    if args.out_prefix:
        band_distribution(df, "_agreement_band", AGREEMENT_LABELS).to_csv(f"{args.out_prefix}_coarse_rows.csv")
        band_distribution(df, "_agreement_band", AGREEMENT_LABELS, weight_col="n_transactions").to_csv(
            f"{args.out_prefix}_coarse_transactions.csv")
        band_distribution(df, "_fine_band", BUCKET_LABELS).to_csv(f"{args.out_prefix}_fine_rows.csv")
        cat_table.to_csv(f"{args.out_prefix}_by_category.csv", index=False)
        if excluded_by_cat is not None:
            excluded_by_cat.to_csv(f"{args.out_prefix}_excluded_variable_amount_by_category.csv", index=False)
        print(f"\nCSV outputs written with prefix: {args.out_prefix}")


if __name__ == "__main__":
    main()
