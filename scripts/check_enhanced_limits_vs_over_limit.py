"""
check_enhanced_limits_vs_over_limit.py
=========================================
Cross-references a file of agents who received a manual "enhanced limit"
(an administrative tier-upgrade override -- e.g. Silver 250,000 -> 350,000,
which is Gold's ceiling) against the "Over limit (110%+)" population
already identified by check_live_shadow_vs_category_limit.py, i.e. agents
whose actual disbursed_amount exceeded 110% of their live assigned_limit.

Question this answers: are the agents who got a manual limit increase the
SAME agents the model already flagged as receiving more than it
recommended, or a largely different population? That distinguishes
"the enhanced-limit program is catching up to agents who needed it" from
"it's a separate, unrelated intervention."

Two input files, both required:

1. --enhanced-limits-file: the manual enhanced-limit roster. Expected
   columns (as supplied): Agent_MSISDN, Category, Sitename, District,
   Previous limit, Increased Limit, Limit application status,
   TargetCategory (the file's raw header has "Category" twice -- the
   second occurrence is TargetCategory, the tier the agent was upgraded
   TO; pandas auto-suffixes the duplicate to "Category.1" on read, which
   this script renames automatically).

2. --live-shadow-file: the per-agent detail CSV written by
   check_live_shadow_vs_category_limit.py (default
   live_shadow_vs_category_limit.csv), which carries pct_of_<ref>_disbursed
   columns for live/shadow_base/shadow_conservative. An agent can appear
   more than once (one row per agent-month); this script treats an agent
   as "ever over limit" on a given reference if ANY of their matched rows
   exceed 110% of that reference, and classifies them into one of the four
   exact over-limit sub-bands (110-125% / 125-150% / 150-200% / 200%+ -- the
   same sub-groups that sum to the ~23-out-of-100 "over limit" figure quoted
   elsewhere) by their WORST (maximum) pct value across matched months.

Reports both a binary over/not-over view and the finer per-band view, each
in both directions (forward: of the roster, where do they fall; reverse:
of each band/population, how much is on the roster) -- NOT a statement
about timing or causation. The enhanced-limits file carries no date for
when the manual increase was applied relative to the Aug 1-14 disbursement
window the over-limit flag is based on, so this is strictly a same-agent
overlap check, not evidence that one caused or followed the other.

CATEGORY_LIMITS is restated here (not imported), matching the same
independent-cross-check convention as every other check_*.py script in
this repo -- it is used only as a sanity note on whether "Previous limit"
/ "Increased Limit" line up with the fixed commission-tier ceilings, not
as part of the over-limit join itself.

Usage:
    python scripts\\check_enhanced_limits_vs_over_limit.py ^
        --enhanced-limits-file Agents_enhanced_limits.csv ^
        --live-shadow-file live_shadow_vs_category_limit.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

CATEGORY_LIMITS = {
    "new bronze": 50_000,
    "bronze": 100_000,
    "silver": 250_000,
    "gold": 350_000,
    "platinum": 500_000,
    "titanium": 750_000,
    "diamond": 1_000_000,
}

OVER_LIMIT_THRESHOLD = 1.10  # matches AGREEMENT_LABELS "Over limit (110%+)" in sibling scripts
PCT_COLUMNS = {
    "live": "pct_of_live_disbursed",
    "shadow_base": "pct_of_shadow_base_disbursed",
    "shadow_conservative": "pct_of_shadow_conservative_disbursed",
}

# The "Over limit (110%+)" population's own finer breakdown, restated from the
# same BUCKET_EDGES/BUCKET_LABELS convention in check_live_shadow_vs_category_limit.py
# (the "over" side only: 110-125%, 125-150%, 150-200%, 200%+ of the reference) --
# i.e. the exact four sub-groups that together sum to the ~23-out-of-100 "over limit"
# figure quoted elsewhere. An agent with more than one matched month is classified by
# the WORST (maximum) pct value across their months, consistent with "ever over limit"
# above also being an ANY-month test.
OVER_BAND_EDGES = [OVER_LIMIT_THRESHOLD, 1.25, 1.50, 2.00, float("inf")]
OVER_BAND_LABELS = [
    "Got 10-25% more than recommended",
    "Got 25-50% more than recommended",
    "Got 50-100% more (up to double) than recommended",
    "Got more than double the recommendation",
]


def _normalize_msisdn(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"[^0-9]", "", regex=True).replace("", pd.NA)


def _category_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def _load_enhanced_limits(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, thousands=",")
    if "Category.1" in df.columns and "TargetCategory" not in df.columns:
        df = df.rename(columns={"Category.1": "TargetCategory"})
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enhanced-limits-file", default="Agents_enhanced_limits.csv")
    ap.add_argument("--live-shadow-file", default="live_shadow_vs_category_limit.csv",
                     help="per-agent detail CSV written by check_live_shadow_vs_category_limit.py")
    ap.add_argument("--out", default="enhanced_limits_vs_over_limit.csv")
    args = ap.parse_args(argv)

    enh_path = Path(args.enhanced_limits_file)
    ls_path = Path(args.live_shadow_file)
    if not enh_path.exists():
        sys.exit(f"ERROR: enhanced-limits file not found: {enh_path}")
    if not ls_path.exists():
        sys.exit(f"ERROR: live/shadow detail file not found: {ls_path} "
                  f"(run check_live_shadow_vs_category_limit.py first).")

    enh = _load_enhanced_limits(enh_path)
    required_enh = {"Agent_MSISDN", "Category", "Previous limit", "Increased Limit", "Limit application status"}
    missing_enh = required_enh - set(enh.columns)
    if missing_enh:
        sys.exit(f"ERROR: {enh_path} is missing required columns {missing_enh}. "
                  f"Columns present: {list(enh.columns)}")

    ls = pd.read_csv(ls_path)
    if "msisdn" not in ls.columns:
        sys.exit(f"ERROR: {ls_path} has no msisdn column. Columns present: {list(ls.columns)}")

    present_pct_cols = {name: col for name, col in PCT_COLUMNS.items() if col in ls.columns}
    if not present_pct_cols:
        sys.exit(f"ERROR: none of {list(PCT_COLUMNS.values())} found in {ls_path}. "
                  f"Was it generated with --monthly-summary-file enrichment? "
                  f"Columns present: {list(ls.columns)}")

    enh["_key"] = _normalize_msisdn(enh["Agent_MSISDN"])
    n_enh_total = len(enh)
    n_enh_dupe_keys = int(enh["_key"].duplicated().sum())
    if n_enh_dupe_keys:
        print(f"NOTE: {n_enh_dupe_keys:,} duplicate Agent_MSISDN row(s) in {enh_path.name} "
              f"(same agent listed more than once).")

    # Sanity note only -- not part of the over-limit join -- on whether the roster's
    # Previous/Increased limit line up with the fixed commission-tier ceiling table.
    enh["_prev_category_limit"] = _category_key(enh["Category"]).map(CATEGORY_LIMITS)
    prev_mismatch = enh["_prev_category_limit"].notna() & (enh["Previous limit"] != enh["_prev_category_limit"])
    n_prev_mismatch = int(prev_mismatch.sum())
    if n_prev_mismatch:
        print(f"NOTE: {n_prev_mismatch:,}/{n_enh_total:,} row(s) have a 'Previous limit' that does not "
              f"match their stated Category's fixed ceiling (informational only).")
    if "TargetCategory" in enh.columns:
        enh["_target_category_limit"] = _category_key(enh["TargetCategory"]).map(CATEGORY_LIMITS)
        target_mismatch = enh["_target_category_limit"].notna() & (enh["Increased Limit"] != enh["_target_category_limit"])
        n_target_mismatch = int(target_mismatch.sum())
        if n_target_mismatch:
            print(f"NOTE: {n_target_mismatch:,}/{n_enh_total:,} row(s) have an 'Increased Limit' that does not "
                  f"match TargetCategory's fixed ceiling (informational only).")

    ls = ls.copy()
    ls["_key"] = _normalize_msisdn(ls["msisdn"])

    # Collapse the (possibly multi-month) live/shadow detail to one row per agent:
    # "ever over limit" = ANY matched agent-month row exceeded the threshold on that reference.
    agg = {"_key": "first"}
    per_agent_rows = []
    for key, grp in ls.groupby("_key", dropna=True):
        row = {"_key": key, "n_months_matched": len(grp)}
        for name, col in present_pct_cols.items():
            vals = grp[col].dropna().astype(float)
            row[f"n_months_checked_{name}"] = len(vals)
            row[f"n_months_over_{name}"] = int((vals > OVER_LIMIT_THRESHOLD).sum())
            row[f"ever_over_{name}"] = bool((vals > OVER_LIMIT_THRESHOLD).any()) if len(vals) else pd.NA
            # float("nan"), not pd.NA -- pd.cut() chokes on pd.NA in older pandas/numpy
            # combinations ("boolean value of NA is ambiguous"); a plain float NaN sorts
            # correctly and pd.cut() treats it as "no band" exactly as intended.
            row[f"worst_pct_{name}"] = float(vals.max()) if len(vals) else float("nan")
        per_agent_rows.append(row)
    ls_per_agent = pd.DataFrame(per_agent_rows)
    for name in present_pct_cols:
        ls_per_agent[f"worst_pct_{name}"] = pd.to_numeric(ls_per_agent[f"worst_pct_{name}"], errors="coerce")
        ls_per_agent[f"over_band_{name}"] = pd.cut(
            ls_per_agent[f"worst_pct_{name}"], bins=OVER_BAND_EDGES, labels=OVER_BAND_LABELS
        )

    merged = enh.merge(ls_per_agent, on="_key", how="left")
    n_matched = int(merged["n_months_matched"].notna().sum())

    print("=" * 78)
    print("Enhanced-limit roster vs. the Over-limit (>110% of reference) population")
    print("=" * 78)
    print(f"Enhanced-limit agents (roster): {n_enh_total:,}")
    print(f"  of those, found with disbursement data to check: {n_matched:,} "
          f"({n_matched / n_enh_total * 100:.1f}% of roster)" if n_enh_total else "")

    summary_rows = []
    for name in present_pct_cols:
        ever_col = f"ever_over_{name}"
        checked = merged[merged[ever_col].notna()]
        n_checked = len(checked)
        n_over = int(checked[ever_col].astype(bool).sum()) if n_checked else 0
        summary_rows.append({
            "reference": name,
            "n_roster_checked": n_checked,
            "n_roster_ever_over_limit": n_over,
            "pct_roster_ever_over_limit": round(n_over / n_checked * 100, 1) if n_checked else float("nan"),
        })
    print()
    print(pd.DataFrame(summary_rows).to_string(index=False))

    # Forward, finer view: of the SAME roster-checked denominator as above, which of the
    # four specific "over" sub-bands (110-125% / 125-150% / 150-200% / 200%+) do they fall
    # into -- the exact sub-groups that sum to the ~23-out-of-100 "over limit" figure.
    print(f"\n{'=' * 78}")
    print("Forward, by exact over-limit band: where in the 23-out-of-100 do roster agents fall?")
    print("(denominator = same roster-checked population as above; rows sum to the")
    print(" 'pct_roster_ever_over_limit' figure above, give or take rounding)")
    print("=" * 78)
    for name in present_pct_cols:
        ever_col = f"ever_over_{name}"
        band_col = f"over_band_{name}"
        checked = merged[merged[ever_col].notna()]
        n_checked = len(checked)
        if n_checked == 0:
            continue
        counts = checked[band_col].value_counts().reindex(OVER_BAND_LABELS).fillna(0).astype(int)
        pct = (counts / n_checked * 100).round(1)
        print(f"\n-- {name} (n_roster_checked={n_checked:,}) --")
        print(pd.DataFrame({"n_roster_agents": counts, "pct_of_roster_checked": pct}).to_string())

    # Reverse direction: of ALL agents in the over-limit population (not just the roster),
    # what share also appear on the enhanced-limit roster? Tells us whether enhanced limits
    # are concentrated inside the already-over-limit group or are a largely separate population.
    print(f"\n{'=' * 78}")
    print("Reverse view: of the full over-limit population, how many are on the enhanced-limit roster?")
    print("=" * 78)
    roster_keys = set(enh["_key"].dropna())
    rev_rows = []
    for name, col in present_pct_cols.items():
        over_keys = set(ls.loc[ls[col].astype(float) > OVER_LIMIT_THRESHOLD, "_key"].dropna())
        n_over_total = len(over_keys)
        n_over_on_roster = len(over_keys & roster_keys)
        rev_rows.append({
            "reference": name,
            "n_over_limit_agents": n_over_total,
            "n_also_on_enhanced_roster": n_over_on_roster,
            "pct_also_on_enhanced_roster": round(n_over_on_roster / n_over_total * 100, 1) if n_over_total else float("nan"),
        })
    print(pd.DataFrame(rev_rows).to_string(index=False))

    # Reverse, finer view: take each of the 4 exact over-limit sub-bands (system-wide, not
    # just the roster) and ask what share of agents IN THAT BAND are on the enhanced-limit
    # roster -- e.g. "of agents who got more than double their recommended limit, how many
    # also received a manual enhanced limit?"
    print(f"\n{'=' * 78}")
    print("Reverse, by exact over-limit band: of each specific band, how many are on the roster?")
    print("=" * 78)
    for name in present_pct_cols:
        band_col = f"over_band_{name}"
        band_rows = []
        for label in OVER_BAND_LABELS:
            band_keys = set(ls_per_agent.loc[ls_per_agent[band_col] == label, "_key"].dropna())
            n_band = len(band_keys)
            n_band_on_roster = len(band_keys & roster_keys)
            band_rows.append({
                "over_band": label,
                "n_agents_in_band": n_band,
                "n_also_on_enhanced_roster": n_band_on_roster,
                "pct_also_on_enhanced_roster": round(n_band_on_roster / n_band * 100, 1) if n_band else float("nan"),
            })
        print(f"\n-- {name} --")
        print(pd.DataFrame(band_rows).to_string(index=False))

    detail_cols = ["Agent_MSISDN", "Category"] + (["TargetCategory"] if "TargetCategory" in enh.columns else [])
    detail_cols += ["Previous limit", "Increased Limit", "Limit application status", "n_months_matched"]
    for name in present_pct_cols:
        detail_cols += [f"n_months_checked_{name}", f"n_months_over_{name}", f"ever_over_{name}",
                         f"worst_pct_{name}", f"over_band_{name}"]
    detail_cols = [c for c in detail_cols if c in merged.columns]
    merged[detail_cols].to_csv(args.out, index=False)
    print(f"\nPer-agent detail written: {args.out}")


if __name__ == "__main__":
    main()
