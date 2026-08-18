"""
Filters --borrower-file down to only the retail-agent msisdns already
computed by apply_retail_agent_filter.py (its retail_agents_filtered.csv
output). Single responsibility: mechanical row filter only -- does NOT
re-implement classify_agent_profile()/is_retail(); trusts the allowlist
already written there.

Why this is needed, not just filtering the transaction file: reading
run_credit_risk_pipeline.py and extrafloat_limit_engine_features.py in
full confirms --borrower-file (df_borrower) is the SOLE base population
for the final scored output -- every merge inside
build_extrafloat_limit_engine_features() is `how="left"` with
borrower_df/merged as the base (transaction_df, loan_df,
loan_history_df are all joined ONTO it, never the other way around), and
pd_scored/seg_out (built from the transaction file) are themselves only
ever left-joined onto that same borrower-based frame later. So a
non-retail agent still present in --borrower-file would still reach the
final output -- just with NaN/zeroed transaction-derived fields via a
failed left-join match -- unless the borrower file itself is filtered
too. --loan-file / --loan-history-file do NOT need filtering: they're
only ever left-joined onto the (now correctly restricted) borrower-based
population, so non-retail rows there are harmless no-ops.

Usage:
    python filter_borrower_file_by_retail_agents.py ^
        --retail-agents-file retail_agents_filtered.csv ^
        --borrower-file data\\borrower_history.csv ^
        --out borrower_history_retail_filtered.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def _norm_msisdn(s: pd.Series) -> pd.Series:
    """Byte-identical to run_credit_risk_pipeline.py's _norm_msisdn
    (lines ~200-209) -- THE normalizer that actually drives the
    pipeline's own Stage 5 join keys. Keep in sync with that function;
    do not substitute extrafloat_limit_engine_features.py's
    _standardize_msisdn or any other file's variant (each does something
    subtly different -- unanchored .0 strip, digit-only strip, no
    sentinel masking, etc.) -- a divergent normalizer here risks a
    msisdn matching in this script but not in the pipeline itself, or
    vice versa, at the margins."""
    out = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return out.mask(out.str.lower().isin({"", "nan", "none", "<na>"}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retail-agents-file", default="retail_agents_filtered.csv")
    ap.add_argument("--agent-msisdn-col", default="agent_msisdn")
    ap.add_argument("--borrower-file", default="data/borrower_history.csv")
    ap.add_argument("--out", default="borrower_history_retail_filtered.csv")
    args = ap.parse_args()

    retail_path = Path(args.retail_agents_file)
    borrower_path = Path(args.borrower_file)
    if not retail_path.exists():
        sys.exit(f"ERROR: retail-agents file not found: {retail_path}")
    if not borrower_path.exists():
        sys.exit(f"ERROR: borrower file not found: {borrower_path}")

    retail_df = pd.read_csv(retail_path, sep=",", encoding="utf-8-sig")
    if args.agent_msisdn_col not in retail_df.columns:
        sys.exit(
            f"ERROR: '{args.agent_msisdn_col}' column not found in retail-agents file. "
            f"Found: {list(retail_df.columns)[:30]}"
        )
    allowlist_norm = _norm_msisdn(retail_df[args.agent_msisdn_col]).dropna()
    allowlist_set = set(allowlist_norm)
    n_allowlist = len(allowlist_set)
    print(f"Retail-agent allowlist: {n_allowlist:,} unique agents (from {retail_path})\n")

    bor = pd.read_csv(borrower_path, sep=",", encoding="utf-8-sig")
    bor_msisdn_col = "msisdn" if "msisdn" in bor.columns else ("phonenumber" if "phonenumber" in bor.columns else None)
    if bor_msisdn_col is None:
        sys.exit(
            f"ERROR: borrower file has neither 'msisdn' nor 'phonenumber' column. "
            f"Found: {list(bor.columns)[:20]}"
        )

    n_borrower_total = len(bor)
    bor["_msisdn_norm"] = _norm_msisdn(bor[bor_msisdn_col])

    n_blank = int(bor["_msisdn_norm"].isna().sum())
    in_allowlist_mask = bor["_msisdn_norm"].isin(allowlist_set)
    kept_mask = bor["_msisdn_norm"].notna() & in_allowlist_mask
    n_kept = int(kept_mask.sum())
    n_non_retail_dropped = n_borrower_total - n_kept - n_blank

    print(f"Borrower file ('{bor_msisdn_col}' column): {n_borrower_total:,} rows")
    print(f"  Kept (retail):            {n_kept:,} ({n_kept/max(1,n_borrower_total):.1%})")
    print(f"  Dropped (non-retail):     {n_non_retail_dropped:,} ({n_non_retail_dropped/max(1,n_borrower_total):.1%})")
    print(f"  Dropped (blank/unparseable msisdn): {n_blank:,} ({n_blank/max(1,n_borrower_total):.1%})")

    borrower_keys_present = set(bor["_msisdn_norm"].dropna())
    n_allowlist_missing = len(allowlist_set - borrower_keys_present)
    print(
        f"\nAllowlist agents NOT found in borrower file: {n_allowlist_missing:,} "
        f"({n_allowlist_missing/max(1,n_allowlist):.1%}) -- informational, this is the "
        f"already-documented borrower-coverage gap, not an error."
    )

    if n_borrower_total > 0 and n_allowlist > 0 and n_kept == 0:
        sys.exit(
            "ERROR: 0 borrower rows survived filtering despite non-empty inputs on both "
            "sides -- likely an msisdn format or column mismatch between the retail-agents "
            "file and the borrower file. Refusing to write an empty output (would make the "
            "full pipeline run against zero borrowers)."
        )

    kept = bor.loc[kept_mask].drop(columns=["_msisdn_norm"])
    kept.to_csv(args.out, index=False)
    print(f"\nFiltered borrower file written to: {args.out} ({n_kept:,} rows)")


if __name__ == "__main__":
    main()
