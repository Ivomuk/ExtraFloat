"""
Filters --borrower-file down to only the retail-agent msisdns already
computed by apply_retail_agent_filter.py (its retail_agents_filtered.csv
output). Single responsibility: mechanical row filter only -- does NOT
re-implement classify_agent_profile()/is_retail(); trusts the allowlist
already written there. Generic on the input file's schema (only requires
an 'msisdn' or 'phonenumber' column), so despite the name this is also
reused to filter the PD-training loan-level file
(--loan-training-file) before pd_model.run_pipeline -- pass --label to
adjust the printed messages for that call site.

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

The identical reasoning applies to pd_model.run_pipeline's loan-level
training mode: reading run_pipeline.py confirms df_pd (built from
--loan-training-file) is the SOLE base population for training -- Phase
2.1 features (from --train-file/--val-file, the agent-mart snapshot
files) are only ever left-joined onto it by (agent_msisdn, split). So
filtering --loan-training-file the same way this script already filters
--borrower-file keeps non-retail agents out of population.

--train-file/--val-file are ALSO filtered (train_retail_filtered.bat
runs this script against them too), even though the left-join alone
already guarantees a non-retail row there can't add a training example
(no match, so it's dropped at merge time). Reason: every feature
pd_model.preprocessing.transaction_features computes today is row-wise
(a ratio/diff over that same row's own columns, no groupby/mean/rank
across agents in this repo's code) -- but a small set of pass-through
input columns it reads if present (commission_cluster_mean,
vol_3m_cluster_mean, cluster_avg_commission, cluster_avg_vol_3m) are not
computed anywhere in this repo; if a real export ever populates them
from an upstream, cross-agent aggregate over the full (unfiltered)
population, that skew would already be baked into each row's value
before this script ever sees the file, and no row-level filter here can
retroactively fix it. Filtering --train-file/--val-file directly doesn't
retroactively fix pre-baked upstream aggregates either -- that would
require the upstream SQL/warehouse query to exclude non-retail agents
from whatever population it aggregates over -- but it does guarantee
every row entering this pipeline's own feature code is retail-only,
closing off any future row-wise feature from being affected and making
the "population is retail-only end to end" property directly verifiable
by inspecting the CSVs themselves rather than resting on a proof about
today's specific merge code.

Usage:
    python filter_borrower_file_by_retail_agents.py ^
        --retail-agents-file retail_agents_filtered.csv ^
        --borrower-file data\\borrower_history.csv ^
        --out borrower_history_retail_filtered.csv

    python filter_borrower_file_by_retail_agents.py ^
        --retail-agents-file retail_agents_filtered.csv ^
        --borrower-file data\\state_data_202608131214.csv ^
        --out data\\state_data_202608131214_retail_filtered.csv ^
        --label "Loan-training file"
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
    ap.add_argument("--label", default="Borrower file",
                     help="Display name used in printed messages (e.g. 'Loan-training file' "
                          "when filtering --loan-training-file instead of the borrower file).")
    args = ap.parse_args()
    label = args.label
    label_lower = label[0].lower() + label[1:] if label else "file"

    retail_path = Path(args.retail_agents_file)
    borrower_path = Path(args.borrower_file)
    if not retail_path.exists():
        sys.exit(f"ERROR: retail-agents file not found: {retail_path}")
    if not borrower_path.exists():
        sys.exit(f"ERROR: {label_lower} not found: {borrower_path}")

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
            f"ERROR: {label_lower} has neither 'msisdn' nor 'phonenumber' column. "
            f"Found: {list(bor.columns)[:20]}"
        )

    n_borrower_total = len(bor)
    bor["_msisdn_norm"] = _norm_msisdn(bor[bor_msisdn_col])

    n_blank = int(bor["_msisdn_norm"].isna().sum())
    in_allowlist_mask = bor["_msisdn_norm"].isin(allowlist_set)
    kept_mask = bor["_msisdn_norm"].notna() & in_allowlist_mask
    n_kept = int(kept_mask.sum())
    n_non_retail_dropped = n_borrower_total - n_kept - n_blank

    print(f"{label} ('{bor_msisdn_col}' column): {n_borrower_total:,} rows")
    print(f"  Kept (retail):            {n_kept:,} ({n_kept/max(1,n_borrower_total):.1%})")
    print(f"  Dropped (non-retail):     {n_non_retail_dropped:,} ({n_non_retail_dropped/max(1,n_borrower_total):.1%})")
    print(f"  Dropped (blank/unparseable msisdn): {n_blank:,} ({n_blank/max(1,n_borrower_total):.1%})")

    borrower_keys_present = set(bor["_msisdn_norm"].dropna())
    n_allowlist_missing = len(allowlist_set - borrower_keys_present)
    print(
        f"\nAllowlist agents NOT found in {label_lower}: {n_allowlist_missing:,} "
        f"({n_allowlist_missing/max(1,n_allowlist):.1%}) -- informational (agents with no "
        f"activity in the classified transaction snapshot), not an error."
    )

    if n_borrower_total > 0 and n_allowlist > 0 and n_kept == 0:
        sys.exit(
            f"ERROR: 0 {label_lower} rows survived filtering despite non-empty inputs on both "
            "sides -- likely an msisdn format or column mismatch between the retail-agents "
            f"file and the {label_lower}. Refusing to write an empty output."
        )

    kept = bor.loc[kept_mask].drop(columns=["_msisdn_norm"])
    kept.to_csv(args.out, index=False)
    print(f"\nFiltered {label_lower} written to: {args.out} ({n_kept:,} rows)")


if __name__ == "__main__":
    main()
