"""
Feature-drift comparison between the two borrower_history.txt versions:
data/borrower_history_original.txt (source: devdata.xtrafloat_daily_trans)
and data/borrower_history.txt (source: analytics.momo_loan_book_tracker_*).

This repo has no live warehouse connection, so this script does not run
either query itself. Run both SQL files manually against the warehouse at
matching cutoffs (same snapshot_dt; for the rewrite, also set as_of_load_ts
to bound the comparison to a single, reproducible load state -- see the
comment at the top of borrower_history.txt), export each result to CSV, then
run this script over the two CSVs. It answers "how much did the feature
values actually change", which data/borrower_history_validation_queries.sql
(population counts, attribution reconciliation) does not cover on its own.

Usage:
    python compare_borrower_history_versions.py ^
        --old-file data\\borrower_history_original_output.csv ^
        --new-file data\\borrower_history_output.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Columns that must be byte-identical in name/order per the rewrite plan --
# treated as the numeric/categorical feature set to check for drift. Anything
# else in either file (the ten new loan_state_daily enrichment columns, plus
# identifiers/timestamps) is reported separately, not drift-checked the same way.
IDENTIFIER_COLS = {"phonenumber", "msisdn", "latest_requestid"}
TIMESTAMP_COLS = {"first_loan_ts", "latest_loan_ts", "latest_disbursement_ts"}
CATEGORICAL_COLS = {"borrower_trend", "borrower_profile_type"}


def _norm_msisdn(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return out.mask(out.str.lower().isin({"", "nan", "none", "<na>"}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-file", default="data/borrower_history_original_output.csv")
    ap.add_argument("--new-file", default="data/borrower_history_output.csv")
    ap.add_argument("--tolerance-pct", type=float, default=0.02, help="Relative diff treated as an exact match")
    ap.add_argument("--out-prefix", default="borrower_history_drift")
    args = ap.parse_args()

    old_path, new_path = Path(args.old_file), Path(args.new_file)
    if not old_path.exists():
        sys.exit(f"ERROR: old-file not found: {old_path}")
    if not new_path.exists():
        sys.exit(f"ERROR: new-file not found: {new_path}")

    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)

    old_key = "phonenumber" if "phonenumber" in old.columns else "msisdn"
    new_key = "phonenumber" if "phonenumber" in new.columns else "msisdn"
    old["_key"] = _norm_msisdn(old[old_key])
    new["_key"] = _norm_msisdn(new[new_key])
    old, new = old[old["_key"].notna()], new[new["_key"].notna()]

    old_keys, new_keys = set(old["_key"]), set(new["_key"])
    both = old_keys & new_keys
    print(f"Old ({old_path.name}): {len(old):,} rows, {len(old_keys):,} unique borrowers")
    print(f"New ({new_path.name}): {len(new):,} rows, {len(new_keys):,} unique borrowers")
    print(f"In both:            {len(both):,}")
    print(f"Only in old:        {len(old_keys - new_keys):,}")
    print(f"Only in new:        {len(new_keys - old_keys):,}")
    print(
        "NOTE: 'only in old'/'only in new' mixes real population differences (ANOMALY_OPEN "
        "exclusion, different source coverage) with join-key normalization mismatches -- cross-check "
        "against Section C of borrower_history_validation_queries.sql before treating this as a real gap."
    )

    shared_cols = [c for c in old.columns if c in new.columns and c not in ("_key",)]
    print(f"\nColumns present in both files: {len(shared_cols)} of {len(old.columns)} old / {len(new.columns)} new")
    only_old = sorted(set(old.columns) - set(new.columns) - {"_key"})
    only_new = sorted(set(new.columns) - set(old.columns) - {"_key"})
    if only_old:
        print(f"Columns only in old (dropped in rewrite -- should be empty per the plan): {only_old}")
    if only_new:
        print(f"Columns only in new (additive enrichment, expected): {only_new}")

    old_i = old[old["_key"].isin(both)].drop_duplicates("_key").set_index("_key")
    new_i = new[new["_key"].isin(both)].drop_duplicates("_key").set_index("_key")
    new_i = new_i.reindex(old_i.index)

    numeric_cols = [
        c for c in shared_cols
        if c not in IDENTIFIER_COLS | TIMESTAMP_COLS | CATEGORICAL_COLS
        and pd.api.types.is_numeric_dtype(pd.to_numeric(old_i[c], errors="coerce"))
    ]
    categorical_cols = [c for c in shared_cols if c in CATEGORICAL_COLS]

    print(f"\n=== Numeric feature drift (n={len(old_i):,} shared borrowers, tolerance={args.tolerance_pct:.0%}) ===")
    rows = []
    for c in numeric_cols:
        o = pd.to_numeric(old_i[c], errors="coerce")
        n = pd.to_numeric(new_i[c], errors="coerce")
        both_present = o.notna() & n.notna()
        diff = (n - o)[both_present]
        rel_diff = (diff / o[both_present].abs().replace(0, np.nan)).abs()
        exact = (diff.abs() <= (o[both_present].abs() * args.tolerance_pct).clip(lower=1e-9)) | (diff == 0)
        rows.append({
            "column": c,
            "old_null_rate": o.isna().mean(),
            "new_null_rate": n.isna().mean(),
            "n_both_present": int(both_present.sum()),
            "exact_match_rate": exact.mean() if len(exact) else np.nan,
            "median_abs_diff": diff.abs().median() if len(diff) else np.nan,
            "p95_abs_diff": diff.abs().quantile(0.95) if len(diff) else np.nan,
            "median_rel_diff": rel_diff.median() if rel_diff.notna().any() else np.nan,
            "p95_rel_diff": rel_diff.quantile(0.95) if rel_diff.notna().any() else np.nan,
        })
    drift_df = pd.DataFrame(rows).sort_values("exact_match_rate")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(drift_df.to_string(index=False))
    drift_out = f"{args.out_prefix}_numeric.csv"
    drift_df.to_csv(drift_out, index=False)
    print(f"\nFull numeric drift table written to: {drift_out}")

    print(f"\n=== Categorical transitions ===")
    for c in categorical_cols:
        ct = pd.crosstab(
            old_i[c].fillna("<NA>"), new_i[c].fillna("<NA>"),
            rownames=["old_value"], colnames=["new_value"],
        )
        trans = ct.stack().rename("count").reset_index()
        trans = trans[trans["count"] > 0].sort_values("count", ascending=False)
        print(f"\n-- {c} --")
        print(trans.head(20).to_string(index=False))
        trans_out = f"{args.out_prefix}_{c}_transitions.csv"
        trans.to_csv(trans_out, index=False)
        print(f"Full transition table written to: {trans_out}")


if __name__ == "__main__":
    main()
