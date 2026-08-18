"""
Investigates the ~2,932 "impossible defaulters" found in the previous
check: blacklisted with reason "Defaulter not paid back in the last 30
days" but absent from --loan-history-file (i.e. no disbursement on
record) -- logically inconsistent, since you can't default on a loan you
never took.

Cross-checks these specific agents against --borrower-file
(borrower_history.csv) to distinguish between the candidate explanations:
  (a) they DID take a loan, but it's specifically missing from the newer
      state_data_*.csv extract (a coverage gap in that file specifically,
      not evidence they never borrowed) -- supported if borrower_history.csv
      shows real loan history (total_loans > 0, non-null first/latest
      loan timestamps) for them.
  (b) they never borrowed at all, and the blacklist "Defaulter" reason is
      mislabeled, reused for a non-ExtraFloat product, or a data-entry
      error -- supported if borrower_history.csv shows no loan history for
      them either, or doesn't have them at all.

Usage:
    python check_impossible_defaulters.py ^
        --blacklist-file data\\blacklist_aug_20260804.csv ^
        --whitelist-file data\\whitelist_aug_20260804.csv ^
        --loan-history-file data\\state_data_202608122330.csv ^
        --borrower-file data\\borrower_history.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pd_model.postprocessing.whitelist_eval import (  # noqa: E402
    _normalize_msisdn,
    load_and_merge_lists,
)

_LOAN_EVIDENCE_COLS = [
    "total_loans", "first_loan_ts", "latest_loan_ts", "latest_disbursement_ts",
    "total_disbursed_amount", "avg_loan_size_lifetime", "max_loan_size_lifetime",
]


def _find_msisdn_col(df: pd.DataFrame, label: str) -> str:
    for candidate in ("msisdn", "agent_msisdn", "customer_msisdn", "phonenumber"):
        if candidate in df.columns:
            return candidate
    sys.exit(f"ERROR: could not find an msisdn column in the {label} file. Found: {list(df.columns)[:30]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blacklist-file", default="data/blacklist_aug_20260804.csv")
    ap.add_argument("--whitelist-file", default="data/whitelist_aug_20260804.csv")
    ap.add_argument("--loan-history-file", default="data/state_data_202608122330.csv")
    ap.add_argument("--borrower-file", default="data/borrower_history.csv")
    ap.add_argument("--reason-match", default="Defaulter not paid back in the last 30 days")
    ap.add_argument("--out", default="impossible_defaulters_check.csv")
    args = ap.parse_args()

    for p, label in [
        (args.blacklist_file, "blacklist"), (args.whitelist_file, "whitelist"),
        (args.loan_history_file, "loan-history"), (args.borrower_file, "borrower"),
    ]:
        if not Path(p).exists():
            sys.exit(f"ERROR: {label} file not found: {p}")

    wl_bl = load_and_merge_lists(args.whitelist_file, args.blacklist_file)
    bl = wl_bl[wl_bl["xtrafloat_list_type"] == "blacklist"].copy()
    target = bl[bl["reason"].astype(str).str.contains(args.reason_match, case=False, na=False)]
    print(f"Blacklisted with reason matching '{args.reason_match}': {len(target):,}\n")

    lh = pd.read_csv(args.loan_history_file)
    lh_col = _find_msisdn_col(lh, "loan-history")
    lh["_key"] = _normalize_msisdn(lh[lh_col])
    disbursed_keys = set(lh["_key"].dropna())

    impossible = target[~target["agent_msisdn_key"].isin(disbursed_keys)].copy()
    n_impossible = len(impossible)
    print(f"Of those, absent from --loan-history-file (the 'impossible defaulter' group): {n_impossible:,}\n")

    if n_impossible == 0:
        print("Nothing to investigate -- every defaulter-reason agent has a disbursement on record.")
        return

    bor = pd.read_csv(args.borrower_file)
    bor_col = _find_msisdn_col(bor, "borrower")
    bor["_key"] = _normalize_msisdn(bor[bor_col])
    n_bor_dupe = int(bor["_key"].duplicated().sum())
    if n_bor_dupe:
        print(f"NOTE: {n_bor_dupe:,} duplicate msisdn rows in borrower file -- keeping first occurrence.")
        bor = bor.drop_duplicates("_key", keep="first")

    present_evidence_cols = [c for c in _LOAN_EVIDENCE_COLS if c in bor.columns]
    if not present_evidence_cols:
        print(f"NOTE: none of the expected loan-evidence columns ({_LOAN_EVIDENCE_COLS}) "
              f"are in the borrower file -- will only check presence, not loan detail.")

    merge_cols = ["_key"] + present_evidence_cols
    merged = impossible.merge(bor[merge_cols], left_on="agent_msisdn_key", right_on="_key", how="left",
                               suffixes=("", "_bor"))

    n_in_borrower = int(merged["_key"].notna().sum())
    n_not_in_borrower = n_impossible - n_in_borrower
    print(f"=== Presence in borrower_history.csv ===")
    print(f"Present in borrower_history.csv: {n_in_borrower:,} ({n_in_borrower/n_impossible:.1%})")
    print(f"Absent from borrower_history.csv too: {n_not_in_borrower:,} ({n_not_in_borrower/n_impossible:.1%})")

    if "total_loans" in merged.columns:
        total_loans_num = pd.to_numeric(merged["total_loans"], errors="coerce")
        n_has_loans = int((total_loans_num > 0).sum())
        n_zero_or_null_loans = n_impossible - n_has_loans
        print(f"\n=== total_loans, within the impossible-defaulter group ===")
        print(f"borrower_history.csv shows total_loans > 0 (explanation (a) -- real loan, "
              f"missing from state_data extract): {n_has_loans:,} ({n_has_loans/n_impossible:.1%})")
        print(f"total_loans == 0, null, or agent absent entirely (explanation (b) -- "
              f"no real ExtraFloat loan on record anywhere): {n_zero_or_null_loans:,} "
              f"({n_zero_or_null_loans/n_impossible:.1%})")

    for ts_col in ("first_loan_ts", "latest_loan_ts", "latest_disbursement_ts"):
        if ts_col in merged.columns:
            n_non_null = int(merged[ts_col].notna().sum())
            print(f"{ts_col} non-null: {n_non_null:,} ({n_non_null/n_impossible:.1%})")

    print(f"\n=== Verdict guide ===")
    print("If most of this group shows total_loans > 0 / non-null loan timestamps in "
          "borrower_history.csv: explanation (a) holds -- state_data_*.csv has a real "
          "coverage gap for these specific agents, not a labeling error.")
    print("If most show no loan evidence anywhere (absent from borrower file, or "
          "total_loans == 0): explanation (b) holds -- the 'Defaulter' reason is likely "
          "mislabeled/reused for something other than an ExtraFloat loan default.")

    merged.to_csv(args.out, index=False)
    print(f"\nFull impossible-defaulter group with borrower_history.csv fields written to: {args.out}")


if __name__ == "__main__":
    main()
