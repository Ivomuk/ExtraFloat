"""
Check the new loan-history-file (state_data_202608122330.csv, built from
analytics.momo_loan_book_tracker_disbursements_daily and
analytics.momo_loan_book_tracker_loan_state_daily) against the business
whitelist/blacklist extracts -- do agents appear consistently across all
three files?

Reuses this repo's own tested whitelist/blacklist loading
(pd_model.postprocessing.whitelist_eval.load_and_merge_lists -- same
MSISDN normalization, same blacklist>whitelist dedup priority) rather
than reimplementing it.

Reports:
  - total unique agents in the loan-history file
  - whitelist coverage: how many whitelist agents appear in this file
  - blacklist coverage: how many blacklist agents appear in this file
  - loan-history agents that are neither whitelisted nor blacklisted
    (unclassified -- expected to be the majority, just for context)

Usage:
    python check_loan_history_vs_whitelist_blacklist.py ^
        --loan-history-file data\\state_data_202608122330.csv ^
        --whitelist-file data\\whitelist_aug_20260804.csv ^
        --blacklist-file data\\blacklist_aug_20260804.csv
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loan-history-file", default="data/state_data_202608122330.csv")
    ap.add_argument("--whitelist-file", default="data/whitelist_aug_20260804.csv")
    ap.add_argument("--blacklist-file", default="data/blacklist_aug_20260804.csv")
    ap.add_argument("--msisdn-col", default=None, help="Override auto-detected msisdn column name")
    ap.add_argument("--out", default="loan_history_vs_wl_bl.csv")
    args = ap.parse_args()

    for p, label in [
        (args.loan_history_file, "loan-history"),
        (args.whitelist_file, "whitelist"),
        (args.blacklist_file, "blacklist"),
    ]:
        if not Path(p).exists():
            sys.exit(f"ERROR: {label} file not found: {p}")

    lh = pd.read_csv(args.loan_history_file)

    msisdn_col = args.msisdn_col
    if msisdn_col is None:
        for candidate in ("msisdn", "agent_msisdn", "customer_msisdn", "phonenumber"):
            if candidate in lh.columns:
                msisdn_col = candidate
                break
    if msisdn_col is None or msisdn_col not in lh.columns:
        sys.exit(
            f"ERROR: could not find an msisdn column in the loan-history file "
            f"(tried msisdn/agent_msisdn/customer_msisdn/phonenumber, or your "
            f"--msisdn-col value). Found columns: {list(lh.columns)[:30]}"
        )
    print(f"Using '{msisdn_col}' as the loan-history msisdn column.\n")

    lh["_key"] = _normalize_msisdn(lh[msisdn_col])
    n_rows = len(lh)
    lh = lh.dropna(subset=["_key"])
    n_dupe = int(lh["_key"].duplicated().sum())
    if n_dupe:
        print(f"NOTE: {n_dupe:,} duplicate msisdn rows in the loan-history file -- "
              f"deduplicating (keeping first) for this comparison.")
        lh = lh.drop_duplicates("_key", keep="first")
    lh_keys = set(lh["_key"])
    print(f"Loan-history file: {n_rows:,} rows -> {len(lh_keys):,} unique agents\n")

    wl_bl = load_and_merge_lists(args.whitelist_file, args.blacklist_file)
    wl = wl_bl[wl_bl["xtrafloat_list_type"] == "whitelist"]
    bl = wl_bl[wl_bl["xtrafloat_list_type"] == "blacklist"]
    wl_keys = set(wl["agent_msisdn_key"])
    bl_keys = set(bl["agent_msisdn_key"])

    print(f"Whitelist: {len(wl_keys):,} agents")
    print(f"Blacklist: {len(bl_keys):,} agents\n")

    wl_in_lh = wl_keys & lh_keys
    bl_in_lh = bl_keys & lh_keys
    wl_missing = wl_keys - lh_keys
    bl_missing = bl_keys - lh_keys

    print("=== Coverage: whitelist / blacklist agents in the loan-history file ===")
    print(
        f"Whitelist agents present in loan-history file: {len(wl_in_lh):,}/{len(wl_keys):,} "
        f"({len(wl_in_lh)/max(1,len(wl_keys)):.1%})"
    )
    print(
        f"Whitelist agents MISSING from loan-history file: {len(wl_missing):,}/{len(wl_keys):,} "
        f"({len(wl_missing)/max(1,len(wl_keys)):.1%})"
    )
    print(
        f"\nBlacklist agents present in loan-history file: {len(bl_in_lh):,}/{len(bl_keys):,} "
        f"({len(bl_in_lh)/max(1,len(bl_keys)):.1%})"
    )
    print(
        f"Blacklist agents MISSING from loan-history file: {len(bl_missing):,}/{len(bl_keys):,} "
        f"({len(bl_missing)/max(1,len(bl_keys)):.1%})"
    )

    lh_unclassified = lh_keys - wl_keys - bl_keys
    print(
        f"\nLoan-history agents that are neither whitelisted nor blacklisted: "
        f"{len(lh_unclassified):,}/{len(lh_keys):,} ({len(lh_unclassified)/max(1,len(lh_keys)):.1%})"
    )

    # -- Export a merged, labeled view for further review -------------------
    lh_out = lh[[msisdn_col, "_key"]].rename(columns={"_key": "agent_msisdn_key"})
    lh_out["in_whitelist"] = lh_out["agent_msisdn_key"].isin(wl_keys)
    lh_out["in_blacklist"] = lh_out["agent_msisdn_key"].isin(bl_keys)
    lh_out = lh_out.merge(
        wl_bl[["agent_msisdn_key", "xtrafloat_list_type", "agent_category"]],
        on="agent_msisdn_key", how="left",
    )
    lh_out.to_csv(args.out, index=False)
    print(f"\nFull loan-history file labeled with whitelist/blacklist status written to: {args.out}")


if __name__ == "__main__":
    main()
