"""
Splits the blacklist into two groups -- agents who actually disbursed an
ExtraFloat loan (present in --loan-history-file, e.g. state_data_202608122330.csv,
scoped to agents with a real disbursement) vs. agents who never did -- and
compares the `reason` breakdown between them.

Tests the hypothesis raised from the earlier coverage check (only 8.2% of
the blacklist has ever disbursed, 91.8% never did): if blacklisting mostly
happens BEFORE disbursement (KYC/fraud/policy screening), the "never
disbursed" 91.8% should skew toward pre-loan reasons, while the "did
disburse" 8.2% should skew toward performance-based reasons (delinquency,
non-repayment, aging). This directly checks that rather than assuming it.

A lightweight keyword heuristic buckets each reason into "likely pre-loan /
administrative" vs "likely loan-performance" vs "unclassified" -- flagged
explicitly as a suggestive aid only, not an authoritative taxonomy, since
the real reason strings haven't all been seen yet. The full, untruncated
reason breakdown for both groups is always shown regardless, so nothing
depends on the heuristic being right.

Usage:
    python check_blacklist_reason_by_disbursement.py ^
        --blacklist-file data\\blacklist_aug_20260804.csv ^
        --whitelist-file data\\whitelist_aug_20260804.csv ^
        --loan-history-file data\\state_data_202608122330.csv
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

# Suggestive only -- see module docstring. Matched case-insensitively as
# substrings against the raw `reason` text.
_PRE_LOAN_KEYWORDS = [
    "kyc", "fraud", "director", "requested", "policy", "duplicate",
    "blacklist", "commission", "active less than", "transacted less than",
    "not eligible", "ineligible", "identity", "document",
]
_PERFORMANCE_KEYWORDS = [
    "delinq", "default", "repay", "aging", "overdue", "dpd", "npl",
    "write off", "write-off", "rollover", "anomaly", "non-repayment",
    "non repayment", "unpaid", "outstanding",
]


def _categorize_reason(reason: str) -> str:
    r = str(reason).lower()
    is_pre = any(kw in r for kw in _PRE_LOAN_KEYWORDS)
    is_perf = any(kw in r for kw in _PERFORMANCE_KEYWORDS)
    if is_perf and not is_pre:
        return "likely loan-performance"
    if is_pre and not is_perf:
        return "likely pre-loan / administrative"
    if is_pre and is_perf:
        return "matches both -- ambiguous"
    return "unclassified"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blacklist-file", default="data/blacklist_aug_20260804.csv")
    ap.add_argument("--whitelist-file", default="data/whitelist_aug_20260804.csv")
    ap.add_argument("--loan-history-file", default="data/state_data_202608122330.csv")
    ap.add_argument("--msisdn-col", default=None, help="Override auto-detected msisdn column name in the loan-history file")
    ap.add_argument("--out", default="blacklist_reason_by_disbursement.csv")
    args = ap.parse_args()

    for p, label in [
        (args.blacklist_file, "blacklist"), (args.whitelist_file, "whitelist"),
        (args.loan_history_file, "loan-history"),
    ]:
        if not Path(p).exists():
            sys.exit(f"ERROR: {label} file not found: {p}")

    wl_bl = load_and_merge_lists(args.whitelist_file, args.blacklist_file)
    bl = wl_bl[wl_bl["xtrafloat_list_type"] == "blacklist"].copy()
    if "reason" not in bl.columns:
        sys.exit("ERROR: blacklist file has no 'reason' column.")
    n_bl = len(bl)
    print(f"Blacklist (deduped): {n_bl:,} agents\n")

    lh = pd.read_csv(args.loan_history_file)
    msisdn_col = args.msisdn_col
    if msisdn_col is None:
        for candidate in ("msisdn", "agent_msisdn", "customer_msisdn", "phonenumber"):
            if candidate in lh.columns:
                msisdn_col = candidate
                break
    if msisdn_col is None or msisdn_col not in lh.columns:
        sys.exit(
            f"ERROR: could not find an msisdn column in the loan-history file. "
            f"Found columns: {list(lh.columns)[:30]}"
        )
    lh["_key"] = _normalize_msisdn(lh[msisdn_col])
    disbursed_keys = set(lh["_key"].dropna())
    print(f"Loan-history file: {len(disbursed_keys):,} unique agents with a disbursement\n")

    bl["has_disbursed"] = bl["agent_msisdn_key"].isin(disbursed_keys)
    disbursed = bl[bl["has_disbursed"]]
    never_disbursed = bl[~bl["has_disbursed"]]
    print(f"Blacklisted AND disbursed: {len(disbursed):,} ({len(disbursed)/n_bl:.1%})")
    print(f"Blacklisted, NEVER disbursed: {len(never_disbursed):,} ({len(never_disbursed)/n_bl:.1%})\n")

    # -- Full, untruncated reason breakdown, side by side --------------------
    disbursed_counts = disbursed["reason"].value_counts(dropna=False)
    never_counts = never_disbursed["reason"].value_counts(dropna=False)
    disbursed_pct = (disbursed_counts / max(1, len(disbursed)) * 100).round(2)
    never_pct = (never_counts / max(1, len(never_disbursed)) * 100).round(2)

    breakdown = pd.DataFrame({
        "disbursed_n": disbursed_counts,
        "disbursed_pct": disbursed_pct,
        "never_disbursed_n": never_counts,
        "never_disbursed_pct": never_pct,
    }).fillna(0)
    breakdown = breakdown.sort_values("never_disbursed_pct", ascending=False)

    print("=== Full reason breakdown: disbursed vs. never-disbursed blacklisted agents ===")
    print(breakdown.to_string())

    # -- Suggestive keyword categorization ------------------------------------
    bl["reason_category"] = bl["reason"].apply(_categorize_reason)
    print("\n=== Suggestive category breakdown (keyword heuristic -- see docstring caveat) ===")
    cat_tbl = (
        bl.groupby(["has_disbursed", "reason_category"])
        .size()
        .rename("n")
        .reset_index()
        .pivot(index="reason_category", columns="has_disbursed", values="n")
        .fillna(0)
        .astype(int)
    )
    cat_tbl = cat_tbl.rename(columns={False: "never_disbursed_n", True: "disbursed_n"})
    for col in ("never_disbursed_n", "disbursed_n"):
        if col not in cat_tbl.columns:
            cat_tbl[col] = 0
    cat_tbl["never_disbursed_pct"] = (cat_tbl["never_disbursed_n"] / max(1, len(never_disbursed)) * 100).round(1)
    cat_tbl["disbursed_pct"] = (cat_tbl["disbursed_n"] / max(1, len(disbursed)) * 100).round(1)
    print(cat_tbl[["never_disbursed_n", "never_disbursed_pct", "disbursed_n", "disbursed_pct"]].to_string())

    bl.to_csv(args.out, index=False)
    print(f"\nFull blacklist with has_disbursed + reason_category written to: {args.out}")


if __name__ == "__main__":
    main()
