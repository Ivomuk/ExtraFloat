"""
Pre-retrain validation for the bad_state_3dpd_30d label fix (removed
"OR ANOMALY_OPEN" -- see loan_state_query_updated_materialized.txt's
ANOMALY_OPEN LABEL DECISION comment). Confirms the new label's bad/good
split aligns with the business's own blacklist ground truth BEFORE
spending a full retrain cycle -- the prior-loan-penalty-history feature
change already burned one retrain on an untested hypothesis that hurt.

Computes blacklist rate for agents with vs. without a bad_state_3dpd_30d=1
loan under the NEW label (data/new_label_vs_blacklist_prevalidation.sql's
export). Compare against this session's earlier numbers for context:
  - Under the OLD label: genuinely_aging_bad agents blacklisted at 25.44%,
    never_bad at 17.71%, rollover_only_bad at 10.47%
    (scripts/check_rollover_only_bad_vs_blacklist.py).
The new label's "has_bad_loan" group should blacklist meaningfully above
its "no_bad_loan" group -- ideally close to or above the old
genuinely_aging_bad rate (25.44%), confirming the fix sharpened the label
rather than just shrinking it.

Reuses pd_model.postprocessing.whitelist_eval's tested MSISDN
normalization and blacklist>whitelist dedup priority.

Usage:
    python scripts\\check_new_label_vs_blacklist.py ^
        --label-export-file data\\new_label_vs_blacklist.csv ^
        --whitelist-file data\\whitelist_aug_20260804.csv ^
        --blacklist-file data\\blacklist_aug_20260804.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pd_model.postprocessing.whitelist_eval import load_and_merge_lists  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label-export-file", required=True,
                     help="CSV export of data/new_label_vs_blacklist_prevalidation.sql")
    ap.add_argument("--whitelist-file", default="data/whitelist_aug_20260804.csv")
    ap.add_argument("--blacklist-file", default="data/blacklist_aug_20260804.csv")
    ap.add_argument("--out", default="new_label_vs_blacklist_result.csv")
    args = ap.parse_args()

    for p, label in [
        (args.label_export_file, "label export"),
        (args.whitelist_file, "whitelist"),
        (args.blacklist_file, "blacklist"),
    ]:
        if not Path(p).exists():
            sys.exit(f"ERROR: {label} file not found: {p}")

    df = pd.read_csv(args.label_export_file)
    required = ["msisdn", "has_bad_loan_new_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: label export file is missing column(s): {missing}. Found: {list(df.columns)}")

    df["group"] = df["has_bad_loan_new_label"].map({1: "has_bad_loan_new_label", 0: "no_bad_loan_new_label"})

    wl_bl = load_and_merge_lists(args.whitelist_file, args.blacklist_file)
    df["agent_msisdn_key"] = df["msisdn"].astype(str).str.replace(".0", "", regex=False).str.strip()

    merged = df.merge(wl_bl, on="agent_msisdn_key", how="inner")
    merged["is_blacklisted"] = (merged["xtrafloat_list_type"] == "blacklist").astype(int)

    print(f"Matched {len(merged):,} of {len(df):,} agents against the whitelist/blacklist ({len(merged)/max(1,len(df)):.1%})\n")

    summary = merged.groupby("group").agg(
        n=("is_blacklisted", "size"),
        blacklist_rate=("is_blacklisted", "mean"),
    ).round(4)
    print("=== Blacklist rate by group (NEW label) ===")
    print(summary.to_string())
    print(
        "\nCompare against the OLD label's breakdown "
        "(check_rollover_only_bad_vs_blacklist.py): genuinely_aging_bad=25.44%, "
        "never_bad=17.71%, rollover_only_bad=10.47%. If has_bad_loan_new_label's "
        "rate here is well above no_bad_loan_new_label's -- ideally close to or "
        "above 25.44% -- the fix sharpened the label. If it's not meaningfully "
        "separated, something else needs investigating before retraining."
    )

    summary.to_csv(args.out)
    print(f"\nWritten to: {args.out}")


if __name__ == "__main__":
    main()
