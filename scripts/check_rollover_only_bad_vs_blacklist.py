"""
Direct test of whether bad_state_3dpd_30d's "OR ANOMALY_OPEN" clause should
stay in the training label, using the business's own blacklist ground truth.

dpd_bucket_split_by_anomaly_open.sql found 68.8% of currently-"bad" loans
are bad ONLY via the rollover_observed_30d=1 branch (ANOMALY_OPEN), with no
elevated repayment lateness (max_days_aging_30d never exceeded 3). If
rollover-only-bad agents blacklist at a rate similar to agents with no bad
loans at all, that's evidence the ANOMALY_OPEN branch is adding false
positives to the label, not catching real Defaulters. If they blacklist at
a rate similar to genuinely-aging-bad agents, that's evidence it belongs.

Categorizes each agent (from data/rollover_only_bad_vs_blacklist_export.sql's
export) into exactly one group:
  - never_bad:          no bad_state_3dpd_30d=1 loans at all
  - rollover_only_bad:  has a bad loan, but ALL bad loans are rollover-only
                         (would not be bad under a stricter days_aging-only
                         definition)
  - genuinely_aging_bad: has at least one bad loan where max_days_aging_30d
                         was independently > 3 (still bad either way)

Reuses pd_model.postprocessing.whitelist_eval's tested MSISDN normalization
and blacklist>whitelist dedup priority -- same machinery as
scripts/check_whitelist_blacklist_eval.py.

Usage:
    python scripts\\check_rollover_only_bad_vs_blacklist.py ^
        --rollover-export-file data\\rollover_only_bad_vs_blacklist.csv ^
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
    ap.add_argument("--rollover-export-file", required=True,
                     help="CSV export of data/rollover_only_bad_vs_blacklist_export.sql")
    ap.add_argument("--whitelist-file", default="data/whitelist_aug_20260804.csv")
    ap.add_argument("--blacklist-file", default="data/blacklist_aug_20260804.csv")
    ap.add_argument("--out", default="rollover_only_bad_vs_blacklist_result.csv")
    args = ap.parse_args()

    for p, label in [
        (args.rollover_export_file, "rollover export"),
        (args.whitelist_file, "whitelist"),
        (args.blacklist_file, "blacklist"),
    ]:
        if not Path(p).exists():
            sys.exit(f"ERROR: {label} file not found: {p}")

    df = pd.read_csv(args.rollover_export_file)
    required = ["msisdn", "has_genuinely_aging_bad_loan", "has_rollover_only_bad_loan", "n_bad_loans"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: rollover export file is missing column(s): {missing}. Found: {list(df.columns)}")

    def categorize(row):
        if row["n_bad_loans"] == 0:
            return "never_bad"
        if row["has_genuinely_aging_bad_loan"] == 1:
            return "genuinely_aging_bad"
        return "rollover_only_bad"

    df["group"] = df.apply(categorize, axis=1)
    print("Agent counts by group:")
    print(df["group"].value_counts().to_string())
    print()

    wl_bl = load_and_merge_lists(args.whitelist_file, args.blacklist_file)
    df["agent_msisdn_key"] = df["msisdn"].astype(str).str.replace(".0", "", regex=False).str.strip()

    merged = df.merge(wl_bl, on="agent_msisdn_key", how="inner")
    merged["is_blacklisted"] = (merged["xtrafloat_list_type"] == "blacklist").astype(int)

    print(f"Matched {len(merged):,} of {len(df):,} agents against the whitelist/blacklist ({len(merged)/max(1,len(df)):.1%})\n")

    summary = merged.groupby("group").agg(
        n=("is_blacklisted", "size"),
        blacklist_rate=("is_blacklisted", "mean"),
    ).round(4)
    print("=== Blacklist rate by group ===")
    print(summary.to_string())
    print(
        "\nRead this as: if rollover_only_bad's blacklist_rate is close to "
        "never_bad's, the ANOMALY_OPEN branch is adding false positives to "
        "the label (not catching real Defaulters). If it's close to "
        "genuinely_aging_bad's, it belongs in the label."
    )

    summary.to_csv(args.out)
    print(f"\nWritten to: {args.out}")


if __name__ == "__main__":
    main()
