"""
Validates 5 candidate prior-loan-history signals against the business's own
blacklist ground truth BEFORE wiring any of them into training/scoring SQL --
same discipline that caught the prior_late_fee_* feature hurting results and
the OR ANOMALY_OPEN label bug: cheap SQL-only validation first, code changes
only for what survives it.

Computes, for each numeric candidate, the Mann-Whitney AUC against
is_blacklisted (0.5 = no discrimination); for the bucketed dpd candidate,
blacklist rate by bucket. Only signals that show real separation (not near-
universal, not near-random) should become the actual feature set for the
next SQL+training+scoring change.

Input: CSV export of
data/prior_loan_history_candidates_vs_blacklist_export.sql.

Reuses pd_model.postprocessing.whitelist_eval's tested MSISDN normalization
and blacklist>whitelist dedup priority -- same machinery as every other
blacklist-cross-reference script this session.

Usage:
    python scripts\\check_prior_loan_history_candidates_vs_blacklist.py ^
        --candidates-file data\\prior_loan_history_candidates_vs_blacklist.csv ^
        --whitelist-file data\\whitelist_aug_20260804.csv ^
        --blacklist-file data\\blacklist_aug_20260804.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pd_model.postprocessing.whitelist_eval import load_and_merge_lists  # noqa: E402

NUMERIC_CANDIDATES = [
    "prior_max_loan_seq",
    "prior_max_principal_outstanding_ugx",
    "prior_max_total_outstanding_ugx",
    "prior_avg_collection_ratio",
    "prior_principal_unsettled_count",
]

BUCKET_CANDIDATE = "prior_max_dpd_bucket_within_30d"
BUCKET_ORDER = ["NEVER_PAST_DUE", "1-2", "3-6", "7-13", "14-29", "30+"]


def _mann_whitney_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mirrors pd_model.postprocessing.whitelist_eval._mann_whitney_auc exactly."""
    ranks = pd.Series(scores).rank(method="average")
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    sum_ranks_pos = float(ranks[labels == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates-file", required=True,
                     help="CSV export of data/prior_loan_history_candidates_vs_blacklist_export.sql")
    ap.add_argument("--whitelist-file", default="data/whitelist_aug_20260804.csv")
    ap.add_argument("--blacklist-file", default="data/blacklist_aug_20260804.csv")
    ap.add_argument("--out", default="prior_loan_history_candidates_result.csv")
    args = ap.parse_args()

    for p, label in [
        (args.candidates_file, "candidates export"),
        (args.whitelist_file, "whitelist"),
        (args.blacklist_file, "blacklist"),
    ]:
        if not Path(p).exists():
            sys.exit(f"ERROR: {label} file not found: {p}")

    df = pd.read_csv(args.candidates_file)
    required = ["msisdn"] + NUMERIC_CANDIDATES + [BUCKET_CANDIDATE]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: candidates export file is missing column(s): {missing}. Found: {list(df.columns)}")

    wl_bl = load_and_merge_lists(args.whitelist_file, args.blacklist_file)
    df["agent_msisdn_key"] = df["msisdn"].astype(str).str.replace(".0", "", regex=False).str.strip()

    merged = df.merge(wl_bl, on="agent_msisdn_key", how="inner")
    merged["is_blacklisted"] = (merged["xtrafloat_list_type"] == "blacklist").astype(int)

    print(f"Matched {len(merged):,} of {len(df):,} agents against the whitelist/blacklist ({len(merged)/max(1,len(df)):.1%})\n")

    print("=" * 78)
    print("AUC per numeric candidate signal (cal_pd-style: 0.5 = no discrimination)")
    print("=" * 78)
    auc_rows = []
    for col in NUMERIC_CANDIDATES:
        sub = merged[[col, "is_blacklisted"]].dropna()
        auc = _mann_whitney_auc(sub[col].to_numpy(), sub["is_blacklisted"].to_numpy())
        pct_nonzero = float((sub[col] != 0).mean()) if len(sub) else np.nan
        auc_rows.append({
            "candidate": col,
            "n": len(sub),
            "auc_vs_blacklist": round(auc, 4) if not np.isnan(auc) else np.nan,
            "pct_nonzero": round(pct_nonzero, 4),
        })
    auc_df = pd.DataFrame(auc_rows).set_index("candidate")
    print(auc_df.to_string())
    print(
        "\nRead this as: AUC near 0.5 or pct_nonzero near 0%/100% (near-universal, "
        "like ever_penalty_1_due/2_due already ruled out) means the signal doesn't "
        "separate blacklisted agents from whitelisted ones. AUC well above 0.5 with "
        "meaningful variation (pct_nonzero well inside 0-100%) means it does."
    )

    print("\n" + "=" * 78)
    print(f"Blacklist rate by {BUCKET_CANDIDATE}")
    print("=" * 78)
    merged[BUCKET_CANDIDATE] = pd.Categorical(
        merged[BUCKET_CANDIDATE], categories=BUCKET_ORDER, ordered=True
    )
    bucket_summary = merged.groupby(BUCKET_CANDIDATE, observed=True).agg(
        n=("is_blacklisted", "size"),
        blacklist_rate=("is_blacklisted", "mean"),
    ).round(4)
    print(bucket_summary.to_string())
    print(
        "\nRead this as: a monotonic increase in blacklist_rate across buckets "
        "(and no bucket anomalously high, unlike the raw dpd_bucket_distribution "
        "diagnostic's old-label result) is what a genuinely discriminating "
        "duration threshold looks like."
    )

    auc_df.to_csv(args.out)
    bucket_summary.to_csv(f"{Path(args.out).stem}_dpd_bucket.csv")
    print(f"\nWritten to: {args.out} and {Path(args.out).stem}_dpd_bucket.csv")


if __name__ == "__main__":
    main()
