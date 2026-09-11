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

The duration candidates (prior_dpd_exceed_3_rate, most_recent_prior_dpd_
bucket_within_30d) use a rate/recency framing rather than a raw MAX across
every prior loan -- an earlier version's MAX-based bucket showed a
non-monotonic, backwards pattern (the "30+" bucket had the LOWEST blacklist
rate of all, while holding 72% of the population) because it was confounded
by prior_max_loan_seq's own finding: agents with more prior loans are safer,
so more prior loans also means more chances for one to have drifted past 30
days purely from exposure.

Input: CSV export of
data/prior_loan_history_candidates_vs_blacklist_export.sql.

Also reports every signal against the "Defaulter not paid back in the last
30 days" blacklist reason specifically (whitelist vs. that reason only,
dropping every other blacklisted agent from the comparison) -- mirrors
check_whitelist_blacklist_eval.py's Part C and check_defaulter_visibility_
at_snapshot.py's cohort split. The full blacklist mixes reasons that have
nothing to do with repayment (e.g. "Average Monthly Commission below 20K",
"Agent active less than 3 months" -- together far outnumbering "Defaulter"),
and prior_max_loan_seq's own inverted result (more prior loans = safer) is
exactly the kind of thing an activity/tenure-driven blacklist reason would
produce independent of actual default risk. The Defaulter-only comparison
is what actually answers whether a candidate predicts default.

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
    "prior_dpd_exceed_3_rate",
]

# Recency-based duration framing (the single most recent prior loan's own
# dpd outcome), not a MAX across every prior loan -- see the diagnostic
# SQL's DURATION FRAMING comment: aggregating MAX across a variable-length
# history confounds duration with prior_max_loan_seq's own tenure/safety
# signal.
BUCKET_CANDIDATE = "most_recent_prior_dpd_bucket_within_30d"
# NO_PRIOR_LOAN / NEVER_PAST_DUE_CONFIRMED / NEVER_PAST_DUE_CENSORED replace
# the old single NEVER_PAST_DUE bucket -- that combined bucket's blacklist
# rate (32%) sat above the mild-lateness buckets (11-12%), and splitting it
# tests whether that was a censoring artifact rather than a genuine "never
# late" signal (see the SQL's NEVER_PAST_DUE SPLIT comment).
BUCKET_ORDER = [
    "NO_PRIOR_LOAN", "NEVER_PAST_DUE_CONFIRMED", "NEVER_PAST_DUE_CENSORED",
    "1-2", "3-6", "7-13", "14-29", "30+",
]

DEFAULTER_REASON = "Defaulter not paid back in the last 30 days"


def _mann_whitney_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mirrors pd_model.postprocessing.whitelist_eval._mann_whitney_auc exactly."""
    ranks = pd.Series(scores).rank(method="average")
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    sum_ranks_pos = float(ranks[labels == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _report(sub_df: pd.DataFrame, label_col: str, out_prefix: str, title: str) -> None:
    print("\n" + "#" * 78)
    print(f"# {title}  (n={len(sub_df):,}, positives={int(sub_df[label_col].sum()):,})")
    print("#" * 78)

    print("=" * 78)
    print(f"AUC per numeric candidate signal vs. {label_col} (0.5 = no discrimination)")
    print("=" * 78)
    auc_rows = []
    for col in NUMERIC_CANDIDATES:
        sub = sub_df[[col, label_col]].dropna()
        auc = _mann_whitney_auc(sub[col].to_numpy(), sub[label_col].to_numpy())
        pct_nonzero = float((sub[col] != 0).mean()) if len(sub) else np.nan
        auc_rows.append({
            "candidate": col,
            "n": len(sub),
            f"auc_vs_{label_col}": round(auc, 4) if not np.isnan(auc) else np.nan,
            "pct_nonzero": round(pct_nonzero, 4),
        })
    auc_df = pd.DataFrame(auc_rows).set_index("candidate")
    print(auc_df.to_string())

    print(f"\n{'=' * 78}")
    print(f"{label_col} rate by {BUCKET_CANDIDATE}")
    print("=" * 78)
    sub_df = sub_df.copy()
    sub_df[BUCKET_CANDIDATE] = pd.Categorical(
        sub_df[BUCKET_CANDIDATE], categories=BUCKET_ORDER, ordered=True
    )
    bucket_summary = sub_df.groupby(BUCKET_CANDIDATE, observed=True).agg(
        n=(label_col, "size"),
        **{f"{label_col}_rate": (label_col, "mean")},
    ).round(4)
    print(bucket_summary.to_string())

    auc_df.to_csv(f"{out_prefix}.csv")
    bucket_summary.to_csv(f"{out_prefix}_dpd_bucket.csv")
    print(f"\nWritten to: {out_prefix}.csv and {out_prefix}_dpd_bucket.csv")


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
    merged["reason"] = merged["reason"].fillna("(whitelist / no reason)")

    print(f"Matched {len(merged):,} of {len(df):,} agents against the whitelist/blacklist ({len(merged)/max(1,len(df)):.1%})\n")

    _report(
        merged, "is_blacklisted", args.out,
        "FULL BLACKLIST (all reasons mixed -- activity/commission reasons dominate)",
    )
    print(
        "\nRead this as: AUC near 0.5 or pct_nonzero near 0%/100% (near-universal, "
        "like ever_penalty_1_due/2_due already ruled out) means the signal doesn't "
        "separate blacklisted agents from whitelisted ones. A monotonic bucket-rate "
        "climb (and no bucket anomalously high) is what a genuinely discriminating "
        "duration threshold looks like -- but see the Defaulter-only section below "
        "for whether that's actually about repayment risk or just activity/tenure."
    )

    defaulter_or_wl = merged[
        (merged["is_blacklisted"] == 0) | (merged["reason"] == DEFAULTER_REASON)
    ].copy()
    defaulter_or_wl["is_defaulter"] = (defaulter_or_wl["reason"] == DEFAULTER_REASON).astype(int)

    out_prefix_defaulter = f"{Path(args.out).stem}_defaulter_only"
    _report(
        defaulter_or_wl, "is_defaulter", out_prefix_defaulter,
        f'DEFAULTER-ONLY ("{DEFAULTER_REASON}" vs. whitelist, other blacklist reasons dropped)',
    )
    print(
        "\nRead this as: compare each candidate's AUC/bucket pattern here against "
        "its full-blacklist counterpart above. A candidate that looks strong above "
        "but weak/flat here (e.g. if prior_max_loan_seq's inversion shrinks toward "
        "0.5) is picking up activity/tenure blacklist reasons, not default risk -- "
        "drop it. A candidate that holds up here is a genuine repayment-risk signal."
    )


if __name__ == "__main__":
    main()
