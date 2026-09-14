"""
Sharpest remaining test in this investigation. Prior checks distinguished
"was there enough elapsed time" (calendar dates) from the more decisive
question: did the agent's actual MOST RECENT closed loan (last_closed_loan_
tenure_days) itself take 30+ days to close? If yes, that loan plausibly IS
what triggered the "Defaulter not paid back in the last 30 days" flag --
this is CURRENT behavior (their latest transaction), not an old, buried
event, and the earlier "old flag, since rehabilitated" explanation does not
apply to this subgroup.

check_defaulter_tenure_matches_definition.py already found 12.0% (580/4,852)
of the WITHOUT-unresolved Defaulter population have last_closed_loan_tenure_days
>= 30 days. This splits the KEY TEST AUC (mirrors check_defaulter_visibility_
at_snapshot.py's _mann_whitney_auc/_auc_for exactly) by that threshold:

  - last_closed_loan_tenure_days >= 30: their last loan itself was severely
    late -- a genuine, recent (if since-closed) risk signal. If the model
    scores THIS group as low-risk too, that's a real predictive gap.
  - last_closed_loan_tenure_days < 30: their last loan closed fast -- the
    "old flag, since rehabilitated" explanation applies here; the model
    scoring them low-risk is expected and likely correct.

Usage:
    python scripts\\check_recent_bad_loan_vs_old_flag_auc.py ^
        --matched-file wl_bl_eval_matched_agents.csv ^
        --loan-history-file data\\loan_history_snapshot_20260817_retail_filtered.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULTER_REASON = "Defaulter not paid back in the last 30 days"


def _normalize_msisdn(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"[^0-9]", "", regex=True).replace("", pd.NA)


def _mann_whitney_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mirrors pd_model.postprocessing.whitelist_eval._mann_whitney_auc exactly."""
    ranks = pd.Series(scores).rank(method="average")
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    sum_ranks_pos = float(ranks[labels == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _auc_for(df: pd.DataFrame, score_col: str, label_col: str = "is_blacklisted") -> tuple[float, int]:
    sub = df[[score_col, label_col]].dropna()
    if sub[label_col].nunique() < 2:
        return np.nan, len(sub)
    return _mann_whitney_auc(sub[score_col].to_numpy(), sub[label_col].to_numpy()), len(sub)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matched-file", default="wl_bl_eval_matched_agents.csv")
    ap.add_argument("--loan-history-file", required=True)
    ap.add_argument("--tenure-threshold", type=float, default=30.0)
    ap.add_argument("--score-col", default="cal_pd")
    args = ap.parse_args()

    matched_path = Path(args.matched_file)
    if not matched_path.exists():
        sys.exit(f"ERROR: matched file not found: {matched_path}")
    lh_path = Path(args.loan_history_file)
    if not lh_path.exists():
        sys.exit(f"ERROR: loan-history file not found: {lh_path}")

    df = pd.read_csv(matched_path)
    required = ["is_blacklisted", "reason", "has_unresolved_loan_at_snapshot", args.score_col]
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        sys.exit(f"ERROR: {matched_path} is missing required column(s): {missing_req}")
    msisdn_col = "msisdn" if "msisdn" in df.columns else ("agent_msisdn" if "agent_msisdn" in df.columns else None)
    if msisdn_col is None:
        sys.exit(f"ERROR: {matched_path} has neither 'msisdn' nor 'agent_msisdn' column")

    df["_key"] = _normalize_msisdn(df[msisdn_col])
    df["reason"] = df["reason"].fillna("(whitelist / no reason)")
    df["_cohort"] = np.where(
        df["is_blacklisted"] == 0, "whitelist",
        np.where(df["reason"] == DEFAULTER_REASON, "blacklist: Defaulter", "blacklist: other reason"),
    )

    lh_df = pd.read_csv(lh_path)
    lh_msisdn_col = "msisdn" if "msisdn" in lh_df.columns else ("phonenumber" if "phonenumber" in lh_df.columns else None)
    if lh_msisdn_col is None:
        sys.exit(f"ERROR: {lh_path} has neither 'msisdn' nor 'phonenumber' column. "
                  f"Columns present: {list(lh_df.columns)}")
    if "last_closed_loan_tenure_days" not in lh_df.columns:
        sys.exit(f"ERROR: {lh_path} has no 'last_closed_loan_tenure_days' column. "
                  f"Columns present: {list(lh_df.columns)}")
    lh_df["_key"] = _normalize_msisdn(lh_df[lh_msisdn_col])
    lh_df["_tenure"] = pd.to_numeric(lh_df["last_closed_loan_tenure_days"], errors="coerce")

    df = df.merge(
        lh_df[["_key", "_tenure"]].drop_duplicates(subset="_key"),
        on="_key", how="left",
    )

    defaulter_or_wl = df[df["_cohort"].isin(["whitelist", "blacklist: Defaulter"])].copy()
    without_unresolved = defaulter_or_wl[
        (defaulter_or_wl["_cohort"] == "whitelist")
        | (
            (defaulter_or_wl["_cohort"] == "blacklist: Defaulter")
            & (defaulter_or_wl["has_unresolved_loan_at_snapshot"] == 0)
        )
    ]

    is_defaulter = without_unresolved["_cohort"] == "blacklist: Defaulter"
    recent_bad = without_unresolved[
        (~is_defaulter) | (without_unresolved["_tenure"] >= args.tenure_threshold)
    ]
    old_flag = without_unresolved[
        (~is_defaulter) | (without_unresolved["_tenure"] < args.tenure_threshold)
    ]

    n_recent_bad_def = int(((recent_bad["_cohort"] == "blacklist: Defaulter")).sum())
    n_old_flag_def = int(((old_flag["_cohort"] == "blacklist: Defaulter")).sum())
    n_missing_tenure = int((is_defaulter & without_unresolved["_tenure"].isna()).sum())

    print(f"WITHOUT-unresolved Defaulter population: {int(is_defaulter.sum()):,}")
    print(f"  last_closed_loan_tenure_days >= {args.tenure_threshold:.0f} days (last loan itself plausibly "
          f"the trigger): {n_recent_bad_def:,}")
    print(f"  last_closed_loan_tenure_days < {args.tenure_threshold:.0f} days (last loan closed fast -- "
          f"an older loan must explain the flag, if valid): {n_old_flag_def:,}")
    if n_missing_tenure:
        print(f"  missing last_closed_loan_tenure_days (excluded from both groups below): {n_missing_tenure:,}")
    print()

    auc_recent_bad, n_recent_bad_total = _auc_for(recent_bad, args.score_col)
    auc_old_flag, n_old_flag_total = _auc_for(old_flag, args.score_col)

    print("=" * 78)
    print("KEY TEST AUC vs. whitelist, split by whether the LAST loan itself was severely late")
    print("=" * 78)
    print(f"  last loan >= {args.tenure_threshold:.0f}d (recent bad loan): "
          f"AUC={auc_recent_bad:.4f} (n_whitelist+n_defaulter={n_recent_bad_total}, n_defaulter={n_recent_bad_def:,})")
    print(f"  last loan <  {args.tenure_threshold:.0f}d (old flag, clean now): "
          f"AUC={auc_old_flag:.4f} (n_whitelist+n_defaulter={n_old_flag_total}, n_defaulter={n_old_flag_def:,})")

    print(
        "\nInterpretation: if the 'recent bad loan' AUC is meaningfully higher than the "
        "'old flag' AUC (ideally well above 0.5), the model DOES distinguish agents whose "
        "most recent loan was itself severely late from those with a merely historical flag "
        "-- the earlier 'old flag, since rehabilitated' explanation holds cleanly. If both "
        "AUCs are similarly poor, the model fails to catch even the agents whose latest "
        "transaction was a genuine, recent 30+-day-late repayment -- a real predictive gap "
        "worth investigating further, separate from the label-staleness story."
    )


if __name__ == "__main__":
    main()
