"""
Closes the loop on check_defaulter_visibility_at_snapshot.py's KEY TEST.
That test found AUC=0.4711-0.4725 for the Defaulter-cohort-WITHOUT-an-
unresolved-loan-at-snapshot subgroup, even after the tenure_days fix gave
the model real closed-loan-history signal. Follow-up diagnostics found:

  - 4.3% (203/4,756) of that subgroup took a NEW loan after the July-31
    snapshot -- for those, the risk event postdates the snapshot entirely,
    so no historical feature could have caught it (check_post_snapshot_
    disbursement.py, after fixing its date_key-as-nanoseconds parsing bug).
  - The remaining ~95.7% show a median of ~39 prior loans with fast
    (median 1-day) recent repayment (check_defaulter_tenure_matches_
    definition.py) -- consistent with a permanently-flagged agent (per the
    product owner: the "Defaulter" tag never clears once a loan crosses
    30+1 days unpaid) whose behavior has since genuinely improved.

This recomputes the exact same KEY TEST AUC (mirrors _mann_whitney_auc /
_auc_for from check_defaulter_visibility_at_snapshot.py precisely) after
excluding the 203 agents with a post-snapshot disbursement, to isolate the
cleanest possible test: does the model correctly score the since-rehabilitated
majority as low risk, once the genuinely-uninformable minority is removed?

Usage:
    python scripts\\check_defaulter_auc_excluding_post_snapshot.py ^
        --matched-file wl_bl_eval_matched_agents.csv ^
        --loan-summary-file data\\loan_summary.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULTER_REASON = "Defaulter not paid back in the last 30 days"

CANDIDATE_DATE_COLS = [
    "last_disbursement_date",
    "Last_disbursement_date",
    "most_recent_disbursement_date",
    "latest_disbursement_date",
]


def _normalize_msisdn(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"[^0-9]", "", regex=True).replace("", pd.NA)


def _parse_date_flexible(s: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(s, errors="coerce")
    if parsed.notna().any() and parsed.dropna().dt.year.max() <= 1971:
        parsed = pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return parsed


def _mann_whitney_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mirrors pd_model.postprocessing.whitelist_eval._mann_whitney_auc exactly
    (same copy used by check_defaulter_visibility_at_snapshot.py)."""
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
    ap.add_argument("--loan-summary-file", required=True)
    ap.add_argument("--date-col", default=None)
    ap.add_argument("--snapshot-date", default="2026-07-31")
    ap.add_argument("--score-col", default="cal_pd")
    args = ap.parse_args()

    matched_path = Path(args.matched_file)
    if not matched_path.exists():
        sys.exit(f"ERROR: matched file not found: {matched_path}")
    ls_path = Path(args.loan_summary_file)
    if not ls_path.exists():
        sys.exit(f"ERROR: loan-summary file not found: {ls_path}")

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

    ls_df = pd.read_csv(ls_path)
    ls_msisdn_col = "msisdn" if "msisdn" in ls_df.columns else ("phonenumber" if "phonenumber" in ls_df.columns else None)
    if ls_msisdn_col is None:
        sys.exit(f"ERROR: {ls_path} has neither 'msisdn' nor 'phonenumber' column. "
                  f"Columns present: {list(ls_df.columns)}")
    date_col = args.date_col or next((c for c in CANDIDATE_DATE_COLS if c in ls_df.columns), None)
    if date_col is None:
        sys.exit(f"ERROR: could not auto-detect a disbursement-date column in {ls_path}. "
                  f"Columns present: {list(ls_df.columns)}. Re-run with --date-col.")
    ls_df["_key"] = _normalize_msisdn(ls_df[ls_msisdn_col])
    ls_df["_date"] = _parse_date_flexible(ls_df[date_col])
    snapshot_date = pd.to_datetime(args.snapshot_date)

    post_snapshot_keys = set(
        ls_df.loc[ls_df["_date"] > snapshot_date, "_key"].dropna()
    )

    defaulter_or_wl = df[df["_cohort"].isin(["whitelist", "blacklist: Defaulter"])].copy()
    without_unresolved = defaulter_or_wl[
        (defaulter_or_wl["_cohort"] == "whitelist")
        | (
            (defaulter_or_wl["_cohort"] == "blacklist: Defaulter")
            & (defaulter_or_wl["has_unresolved_loan_at_snapshot"] == 0)
        )
    ]
    n_before = int((without_unresolved["_cohort"] == "blacklist: Defaulter").sum())
    is_post_snapshot_defaulter = (
        (without_unresolved["_cohort"] == "blacklist: Defaulter")
        & (without_unresolved["_key"].isin(post_snapshot_keys))
    )
    n_excluded = int(is_post_snapshot_defaulter.sum())
    excluded = without_unresolved[~is_post_snapshot_defaulter]
    n_after = int((excluded["_cohort"] == "blacklist: Defaulter").sum())

    print(f"WITHOUT-unresolved Defaulter population before exclusion: {n_before:,}")
    print(f"Excluded (post-snapshot disbursement): {n_excluded:,}")
    print(f"WITHOUT-unresolved Defaulter population after exclusion: {n_after:,}\n")

    auc_before, n_before_total = _auc_for(without_unresolved, args.score_col)
    auc_after, n_after_total = _auc_for(excluded, args.score_col)

    print("=" * 78)
    print("KEY TEST AUC, WITHOUT-unresolved Defaulter vs. whitelist")
    print("=" * 78)
    print(f"  Before exclusion: AUC={auc_before:.4f} (n_whitelist+n_defaulter={n_before_total}, n_defaulter={n_before:,})")
    print(f"  After exclusion:  AUC={auc_after:.4f} (n_whitelist+n_defaulter={n_after_total}, n_defaulter={n_after:,})")
    print(f"  Delta: {auc_after - auc_before:+.4f}")


if __name__ == "__main__":
    main()
