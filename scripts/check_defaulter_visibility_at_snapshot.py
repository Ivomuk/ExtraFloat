"""
Root-cause diagnostic for check_whitelist_blacklist_eval.py's Part C finding:
"Defaulter not paid back in the last 30 days" (a genuine loan-performance
blacklist reason) scores AUC=0.483 vs. cal_pd -- worse than random, meaning
the model scores confirmed defaulters as slightly SAFER than whitelisted
agents on average.

Hypothesis: a defaulter whose bad loan has already CLOSED by the engine's
--snapshot-date is invisible to the point-in-time
has_unresolved_loan_at_snapshot / anomaly_open_at_snapshot signals (from
data/loan_history_snapshot_query.txt) -- the model only "sees" risk from a
loan that is still open AT the snapshot, not one that already resolved
badly. If true, AUC within the Defaulter cohort should be much better for
agents WITH an unresolved loan at snapshot than for those without one.

Two-tier check:
  TIER 1 (always runs): uses only columns already in
  wl_bl_eval_matched_agents.csv (has_unresolved_loan_at_snapshot,
  anomaly_open_at_snapshot, active_loan_days_aging_at_snapshot,
  risk_unresolved_loan_haircut_reason) -- the point-in-time snapshot
  proxy, no extra export needed.

  TIER 2 (optional, --ever-anomaly-open-file): joins in the TRUE
  ever_anomaly_open flag (per-loan-history, not just latest snapshot) --
  see data/ever_anomaly_open_export.sql for the one-line query to export
  this from <schema>.tbl_bh_loan_final (the checkpoint-1 table that stops
  short of vw_bh_output -- this flag never reached borrower_history.csv by
  design, see borrower_history.txt's KNOWN LIMITATION comment). Exact join
  on phonenumber -- unlike ops_scored.csv's msisdn, borrower_history.csv's
  (and therefore engine_test_output.csv's) msisdn is never hashed.

thin_file_flag is NOT included here -- it only exists in
pd_model/artifacts/ops_scored.csv, whose agent_msisdn is deliberately
SHA-256 hashed before writing, and reproducing that hash on the
blacklist's raw msisdns is too fragile to trust without an exact,
verified string-format match to what run_pipeline.py hashed.

Usage:
    python scripts\\check_defaulter_visibility_at_snapshot.py ^
        --matched-file wl_bl_eval_matched_agents.csv

    # with the true ever_anomaly_open flag joined in too:
    python scripts\\check_defaulter_visibility_at_snapshot.py ^
        --matched-file wl_bl_eval_matched_agents.csv ^
        --ever-anomaly-open-file data\\ever_anomaly_open.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pd_model.postprocessing.whitelist_eval import _normalize_msisdn  # noqa: E402

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


def _auc_for(df: pd.DataFrame, score_col: str, label_col: str = "is_blacklisted") -> tuple[float, int]:
    sub = df[[score_col, label_col]].dropna()
    if sub[label_col].nunique() < 2:
        return np.nan, len(sub)
    return _mann_whitney_auc(sub[score_col].to_numpy(), sub[label_col].to_numpy()), len(sub)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matched-file", default="wl_bl_eval_matched_agents.csv")
    ap.add_argument("--ever-anomaly-open-file", default=None,
                     help="Optional CSV with phonenumber + ever_anomaly_open (see "
                          "data/ever_anomaly_open_export.sql). Enables TIER 2.")
    ap.add_argument("--score-col", default="cal_pd", help="Lower = safer")
    ap.add_argument("--out-prefix", default="defaulter_visibility")
    args = ap.parse_args()

    matched_path = Path(args.matched_file)
    if not matched_path.exists():
        sys.exit(f"ERROR: matched file not found: {matched_path}")
    df = pd.read_csv(matched_path)

    required = ["is_blacklisted", "reason", args.score_col]
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        sys.exit(f"ERROR: {matched_path} is missing required column(s): {missing_req}")

    snapshot_cols = [
        "has_unresolved_loan_at_snapshot",
        "anomaly_open_at_snapshot",
        "active_loan_days_aging_at_snapshot",
        "risk_unresolved_loan_haircut_reason",
    ]
    missing_snap = [c for c in snapshot_cols if c not in df.columns]
    if missing_snap:
        print(
            f"WARNING: {matched_path} is missing {missing_snap} -- these come from "
            "data/loan_history_snapshot_query.txt via --loan-history-file. Was "
            "engine_test_output.csv built with that optional input? TIER 1 checks "
            "using these columns will be skipped."
        )

    df["reason"] = df["reason"].fillna("(whitelist / no reason)")
    df["_cohort"] = np.where(
        df["is_blacklisted"] == 0, "whitelist",
        np.where(df["reason"] == DEFAULTER_REASON, "blacklist: Defaulter", "blacklist: other reason"),
    )

    print("=" * 78)
    print(f"TIER 1: point-in-time snapshot signals (as of the engine's --snapshot-date)")
    print("=" * 78)

    have_unresolved = "has_unresolved_loan_at_snapshot" not in missing_snap
    have_anomaly = "anomaly_open_at_snapshot" not in missing_snap
    have_aging = "active_loan_days_aging_at_snapshot" not in missing_snap
    have_haircut = "risk_unresolved_loan_haircut_reason" not in missing_snap

    agg = {"n": (args.score_col, "size"), "mean_cal_pd": (args.score_col, "mean")}
    if have_unresolved:
        agg["pct_unresolved_at_snapshot"] = ("has_unresolved_loan_at_snapshot", "mean")
    if have_anomaly:
        agg["pct_anomaly_open_at_snapshot"] = ("anomaly_open_at_snapshot", "mean")
    if have_aging:
        agg["mean_aging_days_if_unresolved"] = ("active_loan_days_aging_at_snapshot", "mean")
    summary = df.groupby("_cohort").agg(**agg).round(4)
    print(summary.to_string())

    if have_haircut:
        print("\n--- risk_unresolved_loan_haircut_reason value counts, Defaulter cohort ---")
        defaulter_df = df[df["_cohort"] == "blacklist: Defaulter"]
        print(
            defaulter_df["risk_unresolved_loan_haircut_reason"]
            .fillna("(no haircut applied)")
            .value_counts()
            .to_string()
        )

    if have_unresolved:
        print("\n" + "=" * 78)
        print("KEY TEST: AUC within the Defaulter cohort, split by has_unresolved_loan_at_snapshot")
        print("(if the hypothesis is right, WITH an unresolved loan >> WITHOUT one)")
        print("=" * 78)
        defaulter_or_wl = df[df["_cohort"].isin(["whitelist", "blacklist: Defaulter"])].copy()
        for flag_val, label in [(1, "WITH unresolved loan at snapshot"), (0, "WITHOUT unresolved loan at snapshot")]:
            sub = defaulter_or_wl[
                (defaulter_or_wl["_cohort"] == "whitelist")
                | (
                    (defaulter_or_wl["_cohort"] == "blacklist: Defaulter")
                    & (defaulter_or_wl["has_unresolved_loan_at_snapshot"] == flag_val)
                )
            ]
            auc, n = _auc_for(sub, args.score_col)
            n_def = int(((sub["_cohort"] == "blacklist: Defaulter")).sum())
            print(f"  {label}: AUC={auc:.4f} (n_whitelist+n_defaulter={n}, n_defaulter={n_def})")

    out_path = f"{args.out_prefix}_tier1_by_cohort.csv"
    summary.to_csv(out_path)
    print(f"\nTIER 1 summary written to: {out_path}")

    # ------------------------------------------------------------------ #
    # TIER 2 (optional): true ever_anomaly_open, joined on phonenumber
    # ------------------------------------------------------------------ #
    if args.ever_anomaly_open_file:
        eao_path = Path(args.ever_anomaly_open_file)
        if not eao_path.exists():
            sys.exit(f"ERROR: --ever-anomaly-open-file not found: {eao_path}")
        eao = pd.read_csv(eao_path)
        phone_col = next((c for c in ["phonenumber", "msisdn"] if c in eao.columns), None)
        if phone_col is None or "ever_anomaly_open" not in eao.columns:
            sys.exit(
                f"ERROR: {eao_path} must have a 'phonenumber' or 'msisdn' column plus "
                f"'ever_anomaly_open'. Found: {list(eao.columns)}"
            )
        eao["agent_msisdn_key"] = _normalize_msisdn(eao[phone_col])
        if "agent_msisdn_key" not in df.columns:
            if "msisdn" not in df.columns:
                sys.exit("ERROR: matched file has no 'msisdn' column to key the ever_anomaly_open join on.")
            df["agent_msisdn_key"] = _normalize_msisdn(df["msisdn"])
        df2 = df.merge(
            eao[["agent_msisdn_key", "ever_anomaly_open"]].drop_duplicates("agent_msisdn_key"),
            on="agent_msisdn_key", how="left",
        )
        n_matched_eao = int(df2["ever_anomaly_open"].notna().sum())
        print("\n" + "=" * 78)
        print(f"TIER 2: true ever_anomaly_open (matched for {n_matched_eao:,}/{len(df2):,} agents)")
        print("=" * 78)

        summary2 = df2.groupby("_cohort").agg(
            n=(args.score_col, "size"),
            mean_cal_pd=(args.score_col, "mean"),
            pct_ever_anomaly_open=("ever_anomaly_open", "mean"),
        ).round(4)
        print(summary2.to_string())

        print("\n--- AUC within Defaulter cohort, split by TRUE ever_anomaly_open ---")
        defaulter_or_wl2 = df2[df2["_cohort"].isin(["whitelist", "blacklist: Defaulter"])].copy()
        for flag_val, label in [(True, "ever_anomaly_open=True"), (False, "ever_anomaly_open=False")]:
            sub = defaulter_or_wl2[
                (defaulter_or_wl2["_cohort"] == "whitelist")
                | (
                    (defaulter_or_wl2["_cohort"] == "blacklist: Defaulter")
                    & (defaulter_or_wl2["ever_anomaly_open"] == flag_val)
                )
            ]
            auc, n = _auc_for(sub, args.score_col)
            n_def = int((sub["_cohort"] == "blacklist: Defaulter").sum())
            print(f"  {label}: AUC={auc:.4f} (n_whitelist+n_defaulter={n}, n_defaulter={n_def})")

        out_path2 = f"{args.out_prefix}_tier2_by_cohort.csv"
        summary2.to_csv(out_path2)
        print(f"\nTIER 2 summary written to: {out_path2}")
    else:
        print(
            "\n(TIER 2 skipped -- pass --ever-anomaly-open-file to also check against the "
            "TRUE per-loan-history ever_anomaly_open flag; see "
            "data/ever_anomaly_open_export.sql to generate it.)"
        )


if __name__ == "__main__":
    main()
