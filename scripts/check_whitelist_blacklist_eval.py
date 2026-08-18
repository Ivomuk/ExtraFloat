"""
Baseline PD-model evaluation against the business-curated whitelist/blacklist
extracts (whitelist_aug_20260804.csv: agent_msisdn, agent_category --
blacklist_aug_20260804.csv: agent_msisdn, agent_category, reason). Meant to
be re-run after model fixes land, and diffed against the JSON baseline
summary this writes, to check whether discrimination actually improved.

PART A -- confirms/refutes the "by design vs. genuine coverage gap"
question from the borrower_history.csv coverage finding (Issue #3):
of business's OWN approved-eligible whitelist agents, how many are
structurally invisible to the scoring engine because they're missing
from --borrower-file?

PART B -- evaluates PD model discrimination against whitelist (good) /
blacklist (bad) ground truth, reusing this repo's own tested
whitelist-eval machinery (pd_model.postprocessing.whitelist_eval --
same MSISDN normalization, same blacklist>whitelist dedup priority, same
Mann-Whitney AUC formula, same cutoff_sweep function) applied to the
FULL scored population (cal_pd / risk_tier / assigned_limit) rather than
just the thin-file scorecard subset that module was originally built for.
Also flags two business-critical red-flag groups directly: blacklisted
agents who received assigned_limit > 0, and whitelisted agents who got
assigned_limit == 0.

PART C -- breaks AUC/decile discrimination down PER blacklist reason
(whitelist vs. only-that-reason's blacklisted agents each time), to test
whether a specific reason category -- e.g. commission/activity-based
reasons that aren't really a credit-risk judgment -- is diluting or
inverting the overall AUC computed in Part B, versus reasons that are a
genuine loan-performance signal (e.g. "Defaulter...").

Use --score-col to switch the discrimination check between cal_pd and
risk_score. IMPORTANT: cal_pd is "lower = safer" but risk_score is
"higher = safer" (risk_tier_1 requires risk_score >= 0.85) -- opposite
polarity. Pass --higher-is-safer when using risk_score, or every AUC/
decile number in this script will be silently inverted and misleading.

Usage:
    python check_whitelist_blacklist_eval.py ^
        --whitelist-file data\\whitelist_aug_20260804.csv ^
        --blacklist-file data\\blacklist_aug_20260804.csv ^
        --output-file output\\engine_test_output.csv ^
        --borrower-file data\\borrower_history.csv ^
        --score-col cal_pd

    # risk_score instead (note --higher-is-safer):
    python check_whitelist_blacklist_eval.py ^
        --score-col risk_score --higher-is-safer ^
        --whitelist-file data\\whitelist_aug_20260804.csv ^
        --blacklist-file data\\blacklist_aug_20260804.csv ^
        --output-file output\\engine_test_output.csv
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Make pd_model importable even if this repo isn't pip-installed in the
# current environment -- insert the repo root (parent of scripts/) ahead
# of relying on an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pd_model.config.feature_config import NON_PERF_BLACKLIST_REASONS  # noqa: E402
from pd_model.postprocessing.whitelist_eval import (  # noqa: E402
    _normalize_msisdn,
    cutoff_sweep,
    load_and_merge_lists,
)


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
    ap.add_argument("--whitelist-file", default="whitelist_aug_20260804.csv")
    ap.add_argument("--blacklist-file", default="blacklist_aug_20260804.csv")
    ap.add_argument("--output-file", default="output/engine_test_output.csv")
    ap.add_argument("--transaction-file", default="data/mfs_daily_agent_mart_20260731.csv")
    ap.add_argument("--borrower-file", default="data/borrower_history.csv")
    ap.add_argument("--score-col", default="cal_pd", help="Lower = safer, per model convention (see --higher-is-safer)")
    ap.add_argument("--higher-is-safer", action="store_true",
                     help="Set for score columns like risk_score where HIGHER = safer (opposite of cal_pd)")
    ap.add_argument("--min-reason-n", type=int, default=30,
                     help="Minimum blacklisted-agent count for a reason to get its own AUC row in Part C")
    ap.add_argument("--out-prefix", default="wl_bl_eval")
    args = ap.parse_args()

    for p, label in [
        (args.whitelist_file, "whitelist"), (args.blacklist_file, "blacklist"),
        (args.output_file, "output"),
    ]:
        if not Path(p).exists():
            sys.exit(f"ERROR: {label} file not found: {p}")

    wl_bl = load_and_merge_lists(args.whitelist_file, args.blacklist_file)
    n_wl = int((wl_bl["xtrafloat_list_type"] == "whitelist").sum())
    n_bl = int((wl_bl["xtrafloat_list_type"] == "blacklist").sum())
    print(f"Loaded and deduped (blacklist > whitelist priority): {len(wl_bl):,} agents "
          f"(whitelist={n_wl:,}, blacklist={n_bl:,})\n")

    baseline = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_file": args.output_file,
        "score_col": args.score_col,
        "higher_is_safer": args.higher_is_safer,
        "whitelist_n": n_wl,
        "blacklist_n": n_bl,
    }

    # =========================================================================
    # PART A -- coverage vs. borrower_history.csv (Issue #3 confirmation)
    # =========================================================================
    print("=" * 70)
    print("PART A: whitelist/blacklist coverage vs. --borrower-file")
    print("=" * 70)

    bor_path = Path(args.borrower_file)
    if bor_path.exists():
        bor = pd.read_csv(bor_path)
        bor_col = "msisdn" if "msisdn" in bor.columns else ("phonenumber" if "phonenumber" in bor.columns else None)
        if bor_col is None:
            print("NOTE: borrower file has neither 'msisdn' nor 'phonenumber' column -- skipping Part A.")
        else:
            bor_keys = set(_normalize_msisdn(bor[bor_col]).dropna())

            for list_type, n_list in [("whitelist", n_wl), ("blacklist", n_bl)]:
                sub = wl_bl[wl_bl["xtrafloat_list_type"] == list_type]
                missing = sub[~sub["agent_msisdn_key"].isin(bor_keys)]
                n_missing = len(missing)
                baseline[f"{list_type}_missing_from_borrower_n"] = n_missing
                baseline[f"{list_type}_missing_from_borrower_pct"] = n_missing / max(1, n_list)
                print(
                    f"\n{list_type.capitalize()}: {n_list:,} agents -> "
                    f"{n_missing:,} ({n_missing/max(1,n_list):.1%}) missing from borrower_history.csv "
                    f"(structurally invisible to the scoring engine, regardless of activity)"
                )
                if n_missing > 0:
                    sample_cols = [c for c in ["agent_msisdn", "agent_category", "reason"] if c in missing.columns]
                    print(f"  Sample (up to 10): \n{missing[sample_cols].head(10).to_string(index=False)}")
    else:
        print(f"NOTE: borrower file '{args.borrower_file}' not found -- skipping Part A.")

    # =========================================================================
    # PART B -- PD model discrimination vs. whitelist/blacklist ground truth
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART B: PD model discrimination vs. whitelist/blacklist")
    print("=" * 70)

    out = pd.read_csv(args.output_file)
    if "msisdn" not in out.columns:
        sys.exit(f"ERROR: output file has no 'msisdn' column. Found: {list(out.columns)}")
    if args.score_col not in out.columns:
        sys.exit(
            f"ERROR: output file has no '{args.score_col}' column. Found: {list(out.columns)}\n"
            f"Try --score-col risk_score --higher-is-safer if cal_pd isn't present."
        )

    out["agent_msisdn_key"] = _normalize_msisdn(out["msisdn"])
    wl_bl_only = wl_bl[wl_bl["xtrafloat_list_type"].isin(["whitelist", "blacklist"])]

    eval_df = out.merge(
        wl_bl_only[["agent_msisdn_key", "xtrafloat_list_type", "reason"]] if "reason" in wl_bl_only.columns
        else wl_bl_only[["agent_msisdn_key", "xtrafloat_list_type"]],
        on="agent_msisdn_key", how="inner",
    )
    eval_df["is_blacklisted"] = (eval_df["xtrafloat_list_type"] == "blacklist").astype(int)

    # raw_score_col keeps the original, human-readable units for display
    # (decile avg_score, etc.); safety_score_col is what every AUC/rank/
    # decile computation actually sorts on, always oriented "lower = safer"
    # regardless of the source column's natural polarity.
    raw_score_col = f"{args.score_col}_raw"
    safety_score_col = f"{args.score_col}_safety"
    eval_df[raw_score_col] = pd.to_numeric(eval_df[args.score_col], errors="coerce")
    eval_df[safety_score_col] = -eval_df[raw_score_col] if args.higher_is_safer else eval_df[raw_score_col]
    score_num_col = safety_score_col

    n_matched = len(eval_df)
    n_matched_wl = int((eval_df["xtrafloat_list_type"] == "whitelist").sum())
    n_matched_bl = int((eval_df["xtrafloat_list_type"] == "blacklist").sum())
    baseline["matched_n"] = n_matched
    baseline["matched_whitelist_n"] = n_matched_wl
    baseline["matched_blacklist_n"] = n_matched_bl
    print(
        f"\nWhitelist/blacklist agents found in the scoring output: {n_matched:,} "
        f"(whitelist={n_matched_wl:,}/{n_wl:,}={n_matched_wl/max(1,n_wl):.1%}, "
        f"blacklist={n_matched_bl:,}/{n_bl:,}={n_matched_bl/max(1,n_bl):.1%})"
    )
    if args.higher_is_safer:
        print(f"(--higher-is-safer set: {args.score_col} is treated as higher=safer; "
              f"internally negated to '{safety_score_col}' so lower=safer holds throughout this script)")

    # Exclude non-performance blacklist reasons (e.g. "As requested by Director",
    # "Agent active less than 3 months") from the performance evaluation --
    # mirrors run_whitelist_blacklist_eval()'s perf_eval_df exactly, since
    # those blacklist entries aren't a judgment on the model's ability to
    # predict repayment risk.
    if "reason" in eval_df.columns:
        non_perf_mask = (eval_df["is_blacklisted"] == 1) & eval_df["reason"].astype(str).isin(NON_PERF_BLACKLIST_REASONS)
        n_non_perf = int(non_perf_mask.sum())
        eval_df_perf = eval_df[~non_perf_mask].copy()
        if n_non_perf:
            print(f"\nExcluding {n_non_perf:,} blacklisted agents with non-performance reasons "
                  f"({list(NON_PERF_BLACKLIST_REASONS)}) from the performance evaluation below.")
    else:
        eval_df_perf = eval_df

    perf_full = eval_df[[score_num_col, "is_blacklisted"]].dropna()
    perf = eval_df_perf[[score_num_col, "is_blacklisted"]].dropna()
    auc_full = auc_perf = np.nan
    if perf_full.shape[0] > 0 and perf_full["is_blacklisted"].nunique() == 2:
        auc_full = _mann_whitney_auc(perf_full[score_num_col].values, perf_full["is_blacklisted"].values)
        print(f"\nAUC, full labeled set ({args.score_col} vs. is_blacklisted, Mann-Whitney): {auc_full:.4f}")
    if perf.shape[0] > 0 and perf["is_blacklisted"].nunique() == 2:
        auc_perf = _mann_whitney_auc(perf[score_num_col].values, perf["is_blacklisted"].values)
        print(
            f"AUC, performance-filtered: {auc_perf:.4f}\n"
            f"  (0.5 = no discrimination; closer to 1.0 = blacklisted agents correctly score "
            f"riskier than whitelisted ones)"
        )
    else:
        print(f"\nCannot compute AUC -- need both classes present with non-null {args.score_col}.")
    baseline["auc_full"] = None if np.isnan(auc_full) else round(float(auc_full), 4)
    baseline["auc_perf_filtered"] = None if np.isnan(auc_perf) else round(float(auc_perf), 4)

    # -- Decile table: does blacklist rate rise monotonically with score? ----
    decile_records = []
    if perf.shape[0] > 0:
        dec = eval_df_perf.loc[perf.index, [score_num_col, raw_score_col, "is_blacklisted"]].copy()
        dec["score_pct"] = dec[score_num_col].rank(pct=True, method="average")
        dec["decile"] = np.ceil(np.clip(dec["score_pct"], 1e-12, 1.0) * 10).astype(int)
        decile_tbl = (
            dec.groupby("decile")
            .agg(n=("is_blacklisted", "size"), blacklist_rate=("is_blacklisted", "mean"),
                 avg_raw_score=(raw_score_col, "mean"))
            .reset_index()
        )
        print(f"\n=== Decile table ({args.score_col}, decile 1 = safest) ===")
        print(decile_tbl.to_string(index=False))
        decile_records = decile_tbl.round(6).to_dict(orient="records")
    baseline["decile_table"] = decile_records

    # -- risk_tier breakdown, if present -------------------------------------
    if "risk_tier" in eval_df.columns:
        print("\n=== blacklist_rate by risk_tier ===")
        tier_tbl = eval_df.groupby("risk_tier")["is_blacklisted"].agg(n="size", blacklist_rate="mean")
        print(tier_tbl.to_string())
        baseline["blacklist_rate_by_risk_tier"] = tier_tbl.round(6).reset_index().to_dict(orient="records")

    # -- Business-critical red flags ------------------------------------------
    if "assigned_limit" in eval_df.columns:
        bl_with_limit = eval_df[(eval_df["is_blacklisted"] == 1) & (eval_df["assigned_limit"] > 0)]
        wl_zero_limit = eval_df[(eval_df["is_blacklisted"] == 0) & (eval_df["assigned_limit"] == 0)]
        n_bl_with_limit = len(bl_with_limit)
        n_wl_zero_limit = len(wl_zero_limit)
        print(
            f"\n*** RED FLAG: blacklisted agents with assigned_limit > 0: {n_bl_with_limit:,} "
            f"of {n_matched_bl:,} matched blacklisted agents "
            f"({n_bl_with_limit/max(1,n_matched_bl):.1%}) ***"
        )
        print(
            f"Whitelisted agents with assigned_limit == 0: {n_wl_zero_limit:,} "
            f"of {n_matched_wl:,} matched whitelisted agents "
            f"({n_wl_zero_limit/max(1,n_matched_wl):.1%})"
        )
        bl_with_limit.to_csv(f"{args.out_prefix}_blacklisted_with_nonzero_limit.csv", index=False)
        wl_zero_limit.to_csv(f"{args.out_prefix}_whitelisted_with_zero_limit.csv", index=False)
        print(
            f"Written: {args.out_prefix}_blacklisted_with_nonzero_limit.csv, "
            f"{args.out_prefix}_whitelisted_with_zero_limit.csv"
        )
        baseline["blacklisted_with_nonzero_limit_n"] = n_bl_with_limit
        baseline["blacklisted_with_nonzero_limit_pct"] = n_bl_with_limit / max(1, n_matched_bl)
        baseline["whitelisted_with_zero_limit_n"] = n_wl_zero_limit
        baseline["whitelisted_with_zero_limit_pct"] = n_wl_zero_limit / max(1, n_matched_wl)

    # -- Cutoff sweep (reuses the repo's own tested implementation), on the
    #    performance-filtered set -----------------------------------------
    if perf.shape[0] > 0:
        sweep = cutoff_sweep(eval_df_perf, score_col=score_num_col, label_col="is_blacklisted")
        sweep.to_csv(f"{args.out_prefix}_cutoff_sweep.csv", index=False)
        print(f"\nCutoff sweep table written to: {args.out_prefix}_cutoff_sweep.csv")

    eval_df.to_csv(f"{args.out_prefix}_matched_agents.csv", index=False)
    print(f"Full matched whitelist/blacklist eval frame written to: {args.out_prefix}_matched_agents.csv")

    # =========================================================================
    # PART C -- AUC broken down per blacklist reason
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART C: AUC per blacklist reason (whitelist vs. only that reason's blacklisted agents)")
    print("=" * 70)

    reason_rows = []
    if "reason" in eval_df.columns:
        wl_scores = eval_df.loc[eval_df["is_blacklisted"] == 0, score_num_col].dropna()
        reason_counts = eval_df.loc[eval_df["is_blacklisted"] == 1, "reason"].value_counts(dropna=False)
        for reason, n_reason in reason_counts.items():
            if n_reason < args.min_reason_n:
                continue
            bl_scores = eval_df.loc[
                (eval_df["is_blacklisted"] == 1) & (eval_df["reason"] == reason), score_num_col
            ].dropna()
            if len(wl_scores) == 0 or len(bl_scores) == 0:
                continue
            combined_scores = np.concatenate([wl_scores.values, bl_scores.values])
            combined_labels = np.concatenate([np.zeros(len(wl_scores)), np.ones(len(bl_scores))])
            auc_reason = _mann_whitney_auc(combined_scores, combined_labels)
            reason_rows.append({
                "reason": reason,
                "n_blacklisted": int(n_reason),
                "n_matched_with_score": int(len(bl_scores)),
                "auc_vs_whitelist": round(float(auc_reason), 4) if not np.isnan(auc_reason) else None,
            })
        reason_tbl = pd.DataFrame(reason_rows).sort_values("auc_vs_whitelist", ascending=False, na_position="last")
        print(
            f"(only reasons with >= {args.min_reason_n} blacklisted agents shown; "
            f"AUC computed against the full whitelist each time)\n"
        )
        print(reason_tbl.to_string(index=False))
        print(
            "\nRead this as: reasons near 0.5 AUC aren't a credit-risk signal the score can "
            "separate from whitelist at all (commission/activity/admin reasons are expected "
            "here); reasons well above 0.5 are where the score IS discriminating -- if a true "
            "loan-performance reason (e.g. a 'Defaulter...' reason) sits near 0.5 too, that's "
            "the real problem to fix, since that's exactly what cal_pd is meant to predict."
        )
    else:
        print("NOTE: no 'reason' column available -- skipping Part C.")
    baseline["auc_by_blacklist_reason"] = reason_rows

    # -- Write the baseline summary -------------------------------------------
    summary_path = f"{args.out_prefix}_baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(baseline, f, indent=2, default=str)
    print(f"\nBaseline summary written to: {summary_path}")
    print("Keep this file (or rename it, e.g. wl_bl_eval_baseline_summary_PRE_FIX.json) and "
          "re-run this script after your fixes to compare against it directly.")


if __name__ == "__main__":
    main()
