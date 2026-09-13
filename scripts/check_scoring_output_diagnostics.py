"""
Two read-only diagnostics over a completed run_credit_risk_pipeline.py
output, run in one pass:

1. cal_pd mean/median/count broken down by score_source x thin_file_flag
   (falls back to score_source alone if thin_file_flag/is_thin_file isn't
   present in --scoring-output -- confirmed via run_extrafloat_limit_engine.py's
   FINAL_OUTPUT_COLUMNS + Stage 7's re-attached segmentation columns that
   neither is currently exported, so don't assume it's there).

2. Excluded-agent leakage check: confirms every agent_msisdn in
   --excluded-file (retail_agents_excluded.csv, written by
   apply_retail_agent_filter.py) is genuinely ABSENT from
   --scoring-output's msisdn column -- not present with NaN/zeroed
   fields, not present at all. This is the concrete verification that
   the retail-agent pre-filter actually kept non-retail agents out of
   scoring, not just out of the training population.

Usage:
    python scripts\\check_scoring_output_diagnostics.py ^
        --scoring-output output\\engine_test_output.csv ^
        --excluded-file retail_agents_excluded.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def _norm_msisdn(s: pd.Series) -> pd.Series:
    """Byte-identical to run_credit_risk_pipeline.py's _norm_msisdn --
    the normalizer that actually drives the pipeline's own join keys.
    Kept in sync deliberately; see filter_borrower_file_by_retail_agents.py
    for the same duplication and why."""
    out = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return out.mask(out.str.lower().isin({"", "nan", "none", "<na>"}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scoring-output", default="output/engine_test_output.csv")
    ap.add_argument("--excluded-file", default="retail_agents_excluded.csv")
    ap.add_argument("--excluded-msisdn-col", default="agent_msisdn")
    args = ap.parse_args()

    out_path = Path(args.scoring_output)
    excl_path = Path(args.excluded_file)
    if not out_path.exists():
        sys.exit(f"ERROR: scoring output not found: {out_path}")
    if not excl_path.exists():
        sys.exit(f"ERROR: excluded-agents file not found: {excl_path}")

    df = pd.read_csv(out_path, sep=",", encoding="utf-8-sig")
    if "msisdn" not in df.columns:
        sys.exit(f"ERROR: 'msisdn' column not found in {out_path}. Found: {list(df.columns)}")
    if "cal_pd" not in df.columns:
        sys.exit(f"ERROR: 'cal_pd' column not found in {out_path}. Found: {list(df.columns)}")

    # ------------------------------------------------------------------ #
    # 1) cal_pd by score_source x thin_file_flag
    # ------------------------------------------------------------------ #
    print("=" * 78)
    print("1) cal_pd breakdown")
    print("=" * 78)

    if "score_source" not in df.columns:
        print("NOTE: 'score_source' column not found -- skipping this breakdown "
              f"entirely. Columns present: {list(df.columns)}")
    else:
        thin_col = next((c for c in ("thin_file_flag", "is_thin_file") if c in df.columns), None)
        group_cols = ["score_source"] + ([thin_col] if thin_col else [])
        if thin_col is None:
            print("NOTE: neither 'thin_file_flag' nor 'is_thin_file' is present in "
                  f"{out_path} -- run_extrafloat_limit_engine.py's FINAL_OUTPUT_COLUMNS "
                  "and Stage 7's re-attached segmentation columns don't currently "
                  "export it, so this breaks down by score_source alone. Add it to "
                  "Stage 7's re-attach list in run_credit_risk_pipeline.py if you want "
                  "this dimension in future runs.\n")

        cal_pd = pd.to_numeric(df["cal_pd"], errors="coerce")
        summary = (
            df.assign(_cal_pd=cal_pd)
            .groupby(group_cols, dropna=False)["_cal_pd"]
            .agg(n="count", mean="mean", median="median", min="min", max="max")
            .reset_index()
            .sort_values(group_cols)
        )
        print(summary.to_string(index=False))

        overall = cal_pd.agg(["count", "mean", "median", "min", "max"])
        print(f"\nOverall (all {int(overall['count']):,} agents): "
              f"mean={overall['mean']:.4f} median={overall['median']:.4f} "
              f"min={overall['min']:.4f} max={overall['max']:.4f}")

    # ------------------------------------------------------------------ #
    # 2) Excluded-agent leakage check
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 78)
    print("2) Excluded-agent leakage check")
    print("=" * 78)

    excl_df = pd.read_csv(excl_path, sep=",", encoding="utf-8-sig")
    if args.excluded_msisdn_col not in excl_df.columns:
        sys.exit(
            f"ERROR: '{args.excluded_msisdn_col}' column not found in {excl_path}. "
            f"Found: {list(excl_df.columns)[:30]}"
        )
    excluded_norm = _norm_msisdn(excl_df[args.excluded_msisdn_col]).dropna()
    excluded_set = set(excluded_norm)
    n_excluded = len(excluded_set)

    output_norm = _norm_msisdn(df["msisdn"]).dropna()
    output_set = set(output_norm)

    leaked = excluded_set & output_set
    n_leaked = len(leaked)

    print(f"Excluded agents (allowlist-out): {n_excluded:,} unique msisdns (from {excl_path})")
    print(f"Scoring output:                  {len(output_set):,} unique msisdns (from {out_path})")
    print(f"Excluded agents found in scoring output: {n_leaked:,}")

    if n_leaked > 0:
        sample = sorted(leaked)[:10]
        print(f"\nFAIL: {n_leaked} excluded agent(s) leaked into scoring output. "
              f"Sample msisdns: {sample}")
        sys.exit(2)

    print("\nPASS: no excluded agent's msisdn appears in the scoring output at all.")


if __name__ == "__main__":
    main()
