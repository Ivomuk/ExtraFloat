"""
Compare actual ExtraFloat amounts issued to agents (fin_log_cfw_*.csv) against
this pipeline's assigned_limit output (e.g. output/engine_test_output.csv).

Usage:
    python compare_issued_vs_assigned.py ^
        --fin-log-file fin_log_cfw_202607211227.csv ^
        --output-file output/engine_test_output.csv ^
        --out compare_issued_vs_assigned_result.csv

Notes on the comparison:
  - fin_log_cfw_*.csv reflects a real issuance snapshot as of 2026-05-31.
  - The pipeline output reflects the model's recommended limit as of whatever
    --snapshot-date run.bat was last run with -- roughly two months later, if
    unchanged from earlier in this session. A mismatch is informative, not
    automatically wrong: agent risk/activity/commission can genuinely shift
    over two months, and the two numbers answer different questions (what was
    actually given historically vs. what the model would assign today).
  - fin_log may contain more than one issuance row per agent (repeat
    instructions). Rows are summed per msisdn before comparing, and the
    per-agent instruction count is reported so you can sanity-check that
    assumption against the real data.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _norm_msisdn(s: pd.Series) -> pd.Series:
    """Strip whitespace and trailing .0 suffixes -- same normalization used
    elsewhere in this repo (run_credit_risk_pipeline.py's _norm_msisdn), so
    join behavior here matches the rest of the pipeline."""
    out = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return out.mask(out.str.lower().isin({"", "nan", "none", "<na>"}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fin-log-file", default="fin_log_cfw_202607211227.csv")
    ap.add_argument("--output-file", default="output/engine_test_output.csv")
    ap.add_argument(
        "--out",
        default="compare_issued_vs_assigned_result.csv",
        help="Path to write the full merged comparison CSV",
    )
    args = ap.parse_args()

    fin_path = Path(args.fin_log_file)
    out_path = Path(args.output_file)
    if not fin_path.exists():
        sys.exit(f"ERROR: fin-log file not found: {fin_path}")
    if not out_path.exists():
        sys.exit(f"ERROR: output file not found: {out_path}")

    fin = pd.read_csv(fin_path)
    out = pd.read_csv(out_path)

    required_fin_cols = ["instruct_amount", "instruct_amount_to_fro_user_prf", "instruct_to_fro_msisdn"]
    missing_fin = [c for c in required_fin_cols if c not in fin.columns]
    if missing_fin:
        sys.exit(
            f"ERROR: fin-log file is missing expected columns: {missing_fin}\n"
            f"Found columns: {list(fin.columns)}"
        )

    if "msisdn" not in out.columns:
        sys.exit(f"ERROR: output file has no 'msisdn' column. Found columns: {list(out.columns)}")
    if "assigned_limit" not in out.columns:
        sys.exit(f"ERROR: output file has no 'assigned_limit' column. Found columns: {list(out.columns)}")

    fin["_msisdn_norm"] = _norm_msisdn(fin["instruct_to_fro_msisdn"])
    out["_msisdn_norm"] = _norm_msisdn(out["msisdn"])

    n_fin_null = int(fin["_msisdn_norm"].isna().sum())
    n_out_null = int(out["_msisdn_norm"].isna().sum())
    if n_fin_null:
        print(f"WARNING: {n_fin_null} rows in fin-log have an unparseable/blank msisdn -- dropped.")
    if n_out_null:
        print(f"WARNING: {n_out_null} rows in output file have an unparseable/blank msisdn -- dropped.")
    fin = fin[fin["_msisdn_norm"].notna()]
    out = out[out["_msisdn_norm"].notna()]

    # -- Aggregate fin-log to one row per agent (it may carry multiple
    #    issuance instructions per msisdn) ----------------------------------
    n_fin_rows = len(fin)
    fin_agg = (
        fin.groupby("_msisdn_norm")
        .agg(
            issued_amount_total=("instruct_amount", "sum"),
            issued_amount_max=("instruct_amount", "max"),
            issued_instruction_count=("instruct_amount", "size"),
            agent_profile=("instruct_amount_to_fro_user_prf", "first"),
        )
        .reset_index()
    )
    n_agents_multi = int((fin_agg["issued_instruction_count"] > 1).sum())
    print(
        f"fin-log: {n_fin_rows:,} instruction rows -> {len(fin_agg):,} unique agents "
        f"({n_agents_multi:,} agents had more than one instruction row; amounts summed per agent)."
    )

    # -- Dedup output on msisdn (should already be one row per agent; guard anyway) --
    n_out_dupe = int(out["_msisdn_norm"].duplicated().sum())
    if n_out_dupe:
        print(f"WARNING: {n_out_dupe} duplicate msisdn rows in output file -- keeping first occurrence.")
        out = out.drop_duplicates("_msisdn_norm", keep="first")

    out_cols = ["_msisdn_norm", "assigned_limit"]
    for extra in ("risk_tier", "agent_category", "is_thin_file"):
        if extra in out.columns:
            out_cols.append(extra)
    out_small = out[out_cols]

    merged = fin_agg.merge(out_small, on="_msisdn_norm", how="outer", indicator=True)
    merged = merged.rename(columns={"_msisdn_norm": "msisdn"})

    fin_only = merged["_merge"] == "left_only"
    out_only = merged["_merge"] == "right_only"
    both = merged["_merge"] == "both"

    print("\n=== Coverage ===")
    print(f"Agents in fin-log only (issued, not in current scoring output): {int(fin_only.sum()):,}")
    print(f"Agents in output only (scored, no issuance record):             {int(out_only.sum()):,}")
    print(f"Agents in both:                                                 {int(both.sum()):,}")

    cmp_df = merged[both].copy()
    cmp_df["diff"] = cmp_df["assigned_limit"] - cmp_df["issued_amount_total"]
    cmp_df["abs_diff"] = cmp_df["diff"].abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        cmp_df["pct_diff"] = np.where(
            cmp_df["issued_amount_total"] > 0,
            cmp_df["diff"] / cmp_df["issued_amount_total"] * 100.0,
            np.nan,
        )

    print(f"\n=== Comparison, agents present in both files (n={len(cmp_df):,}) ===")
    print(cmp_df[["issued_amount_total", "assigned_limit", "diff", "abs_diff"]].describe().to_string())

    print("\n=== Direction of mismatch ===")
    n_higher = int((cmp_df["assigned_limit"] > cmp_df["issued_amount_total"]).sum())
    n_lower = int((cmp_df["assigned_limit"] < cmp_df["issued_amount_total"]).sum())
    n_equal = int((cmp_df["assigned_limit"] == cmp_df["issued_amount_total"]).sum())
    print(f"assigned_limit > issued_amount_total: {n_higher:,}")
    print(f"assigned_limit < issued_amount_total: {n_lower:,}  <- model would now give LESS than was actually issued")
    print(f"assigned_limit == issued_amount_total: {n_equal:,}")

    n_issued_but_zero_now = int(
        ((cmp_df["issued_amount_total"] > 0) & (cmp_df["assigned_limit"] == 0)).sum()
    )
    print(
        f"\nIssued > 0 historically but assigned_limit == 0 now: {n_issued_but_zero_now:,} "
        f"(agents who received real float but the current model would give them nothing)"
    )

    bands = [-np.inf, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, np.inf]
    labels = [
        "<= -50%", "-50% to -20%", "-20% to -5%", "-5% to +5% (~match)",
        "+5% to +20%", "+20% to +50%", "> +50%",
    ]
    cmp_df["pct_diff_band"] = pd.cut(cmp_df["pct_diff"], bins=bands, labels=labels)
    print("\n=== pct_diff bands ((assigned_limit - issued) / issued) ===")
    print(cmp_df["pct_diff_band"].value_counts().reindex(labels).to_string())

    if "risk_tier" in cmp_df.columns:
        print("\n=== Mean issued vs. assigned, by risk_tier (agents in both files) ===")
        print(
            cmp_df.groupby("risk_tier")[["issued_amount_total", "assigned_limit"]]
            .mean()
            .round(0)
            .to_string()
        )

    if "agent_category" in cmp_df.columns:
        print("\n=== Mean issued vs. assigned, by agent_category (agents in both files) ===")
        print(
            cmp_df.groupby("agent_category")[["issued_amount_total", "assigned_limit"]]
            .mean()
            .round(0)
            .to_string()
        )

    merged.to_csv(args.out, index=False)
    print(f"\nFull merged comparison (all agents, both/left_only/right_only) written to: {args.out}")


if __name__ == "__main__":
    main()
