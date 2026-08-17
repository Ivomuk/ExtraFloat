"""
For agents currently classified agent_category == "Below Threshold" (July
scoring run) who WERE previously issued real ExtraFloat (per the fin_log_cfw
extract, snapshot as of 2026-05-31): pull their 1m/3m/6m commission trend from
the same transaction-file extract used for scoring, to distinguish two
hypotheses --

  (a) "Model regression": these agents were already low/no-commission earners
      throughout the window that includes May (i.e. this isn't new -- the old
      process extended them float anyway, and the new commission-tier gate is
      just newly enforcing a rule that would have applied in May too).
  (b) "Genuine drift": these agents had real commission activity earlier
      (including in/around May) and have since gone quiet by July -- in which
      case "Below Threshold" today is an accurate, and arguably correct,
      reflection of a real behavior change, not evidence against the new
      model.

Important limitation: the transaction-file extract is a single snapshot as of
2026-07-31, with TRAILING windows measured backward from that date (1m ~=
July, 3m ~= May-July, 6m ~= Feb-July). There is no standalone "as of
2026-05-31" mart snapshot to check directly -- this script approximates by
comparing the recent (1m) window against the longer trailing windows that
partially overlap May. It cannot fully separate "earned nothing in May
specifically" from "earned something in May but it's buried inside a 3-6
month sum" -- treat the classification bands below as a lean, not proof.

Usage:
    python check_below_threshold_commission_trend.py ^
        --fin-log-file data/fin_log_cfw_202607211227.csv ^
        --output-file output/engine_test_output.csv ^
        --transaction-file data/mfs_daily_agent_mart_20260731.csv

Requires --keep-intermediate to have been used for the run.bat run that
produced --output-file, since agent_category is an intermediate column.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _norm_msisdn(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return out.mask(out.str.lower().isin({"", "nan", "none", "<na>"}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fin-log-file", default="data/fin_log_cfw_202607211227.csv")
    ap.add_argument("--output-file", default="output/engine_test_output.csv")
    ap.add_argument("--transaction-file", default="data/mfs_daily_agent_mart_20260731.csv")
    ap.add_argument("--out", default="below_threshold_commission_trend.csv")
    args = ap.parse_args()

    fin_path = Path(args.fin_log_file)
    out_path = Path(args.output_file)
    txn_path = Path(args.transaction_file)
    for p, label in [(fin_path, "fin-log"), (out_path, "output"), (txn_path, "transaction")]:
        if not p.exists():
            sys.exit(f"ERROR: {label} file not found: {p}")

    # -- fin-log: sum instruct_amount per agent -------------------------------
    fin = pd.read_csv(fin_path)
    required_fin_cols = ["instruct_amount", "instruct_to_fro_msisdn"]
    missing_fin = [c for c in required_fin_cols if c not in fin.columns]
    if missing_fin:
        sys.exit(f"ERROR: fin-log file missing columns: {missing_fin}\nFound: {list(fin.columns)}")
    fin["_msisdn_norm"] = _norm_msisdn(fin["instruct_to_fro_msisdn"])
    fin = fin[fin["_msisdn_norm"].notna()]
    fin_agg = fin.groupby("_msisdn_norm").agg(issued_amount_total=("instruct_amount", "sum")).reset_index()

    # -- scoring output: need agent_category ----------------------------------
    out = pd.read_csv(out_path)
    if "msisdn" not in out.columns:
        sys.exit(f"ERROR: output file has no 'msisdn' column. Found: {list(out.columns)}")
    if "agent_category" not in out.columns:
        sys.exit(
            "ERROR: output file has no 'agent_category' column -- re-run run.bat "
            "with --keep-intermediate first, then re-run this script."
        )
    out["_msisdn_norm"] = _norm_msisdn(out["msisdn"])
    out = out[out["_msisdn_norm"].notna()]
    out_cols = ["_msisdn_norm", "agent_category", "assigned_limit"]
    for extra in ("risk_tier", "is_thin_file"):
        if extra in out.columns:
            out_cols.append(extra)
    out_small = out[out_cols].drop_duplicates("_msisdn_norm")

    # -- transaction file: recompute commission per window from parts ---------
    txn = pd.read_csv(txn_path, sep=",")
    if "agent_msisdn" not in txn.columns:
        sys.exit(f"ERROR: transaction file has no 'agent_msisdn' column. Found: {list(txn.columns)[:20]}")
    txn["_msisdn_norm"] = _norm_msisdn(txn["agent_msisdn"])
    txn = txn[txn["_msisdn_norm"].notna()]

    windows = ["1m", "3m", "6m"]
    comm_parts = ["cash_out_comm", "cash_in_comm", "voucher_comm", "payment_comm"]
    for w in windows:
        cols = [f"{p}_{w}" for p in comm_parts]
        missing = [c for c in cols if c not in txn.columns]
        if missing:
            sys.exit(f"ERROR: transaction file missing commission columns for {w}: {missing}")
        txn[f"commission_{w}"] = sum(pd.to_numeric(txn[c], errors="coerce").fillna(0) for c in cols)

    txn_cols = ["_msisdn_norm", "commission_1m", "commission_3m", "commission_6m"]
    if "commission" in txn.columns:
        # Raw pre-computed field the engine actually reads for agent_category
        # tiering (prepare_transaction_capacity_features() -- see
        # extrafloat_limit_engine_features.py). Included as a cross-check
        # against the recomputed commission_6m above.
        txn["commission_raw"] = pd.to_numeric(txn["commission"], errors="coerce").fillna(0)
        txn_cols.append("commission_raw")
    txn_small = txn[txn_cols].drop_duplicates("_msisdn_norm")

    # -- Join all three on normalized msisdn -----------------------------------
    merged = fin_agg.merge(out_small, on="_msisdn_norm", how="inner").merge(
        txn_small, on="_msisdn_norm", how="inner"
    )

    group = merged[
        (merged["agent_category"] == "Below Threshold") & (merged["issued_amount_total"] > 0)
    ].copy()

    print(f"Agents: Below Threshold (current) AND previously issued float (fin-log) "
          f"AND present in transaction file: {len(group):,}\n")

    if len(group) == 0:
        print("No agents matched all three conditions -- nothing further to report.")
        return

    desc_cols = ["issued_amount_total", "commission_1m", "commission_3m", "commission_6m"]
    if "commission_raw" in group.columns:
        desc_cols.append("commission_raw")
    print("=== Distribution within this group ===")
    print(group[desc_cols].describe().to_string())

    # -- Lean classification (see module docstring for the caveat) ------------
    # avg_monthly_rate_3m: what their monthly commission rate looked like over
    # the May-July window, as a whole. Comparing it to the most recent 1-month
    # window gives a rough "did the rate drop off by July" signal.
    group["avg_monthly_rate_3m"] = group["commission_3m"] / 3.0
    group["avg_monthly_rate_6m"] = group["commission_6m"] / 6.0

    # Reference point from earlier analysis this session: median 1-month
    # commission across the full agent population was ~6,801 UGX. Used here
    # only as a rough "meaningfully active" bar, not a hard cutoff.
    active_floor = 6_801.0

    # Mutually exclusive, checked in this priority order: a real 3m rate that
    # has since dropped off is the more specific/informative signal, so it's
    # checked first and consistently_low only claims what's left over.
    recent_decline = (group["avg_monthly_rate_3m"] > active_floor) & (group["commission_1m"] <= active_floor * 0.5)
    consistently_low = (
        ~recent_decline
        & (group["avg_monthly_rate_6m"] <= active_floor)
        & (group["commission_1m"] <= active_floor)
    )
    other = ~consistently_low & ~recent_decline

    print(f"\n=== Lean classification (reference floor = {active_floor:,.0f} UGX/month, "
          f"the population median 1-month commission from earlier analysis) ===")
    print(f"Consistently low/no commission across the whole 6m window (leans: model regression, "
          f"not new): {int(consistently_low.sum()):,} ({consistently_low.mean():.1%})")
    print(f"Recent drop-off -- meaningful 3m rate but 1m has fallen well below it "
          f"(leans: genuine drift): {int(recent_decline.sum()):,} ({recent_decline.mean():.1%})")
    print(f"Other / ambiguous: {int(other.sum()):,} ({other.mean():.1%})")

    group.to_csv(args.out, index=False)
    print(f"\nFull group written to: {args.out}")


if __name__ == "__main__":
    main()
