#!/usr/bin/env python3
"""
Thin-file threshold validation: slice XGBoost AUC and bad-rate by
distinct_loan_months x total_loans_6m to confirm whether the production
thresholds (3 months, 10 loans) are appropriate or should be adjusted.

Two outputs:
  1. AUC table for the current THICK-FILE population (sliced by months/loans)
     If AUC is stable at months=3 vs months=6, the threshold is defensible.
     If AUC collapses near the boundary, raise the threshold.

  2. Bad-rate table for BOUNDARY thin-file agents (just below threshold)
     Shows whether agents near the boundary have similar risk profiles to
     thick-file agents — if so, lowering the threshold is safe.

Usage (Windows):
    python scripts/validate_thinfile_thresholds.py ^
        --val-file data/mfs_daily_agent_mart_20251115.csv ^
        --repayment-file data/snapshots_202601112144.csv ^
        --ops-scored pd_model/artifacts/ops_scored.csv ^
        --val-snapshot-date 20251115
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    if mask.sum() < 20 or len(np.unique(y_true[mask])) < 2:
        return np.nan
    return float(roc_auc_score(y_true[mask], y_score[mask]))


def _fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if not np.isnan(v) else "  —  "


def _run_phase22(val_file, repayment_file, snapshot_date):
    from pd_model.preprocessing.loan_features import run_phase_2_2_repayment_pd_features
    from pd_model.config.model_config import DEFAULT_CONFIG

    print(f"Loading val file:        {val_file}")
    df_val = pd.read_csv(val_file, low_memory=False)
    df_val["snapshot_dt"] = pd.to_datetime(str(snapshot_date), format="%Y%m%d")

    print(f"Loading repayment file:  {repayment_file}")
    df_rep = pd.read_csv(repayment_file, low_memory=False)
    if "snapshot_dt" not in df_rep.columns:
        df_rep["snapshot_dt"] = pd.to_datetime(str(snapshot_date), format="%Y%m%d")

    print("Running Phase 2.2 (loan features)...")
    df_out, _ = run_phase_2_2_repayment_pd_features(df_val, df_rep, cfg=DEFAULT_CONFIG, verbose=False)
    return df_out


# --------------------------------------------------------------------------- #
# Analysis 1: thick-file AUC slice
# --------------------------------------------------------------------------- #

MONTH_BINS = [(3, 3), (4, 4), (5, 5), (6, 99)]
LOAN_BINS  = [(10, 14), (15, 19), (20, 29), (30, 9999)]


def _thick_auc_table(df, score_col):
    rows = []
    for m_lo, m_hi in MONTH_BINS:
        m_label = str(m_lo) if m_lo == m_hi else f"{m_lo}+"
        row = {"distinct_loan_months": m_label}
        for l_lo, l_hi in LOAN_BINS:
            l_label = f"{l_lo}-{l_hi}" if l_hi < 9999 else f"{l_lo}+"
            mask = (
                df["distinct_loan_months"].between(m_lo, m_hi)
                & df["total_loans_6m"].between(l_lo, l_hi)
            )
            sub = df[mask]
            n = len(sub)
            if n < 20:
                row[l_label] = f"n={n}"
            else:
                auc = _auc(sub["bad_state"], sub[score_col])
                row[l_label] = f"{_fmt(auc)} (n={n:,})"
        rows.append(row)

    print("\n╔══ AUC by distinct_loan_months × total_loans_6m (THICK-FILE val population) ══╗\n")
    print("  Columns = total_loans_6m buckets\n")
    tbl = pd.DataFrame(rows).set_index("distinct_loan_months")
    print(tbl.to_string())

    # Marginal by months
    print("\n── Marginal AUC by distinct_loan_months ──\n")
    marg = []
    for m_lo, m_hi in MONTH_BINS:
        m_label = str(m_lo) if m_lo == m_hi else f"{m_lo}+"
        mask = df["distinct_loan_months"].between(m_lo, m_hi)
        sub = df[mask]
        if len(sub) == 0:
            continue
        marg.append({
            "months": m_label,
            "n": len(sub),
            "bad_rate_%": f"{sub['bad_state'].mean()*100:.2f}",
            "AUC": _fmt(_auc(sub["bad_state"], sub[score_col])),
        })
    print(pd.DataFrame(marg).to_string(index=False))

    # Marginal by loans
    print("\n── Marginal AUC by total_loans_6m ──\n")
    marg2 = []
    for l_lo, l_hi in LOAN_BINS:
        l_label = f"{l_lo}-{l_hi}" if l_hi < 9999 else f"{l_lo}+"
        mask = df["total_loans_6m"].between(l_lo, l_hi)
        sub = df[mask]
        if len(sub) == 0:
            continue
        marg2.append({
            "total_loans_6m": l_label,
            "n": len(sub),
            "bad_rate_%": f"{sub['bad_state'].mean()*100:.2f}",
            "AUC": _fmt(_auc(sub["bad_state"], sub[score_col])),
        })
    print(pd.DataFrame(marg2).to_string(index=False))


# --------------------------------------------------------------------------- #
# Analysis 2: boundary thin-file bad-rate
# --------------------------------------------------------------------------- #

def _boundary_table(df_all):
    thick_mask = df_all["distinct_loan_months"].ge(3) & df_all["total_loans_6m"].ge(10)
    thick_br = df_all.loc[thick_mask, "bad_state"].mean()

    segments = [
        ("thick-file (months>=3 AND loans>=10) [REFERENCE]",
         thick_mask),
        ("months>=3 AND loans in [5,9]  — fails loans only",
         df_all["distinct_loan_months"].ge(3) & df_all["total_loans_6m"].between(5, 9)),
        ("months=2 AND loans>=10        — fails months only",
         df_all["distinct_loan_months"].eq(2) & df_all["total_loans_6m"].ge(10)),
        ("months=2 AND loans in [5,9]   — fails both (near)",
         df_all["distinct_loan_months"].eq(2) & df_all["total_loans_6m"].between(5, 9)),
        ("months=1 AND loans>=10        — far from boundary",
         df_all["distinct_loan_months"].eq(1) & df_all["total_loans_6m"].ge(10)),
        ("months>=3 AND loans in [1,4]  — very low depth",
         df_all["distinct_loan_months"].ge(3) & df_all["total_loans_6m"].between(1, 4)),
        ("no loan history (months=0)",
         df_all["distinct_loan_months"].eq(0)),
    ]

    rows = []
    for label, mask in segments:
        sub = df_all[mask]
        n = len(sub)
        if n == 0:
            continue
        br = sub["bad_state"].mean() if sub["bad_state"].notna().any() else np.nan
        diff = (br - thick_br) * 100 if not np.isnan(br) else np.nan
        rows.append({
            "segment": label,
            "n": f"{n:,}",
            "bad_rate_%": f"{br*100:.2f}" if not np.isnan(br) else "—",
            "diff_vs_thick_pp": f"{diff:+.2f}" if not np.isnan(diff) else "—",
        })

    print("\n╔══ Bad-rate for BOUNDARY thin-file agents (vs thick-file reference) ══╗\n")
    print("  Positive diff = riskier than thick-file; negative = safer.\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        "\nInterpretation guide:"
        "\n  diff ≈ 0pp  → safe to lower the threshold for that segment"
        "\n  diff > 3pp  → keep or raise the threshold"
        "\n  diff > 8pp  → segment is materially riskier; threshold is protecting well"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Validate thin-file routing thresholds")
    ap.add_argument("--val-file", required=True, help="Val snapshot CSV (same file used in training)")
    ap.add_argument("--repayment-file", required=True, help="Repayment snapshot CSV")
    ap.add_argument("--ops-scored", required=True, help="pd_model/artifacts/ops_scored.csv from training run")
    ap.add_argument("--val-snapshot-date", default="20251115", help="Snapshot date YYYYMMDD")
    args = ap.parse_args()

    # -- Phase 2.2: get distinct_loan_months + total_loans_6m for ALL val agents
    df_all = _run_phase22(args.val_file, args.repayment_file, args.val_snapshot_date)
    print(f"\nPhase 2.2 complete: {len(df_all):,} agents")
    print(f"  distinct_loan_months present: {'distinct_loan_months' in df_all.columns}")
    print(f"  total_loans_6m present:       {'total_loans_6m' in df_all.columns}")

    if "distinct_loan_months" not in df_all.columns or "total_loans_6m" not in df_all.columns:
        print("\nERROR: distinct_loan_months / total_loans_6m not produced by Phase 2.2.")
        print("Ensure the repayment file contains disbursement_vol_m1 ... disbursement_vol_m6 columns.")
        sys.exit(1)

    # -- Boundary analysis: defer until after ops_scored is loaded so we can use
    #    the trained bad_state label rather than the raw Phase 2.2 label.
    #    (Raw bad_state_30D in the val file can differ from the cleaned label used
    #    in training, producing implausible bad rates like 70%.)
    pass  # boundary table built below after ops_scored is loaded

    # -- Thick-file AUC slice uses ops_scored (thick-file agents with XGBoost scores)
    ops_path = Path(args.ops_scored)
    if not ops_path.exists():
        print(f"\nops_scored file not found at {ops_path} — skipping AUC slice.")
        print("Run training with output directed to an artifacts dir that includes ops_scored.csv.")
        return

    print(f"\nLoading ops_scored: {ops_path}")
    ops = pd.read_csv(ops_path, low_memory=False)
    print(f"  ops_scored shape: {ops.shape}")

    # Identify columns
    id_col = next((c for c in ["agent_msisdn", "msisdn"] if c in ops.columns), None)
    score_col = next((c for c in ["cal_pd", "model_score", "raw_score"] if c in ops.columns), None)
    thin_col = next((c for c in ["thin_file_flag", "thin_file"] if c in ops.columns), None)

    if id_col is None or score_col is None:
        print(f"ops_scored columns: {ops.columns.tolist()}")
        print("ERROR: cannot find agent ID or score column in ops_scored.")
        sys.exit(1)

    # Filter to thick-file only
    if thin_col:
        ops_thick = ops[ops[thin_col].eq(0)].copy()
        print(f"  Thick-file agents in ops_scored: {len(ops_thick):,} / {len(ops):,}")
    else:
        ops_thick = ops.copy()
        print("  thin_file_flag not found — treating all ops_scored rows as thick-file")

    # Merge in distinct_loan_months / total_loans_6m
    phase22_id = next((c for c in ["agent_msisdn", "msisdn"] if c in df_all.columns), None)
    join_cols = [phase22_id, "distinct_loan_months", "total_loans_6m"]
    join_cols = [c for c in join_cols if c in df_all.columns]

    # ops_scored agent_msisdn are SHA-256 prefixes (16 hex chars) written by
    # run_pipeline.py for PII compliance. Hash Phase 2.2 raw MSISDNs to match.
    def _sha256_16(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).apply(
            lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]
        )

    df_merge = df_all[join_cols].copy()
    df_merge[phase22_id] = _sha256_16(df_merge[phase22_id])
    ops_thick = ops_thick.copy()
    ops_thick[id_col] = ops_thick[id_col].astype(str).str.strip()

    # Diagnostic: show sample IDs from each side
    sample_ops = ops_thick[id_col].dropna().unique()[:3].tolist()
    sample_p22 = df_merge[phase22_id].dropna().unique()[:3].tolist()
    print(f"  ID samples — ops_scored (hashed): {sample_ops}")
    print(f"  ID samples — phase22    (hashed): {sample_p22}")

    ops_thick = ops_thick.merge(
        df_merge.rename(columns={phase22_id: id_col}),
        on=id_col,
        how="left",
    )

    matched = ops_thick["distinct_loan_months"].notna().sum()
    total = len(ops_thick)
    print(f"  Merge result: {matched:,}/{total:,} thick-file agents matched ({100*matched/max(total,1):.1f}%)")
    ops_thick = ops_thick.dropna(subset=["distinct_loan_months", "total_loans_6m"])

    if "bad_state" not in ops_thick.columns:
        print("ERROR: bad_state column not found in ops_scored — cannot compute AUC.")
        sys.exit(1)

    # -- Boundary table: build from ops_scored bad_state + Phase 2.2 month/loan columns
    # Merge ALL ops_scored agents (thin + thick) with Phase 2.2 month/loan columns.
    ops_all = ops.copy()
    ops_all[id_col] = ops_all[id_col].astype(str).str.strip()
    ops_all = ops_all.merge(
        df_merge.rename(columns={phase22_id: id_col}),
        on=id_col,
        how="left",
    )
    ops_all = ops_all.dropna(subset=["distinct_loan_months", "total_loans_6m", "bad_state"])
    print(f"\n  Boundary table will use {len(ops_all):,} ops_scored agents with matched month/loan data")
    _boundary_table(ops_all)

    _thick_auc_table(ops_thick, score_col)

    print("\n── Done ──\n")
    print("How to read the results:")
    print("  • AUC stable across month/loan buckets → thresholds are appropriate")
    print("  • AUC drops sharply at months=3 → consider raising to 4")
    print("  • AUC drops sharply at loans<15  → consider raising to 12-15")
    print("  • Boundary bad_rate ≈ thick-file  → safe to lower that threshold")


if __name__ == "__main__":
    main()
