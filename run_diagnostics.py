"""
Leakage & AUC diagnostic script.

Reruns pipeline data-prep (steps 1-7 only — no model training) to obtain
the real X_train / y_train feature matrix, then runs three diagnostic checks:

  Check 1 — Single-feature AUC: does any feature alone reach > 0.90 AUC?
  Check 2 — Spearman/Pearson correlation with label: any feature with |r| > 0.80?
  Check 3 — Business-process co-definition: bad rate by outstanding-debt status

Usage (Windows CMD, same args as run_pipeline.py):
    python run_diagnostics.py ^
        --train-file    data\snapshot_20250930.csv ^
        --val-file      data\snapshot_20251115.csv ^
        --repayment-file data\repayments.csv ^
        --train-snapshot-date 20250930 ^
        --val-snapshot-date   20251115 ^
        --train-cutoff  2025-10-31 ^
        --output-dir    artifacts

PowerShell:
    python run_diagnostics.py `
        --train-file    data/snapshot_20250930.csv `
        --val-file      data/snapshot_20251115.csv `
        --repayment-file data/repayments.csv `
        --train-snapshot-date 20250930 `
        --val-snapshot-date   20251115 `
        --train-cutoff  2025-10-31 `
        --output-dir    artifacts
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from pd_model.config import feature_config
from pd_model.config.model_config import DEFAULT_CONFIG
from pd_model.data_prep.pipeline_data import prepare_pd_training_and_validation_data
from pd_model.logging_config import get_logger
from pd_model.modeling.scorecard import add_never_loan_scorecard_from_phase_2_1
from pd_model.preprocessing.loan_features import (
    add_thin_file_flags,
    classify_agent_loan_status,
    leakage_audit_phase_2_2,
    run_phase_2_2_repayment_pd_features,
)
from pd_model.preprocessing.transaction_features import run_phase_2_1_richer_tx_behaviour
from pd_model.preprocessing.transformations import (
    build_transformed_dataframe,
    get_and_classify_pd_features,
)
from pd_model.scoring.iv_selector import iv_filter_phase_2
from pd_model.tests.test_single_feature_auc import single_feature_aucs
from pd_model.tests.test_feature_target_correlation import feature_label_correlations
from pd_model.tests.test_business_process_codef import business_process_codef_report

logger = get_logger(__name__)

DIVIDER = "=" * 70


def _load_snapshot(path: str, snapshot_date_int: int, split_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["snapshot_dt"] = snapshot_date_int
    df["split"] = split_label
    df["tbl_dt"] = pd.to_numeric(df.get("tbl_dt", snapshot_date_int), errors="coerce")
    logger.info("Loaded %s: %d rows, %d cols", path, len(df), df.shape[1])
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "tbl_dt" in df.columns:
        df["tbl_dt"] = pd.to_datetime(df["tbl_dt"].astype(str), errors="coerce")
    if "snapshot_dt" in df.columns:
        df["snapshot_dt"] = pd.to_datetime(df["snapshot_dt"].astype(str), errors="coerce")
    if "activation_dt" in df.columns:
        df["activation_dt"] = df["activation_dt"].astype(str).str.split(".").str[0].replace("", pd.NA)
        df["activation_dt"] = pd.to_datetime(df["activation_dt"], format="%Y%m%d", errors="coerce")
    for c in ["date_of_birth", "payment_last", "cash_in_last", "cash_out_last"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str), errors="coerce")
    return df


def run_diagnostics(args: argparse.Namespace) -> None:
    cfg = DEFAULT_CONFIG

    # ------------------------------------------------------------------ #
    # Steps 1-3: Load data, run feature engineering, join labels
    # ------------------------------------------------------------------ #
    print(f"\n{DIVIDER}")
    print("STEP 1-3: Loading data and building features ...")
    print(DIVIDER)

    train_df = _load_snapshot(args.train_file, args.train_snapshot_date, "train")
    val_df   = _load_snapshot(args.val_file,   args.val_snapshot_date,   "validation")

    df_pd = pd.concat([train_df, val_df], ignore_index=True, sort=False)
    df_pd = _parse_dates(df_pd)

    df_pd = run_phase_2_1_richer_tx_behaviour(df_pd, cfg=cfg)

    df_repayments = pd.read_csv(args.repayment_file)
    logger.info("Repayments loaded: %d rows", len(df_repayments))

    df_pd, _ = run_phase_2_2_repayment_pd_features(df_pd, df_repayments, cfg=cfg, verbose=True)
    df_pd = classify_agent_loan_status(df_pd)
    df_pd = add_thin_file_flags(df_pd, cfg=cfg)

    leakage_audit_phase_2_2(
        df_pd, hard_fail=True, cfg=cfg,
        pd_feature_blacklist=feature_config.PD_FEATURE_BLACKLIST,
    )

    df_pd = add_never_loan_scorecard_from_phase_2_1(df_pd, cfg=cfg)

    # ------------------------------------------------------------------ #
    # Steps 4-5: Transform features (fit on train only)
    # ------------------------------------------------------------------ #
    train_cutoff = pd.Timestamp(args.train_cutoff)
    split_mask   = pd.to_datetime(df_pd["snapshot_dt"], errors="coerce") <= train_cutoff
    df_train_raw = df_pd[split_mask].copy()
    df_val_raw   = df_pd[~split_mask].copy()

    pd_features, log_cols, cap_cols, _, signed_log_cols, _ = get_and_classify_pd_features(df_train_raw)

    df_train_trans, _, transform_report = build_transformed_dataframe(
        df_train_raw, pd_features=pd_features, log_cols=log_cols,
        cap_cols=cap_cols, signed_log_cols=signed_log_cols, cfg=cfg,
    )
    fitted_params = {
        row["feature"]: {
            "action": row["action"],
            "lo": row.get("lo") if not pd.isna(row.get("lo", float("nan"))) else None,
            "hi": row.get("hi") if not pd.isna(row.get("hi", float("nan"))) else None,
        }
        for _, row in transform_report.iterrows()
        if pd.notna(row.get("action"))
    }
    df_val_trans, _, _ = build_transformed_dataframe(
        df_val_raw, pd_features=pd_features, log_cols=log_cols,
        cap_cols=cap_cols, signed_log_cols=signed_log_cols, cfg=cfg,
        fitted_params=fitted_params,
    )
    val_cols = [c for c in df_train_trans.columns if c in df_val_trans.columns]
    df_val_trans = df_val_trans[val_cols]

    df_pd_raw        = pd.concat([df_train_raw, df_val_raw],   ignore_index=True)
    df_pd_transformed = pd.concat([df_train_trans, df_val_trans], ignore_index=True)

    # ------------------------------------------------------------------ #
    # Step 6: Train/val split + IV feature selection
    # ------------------------------------------------------------------ #
    (
        X_train_raw, X_train_trans, y_train,
        X_val_raw,   X_val_trans,   y_val,
        candidate_features, *_
    ) = prepare_pd_training_and_validation_data(
        df_pd_raw=df_pd_raw,
        df_pd_transformed=df_pd_transformed,
        target_col=feature_config.TARGET_COL,
        train_cutoff=train_cutoff,
        id_cols=[feature_config.AGENT_KEY],
        protected_cols=list(feature_config.PD_FEATURE_BLACKLIST),
        pd_feature_blacklist=feature_config.PD_FEATURE_BLACKLIST,
        forbidden_feature_patterns=feature_config.LEAKAGE_PATTERNS,
        date_cols=feature_config.DATE_COLS,
    )

    selected_features, _ = iv_filter_phase_2(
        X_train_raw=X_train_raw,
        X_train_transformed=X_train_trans,
        y_train=y_train,
        cfg=cfg,
        pd_feature_blacklist=feature_config.PD_FEATURE_BLACKLIST,
        forbidden_feature_patterns=feature_config.LEAKAGE_PATTERNS,
        target_col=feature_config.TARGET_COL,
    )

    X_train = X_train_trans[selected_features].copy()
    X_val   = X_val_trans[[c for c in selected_features if c in X_val_trans.columns]].copy()

    print(f"\nTraining matrix: {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
    print(f"Bad rate (train): {y_train.mean():.2%}  |  Bad rate (val): {y_val.mean():.2%}")

    # ------------------------------------------------------------------ #
    # CHECK 1: Single-feature AUC
    # ------------------------------------------------------------------ #
    print(f"\n{DIVIDER}")
    print("CHECK 1: Single-feature AUC  (ALERT threshold > 0.90)")
    print(DIVIDER)
    auc_report = single_feature_aucs(X_train, y_train)
    print(auc_report.head(15).to_string(index=False))

    alert_feats = auc_report[auc_report["flag"] == "ALERT"]
    warn_feats  = auc_report[auc_report["flag"] == "WARN"]
    print(f"\n  ALERT features (AUC > 0.90): {len(alert_feats)}")
    print(f"  WARN  features (AUC > 0.85): {len(warn_feats)}")
    if not alert_feats.empty:
        print("\n  *** ALERT — investigate these features for leakage or co-definition: ***")
        print(alert_feats[["feature", "auc"]].to_string(index=False))

    # ------------------------------------------------------------------ #
    # CHECK 2: Feature-label Spearman / Pearson correlation
    # ------------------------------------------------------------------ #
    print(f"\n{DIVIDER}")
    print("CHECK 2: Feature-label correlation  (ALERT threshold |r| > 0.80)")
    print(DIVIDER)
    corr_report = feature_label_correlations(X_train, y_train)
    print(corr_report.head(15).to_string(index=False))

    alert_corr = corr_report[corr_report["flag"] == "ALERT"]
    print(f"\n  ALERT features (|r| > 0.80): {len(alert_corr)}")
    if not alert_corr.empty:
        print("\n  *** ALERT — high correlation with label: ***")
        print(alert_corr[["feature", "spearman_r", "pearson_r"]].to_string(index=False))

    # ------------------------------------------------------------------ #
    # CHECK 3: Business-process co-definition
    # ------------------------------------------------------------------ #
    print(f"\n{DIVIDER}")
    print("CHECK 3: Business-process co-definition (net_exposure_6M vs bad_state)")
    print(DIVIDER)

    codef_df = df_train_raw[
        [c for c in ["net_exposure_6M", "currently_outstanding_flag", "bad_state"]
         if c in df_train_raw.columns]
    ].copy()

    if "bad_state" not in codef_df.columns and feature_config.TARGET_COL in df_train_raw.columns:
        codef_df["bad_state"] = df_train_raw[feature_config.TARGET_COL].values

    if "bad_state" in codef_df.columns and "net_exposure_6M" in codef_df.columns:
        codef_report = business_process_codef_report(
            codef_df,
            outstanding_col="currently_outstanding_flag",
            net_exposure_col="net_exposure_6M",
            label_col="bad_state",
        )
        print(codef_report.to_string(index=False))
        outstanding_row = codef_report[codef_report["segment"] == "outstanding"]
        if not outstanding_row.empty:
            out_br   = outstanding_row["bad_rate"].iloc[0]
            out_lift = outstanding_row["lift"].iloc[0]
            print(f"\n  Outstanding agents bad rate : {out_br:.2%}")
            print(f"  Lift vs overall             : {out_lift:.2f}x")
            if out_br > 0.30:
                print("  *** WARNING: bad rate > 30% when outstanding — strong co-definition ***")
            elif out_br > 0.15:
                print("  NOTE: moderate lift; net_exposure_6M is a strong but not deterministic signal")
            else:
                print("  OK: outstanding status is not near-deterministic of default")
    else:
        print("  SKIP: net_exposure_6M or bad_state not found in training DataFrame")

    # ------------------------------------------------------------------ #
    # CHECK 4: Thin-file vs thick-file population breakdown
    # ------------------------------------------------------------------ #
    print(f"\n{DIVIDER}")
    print("CHECK 4: Thin-file vs thick-file population breakdown")
    print(DIVIDER)

    thin_col = feature_config.THIN_FILE_COL  # "thin_file_flag"
    label_col = feature_config.TARGET_COL    # "bad_state"

    if thin_col in df_train_raw.columns and label_col in df_train_raw.columns:
        from sklearn.metrics import roc_auc_score

        tf = df_train_raw[[thin_col, label_col]].copy()
        tf[thin_col] = pd.to_numeric(tf[thin_col], errors="coerce")
        tf[label_col] = pd.to_numeric(tf[label_col], errors="coerce")
        tf = tf.dropna()

        n_total     = len(tf)
        n_thin      = int((tf[thin_col] == 1).sum())
        n_thick     = int((tf[thin_col] == 0).sum())
        br_thin     = float(tf.loc[tf[thin_col] == 1, label_col].mean()) if n_thin else float("nan")
        br_thick    = float(tf.loc[tf[thin_col] == 0, label_col].mean()) if n_thick else float("nan")
        br_overall  = float(tf[label_col].mean())

        print(f"  {'Segment':<20} {'n':>10} {'share':>8} {'bad_rate':>10}")
        print(f"  {'-'*50}")
        print(f"  {'thin-file (scorecard)':<20} {n_thin:>10,} {n_thin/n_total:>8.1%} {br_thin:>10.2%}")
        print(f"  {'thick-file (PD model)':<20} {n_thick:>10,} {n_thick/n_total:>8.1%} {br_thick:>10.2%}")
        print(f"  {'overall':<20} {n_total:>10,} {'100%':>8} {br_overall:>10.2%}")

        # AUC per segment using model score from X_train if available
        print()
        # Thin-file single-feature AUC using only transaction volume features
        thin_mask  = (df_train_raw[thin_col] == 1) if thin_col in df_train_raw.columns else pd.Series(False, index=df_train_raw.index)
        thick_mask = ~thin_mask

        y_all = pd.to_numeric(df_train_raw[label_col], errors="coerce")

        # Check top feature AUC on thin vs thick separately
        top_feat = auc_report.iloc[0]["feature"]
        if top_feat in df_train_raw.columns:
            x_feat = pd.to_numeric(df_train_raw[top_feat], errors="coerce")
            for seg_name, mask in [("thin-file", thin_mask), ("thick-file", thick_mask)]:
                valid = mask & x_feat.notna() & y_all.notna()
                if valid.sum() >= 50 and y_all[valid].nunique() == 2:
                    auc = roc_auc_score(y_all[valid], x_feat[valid])
                    auc = max(auc, 1 - auc)
                    print(f"  Top feature '{top_feat}' AUC on {seg_name}: {auc:.4f}")
                else:
                    print(f"  Top feature '{top_feat}' AUC on {seg_name}: insufficient data")

        if n_thin > 0:
            print(f"\n  NOTE: {n_thin:,} thin-file agents ({n_thin/n_total:.1%}) use the scorecard,")
            print(f"  not XGBoost. The 0.989 AUC is measured across all {n_total:,} agents.")
            print(f"  Thin-file bad rate ({br_thin:.2%}) vs thick-file ({br_thick:.2%}) — separate")
            print(f"  AUC estimates per segment would confirm scorecard discrimination quality.")
    else:
        print(f"  SKIP: '{thin_col}' or '{label_col}' not found in training DataFrame")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(f"\n{DIVIDER}")
    print("SUMMARY")
    print(DIVIDER)
    top_auc = auc_report.iloc[0]
    print(f"  Top single-feature AUC : {top_auc['feature']}  =  {top_auc['auc']:.4f}  [{top_auc['flag']}]")
    top_corr = corr_report.iloc[0]
    print(f"  Top label correlation  : {top_corr['feature']}  Spearman r = {top_corr['spearman_r']:.4f}  [{top_corr['flag']}]")
    print(f"  ALERT features (AUC)   : {len(alert_feats)}")
    print(f"  ALERT features (corr)  : {len(alert_corr)}")
    if len(alert_feats) == 0 and len(alert_corr) == 0:
        print("\n  VERDICT: No single feature drives AUC above the leakage threshold.")
        print("  The 0.989 AUC likely reflects a combination of genuine pre-snapshot")
        print("  behavioural signals, not a single leaked or co-defined feature.")
    else:
        print("\n  VERDICT: One or more features require investigation (see ALERT above).")
    print(DIVIDER)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Leakage & AUC diagnostic — reruns data-prep only, no model training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-file",           required=True,  help="Training snapshot CSV")
    p.add_argument("--val-file",             required=True,  help="Validation snapshot CSV")
    p.add_argument("--repayment-file",       required=True,  help="Repayments CSV")
    p.add_argument("--train-snapshot-date",  required=True,  type=int, help="YYYYMMDD int")
    p.add_argument("--val-snapshot-date",    required=True,  type=int, help="YYYYMMDD int")
    p.add_argument("--train-cutoff",         required=True,  help="YYYY-MM-DD upper bound for train rows")
    p.add_argument("--output-dir",           default="artifacts")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    args = _build_parser().parse_args()
    try:
        run_diagnostics(args)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
