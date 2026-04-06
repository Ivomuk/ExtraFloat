"""
PD Model Feature Pipeline — CLI entry point.

Usage
-----
::

    python -m pd_model.run_pipeline \\
        --train-file  data/snapshot_20250930.csv \\
        --val-file    data/snapshot_20251115.csv \\
        --repayment-file data/repayments.csv \\
        --train-snapshot-date 20250930 \\
        --val-snapshot-date   20251115 \\
        --train-cutoff        2025-09-30 \\
        --output-dir          artifacts/ \\
        --log-level           INFO

The pipeline:
1.  Load raw snapshots and stack into a combined modelling DataFrame.
2.  Phase 2.1 — transaction behaviour features.
3.  Phase 2.2 — loan repayment features + labelling.
4.  Thin-file scorecard (agents with no loan history).
5.  Feature classification + transformation.
6.  IV-based feature selection (train-only).
7.  Train / validation split + final data prep.
8.  Write ``feature_order.json`` and ``model_metadata.json`` to *output-dir*.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from pd_model.config import feature_config
from pd_model.config.model_config import DEFAULT_CONFIG
from pd_model.data_prep.pipeline_data import prepare_pd_training_and_validation_data
from pd_model.logging_config import configure_root_level, get_logger
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

logger = get_logger(__name__)


# ======================================================================== #
# Data loading helpers
# ======================================================================== #

def _load_snapshot(path: str, snapshot_date_int: int, split_label: str) -> pd.DataFrame:
    """Load a single snapshot CSV and tag it with snapshot_dt and split."""
    df = pd.read_csv(path)
    df["snapshot_dt"] = snapshot_date_int
    df["split"] = split_label
    df["tbl_dt"] = pd.to_numeric(df.get("tbl_dt", snapshot_date_int), errors="coerce")
    logger.info("Loaded %s: %d rows, %d cols", path, len(df), df.shape[1])
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce key date columns to datetime."""
    if "tbl_dt" in df.columns:
        df["tbl_dt"] = pd.to_datetime(df["tbl_dt"].astype(str), errors="coerce")
    if "snapshot_dt" in df.columns:
        df["snapshot_dt"] = pd.to_datetime(df["snapshot_dt"].astype(str), errors="coerce")
    if "activation_dt" in df.columns:
        df["activation_dt"] = (
            df["activation_dt"]
            .astype(str)
            .str.split(".")
            .str[0]
            .replace("", pd.NA)
        )
        df["activation_dt"] = pd.to_datetime(df["activation_dt"], format="%Y%m%d", errors="coerce")
    for c in ["date_of_birth", "payment_last", "cash_in_last", "cash_out_last"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str), errors="coerce")
    return df


# ======================================================================== #
# Pipeline
# ======================================================================== #

def run_pipeline(args: argparse.Namespace) -> None:
    cfg = DEFAULT_CONFIG
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1) Load and stack snapshots
    # ------------------------------------------------------------------ #
    logger.info("=== Step 1: Load snapshots ===")
    train_df = _load_snapshot(args.train_file, args.train_snapshot_date, "train")
    val_df = _load_snapshot(args.val_file, args.val_snapshot_date, "validation")

    df_pd = pd.concat([train_df, val_df], ignore_index=True, sort=False)
    df_pd = _parse_dates(df_pd)

    dup_cnt = df_pd.duplicated(subset=[feature_config.AGENT_KEY, "snapshot_dt"]).sum()
    if dup_cnt > 0:
        logger.warning("Duplicate (agent, snapshot_dt) rows detected: %d", dup_cnt)

    logger.info("Combined DataFrame: %d rows, %d cols", *df_pd.shape)

    # ------------------------------------------------------------------ #
    # 2) Phase 2.1 — transaction behaviour features
    # ------------------------------------------------------------------ #
    logger.info("=== Step 2: Phase 2.1 — transaction behaviour features ===")
    df_pd = run_phase_2_1_richer_tx_behaviour(df_pd, cfg=cfg)

    # ------------------------------------------------------------------ #
    # 3) Load repayments and run Phase 2.2
    # ------------------------------------------------------------------ #
    logger.info("=== Step 3: Phase 2.2 — repayment features + labelling ===")
    df_repayments = pd.read_csv(args.repayment_file)
    logger.info("Repayments loaded: %d rows", len(df_repayments))

    df_pd, _df_repayments_out = run_phase_2_2_repayment_pd_features(
        df_pd, df_repayments, cfg=cfg, verbose=True
    )
    df_pd = classify_agent_loan_status(df_pd)
    df_pd = add_thin_file_flags(df_pd, cfg=cfg)

    # ------------------------------------------------------------------ #
    # 4) Leakage audit
    # ------------------------------------------------------------------ #
    logger.info("=== Step 4: Leakage audit ===")
    leakage_audit_phase_2_2(
        df_pd,
        hard_fail=args.hard_fail_leakage,
        cfg=cfg,
        verbose=True,
    )

    # ------------------------------------------------------------------ #
    # 5) Thin-file scorecard
    # ------------------------------------------------------------------ #
    logger.info("=== Step 5: Thin-file scorecard ===")
    df_pd = add_never_loan_scorecard_from_phase_2_1(df_pd, cfg=cfg)

    # ------------------------------------------------------------------ #
    # 6) Feature classification + transformation
    # ------------------------------------------------------------------ #
    logger.info("=== Step 6: Feature classification + transformation ===")
    (
        pd_features,
        log_cols,
        cap_cols,
        _protected,
        signed_log_cols,
        _excluded_df,
    ) = get_and_classify_pd_features(df_pd)

    df_pd_raw = df_pd.copy(deep=True)

    df_pd_transformed, pd_features_pruned, transform_report = build_transformed_dataframe(
        df_pd,
        pd_features=pd_features,
        log_cols=log_cols,
        cap_cols=cap_cols,
        signed_log_cols=signed_log_cols,
        cfg=cfg,
    )

    # ------------------------------------------------------------------ #
    # 7) Train/val split + final data prep
    # ------------------------------------------------------------------ #
    logger.info("=== Step 7: Train/val split ===")
    train_cutoff = pd.Timestamp(args.train_cutoff)

    (
        X_train_raw, X_train_trans, y_train,
        X_val_raw, X_val_trans, y_val,
        candidate_features,
        thin_train, thin_val,
        agent_train, agent_val,
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

    # ------------------------------------------------------------------ #
    # 8) IV feature selection
    # ------------------------------------------------------------------ #
    logger.info("=== Step 8: IV feature selection ===")
    selected_features, iv_table = iv_filter_phase_2(
        X_train_raw=X_train_raw,
        X_train_transformed=X_train_trans,
        y_train=y_train,
        cfg=cfg,
        pd_feature_blacklist=feature_config.PD_FEATURE_BLACKLIST,
        forbidden_feature_patterns=feature_config.LEAKAGE_PATTERNS,
        target_col=feature_config.TARGET_COL,
    )

    # ------------------------------------------------------------------ #
    # 9) Write artifacts
    # ------------------------------------------------------------------ #
    logger.info("=== Step 9: Write artifacts ===")

    feature_order_path = output_dir / "feature_order.json"
    feature_order_path.write_text(
        json.dumps({"selected_features": selected_features}, indent=2)
    )
    logger.info("Wrote %s (%d features)", feature_order_path, len(selected_features))

    metadata = {
        "train_rows": len(X_train_raw),
        "val_rows": len(X_val_raw),
        "candidate_features": len(candidate_features),
        "selected_features": len(selected_features),
        "bad_rate_train": float(y_train.mean()),
        "bad_rate_val": float(y_val.mean()),
        "train_cutoff": str(train_cutoff.date()),
        "thin_file_train": int(thin_train.sum()),
        "thin_file_val": int(thin_val.sum()),
        "iv_table": iv_table.to_dict(orient="records"),
        "transform_report": transform_report.to_dict(orient="records"),
    }

    metadata_path = output_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    logger.info("Wrote %s", metadata_path)

    logger.info("Pipeline complete.")


# ======================================================================== #
# CLI
# ======================================================================== #

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PD Model Feature Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-file", required=True, help="Path to training snapshot CSV")
    parser.add_argument("--val-file", required=True, help="Path to validation snapshot CSV")
    parser.add_argument("--repayment-file", required=True, help="Path to repayments CSV")
    parser.add_argument(
        "--train-snapshot-date",
        required=True,
        type=int,
        help="Snapshot date for training set (YYYYMMDD int)",
    )
    parser.add_argument(
        "--val-snapshot-date",
        required=True,
        type=int,
        help="Snapshot date for validation set (YYYYMMDD int)",
    )
    parser.add_argument(
        "--train-cutoff",
        required=True,
        help="Inclusive upper bound for training rows (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where feature_order.json and model_metadata.json are written",
    )
    parser.add_argument(
        "--hard-fail-leakage",
        action="store_true",
        default=True,
        help="Raise an error on HIGH leakage findings (default: True)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    configure_root_level(level)

    try:
        run_pipeline(args)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
