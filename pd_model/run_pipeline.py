"""
PD Model Feature + Training Pipeline -- CLI entry point.

Usage (original, agent-snapshot grain)
---------------------------------------
::

    python -m pd_model.run_pipeline \\
        --train-file  data/snapshot_20250930.csv \\
        --val-file    data/snapshot_20251115.csv \\
        --repayment-file data/repayments.csv \\
        --train-snapshot-date 20250930 \\
        --val-snapshot-date   20251115 \\
        --train-cutoff        2025-09-30 \\
        --output-dir          artifacts/ \\
        --champion            xgb \\
        --log-level           INFO

Usage (loan-level grain, data/loan_state_query_updated_materialized.txt)
--------------------------------------------------------------------------
::

    python -m pd_model.run_pipeline \\
        --loan-training-file data/loan_state_training_export.csv \\
        --train-file  data/snapshot_20250930.csv \\
        --val-file    data/snapshot_20251115.csv \\
        --train-snapshot-date 20250930 \\
        --val-snapshot-date   20251115 \\
        --output-dir          artifacts/ \\
        --champion            xgb \\
        --log-level           INFO

``--loan-training-file`` and ``--repayment-file`` are mutually exclusive --
exactly one must be supplied. ``--train-file``/``--val-file``/
``--train-snapshot-date``/``--val-snapshot-date`` are required in both
modes: in loan-level mode they still supply Phase 2.1 (commission/balance/
capacity) features, frozen at one cohort-representative date per split and
joined onto the loan grain by agent_msisdn -- a real point-in-time Phase 2.1
join is a follow-up, not built here. ``--train-cutoff`` is required only in
the original mode; in loan-level mode the query's own ``split`` column is
authoritative and a representative cutoff is derived automatically for
metadata/logging only.

The pipeline (loan-level mode differences noted inline):
1.  Load raw snapshots and stack into a combined modelling DataFrame.
    Loan-level mode: load the loan-level training file directly (already
    has ``split`` and the label per row).
2.  Phase 2.1 -- transaction behaviour features.
    Loan-level mode: joined onto the loan grain by (agent_msisdn, split).
3.  Phase 2.2 -- loan repayment features + labelling.
    Loan-level mode: ``run_phase_2_2_loan_history_pd_features()`` instead of
    ``run_phase_2_2_repayment_pd_features()`` -- see
    ``pd_model/preprocessing/loan_history_features.py``.
4.  Thin-file scorecard (agents with no loan history).
5.  Feature classification + transformation.
6.  IV-based feature selection (train-only).
7.  Train / validation split + final data prep.
    Loan-level mode: the ``split`` column is used directly
    (``precomputed_train_mask``), not a date-cutoff comparison.
8.  Write ``feature_order.json`` and ``model_metadata.json``.
9.  Train XGBoost model.
10. Train LightGBM model.
11. Bootstrap AUC comparison (skippable with --skip-bootstrap).
12. Calibration + policy pipeline.
13. Build unified ops_scored table.
14. Serialize all artifacts for inference.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pd_model.config import feature_config
from pd_model.config.model_config import DEFAULT_CONFIG
from pd_model.data_prep.pipeline_data import prepare_pd_training_and_validation_data
from pd_model.exceptions import DataAlignmentError
from pd_model.logging_config import configure_root_level, get_logger
from pd_model.modeling.calibration import run_bootstrap_comparison, run_locked_policy_pipeline
from pd_model.modeling.evaluation import compare_models_deciles
from pd_model.modeling.lgbm_model import train_lgbm
from pd_model.modeling.scorecard import (
    add_never_loan_scorecard_from_phase_2_1,
    apply_thin_file_lr,
    fit_thin_file_lr,
)
from pd_model.modeling.xgb_model import train_xgb
from pd_model.postprocessing.ops_scored import build_exec_summary, build_ops_scored_table
from pd_model.preprocessing.loan_features import (
    add_thin_file_flags,
    classify_agent_loan_status,
    leakage_audit_phase_2_2,
    run_phase_2_2_repayment_pd_features,
)
from pd_model.preprocessing.loan_history_features import run_phase_2_2_loan_history_pd_features
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
        df["activation_dt"] = df["activation_dt"].astype(str).str.split(".").str[0].replace("", pd.NA)
        df["activation_dt"] = pd.to_datetime(df["activation_dt"], format="%Y%m%d", errors="coerce")
    for c in ["date_of_birth", "payment_last", "cash_in_last", "cash_out_last"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str), errors="coerce")
    return df


def _downcast_float64_to_float32(df: pd.DataFrame) -> None:
    """
    In-place: halve memory for all float64 columns.

    The loan-level grain (one row per loan) can be an order of magnitude
    more rows than the original agent-snapshot grain, so full-precision
    copies of the whole modelling frame (e.g. add_never_loan_scorecard_from_phase_2_1's
    df_pd_in.copy(), or the train/val split further down) can require a
    multi-GiB single allocation. Feature columns get cast to float32 again
    later in build_transformed_dataframe regardless, so downcasting here
    changes nothing downstream -- it just does it before the first
    full-frame copy at scale rather than after. Idempotent / near-free to
    call again once already downcast.
    """
    float64_cols = df.select_dtypes(include="float64").columns
    if len(float64_cols) > 0:
        df[float64_cols] = df[float64_cols].astype("float32")


# Loan-level bookkeeping/join-key columns from
# data/loan_state_query_updated_materialized.txt that are (a) in
# PD_FEATURE_BLACKLIST -- never candidate features -- and (b) not read
# anywhere else in pd_model/ (confirmed by grep). They're internal
# artifacts of the SQL's loan-to-state mapping, not human-interpretable
# on their own (unlike disbursement_fid, the canonical loan id, which IS
# kept for the duplicate-row diagnostic and general audit value). Object
# (string) columns are far more memory-expensive per row than numeric
# ones -- each value is a separate Python heap object, not just a few
# bytes -- so carrying millions of rows of these through the whole
# pipeline for zero downstream benefit is pure waste.
_UNUSED_LOAN_LEVEL_ID_COLUMNS: tuple[str, ...] = (
    "disbursement_uid",
    "target_loan_uid",
    "scoring_state_loan_uid",
    "last_closed_loan_uid",
)

# Low-cardinality string columns worth converting to category dtype
# in-place: collapses millions of repeated string objects down to a
# handful of category levels + a small integer code per row.
_LOW_CARDINALITY_STRING_COLUMNS: tuple[str, ...] = (
    "split",
    "sales_region",
    "sales_territory",
    "district",
)


def _reduce_object_column_memory(df: pd.DataFrame) -> None:
    """In-place: drop unused high-cardinality id columns, categorize
    known low-cardinality ones. See the constants above for rationale."""
    present_unused = [c for c in _UNUSED_LOAN_LEVEL_ID_COLUMNS if c in df.columns]
    if present_unused:
        df.drop(columns=present_unused, inplace=True)
        logger.info(
            "Dropped %d unused id column(s) to reduce memory: %s",
            len(present_unused),
            present_unused,
        )
    for c in _LOW_CARDINALITY_STRING_COLUMNS:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype("category")


# ======================================================================== #
# Pipeline
# ======================================================================== #


def run_pipeline(args: argparse.Namespace) -> None:
    cfg = DEFAULT_CONFIG
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    using_loan_level_grain = getattr(args, "loan_training_file", None) is not None
    train_cutoff: pd.Timestamp | None = None

    if using_loan_level_grain:
        # ------------------------------------------------------------------ #
        # 1) Load the loan-level training file
        #
        # Grain: one row per disbursement_fid, with split and the label
        # already assigned by data/loan_state_query_updated_materialized.txt.
        # ------------------------------------------------------------------ #
        logger.info("=== Step 1: Load loan-level training file ===")
        df_loans = pd.read_csv(args.loan_training_file)
        logger.info("Loaded %s: %d rows, %d cols", args.loan_training_file, *df_loans.shape)

        df_pd, _df_label_diagnostics = run_phase_2_2_loan_history_pd_features(
            df_loans, cfg=cfg, verbose=True
        )

        # disbursement_fid is the loan-level primary key (unique per loan,
        # source: analytics.momo_loan_book_tracker_disbursements_daily) --
        # check it alone, not the (agent, disbursement_fid) compound. The
        # compound key is strictly weaker: it would miss the same
        # disbursement_fid appearing under two different agent_msisdn
        # values, which is a genuine data-integrity bug (a loan attached to
        # the wrong agent), not a benign edge case the compound key should
        # be tolerating.
        dup_cnt = df_pd.duplicated(subset=["disbursement_fid"]).sum()
        if dup_cnt > 0:
            logger.warning(
                "Duplicate disbursement_fid rows detected: %d -- disbursement_fid should be "
                "unique per loan; check the export for schema/join issues upstream.",
                dup_cnt,
            )

        _reduce_object_column_memory(df_pd)

        logger.info("Combined DataFrame: %d rows, %d cols", *df_pd.shape)

        # ------------------------------------------------------------------ #
        # 2) Phase 2.1 -- transaction behaviour features
        #
        # Frozen at one cohort-representative date per split (train/val
        # snapshot files, same as the original mode) and joined onto the
        # loan grain by (agent_msisdn, split). This is coarser than true
        # point-in-time (each loan's own disbursement date), and is a known,
        # explicitly accepted limitation for this first version -- see the
        # module docstring. Labels and loan-history features are point-in-time
        # correct via the SQL regardless.
        # ------------------------------------------------------------------ #
        logger.info("=== Step 2: Phase 2.1 -- transaction behaviour features (frozen per-split date) ===")
        train_snap_df = _load_snapshot(args.train_file, args.train_snapshot_date, "train")
        val_snap_df = _load_snapshot(args.val_file, args.val_snapshot_date, "validation")
        snap_df = pd.concat([train_snap_df, val_snap_df], ignore_index=True, sort=False)
        snap_df = _parse_dates(snap_df)
        snap_df = run_phase_2_1_richer_tx_behaviour(snap_df, cfg=cfg)

        snap_df[feature_config.AGENT_KEY] = snap_df[feature_config.AGENT_KEY].astype(str).str.strip()
        df_pd[feature_config.AGENT_KEY] = df_pd[feature_config.AGENT_KEY].astype(str).str.strip()

        # One Phase 2.1 row per (agent, split) -- keep last on duplicates.
        snap_df = snap_df.drop_duplicates(subset=[feature_config.AGENT_KEY, "split"], keep="last")

        phase21_cols = [c for c in snap_df.columns if c not in (feature_config.AGENT_KEY, "split")]
        overlap = [c for c in phase21_cols if c in df_pd.columns]
        if overlap:
            logger.info(
                "Step 2: %d Phase 2.1 column(s) overlap loan-level columns by name; "
                "loan-level values kept, Phase 2.1 duplicates dropped: %s",
                len(overlap),
                overlap,
            )
            phase21_cols = [c for c in phase21_cols if c not in overlap]

        n_before_join = len(df_pd)
        df_pd = df_pd.merge(
            snap_df[[feature_config.AGENT_KEY, "split"] + phase21_cols],
            on=[feature_config.AGENT_KEY, "split"],
            how="left",
        )
        if len(df_pd) != n_before_join:
            raise DataAlignmentError(
                f"[run_pipeline] Phase 2.1 join changed row count: {n_before_join} -> {len(df_pd)} "
                "(snap_df not unique per (agent_msisdn, split) after dedup)"
            )
        logger.info("After Phase 2.1 join: %d rows, %d cols", *df_pd.shape)

        # ------------------------------------------------------------------ #
        # 3) Phase 2.2 already applied in Step 1 above
        #    (run_phase_2_2_loan_history_pd_features derives bad_state and
        #    thin_file_flag directly from the loan-level query output).
        # ------------------------------------------------------------------ #
        logger.info("=== Step 3: Phase 2.2 -- loan history features (applied in Step 1) ===")

        # ------------------------------------------------------------------ #
        # 4) Leakage audit
        # ------------------------------------------------------------------ #
        logger.info("=== Step 4: Leakage audit ===")
        leakage_audit_phase_2_2(
            df_pd,
            hard_fail=args.hard_fail_leakage,
            cfg=cfg,
            verbose=True,
            pd_feature_blacklist=feature_config.PD_FEATURE_BLACKLIST,
        )

        # ------------------------------------------------------------------ #
        # 5) Thin-file scorecard
        # ------------------------------------------------------------------ #
        logger.info("=== Step 5: Thin-file scorecard ===")
        _downcast_float64_to_float32(df_pd)
        df_pd = add_never_loan_scorecard_from_phase_2_1(df_pd, cfg=cfg)

        # Derive a representative train_cutoff for metadata/logging only --
        # the actual split uses the split column directly (Step 7), not this
        # date. See prepare_pd_training_and_validation_data's
        # precomputed_train_mask parameter.
        train_loan_dates = pd.to_datetime(
            df_pd.loc[df_pd["split"].eq("train"), "loan_date"], errors="coerce"
        )
        train_cutoff = train_loan_dates.max()
        if pd.isna(train_cutoff):
            train_cutoff = pd.Timestamp.today().normalize()

    else:
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
        # 2) Phase 2.1 -- transaction behaviour features
        # ------------------------------------------------------------------ #
        logger.info("=== Step 2: Phase 2.1 -- transaction behaviour features ===")
        df_pd = run_phase_2_1_richer_tx_behaviour(df_pd, cfg=cfg)

        # ------------------------------------------------------------------ #
        # 3) Load repayments and run Phase 2.2
        # ------------------------------------------------------------------ #
        logger.info("=== Step 3: Phase 2.2 -- repayment features + labelling ===")
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
            pd_feature_blacklist=feature_config.PD_FEATURE_BLACKLIST,
        )

        # ------------------------------------------------------------------ #
        # 5) Thin-file scorecard
        # ------------------------------------------------------------------ #
        logger.info("=== Step 5: Thin-file scorecard ===")
        _downcast_float64_to_float32(df_pd)
        df_pd = add_never_loan_scorecard_from_phase_2_1(df_pd, cfg=cfg)

        train_cutoff = pd.Timestamp(args.train_cutoff)

    # ------------------------------------------------------------------ #
    # 6) Feature classification + transformation (fit on train only)
    # ------------------------------------------------------------------ #
    logger.info("=== Step 6: Feature classification + transformation ===")

    # Re-downcast: Step 5's scorecard adds its own float64 columns
    # (never_loan_points, never_loan_score_0_100, never_loan_pd_like), so
    # this catches those too, on top of the pre-Step-5 downcast above.
    # Cheap / near-free if there's nothing new to convert.
    _downcast_float64_to_float32(df_pd)

    # Determine the temporal split boundary up-front so winsorization bounds
    # are fitted on training data only and applied to validation -- prevents
    # validation-distribution leakage into the transform_report saved to disk.
    # Loan-level mode: the SQL's own split column is authoritative (Decision 5);
    # original mode: date-cutoff comparison as before.
    if using_loan_level_grain:
        split_mask = df_pd["split"].eq("train")
    else:
        split_mask = pd.to_datetime(df_pd["snapshot_dt"], errors="coerce") <= train_cutoff
    # Boolean-mask row selection already returns an independent copy in
    # pandas (never a view), so the .copy() previously chained here was
    # redundant -- it forced a second full consolidate+copy pass right
    # after the first, doubling peak memory at exactly this line (the
    # ArrayMemoryError above pointed here).
    df_train_raw = df_pd[split_mask]
    df_val_raw = df_pd[~split_mask]
    # df_pd is never read again after this point (confirmed by grep) -- it's
    # likely the single largest object in the run (full loan-level row count,
    # widest column count before any narrowing), and it would otherwise sit
    # alive in scope through Steps 6-13 for no reason.
    del df_pd
    gc.collect()
    logger.info(
        "Pre-transform split: %d train rows / %d val rows (cutoff=%s)",
        len(df_train_raw),
        len(df_val_raw),
        train_cutoff.date(),
    )

    # ------------------------------------------------------------------ #
    # 5b) Fit thin-file logistic regression (train split only)
    #
    # Replaces the manual sigmoid with a data-driven balanced LR so the
    # thin-file scorecard AUC improves beyond the 0.617 baseline.
    # Applied before transformation so the fitted model uses raw features.
    # ------------------------------------------------------------------ #
    logger.info("=== Step 5b: Fit thin-file logistic regression ===")
    _thin_lr_pipeline, _thin_lr_features = fit_thin_file_lr(
        df_train_raw, df_val_raw=df_val_raw, label_col=feature_config.TARGET_COL
    )
    if _thin_lr_pipeline is not None:
        df_train_raw = apply_thin_file_lr(df_train_raw, _thin_lr_pipeline, _thin_lr_features)
        df_val_raw = apply_thin_file_lr(df_val_raw, _thin_lr_pipeline, _thin_lr_features)
        logger.info("Step 5b: thin-file LR applied to train and val")
        # Compute val AUC for the metadata
        _thin_mask_val_lr = pd.to_numeric(
            df_val_raw.get(feature_config.THIN_FILE_COL, 0), errors="coerce"
        ).fillna(0).eq(1)
        _thin_val_lr = df_val_raw[_thin_mask_val_lr]
        if (
            _thin_mask_val_lr.sum() > 1
            and feature_config.TARGET_COL in _thin_val_lr.columns
            and _thin_val_lr[feature_config.TARGET_COL].nunique() > 1
        ):
            from sklearn.metrics import roc_auc_score as _roc_auc_sc
            _thin_lr_val_auc: float | None = float(
                _roc_auc_sc(
                    _thin_val_lr[feature_config.TARGET_COL].fillna(0).astype(int),
                    _thin_val_lr["never_loan_pd_like"],
                )
            )
            logger.info("Step 5b: thin-file LR val AUC=%.4f", _thin_lr_val_auc)
        else:
            _thin_lr_val_auc = None
    else:
        _thin_lr_val_auc = None

    (
        pd_features,
        log_cols,
        cap_cols,
        _protected,
        signed_log_cols,
        _excluded_df,
    ) = get_and_classify_pd_features(df_train_raw)

    # Fit transformations on training data only
    df_train_trans, pd_features_pruned, transform_report = build_transformed_dataframe(
        df_train_raw,
        pd_features=pd_features,
        log_cols=log_cols,
        cap_cols=cap_cols,
        signed_log_cols=signed_log_cols,
        cfg=cfg,
    )

    # Replay training-fitted params on the validation set so clip bounds and
    # transform choices are fixed to the training distribution.
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
        df_val_raw,
        pd_features=pd_features,
        log_cols=log_cols,
        cap_cols=cap_cols,
        signed_log_cols=signed_log_cols,
        cfg=cfg,
        fitted_params=fitted_params,
    )
    # Restrict val columns to the training feature set (prune skips features
    # with fitted_params, so val may have fewer columns than train).
    val_cols = [c for c in df_train_trans.columns if c in df_val_trans.columns]
    df_val_trans = df_val_trans[val_cols]

    # Re-combine into the unified frames that prepare_pd_training_and_validation_data
    # expects; it will re-split internally using the same train_cutoff (or, in
    # loan-level mode, the precomputed_train_mask derived below).
    df_pd_raw = pd.concat([df_train_raw, df_val_raw], ignore_index=True)
    df_pd_transformed = pd.concat([df_train_trans, df_val_trans], ignore_index=True)
    precomputed_train_mask = (
        pd.Series(
            [True] * len(df_train_raw) + [False] * len(df_val_raw),
            index=df_pd_raw.index,
        )
        if using_loan_level_grain
        else None
    )

    # After the prepare_pd_training_and_validation_data() call below, df_pd_raw
    # is only ever needed again for one thing further down (building df_sc_thin
    # for the thin-file half of ops_scored), and that only ever reads 7 columns
    # (build_ops_scored_table's _THIN_SCORECARD_COLS + agent_msisdn). Extract
    # that tiny lookup now, before df_pd_raw's full width becomes dead weight
    # carried through Steps 8-13 (IV selection, model training, calibration).
    _thin_lookup_cols = [
        c
        for c in [
            feature_config.AGENT_KEY,
            feature_config.THIN_FILE_COL,
            "bad_state",
            "never_loan_points",
            "never_loan_score_0_100",
            "never_loan_pd_like",
            "never_loan_top_drivers",
        ]
        if c in df_pd_raw.columns
    ]
    df_thin_lookup = df_pd_raw[_thin_lookup_cols].copy()

    # ------------------------------------------------------------------ #
    # 7) Train/val split + final data prep
    # ------------------------------------------------------------------ #
    logger.info("=== Step 7: Train/val split ===")
    # train_cutoff already defined above

    (
        X_train_raw,
        X_train_trans,
        y_train,
        X_val_raw,
        X_val_trans,
        y_val,
        candidate_features,
        thin_train,
        thin_val,
        agent_train,
        agent_val,
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
        precomputed_train_mask=precomputed_train_mask,
    )

    # df_pd_raw/df_pd_transformed/df_train_raw/df_val_raw/df_train_trans/
    # df_val_trans are never read again -- everything needed going forward
    # is in the X_*/y_*/candidate_features/thin_*/agent_* return values above,
    # plus df_thin_lookup (extracted just above) for the one remaining need.
    del df_pd_raw, df_pd_transformed, df_train_raw, df_val_raw, df_train_trans, df_val_trans
    gc.collect()

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
    # 9) Train XGBoost  (thick-file agents only)
    # ------------------------------------------------------------------ #
    logger.info("=== Step 9: Train XGBoost ===")

    # Exclude thin-file agents before fitting. XGBoost handles NaN natively
    # via default branch directions, so including thin-file agents (whose
    # repayment features are all NaN) teaches the model to use missingness
    # as a proxy for thin-file status rather than genuine credit risk.
    thick_train_mask = thin_train.eq(0)
    thick_val_mask = thin_val.eq(0)
    logger.info(
        "Thick-file population: train=%d/%d | val=%d/%d",
        int(thick_train_mask.sum()), len(thin_train),
        int(thick_val_mask.sum()), len(thin_val),
    )

    # No .copy() needed -- .loc[row_mask, col_list] is already independent
    # under pandas' copy-on-write (same reasoning as Steps 6/7 above).
    Xtr = X_train_trans.loc[thick_train_mask, selected_features]
    y_train_thick = y_train.loc[thick_train_mask]
    Xva = X_val_trans.loc[thick_val_mask, selected_features]
    y_val_thick = y_val.loc[thick_val_mask]
    agent_train_thick = agent_train.loc[thick_train_mask]
    thin_train_thick = thin_train.loc[thick_train_mask]
    agent_val_thick = agent_val.loc[thick_val_mask]
    thin_val_thick = thin_val.loc[thick_val_mask]

    xgb_model, xgb_train_scored, xgb_val_scored = train_xgb(
        Xtr, y_train_thick, Xva, y_val_thick, cfg=cfg
    )
    for scored_df, agent_ids, thin_flags in [
        (xgb_train_scored, agent_train_thick, thin_train_thick),
        (xgb_val_scored, agent_val_thick, thin_val_thick),
    ]:
        scored_df.insert(0, feature_config.AGENT_KEY, agent_ids.values)
        scored_df[feature_config.THIN_FILE_COL] = thin_flags.values

    # ------------------------------------------------------------------ #
    # 10) Train LightGBM  (same thick-file Xtr / Xva)
    # ------------------------------------------------------------------ #
    logger.info("=== Step 10: Train LightGBM ===")
    lgb_model, lgb_train_scored, lgb_val_scored = train_lgbm(
        Xtr, y_train_thick, Xva, y_val_thick, cfg=cfg
    )
    for scored_df, agent_ids, thin_flags in [
        (lgb_train_scored, agent_train_thick, thin_train_thick),
        (lgb_val_scored, agent_val_thick, thin_val_thick),
    ]:
        scored_df.insert(0, feature_config.AGENT_KEY, agent_ids.values)
        scored_df[feature_config.THIN_FILE_COL] = thin_flags.values

    # ------------------------------------------------------------------ #
    # 10b) Split val 50/50: selection half (AUC/bootstrap) vs calibration half
    #
    # Using the same val set for both model selection and isotonic calibration
    # introduces a subtle bias.  Splitting here ensures the calibration map
    # is never fit on data that influenced model selection.
    # ------------------------------------------------------------------ #
    from sklearn.model_selection import train_test_split as _tts

    _val_idx = np.arange(len(xgb_val_scored))
    _bad_states = xgb_val_scored["bad_state"].fillna(0).astype(int).values
    _idx_sel, _idx_cal = _tts(_val_idx, test_size=0.5, stratify=_bad_states, random_state=cfg.random_state)

    xgb_val_select = xgb_val_scored.iloc[_idx_sel].reset_index(drop=True)
    lgb_val_select = lgb_val_scored.iloc[_idx_sel].reset_index(drop=True)
    xgb_val_cal = xgb_val_scored.iloc[_idx_cal].reset_index(drop=True)
    lgb_val_cal = lgb_val_scored.iloc[_idx_cal].reset_index(drop=True)

    logger.info(
        "Val split: selection=%d rows | calibration=%d rows | bad_rate_sel=%.4f | bad_rate_cal=%.4f",
        len(xgb_val_select),
        len(xgb_val_cal),
        float(xgb_val_select["bad_state"].mean()),
        float(xgb_val_cal["bad_state"].mean()),
    )

    # Thick-file-only XGBoost AUC -- removes population mixing effect so the
    # reported metric reflects genuine credit-risk discrimination.
    _thick_mask_sel = xgb_val_select[feature_config.THIN_FILE_COL].eq(0)
    _xgb_sel_thick = xgb_val_select[_thick_mask_sel]
    if _thick_mask_sel.sum() > 1 and _xgb_sel_thick["bad_state"].nunique() > 1:
        from sklearn.metrics import roc_auc_score as _roc_auc_thick
        _thick_val_auc: float | None = float(
            _roc_auc_thick(_xgb_sel_thick["bad_state"], _xgb_sel_thick["raw_score"])
        )
        logger.info(
            "XGBoost thick-file-only val AUC=%.4f (n=%d, bad_rate=%.4f)",
            _thick_val_auc,
            int(_thick_mask_sel.sum()),
            float(_xgb_sel_thick["bad_state"].mean()),
        )
    else:
        _thick_val_auc = None
        logger.warning(
            "XGBoost thick-file-only AUC: insufficient data (n=%d)", int(_thick_mask_sel.sum())
        )

    # ------------------------------------------------------------------ #
    # 11) Bootstrap AUC comparison  (selection half only)
    # ------------------------------------------------------------------ #
    if not getattr(args, "skip_bootstrap", False):
        logger.info("=== Step 11: Bootstrap AUC comparison ===")
        bootstrap_tbl = run_bootstrap_comparison(xgb_val_select, lgb_val_select, cfg=cfg)
        logger.info("Bootstrap results:\n%s", bootstrap_tbl.to_string(index=False))
    else:
        logger.info("=== Step 11: Bootstrap skipped (--skip-bootstrap) ===")
        bootstrap_tbl = pd.DataFrame()

    # Model comparison deciles  (selection half only)
    cmp_summary, _xgb_dec, _lgb_dec = compare_models_deciles(
        xgb_val_select, lgb_val_select, name_a="xgb", name_b="lgb"
    )
    logger.info("Model comparison:\n%s", cmp_summary.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 12) Calibration + policy pipeline  (calibration half only)
    # ------------------------------------------------------------------ #
    logger.info("=== Step 12: Calibration + policy pipeline ===")
    locked_artifacts = run_locked_policy_pipeline(xgb_val_cal, lgb_val_cal, cfg=cfg, output_dir=output_dir)

    # ------------------------------------------------------------------ #
    # 13) Build unified ops_scored table
    # ------------------------------------------------------------------ #
    logger.info("=== Step 13: Build ops_scored table ===")
    champion = getattr(args, "champion", "xgb")

    from pd_model.modeling.calibration import add_policy_flags, make_policy_bucket

    try:
        # run_locked_policy_pipeline already calibrated both models and returned
        # sorted DataFrames with cal_pd attached; apply policy flags for both.
        xgb_sorted = locked_artifacts["xgb_policy_df_sorted"].copy()
        lgb_sorted = locked_artifacts["lgb_policy_df_sorted"].copy()

        xgb_sorted = add_policy_flags(
            xgb_sorted, locked_artifacts["xgb_policy_threshold_tbl"], prefix="xgb", cfg=cfg
        )
        lgb_sorted = add_policy_flags(
            lgb_sorted, locked_artifacts["lgb_policy_threshold_tbl"], prefix="lgb", cfg=cfg
        )

        # Rename model-specific cal_pd before merging
        xgb_sorted = xgb_sorted.rename(columns={feature_config.CAL_PD_COL: "xgb_cal_pd"})
        lgb_sorted = lgb_sorted.rename(columns={feature_config.CAL_PD_COL: "lgb_cal_pd"})

        # Merge LGB columns onto XGB base (both cover the same thick-file val agents)
        lgb_merge_cols = [feature_config.AGENT_KEY, "lgb_cal_pd"] + [
            c for c in lgb_sorted.columns if c.startswith("lgb_approved")
        ]
        thick_df = xgb_sorted.merge(lgb_sorted[lgb_merge_cols], on=feature_config.AGENT_KEY, how="left")

        # Promote champion's cal_pd to shared cal_pd column
        thick_df[feature_config.CAL_PD_COL] = thick_df[f"{champion}_cal_pd"]
        thick_df[feature_config.POLICY_BUCKET_COL] = make_policy_bucket(thick_df, prefix=champion, cfg=cfg)
        thick_df["final_approved"] = thick_df.get(f"{champion}_approved_at_50", 0)

        # thin-file scorecard rows
        thin_mask_val = thin_val.astype(bool)
        df_sc_thin = (
            df_thin_lookup.loc[X_val_raw.index[thin_mask_val]].copy()
            if thin_mask_val.sum() > 0
            else pd.DataFrame()
        )

        if df_sc_thin.shape[0] > 0:
            ops_scored = build_ops_scored_table(thick_df, df_sc_thin)
        else:
            ops_scored = thick_df.copy()
            ops_scored[feature_config.DECISION_SOURCE_COL] = "PD_MODEL"

        exec_summary = build_exec_summary(ops_scored)
        logger.info("Exec summary:\n%s", exec_summary.to_string(index=False))

        # Pseudonymise MSISDN before writing: replace with first 16 hex chars of
        # SHA-256 so the file does not contain raw phone numbers.
        if feature_config.AGENT_KEY in ops_scored.columns:
            ops_scored = ops_scored.copy()
            ops_scored[feature_config.AGENT_KEY] = (
                ops_scored[feature_config.AGENT_KEY]
                .astype(str)
                .apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])
            )
        ops_path = output_dir / "ops_scored.csv"
        ops_scored.to_csv(ops_path, index=False)
        logger.info("Wrote %s (%d rows) [agent_msisdn pseudonymised]", ops_path, len(ops_scored))

    except Exception as exc:
        logger.warning("ops_scored build failed (calibration may need more data): %s", exc)
        ops_scored = pd.DataFrame()

    # ------------------------------------------------------------------ #
    # 14) Serialize artifacts
    # ------------------------------------------------------------------ #
    logger.info("=== Step 14: Serialize artifacts ===")

    xgb_path = output_dir / "xgb_model.joblib"
    lgb_path = output_dir / "lgbm_model.joblib"
    joblib.dump(xgb_model, xgb_path)
    joblib.dump(lgb_model, lgb_path)
    logger.info("Saved XGBoost model -> %s", xgb_path)
    logger.info("Saved LightGBM model -> %s", lgb_path)

    if _thin_lr_pipeline is not None:
        lr_path = output_dir / "thin_file_lr.joblib"
        lr_features_path = output_dir / "thin_file_lr_features.json"
        joblib.dump(_thin_lr_pipeline, lr_path)
        lr_features_path.write_text(json.dumps(_thin_lr_features, indent=2))
        logger.info("Saved thin-file LR -> %s", lr_path)

    # transform_report for inference
    tr_path = output_dir / "transform_report.csv"
    transform_report.to_csv(tr_path, index=False)
    logger.info("Saved transform_report -> %s", tr_path)

    # feature_order.json
    feature_order_path = output_dir / "feature_order.json"
    feature_order_path.write_text(json.dumps({"selected_features": selected_features}, indent=2))
    logger.info("Wrote %s (%d features)", feature_order_path, len(selected_features))

    # Lineage helpers
    def _sha256_file(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def _git_commit() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                cwd=str(output_dir.parent),
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return "unknown"

    # Full metadata
    metadata = {
        # Lineage
        "training_completed_at": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_commit(),
        "feature_schema_version": 1,
        "preprocessor_version": 1,
        # Training stats
        "train_rows": len(X_train_raw),
        "val_rows": len(X_val_raw),
        "val_select_rows": len(xgb_val_select),
        "val_cal_rows": len(xgb_val_cal),
        "candidate_features": len(candidate_features),
        "selected_features": len(selected_features),
        "bad_rate_train": float(y_train.mean()),
        "bad_rate_val": float(y_val.mean()),
        "train_cutoff": str(train_cutoff.date()),
        "thin_file_train": int(thin_train.sum()),
        "thin_file_val": int(thin_val.sum()),
        "thin_file_lr_val_auc": _thin_lr_val_auc,
        "champion": champion,
        "xgb_val_auc": float(cmp_summary.loc[cmp_summary["model"] == "xgb", "auc"].iloc[0])
        if "xgb" in cmp_summary["model"].values
        else None,
        "xgb_val_auc_thick_only": _thick_val_auc,
        "lgb_val_auc": float(cmp_summary.loc[cmp_summary["model"] == "lgb", "auc"].iloc[0])
        if "lgb" in cmp_summary["model"].values
        else None,
        "iv_table": iv_table.to_dict(orient="records"),
        "transform_report": transform_report.to_dict(orient="records"),
        "bootstrap": bootstrap_tbl.to_dict(orient="records") if not bootstrap_tbl.empty else [],
    }

    metadata_path = output_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    # Artifact checksums (computed after all files are written, excluding metadata itself)
    artifact_sha256 = {
        f.name: _sha256_file(f)
        for f in sorted(output_dir.iterdir())
        if f.suffix in {".joblib", ".csv", ".json"} and f.name != "model_metadata.json"
    }
    metadata["artifact_sha256"] = artifact_sha256
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    logger.info("Wrote %s (sha256 of %d artifacts)", metadata_path, len(artifact_sha256))

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
    parser.add_argument(
        "--repayment-file",
        default=None,
        help="Path to repayments CSV (original agent-snapshot grain mode). "
        "Mutually exclusive with --loan-training-file; exactly one is required.",
    )
    parser.add_argument(
        "--loan-training-file",
        default=None,
        help="Path to the loan-level training export from "
        "data/loan_state_query_updated_materialized.txt (loan-level grain mode). "
        "Mutually exclusive with --repayment-file; exactly one is required.",
    )
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
        default=None,
        help="Inclusive upper bound for training rows (YYYY-MM-DD). Required "
        "only in original mode (--repayment-file); ignored in loan-level "
        "mode, where the loan-level file's own split column is authoritative.",
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
        "--champion",
        default="xgb",
        choices=["xgb", "lgb"],
        help="Champion model for policy placement and inference (default: xgb)",
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        default=False,
        help="Skip the 2000-sample bootstrap AUC comparison (faster dev runs)",
    )
    parser.add_argument(
        "--whitelist-file",
        default=None,
        help="Optional path to whitelist CSV for thin-file scorecard evaluation",
    )
    parser.add_argument(
        "--blacklist-file",
        default=None,
        help="Optional path to blacklist CSV for thin-file scorecard evaluation",
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

    if bool(args.repayment_file) == bool(args.loan_training_file):
        parser.error(
            "exactly one of --repayment-file (original mode) or "
            "--loan-training-file (loan-level mode) must be supplied"
        )
    if args.loan_training_file is None and args.train_cutoff is None:
        parser.error("--train-cutoff is required in original mode (--repayment-file)")

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    configure_root_level(level)

    try:
        run_pipeline(args)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
