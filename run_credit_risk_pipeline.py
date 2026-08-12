"""
run_credit_risk_pipeline.py
============================
End-to-end CreditRisk pipeline: segmentation -> PD model -> credit limit engine.

Pipeline stages
---------------
1. Load three input files (transaction, loan, borrower).
2. Run the agent segmentation pipeline -> capacity_tier per agent.
3. Run the PD model inference pipeline -> cal_pd per agent.
4. Build engine features from the three CSVs.
5. Join cal_pd and capacity_tier onto the engine features DataFrame.
6. Run the credit limit engine -> assigned_limit + risk_tier + cal_pd per agent.
7. Optionally write the result to a CSV.

Input files
-----------
``--transaction-file``
    Agent profile snapshot -- MTN MoMo agent behavioural features:
    commission, account_balance, cash_in/out volumes, customer counts, etc.
    Read twice: once raw (agent_msisdn key) for PD model Phase 2.1 feature
    engineering, and once via the capacity loader (renamed to msisdn) for
    the engine's capacity cap.

``--loan-file``
    XtraFloat loan summary -- disbursement/repayment volumes and penalty
    counts over 1m/3m windows.  Fed to the engine's recent usage cap.

``--borrower-file``
    Borrower credit history -- on_time_repayment_rate, lifetime_default_rate,
    prior loan sizes, lifetime_loan_count.  Fed to the engine's prior
    exposure cap and risk signal fallback.

Usage
-----
::

    python run_credit_risk_pipeline.py \\
        --transaction-file  data/agent_profile_snapshot.csv \\
        --loan-file         data/loan_summary.csv \\
        --borrower-file     data/borrower_credit.csv \\
        --artifacts-dir     pd_model/artifacts/ \\
        --output            output/credit_risk_output.csv

All --*-file arguments accept CSV or tab-delimited text (auto-detected).
--artifacts-dir must contain the trained model artifacts produced by
``python -m pd_model.run_pipeline``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from extrafloat.engine.extrafloat_limit_engine_features import (
    build_extrafloat_limit_engine_features,
)
from extrafloat.engine.run_extrafloat_limit_engine import run_extrafloat_limit_engine
from extrafloat.io.extrafloat_data_loaders import (
    load_borrower_limit_features,
    load_loan_summary_recent_features,
    load_transaction_capacity_features,
)
from pd_model.exceptions import ArtifactVerificationError, DataAlignmentError, MissingArtifactError
from pd_model.logging_config import install_pii_filter
from pd_model.modeling.inference import run_inference_pipeline
from run_extrafloat_segmentation import run_extrafloat_segmentation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
install_pii_filter()  # protect root handler used by extrafloat/ and propagating loggers
logger = logging.getLogger("credit_risk_pipeline")

# Artifact files produced by pd_model.run_pipeline that must exist before scoring.
_REQUIRED_ARTIFACTS = [
    "xgb_model.joblib",
    "lgbm_model.joblib",
    "feature_order.json",
    "pd_calibration_map.csv",
    "xgb_policy_thresholds.csv",
    "lgb_policy_thresholds.csv",
    "transform_report.csv",
    "model_metadata.json",
]


def _check_artifacts(artifacts_dir: Path, allow_unverified: bool = False) -> None:
    """
    Raise MissingArtifactError early if any required PD model artifact is missing,
    then verify sha256 checksums against the values stored in model_metadata.json.

    Called before run_inference_pipeline() so failures surface with a clear
    message and the exact training command, rather than crashing deep inside
    load_artifacts() with a MissingArtifactError.

    Parameters
    ----------
    artifacts_dir    : directory to inspect
    allow_unverified : if True, missing or incomplete checksums produce a warning
                       instead of a RuntimeError.  Use ONLY for development or
                       testing with placeholder metadata.  Production inference
                       must never pass True here.
    """
    missing = [f for f in _REQUIRED_ARTIFACTS if not (artifacts_dir / f).exists()]
    if missing:
        raise MissingArtifactError(
            f"Missing PD model artifacts in {artifacts_dir}:\n" + "  " + ", ".join(missing) + "\n\n"
            "Train the model first:\n"
            "  python -m pd_model.run_pipeline \\\n"
            "      --train-file  <agent_snapshot_train.csv> \\\n"
            "      --val-file    <agent_snapshot_val.csv> \\\n"
            f"      --output-dir  {artifacts_dir}\n\n"
            "Or run:  make train"
        )

    # Verify sha256 checksums against model_metadata.json.
    meta_path = artifacts_dir / "model_metadata.json"
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        raise ArtifactVerificationError(
            f"[preflight] model_metadata.json is present but contains invalid JSON: {exc}\n"
            "Re-train to regenerate a valid metadata file."
        ) from exc

    stored_hashes = meta.get("artifact_sha256", {})
    if not stored_hashes:
        _unverified_warn_or_raise(
            allow_unverified,
            "[preflight] model_metadata.json contains no artifact_sha256 checksums.\n"
            "Re-train to generate checksums, or run with --allow-unverified-artifacts "
            "(development / testing only).",
        )
        return

    mismatches = []
    for fname in _REQUIRED_ARTIFACTS:
        if fname == "model_metadata.json":
            continue
        if fname not in stored_hashes:
            _unverified_warn_or_raise(
                allow_unverified,
                f"[preflight] No stored checksum for {fname} in model_metadata.json.\n"
                "Re-train to generate checksums, or run with --allow-unverified-artifacts "
                "(development / testing only).",
            )
            continue
        actual = hashlib.sha256((artifacts_dir / fname).read_bytes()).hexdigest()
        if actual != stored_hashes[fname]:
            mismatches.append(fname)

    if mismatches:
        raise ArtifactVerificationError(
            f"[preflight] Artifact checksum mismatch for: {', '.join(mismatches)}\n"
            "Artifacts may be corrupted, partially copied, or from a different training run.\n"
            "Re-train or restore from backup."
        )


def _unverified_warn_or_raise(allow_unverified: bool, message: str) -> None:
    """Emit a warning when allow_unverified=True; raise ArtifactVerificationError otherwise."""
    if allow_unverified:
        logger.warning("%s", message)
    else:
        raise ArtifactVerificationError(message)


def _norm_msisdn(s: pd.Series) -> pd.Series:
    """Normalise MSISDN strings: strip whitespace and remove trailing .0 suffixes.

    Uses StringDtype so pd.NA is preserved through chained str operations,
    keeping isna() accurate after normalisation.  Values that stringify to
    known null sentinels ("nan", "none", "<na>", "") are also masked to NA
    so the null-key guard catches them correctly.
    """
    out = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return out.mask(out.str.lower().isin({"", "nan", "none", "<na>"}))


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------


def run_credit_risk_pipeline(
    transaction_file: str | Path,
    loan_file: str | Path,
    borrower_file: str | Path,
    artifacts_dir: str | Path,
    repayment_file: str | Path | None = None,
    snapshot_date: str | int | None = None,
    champion: str = "xgb",
    keep_intermediate: bool = False,
    engine_config: dict | None = None,
    allow_unverified_artifacts: bool = False,
    scorecard_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Run the full PD model -> credit limit engine pipeline.

    Parameters
    ----------
    transaction_file           : Agent profile snapshot CSV -- MTN MoMo behavioural features
                                 (commission, account_balance, cash_in/out, customer counts, …).
                                 Read twice from the same path:
                                   - raw (agent_msisdn key) -> PD model Phase 2.1 feature engineering
                                   - via capacity loader (msisdn key) -> engine capacity cap
    loan_file                  : XtraFloat loan summary CSV -- disbursement/repayment
                                 volumes and penalty counts over 1m/3m windows.
                                 Fed to the engine recent usage cap.
    borrower_file              : Borrower credit history CSV -- on_time_repayment_rate,
                                 lifetime_default_rate, prior loan sizes, etc.
                                 Fed to the engine prior exposure cap.
    artifacts_dir              : directory containing trained PD model artifacts
                                 (xgb_model.joblib, lgbm_model.joblib, pd_calibration_map.csv, …)
    repayment_file             : optional repayment history CSV for Phase 2.2 PD features.
                                 Agents without repayment rows are treated as thin-file.
    champion                   : "xgb" or "lgb" -- which model's cal_pd feeds into the engine
    keep_intermediate          : if True, all engine intermediate columns are retained
    engine_config              : optional dict to override DEFAULT_CAP_CONFIG values
    allow_unverified_artifacts : if True, missing or incomplete artifact checksums produce a
                                 warning instead of a RuntimeError.  Use ONLY for development
                                 or testing.  Never set True in production.
    scorecard_path             : optional path to the capacity scorecard JSON produced by
                                 calibrate_scorecard.py.  When provided, the segmentation stage
                                 produces deterministic capacity_tier values.  When omitted,
                                 segmentation runs in degraded mode (allow_missing_scorecard=True)
                                 and capacity_tier is derived from heuristic scoring only.

    Returns
    -------
    DataFrame with one row per agent:
        msisdn, assigned_limit, risk_tier, cal_pd, final_decision_reason,
        and (if keep_intermediate=True) all intermediate cap and feature columns.
    """
    artifacts_dir = Path(artifacts_dir)

    # -- Stage 1: Load raw data ----------------------------------------------
    logger.info("Stage 1: loading raw data")
    # Raw read preserves agent_msisdn -- required by PD model Phase 2.1
    df_agent = pd.read_csv(transaction_file, sep=None, engine="python")
    # Loader renames agent_msisdn -> msisdn for the engine
    df_txn = load_transaction_capacity_features(transaction_file)
    df_loan = load_loan_summary_recent_features(loan_file)
    df_borrower = load_borrower_limit_features(borrower_file)
    logger.info(
        "Loaded -- agents: %d rows | txn: %d rows | loan: %d rows | borrower: %d rows",
        len(df_agent),
        len(df_txn),
        len(df_loan),
        len(df_borrower),
    )

    # Tag both dataframes with snapshot_dt so Phase 2.2 can join them.
    # Mirrors what run_pipeline._load_snapshot() does during training.
    if snapshot_date is not None:
        snap_ts = pd.to_datetime(str(snapshot_date), format="%Y%m%d", errors="coerce")
        if pd.isna(snap_ts):
            snap_ts = pd.to_datetime(str(snapshot_date), errors="coerce")
        df_agent["snapshot_dt"] = snap_ts

    repayment_df = None
    if repayment_file is not None:
        repayment_df = pd.read_csv(repayment_file, sep=None, engine="python")
        logger.info("Repayment file loaded: %d rows", len(repayment_df))
        if snapshot_date is not None:
            if "snapshot_dt" not in repayment_df.columns and "tbl_dt" not in repayment_df.columns:
                repayment_df["snapshot_dt"] = snap_ts

    # -- Stage 2: Agent segmentation -----------------------------------------
    logger.info("Stage 2: running agent segmentation")
    seg_config: dict = {
        "scoring": {
            "scorecard_path": str(scorecard_path) if scorecard_path else "",
            "allow_missing_scorecard": scorecard_path is None,
        },
        "clustering": {"enable_diagnostics": False},
    }
    try:
        seg_df = run_extrafloat_segmentation(df_agent, config=seg_config)
        seg_cols = ["agent_msisdn"] + [
            c for c in (
                "capacity_score",
                "capacity_tier",
                "capacity_tier_raw",
                "capacity_safety_flags",
                # Anomaly flags: independent of capacity_tier; an agent can be
                # Gold + is_anomaly=True simultaneously. Never changes tier.
                "is_anomaly",
                "is_global_anomaly",
                "is_local_anomaly",
                "lof_score",
            )
            if c in seg_df.columns
        ]
        seg_out = seg_df[seg_cols].copy()
        n_anomalies = int(seg_out["is_anomaly"].sum()) if "is_anomaly" in seg_out.columns else 0
        logger.info(
            "Segmentation complete: %d agents | capacity_tier distribution: %s | anomalies: %d",
            len(seg_out),
            seg_out["capacity_tier"].value_counts().to_dict() if "capacity_tier" in seg_out.columns else "n/a",
            n_anomalies,
        )
    except ValueError as exc:
        logger.warning(
            "Stage 2: segmentation skipped — %s. "
            "capacity_tier will be absent from output. "
            "Ensure the transaction file contains: agent_msisdn, commission, "
            "cash_out_vol_1m, cash_out_value_1m, cash_in_value_1m.",
            exc,
        )
        seg_out = pd.DataFrame(columns=["agent_msisdn"])

    # -- Stage 3: PD model scoring -------------------------------------------
    logger.info("Stage 3: running PD model inference (champion=%s)", champion)
    _check_artifacts(artifacts_dir, allow_unverified=allow_unverified_artifacts)
    pd_scored = run_inference_pipeline(
        df_raw=df_agent,
        artifacts_dir=artifacts_dir,
        repayment_df=repayment_df,
        champion=champion,
    )
    logger.info(
        "PD model scored %d agents | cal_pd mean=%.4f",
        len(pd_scored),
        pd_scored["cal_pd"].mean() if "cal_pd" in pd_scored.columns else float("nan"),
    )

    # -- Stage 4: Engine feature engineering --------------------------------
    logger.info("Stage 4: building engine features")
    features_df = build_extrafloat_limit_engine_features(
        borrower_limit_df=df_borrower,
        transaction_capacity_df=df_txn,
        loan_summary_df=df_loan,
    )
    logger.info("Engine features: %d agents", len(features_df))

    # -- Stage 5: Join cal_pd and capacity_tier onto engine features ---------
    # PD model key: agent_msisdn  |  engine key: msisdn  (same identifier)
    logger.info("Stage 5: joining PD scores and segmentation onto engine features")
    pd_join_cols = [c for c in ["agent_msisdn", "cal_pd", "thin_file_flag"] if c in pd_scored.columns]
    pd_join = pd_scored[pd_join_cols].rename(columns={"agent_msisdn": "msisdn"})

    # Normalise MSISDN on both sides before joining to prevent .0-suffix /
    # whitespace mismatches that would silently leave agents without cal_pd.
    features_df["msisdn"] = _norm_msisdn(features_df["msisdn"])
    pd_join["msisdn"] = _norm_msisdn(pd_join["msisdn"])

    # Guard: null join keys cannot be meaningfully joined.
    null_engine = features_df["msisdn"].isna().sum()
    if null_engine > 0:
        raise DataAlignmentError(
            f"{null_engine} null msisdn values in engine features -- cannot join. "
            "Check load_transaction_capacity_features / build_extrafloat_limit_engine_features."
        )
    null_pd = pd_join["msisdn"].isna().sum()
    if null_pd > 0:
        logger.warning("PD output contains %d null agent_msisdn rows -- dropping before join", null_pd)
        pd_join = pd_join.dropna(subset=["msisdn"])

    # Guard: duplicates on either side would silently expand rows (many-to-many).
    dup_engine = features_df["msisdn"].duplicated().sum()
    if dup_engine > 0:
        raise DataAlignmentError(
            f"Duplicate msisdn in engine features ({dup_engine} rows). "
            "Check build_extrafloat_limit_engine_features for fan-out."
        )
    dup_pd = pd_join["msisdn"].duplicated().sum()
    if dup_pd > 0:
        raise DataAlignmentError(
            f"Duplicate agent_msisdn in PD output ({dup_pd} rows). "
            "Check run_inference_pipeline for duplicate agents."
        )

    pre_join = len(features_df)
    features_df = features_df.merge(pd_join, on="msisdn", how="left", validate="one_to_one")

    # Left join must never expand rows.
    if len(features_df) != pre_join:
        raise DataAlignmentError(
            f"Stage 4 join expanded rows: pre={pre_join}, post={len(features_df)}. "
            "Duplicate msisdn values may have bypassed the pre-join checks."
        )

    covered = features_df["cal_pd"].notna().sum()
    logger.info(
        "PD join: %d / %d agents have cal_pd (%.1f%% coverage); %d will use the 7-signal fallback",
        covered,
        pre_join,
        100.0 * covered / max(pre_join, 1),
        pre_join - covered,
    )

    # -- Thin-file reconciliation: PD model is authoritative ----------------
    # The PD model's thin_file_flag determines which scoring path was used
    # (LR vs XGBoost). The engine's is_thin_file is computed independently
    # from total_loans < 3 (borrower credit history). Align them so the
    # engine's cap weighting matches the scoring path actually applied.
    if "thin_file_flag" in features_df.columns:
        if "is_thin_file" in features_df.columns:
            disagree = (
                features_df["thin_file_flag"].fillna(0).astype(int)
                != features_df["is_thin_file"].fillna(0).astype(int)
            ).sum()
            logger.info(
                "thin-file reconciliation: %d/%d agents differ between PD model "
                "and engine definitions; overriding is_thin_file with PD model thin_file_flag",
                int(disagree),
                len(features_df),
            )
        features_df["is_thin_file"] = features_df["thin_file_flag"].fillna(0).astype(int)

    # Join segmentation output (capacity_tier, capacity_score) onto features_df.
    # Left join so engine agents with no segmentation row are unaffected.
    seg_join = seg_out.copy()
    seg_join["agent_msisdn"] = seg_join["agent_msisdn"].astype(str).str.strip()
    seg_join = seg_join.rename(columns={"agent_msisdn": "msisdn"})
    seg_join["msisdn"] = _norm_msisdn(seg_join["msisdn"])
    seg_cols_to_join = [c for c in seg_join.columns if c != "msisdn"]
    existing_seg_cols = [c for c in seg_cols_to_join if c in features_df.columns]
    if existing_seg_cols:
        features_df = features_df.drop(columns=existing_seg_cols)
    features_df = features_df.merge(
        seg_join[["msisdn"] + seg_cols_to_join], on="msisdn", how="left"
    )
    matched = features_df["capacity_tier"].notna().sum() if "capacity_tier" in features_df.columns else 0
    logger.info(
        "Segmentation join: %d/%d engine agents matched a capacity_tier",
        int(matched), len(features_df),
    )

    # -- Stage 6: Run credit limit engine -----------------------------------
    logger.info("Stage 6: running credit limit engine")
    result_df = run_extrafloat_limit_engine(
        features_df,
        config=engine_config,
        keep_intermediate=keep_intermediate,
    )
    logger.info(
        "Engine complete: %d agents | mean_limit=%.0f | risk_tier distribution:\n%s",
        len(result_df),
        result_df["assigned_limit"].mean() if "assigned_limit" in result_df.columns else float("nan"),
        result_df["risk_tier"].value_counts().to_string() if "risk_tier" in result_df.columns else "n/a",
    )

    # -- Stage 7: Re-attach segmentation columns to engine output -------------
    # The engine does not pass through extra columns, so join capacity_tier,
    # is_anomaly, is_global_anomaly, is_local_anomaly back from seg_out.
    seg_passthrough_cols = [
        c for c in (
            "capacity_tier", "capacity_score", "capacity_tier_raw",
            "is_anomaly", "is_global_anomaly", "is_local_anomaly",
        )
        if c in seg_out.columns
    ]
    if seg_passthrough_cols and "agent_msisdn" in seg_out.columns:
        seg_reattach = seg_out[["agent_msisdn"] + seg_passthrough_cols].copy()
        seg_reattach = seg_reattach.rename(columns={"agent_msisdn": "msisdn"})
        seg_reattach["msisdn"] = _norm_msisdn(seg_reattach["msisdn"].astype(str).str.strip())
        result_df = result_df.merge(seg_reattach, on="msisdn", how="left")
        logger.info(
            "Stage 7: re-attached segmentation columns %s to output",
            seg_passthrough_cols,
        )

    # -- Stage 8: Audit columns ---------------------------------------------
    # score_source distinguishes legitimate population misses (agents not in
    # PD output -> "7_signal_fallback") from calibration failures that now
    # propagate as exceptions rather than silent NaN.
    result_df["score_source"] = np.where(result_df["cal_pd"].notna(), "pd_model", "7_signal_fallback")
    result_df["scored_at"] = _dt.datetime.utcnow().isoformat() + "Z"

    return result_df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CreditRisk pipeline: PD model -> credit limit engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--transaction-file",
        required=True,
        help="Agent profile snapshot CSV -- read raw for PD model, via loader for engine capacity cap",
    )
    p.add_argument("--loan-file", required=True, help="XtraFloat loan summary CSV (engine recent usage cap)")
    p.add_argument(
        "--borrower-file", required=True, help="Borrower credit history CSV (engine prior exposure cap)"
    )
    p.add_argument("--artifacts-dir", required=True, help="Trained PD model artifacts directory")
    p.add_argument(
        "--repayment-file",
        default=None,
        help="Optional repayment history CSV for PD model Phase 2.2 features",
    )
    p.add_argument(
        "--snapshot-date",
        default=None,
        help="Snapshot date for the transaction file (YYYYMMDD). Required when using --repayment-file "
             "so Phase 2.2 can join the two files on snapshot_dt.",
    )
    p.add_argument(
        "--scorecard-path",
        default=None,
        help="Path to capacity scorecard JSON (produced by calibrate_scorecard.py). "
             "When omitted, segmentation runs in degraded mode without deterministic tier scoring.",
    )
    p.add_argument("--output", default=None, help="Output CSV path (omit to print summary only)")
    p.add_argument("--champion", default="xgb", choices=["xgb", "lgb"])
    p.add_argument(
        "--keep-intermediate", action="store_true", help="Retain all intermediate engine columns in output"
    )
    p.add_argument(
        "--allow-unverified-artifacts",
        action="store_true",
        help=(
            "Skip sha256 checksum enforcement when model_metadata.json has no "
            "stored hashes.  Use ONLY for development or testing with placeholder "
            "artifacts.  Never use in production."
        ),
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    result = run_credit_risk_pipeline(
        transaction_file=args.transaction_file,
        loan_file=args.loan_file,
        borrower_file=args.borrower_file,
        artifacts_dir=args.artifacts_dir,
        repayment_file=args.repayment_file,
        snapshot_date=args.snapshot_date,
        champion=args.champion,
        keep_intermediate=args.keep_intermediate,
        allow_unverified_artifacts=args.allow_unverified_artifacts,
        scorecard_path=args.scorecard_path,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        logger.info("Output written to %s (%d rows, %d columns)", out_path, len(result), len(result.columns))

    # Always print a screen summary regardless of whether --output was given
    preview_cols = ["msisdn", "assigned_limit", "risk_tier", "cal_pd", "thin_file_flag", "capacity_tier", "is_anomaly", "final_decision_reason", "score_source"]
    display = result[[c for c in preview_cols if c in result.columns]]
    print("\n=== Credit Risk Pipeline Output ===")
    print(f"Agents scored: {len(display)}")
    if "risk_tier" in display.columns:
        print("\nRisk tier distribution:")
        print(display["risk_tier"].value_counts().to_string())
    if "assigned_limit" in display.columns:
        print(f"\nAssigned limit  mean={display['assigned_limit'].mean():,.0f}  "
              f"min={display['assigned_limit'].min():,.0f}  "
              f"max={display['assigned_limit'].max():,.0f}")
    if "cal_pd" in display.columns:
        print(f"Calibrated PD   mean={display['cal_pd'].mean():.4f}  "
              f"min={display['cal_pd'].min():.4f}  "
              f"max={display['cal_pd'].max():.4f}")
    if "thin_file_flag" in display.columns:
        thin_n = display["thin_file_flag"].sum()
        thick_n = len(display) - thin_n
        print(f"\nScoring model:  thick-file (XGBoost)={thick_n:,}  thin-file (LR)={thin_n:,}")
    print("\nSample output (first 20 rows):")
    print(display.head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
