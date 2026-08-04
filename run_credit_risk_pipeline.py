"""
run_credit_risk_pipeline.py
============================
End-to-end CreditRisk pipeline: PD model → credit limit engine.

Pipeline stages
---------------
1. Load three input files (transaction, loan, borrower).
2. Run the PD model inference pipeline → cal_pd per agent.
3. Build engine features from the three CSVs.
4. Join cal_pd onto the engine features DataFrame (on agent_msisdn / msisdn).
5. Run the credit limit engine → assigned_limit + risk_tier + cal_pd per agent.
6. Optionally write the result to a CSV.

Input files
-----------
``--transaction-file``
    Agent profile snapshot — MTN MoMo agent behavioural features:
    commission, account_balance, cash_in/out volumes, customer counts, etc.
    Read twice: once raw (agent_msisdn key) for PD model Phase 2.1 feature
    engineering, and once via the capacity loader (renamed to msisdn) for
    the engine's capacity cap.

``--loan-file``
    XtraFloat loan summary — disbursement/repayment volumes and penalty
    counts over 1m/3m windows.  Fed to the engine's recent usage cap.

``--borrower-file``
    Borrower credit history — on_time_repayment_rate, lifetime_default_rate,
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
from pd_model.exceptions import ArtifactVerificationError, DataAlignmentError
from pd_model.modeling.inference import run_inference_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
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
    Raise FileNotFoundError early if any required PD model artifact is missing,
    then verify sha256 checksums against the values stored in model_metadata.json.

    Called before run_inference_pipeline() so failures surface with a clear
    message and the exact training command, rather than crashing deep inside
    load_artifacts() with a generic FileNotFoundError.

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
        raise FileNotFoundError(
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


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


def run_credit_risk_pipeline(
    transaction_file: str | Path,
    loan_file: str | Path,
    borrower_file: str | Path,
    artifacts_dir: str | Path,
    repayment_file: str | Path | None = None,
    champion: str = "xgb",
    keep_intermediate: bool = False,
    engine_config: dict | None = None,
    allow_unverified_artifacts: bool = False,
) -> pd.DataFrame:
    """
    Run the full PD model → credit limit engine pipeline.

    Parameters
    ----------
    transaction_file           : Agent profile snapshot CSV — MTN MoMo behavioural features
                                 (commission, account_balance, cash_in/out, customer counts, …).
                                 Read twice from the same path:
                                   - raw (agent_msisdn key) → PD model Phase 2.1 feature engineering
                                   - via capacity loader (msisdn key) → engine capacity cap
    loan_file                  : XtraFloat loan summary CSV — disbursement/repayment
                                 volumes and penalty counts over 1m/3m windows.
                                 Fed to the engine recent usage cap.
    borrower_file              : Borrower credit history CSV — on_time_repayment_rate,
                                 lifetime_default_rate, prior loan sizes, etc.
                                 Fed to the engine prior exposure cap.
    artifacts_dir              : directory containing trained PD model artifacts
                                 (xgb_model.joblib, lgbm_model.joblib, pd_calibration_map.csv, …)
    repayment_file             : optional repayment history CSV for Phase 2.2 PD features.
                                 Agents without repayment rows are treated as thin-file.
    champion                   : "xgb" or "lgb" — which model's cal_pd feeds into the engine
    keep_intermediate          : if True, all engine intermediate columns are retained
    engine_config              : optional dict to override DEFAULT_CAP_CONFIG values
    allow_unverified_artifacts : if True, missing or incomplete artifact checksums produce a
                                 warning instead of a RuntimeError.  Use ONLY for development
                                 or testing.  Never set True in production.

    Returns
    -------
    DataFrame with one row per agent:
        msisdn, assigned_limit, risk_tier, cal_pd, final_decision_reason,
        and (if keep_intermediate=True) all intermediate cap and feature columns.
    """
    artifacts_dir = Path(artifacts_dir)

    # ── Stage 1: Load raw data ──────────────────────────────────────────────
    logger.info("Stage 1: loading raw data")
    # Raw read preserves agent_msisdn — required by PD model Phase 2.1
    df_agent = pd.read_csv(transaction_file, sep=None, engine="python")
    # Loader renames agent_msisdn → msisdn for the engine
    df_txn = load_transaction_capacity_features(transaction_file)
    df_loan = load_loan_summary_recent_features(loan_file)
    df_borrower = load_borrower_limit_features(borrower_file)
    logger.info(
        "Loaded — agents: %d rows | txn: %d rows | loan: %d rows | borrower: %d rows",
        len(df_agent),
        len(df_txn),
        len(df_loan),
        len(df_borrower),
    )

    repayment_df = None
    if repayment_file is not None:
        repayment_df = pd.read_csv(repayment_file, sep=None, engine="python")
        logger.info("Repayment file loaded: %d rows", len(repayment_df))

    # ── Stage 2: PD model scoring ───────────────────────────────────────────
    logger.info("Stage 2: running PD model inference (champion=%s)", champion)
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

    # ── Stage 3: Engine feature engineering ────────────────────────────────
    logger.info("Stage 3: building engine features")
    features_df = build_extrafloat_limit_engine_features(
        df_txn=df_txn,
        df_loan=df_loan,
        df_borrower=df_borrower,
    )
    logger.info("Engine features: %d agents", len(features_df))

    # ── Stage 4: Join cal_pd onto engine features ───────────────────────────
    # PD model key: agent_msisdn  |  engine key: msisdn  (same identifier)
    logger.info("Stage 4: joining PD scores onto engine features")
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
            f"{null_engine} null msisdn values in engine features — cannot join. "
            "Check load_transaction_capacity_features / build_extrafloat_limit_engine_features."
        )
    null_pd = pd_join["msisdn"].isna().sum()
    if null_pd > 0:
        logger.warning("PD output contains %d null agent_msisdn rows — dropping before join", null_pd)
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

    # ── Stage 5: Run credit limit engine ───────────────────────────────────
    logger.info("Stage 5: running credit limit engine")
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

    # ── Stage 6: Audit columns ─────────────────────────────────────────────
    # score_source distinguishes legitimate population misses (agents not in
    # PD output → "7_signal_fallback") from calibration failures that now
    # propagate as exceptions rather than silent NaN.
    result_df["score_source"] = np.where(result_df["cal_pd"].notna(), "pd_model", "7_signal_fallback")
    result_df["scored_at"] = _dt.datetime.utcnow().isoformat() + "Z"

    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CreditRisk pipeline: PD model → credit limit engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--transaction-file",
        required=True,
        help="Agent profile snapshot CSV — read raw for PD model, via loader for engine capacity cap",
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
        champion=args.champion,
        keep_intermediate=args.keep_intermediate,
        allow_unverified_artifacts=args.allow_unverified_artifacts,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        logger.info("Output written to %s (%d rows, %d columns)", out_path, len(result), len(result.columns))
    else:
        print(
            result[["msisdn", "assigned_limit", "risk_tier", "cal_pd", "final_decision_reason"]].to_string(
                index=False
            )
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
