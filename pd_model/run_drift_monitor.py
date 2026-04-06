"""
Drift monitoring CLI — designed to be called on a scheduled cadence
(cron, Airflow, GitHub Actions, etc.).

What it does
------------
1. Loads reference scored data (training-time snapshot) from artifacts dir
2. Loads the latest production snapshot (new scored file)
3. Runs PSI (score distribution) + CSI (per-feature distribution) checks
4. Checks thresholds and generates a structured alert payload
5. Writes a JSON alert report to the output dir
6. Optionally POSTs a Slack message if a webhook URL is provided

Typical cron entry (daily at 06:00 UTC)
----------------------------------------
::

    0 6 * * * python -m pd_model.run_drift_monitor \\
        --reference-file  artifacts/train_scored.csv \\
        --monitoring-file data/latest_scored.csv \\
        --artifacts-dir   artifacts/ \\
        --output-dir      monitoring/ \\
        --score-col       xgb_raw_score \\
        --slack-webhook   ${SLACK_WEBHOOK_URL}

Exit codes
----------
0 — no alerts or warnings only
1 — at least one CRITICAL alert (score PSI ≥ 0.25 or ≥1 feature CSI ≥ 0.25)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from pd_model.logging_config import configure_root_level, get_logger
from pd_model.monitoring.alert import check_drift_alerts, send_slack_alert, write_alert_report
from pd_model.monitoring.drift import run_drift_report

logger = get_logger(__name__)


def _load_feature_order(artifacts_dir: Path) -> list[str]:
    """Load feature_order.json from artifacts dir if available."""
    p = artifacts_dir / "feature_order.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return data.get("selected_features", data) if isinstance(data, dict) else data


def run_drift_monitor(
    reference_file: Path,
    monitoring_file: Path,
    artifacts_dir: Path,
    output_dir: Path,
    score_col: str = "xgb_raw_score",
    psi_warn: float = 0.10,
    psi_critical: float = 0.25,
    slack_webhook: str | None = None,
    report_date: str | None = None,
    bins: int = 10,
) -> dict:
    """
    Core drift monitoring logic (importable for testing / programmatic use).

    Returns the alert payload dict.
    """
    reference_df = pd.read_csv(reference_file)
    monitoring_df = pd.read_csv(monitoring_file)

    logger.info(
        "Drift monitor: reference=%d rows | monitoring=%d rows",
        len(reference_df), len(monitoring_df),
    )

    feature_cols = _load_feature_order(artifacts_dir)
    if not feature_cols:
        # Fall back to numeric columns present in both DataFrames
        feature_cols = [
            c for c in reference_df.select_dtypes(include="number").columns
            if c in monitoring_df.columns and c != score_col
        ]
        logger.warning(
            "feature_order.json not found — monitoring %d numeric columns",
            len(feature_cols),
        )

    drift_report = run_drift_report(
        reference_df=reference_df,
        monitoring_df=monitoring_df,
        feature_cols=feature_cols,
        score_col=score_col,
        bins=bins,
    )

    alerts = check_drift_alerts(
        drift_report,
        report_date=report_date,
        psi_warn=psi_warn,
        psi_critical=psi_critical,
    )

    # Write alert JSON
    output_dir = Path(output_dir)
    date_str = alerts["report_date"].replace("-", "")
    alert_path = output_dir / f"drift_alert_{date_str}.json"
    write_alert_report(alerts, output_path=alert_path)

    # Write CSI table alongside alert
    csi_path = output_dir / f"csi_table_{date_str}.csv"
    drift_report["csi_table"].to_csv(csi_path, index=False)
    logger.info("CSI table written → %s", csi_path)

    # Optional Slack notification
    if slack_webhook and alerts["has_warning"]:
        send_slack_alert(alerts["message"], webhook_url=slack_webhook)
    elif slack_webhook:
        logger.info("No warnings — Slack notification suppressed")

    logger.info("Drift monitor complete: has_critical=%s", alerts["has_critical"])
    return alerts


# ======================================================================== #
# CLI
# ======================================================================== #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PD Model Drift Monitor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--reference-file",  required=True, type=Path,
                   help="CSV of training/reference scored snapshot")
    p.add_argument("--monitoring-file", required=True, type=Path,
                   help="CSV of latest production scored snapshot")
    p.add_argument("--artifacts-dir",   required=True, type=Path,
                   help="Directory containing feature_order.json")
    p.add_argument("--output-dir",      required=True, type=Path,
                   help="Directory to write alert JSON and CSI table")
    p.add_argument("--score-col",       default="xgb_raw_score",
                   help="Score column name present in both CSVs")
    p.add_argument("--psi-warn",        type=float, default=0.10)
    p.add_argument("--psi-critical",    type=float, default=0.25)
    p.add_argument("--slack-webhook",   default=os.environ.get("SLACK_WEBHOOK"),
                   help="Slack incoming webhook URL (or set SLACK_WEBHOOK env var)")
    p.add_argument("--report-date",     default=None,
                   help="Override report date (YYYY-MM-DD); default: today")
    p.add_argument("--bins",            type=int, default=10,
                   help="Number of bins for PSI/CSI calculation")
    p.add_argument("--log-level",       default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_root_level(args.log_level)

    alerts = run_drift_monitor(
        reference_file=args.reference_file,
        monitoring_file=args.monitoring_file,
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
        score_col=args.score_col,
        psi_warn=args.psi_warn,
        psi_critical=args.psi_critical,
        slack_webhook=args.slack_webhook,
        report_date=args.report_date,
        bins=args.bins,
    )

    return 1 if alerts["has_critical"] else 0


if __name__ == "__main__":
    sys.exit(main())
