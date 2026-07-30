"""
Alert generation from drift reports.

Checks PSI/CSI against configurable thresholds and surfaces alerts via:
  - Structured JSON alert file (always written when output_path provided)
  - Slack webhook POST (optional — pass webhook_url)
  - Structured log messages (always)

Typical usage in a scheduled job
---------------------------------
::

    from pd_model.monitoring.drift import run_drift_report
    from pd_model.monitoring.alert import check_drift_alerts, write_alert_report, send_slack_alert

    report  = run_drift_report(reference_df, monitoring_df, feature_cols, score_col)
    alerts  = check_drift_alerts(report, report_date="2026-01-15")
    write_alert_report(alerts, output_path=Path("monitoring/alerts_2026-01-15.json"))
    if alerts["has_critical"]:
        send_slack_alert(alerts["message"], webhook_url=os.environ["SLACK_WEBHOOK"])
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from pd_model.logging_config import get_logger

logger = get_logger(__name__)

# Default thresholds (same as drift.py conventions)
_WARN_THRESHOLD = 0.10
_CRITICAL_THRESHOLD = 0.25


# ======================================================================== #
# Alert checking
# ======================================================================== #

def check_drift_alerts(
    drift_report: dict[str, Any],
    report_date: str | None = None,
    psi_warn: float = _WARN_THRESHOLD,
    psi_critical: float = _CRITICAL_THRESHOLD,
    csi_warn: float = _WARN_THRESHOLD,
    csi_critical: float = _CRITICAL_THRESHOLD,
    top_n_features: int = 5,
) -> dict[str, Any]:
    """
    Inspect a drift report and produce a structured alert payload.

    Parameters
    ----------
    drift_report   : output of ``run_drift_report``
    report_date    : ISO date string for the monitoring snapshot (default: today)
    psi_warn       : PSI threshold for WARNING level (default 0.10)
    psi_critical   : PSI threshold for CRITICAL level (default 0.25)
    csi_warn       : CSI threshold for WARNING level per feature
    csi_critical   : CSI threshold for CRITICAL level per feature
    top_n_features : number of most-drifted features to include in the message

    Returns
    -------
    dict with keys:
        report_date, has_warning, has_critical,
        score_psi, score_level,
        n_features_critical, n_features_warning,
        top_drifted_features  (list of dicts),
        message               (human-readable summary string),
        raw_report            (the input drift_report)
    """
    report_date = report_date or datetime.utcnow().strftime("%Y-%m-%d")
    score_psi = drift_report.get("score_psi", np.nan)
    csi_table = drift_report.get("csi_table")

    # Score-level classification
    score_level = _classify(score_psi, psi_warn, psi_critical)

    # Feature-level counts from the CSI table
    n_feat_critical = 0
    n_feat_warning = 0
    top_drifted: list[dict] = []

    if csi_table is not None and len(csi_table) > 0:
        for _, row in csi_table.iterrows():
            csi_val = row.get("csi", np.nan)
            level = _classify(csi_val, csi_warn, csi_critical)
            if level == "critical":
                n_feat_critical += 1
            elif level == "warning":
                n_feat_warning += 1

        top_rows = csi_table.head(top_n_features)
        for _, row in top_rows.iterrows():
            csi_val = row.get("csi", np.nan)
            top_drifted.append({
                "feature": str(row.get("feature", "")),
                "csi": round(float(csi_val), 4) if not np.isnan(csi_val) else None,
                "stability": str(row.get("stability", "unknown")),
            })

    has_warning = score_level in ("warning", "critical") or n_feat_warning > 0 or n_feat_critical > 0
    has_critical = score_level == "critical" or n_feat_critical > 0

    message = _format_message(
        report_date=report_date,
        score_psi=score_psi,
        score_level=score_level,
        n_feat_critical=n_feat_critical,
        n_feat_warning=n_feat_warning,
        n_feat_monitored=drift_report.get("n_features_monitored", 0),
        top_drifted=top_drifted,
        has_critical=has_critical,
    )

    level_log = logger.warning if has_critical else (logger.info if has_warning else logger.info)
    level_log("drift_alert [%s]: score_psi=%.4f (%s) | feat_critical=%d | feat_warning=%d",
              report_date, score_psi if not np.isnan(score_psi) else -1,
              score_level, n_feat_critical, n_feat_warning)

    return {
        "report_date": report_date,
        "has_warning": has_warning,
        "has_critical": has_critical,
        "score_psi": float(score_psi) if not np.isnan(score_psi) else None,
        "score_level": score_level,
        "n_features_critical": n_feat_critical,
        "n_features_warning": n_feat_warning,
        "top_drifted_features": top_drifted,
        "message": message,
        "raw_report": drift_report,
    }


# ======================================================================== #
# Output sinks
# ======================================================================== #

def write_alert_report(
    alerts: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Persist the alert payload as JSON.

    The ``raw_report`` key (which contains a DataFrame) is excluded to keep
    the file portable — use ``drift_report["csi_table"].to_csv(...)`` separately
    if the full CSI table is needed alongside the alert.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {k: v for k, v in alerts.items() if k != "raw_report"}
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Alert report written → %s", output_path)


def send_slack_alert(message: str, webhook_url: str) -> bool:
    """
    POST a message to a Slack incoming webhook.

    Parameters
    ----------
    message     : plain-text or mrkdwn-formatted string
    webhook_url : Slack incoming webhook URL

    Returns
    -------
    True if POST succeeded (HTTP 200), False otherwise.
    Failures are logged as warnings and do not raise.
    """
    try:
        import urllib.request
        payload = json.dumps({"text": message}).encode()
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            success = resp.status == 200
        if success:
            logger.info("Slack alert sent successfully")
        else:
            logger.warning("Slack webhook returned non-200 status")
        return success
    except Exception as exc:
        logger.warning("Slack alert failed — continuing without notification: %s", exc)
        return False


# ======================================================================== #
# Helpers
# ======================================================================== #

def _classify(value: float, warn: float, critical: float) -> str:
    if np.isnan(value):
        return "unknown"
    if value >= critical:
        return "critical"
    if value >= warn:
        return "warning"
    return "ok"


def _format_message(
    report_date: str,
    score_psi: float,
    score_level: str,
    n_feat_critical: int,
    n_feat_warning: int,
    n_feat_monitored: int,
    top_drifted: list[dict],
    has_critical: bool,
) -> str:
    header = f"[{'CRITICAL' if has_critical else 'WARNING' if n_feat_warning else 'OK'}] "
    header += f"PD Model Drift Report — {report_date}"

    psi_str = f"{score_psi:.4f}" if score_psi is not None and not np.isnan(score_psi) else "n/a"
    lines = [
        header,
        f"  Score PSI : {psi_str} ({score_level.upper()})",
        f"  Features  : {n_feat_monitored} monitored | "
        f"{n_feat_critical} critical | {n_feat_warning} warning",
    ]

    if top_drifted:
        lines.append("  Top drifted features:")
        for row in top_drifted:
            csi_str = f"{row['csi']:.4f}" if row["csi"] is not None else "n/a"
            lines.append(f"    • {row['feature']}: CSI={csi_str} ({row['stability']})")

    return "\n".join(lines)
