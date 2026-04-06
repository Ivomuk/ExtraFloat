"""Tests for pd_model.monitoring.alert."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pd_model.monitoring.alert import check_drift_alerts, write_alert_report, _classify


def _make_drift_report(score_psi: float = 0.05, feat_csi: float = 0.05):
    csi_table = pd.DataFrame({
        "feature": ["feat_a", "feat_b"],
        "csi": [feat_csi, feat_csi * 0.5],
        "stability": ["stable", "stable"],
    })
    return {
        "score_psi": score_psi,
        "score_stability": "stable",
        "csi_table": csi_table,
        "n_features_stable": 2,
        "n_features_moderate": 0,
        "n_features_significant": 0,
        "n_features_monitored": 2,
    }


class TestClassify:
    def test_ok(self):
        assert _classify(0.05, 0.10, 0.25) == "ok"

    def test_warning(self):
        assert _classify(0.15, 0.10, 0.25) == "warning"

    def test_critical(self):
        assert _classify(0.30, 0.10, 0.25) == "critical"

    def test_nan_returns_unknown(self):
        assert _classify(np.nan, 0.10, 0.25) == "unknown"


class TestCheckDriftAlerts:
    def test_no_alerts_when_stable(self):
        report = _make_drift_report(score_psi=0.02, feat_csi=0.02)
        alerts = check_drift_alerts(report, report_date="2026-01-01")
        assert not alerts["has_warning"]
        assert not alerts["has_critical"]

    def test_has_warning_when_psi_moderate(self):
        report = _make_drift_report(score_psi=0.15)
        alerts = check_drift_alerts(report, report_date="2026-01-01")
        assert alerts["has_warning"]

    def test_has_critical_when_psi_high(self):
        report = _make_drift_report(score_psi=0.30)
        alerts = check_drift_alerts(report, report_date="2026-01-01")
        assert alerts["has_critical"]
        assert alerts["score_level"] == "critical"

    def test_feature_critical_counted(self):
        report = _make_drift_report(score_psi=0.02, feat_csi=0.30)
        alerts = check_drift_alerts(report, report_date="2026-01-01")
        assert alerts["n_features_critical"] >= 1
        assert alerts["has_critical"]

    def test_message_is_string(self):
        report = _make_drift_report()
        alerts = check_drift_alerts(report, report_date="2026-01-01")
        assert isinstance(alerts["message"], str)
        assert "2026-01-01" in alerts["message"]

    def test_top_drifted_features_populated(self):
        report = _make_drift_report(feat_csi=0.15)
        alerts = check_drift_alerts(report, report_date="2026-01-01", top_n_features=2)
        assert len(alerts["top_drifted_features"]) == 2

    def test_report_date_defaults_to_today(self):
        report = _make_drift_report()
        alerts = check_drift_alerts(report)
        assert alerts["report_date"] is not None
        assert len(alerts["report_date"]) == 10   # YYYY-MM-DD


class TestWriteAlertReport:
    def test_writes_json(self, tmp_path):
        report = _make_drift_report()
        alerts = check_drift_alerts(report, report_date="2026-01-01")
        out = tmp_path / "alert.json"
        write_alert_report(alerts, output_path=out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "report_date" in data
        assert "has_critical" in data

    def test_raw_report_excluded(self, tmp_path):
        report = _make_drift_report()
        alerts = check_drift_alerts(report, report_date="2026-01-01")
        out = tmp_path / "alert.json"
        write_alert_report(alerts, output_path=out)
        data = json.loads(out.read_text())
        assert "raw_report" not in data
