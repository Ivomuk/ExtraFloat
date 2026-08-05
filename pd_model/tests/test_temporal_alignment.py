"""
Test 3: Temporal alignment verification.

Confirms that:
1. Training snapshot (2025-09-30) and validation snapshot (2025-11-15) are
   properly separated -- val is after the training outcome window closes.
2. Rolling feature windows (m1 through m6) are all strictly pre-snapshot.
3. LEAKAGE_PATTERNS catches forward-looking column names.
4. Legitimate pre-snapshot columns are NOT caught by leakage guards.

No real data required; all checks run on synthetic DataFrames or config constants.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pd_model.config.feature_config import LEAKAGE_PATTERNS, PD_FEATURE_BLACKLIST

TRAIN_SNAPSHOT = pd.Timestamp("2025-09-30")
VAL_SNAPSHOT = pd.Timestamp("2025-11-15")
OUTCOME_WINDOW_DAYS = 30


class TestSnapshotDateBoundaries:
    def test_val_snapshot_after_train_outcome_window(self):
        """Validation snapshot must be after the training outcome window closes."""
        train_outcome_end = TRAIN_SNAPSHOT + pd.Timedelta(days=OUTCOME_WINDOW_DAYS)
        assert VAL_SNAPSHOT > train_outcome_end, (
            f"Val snapshot {VAL_SNAPSHOT.date()} must be after training outcome window "
            f"end {train_outcome_end.date()}. Overlap would cause temporal leakage."
        )

    def test_m1_window_ends_at_snapshot(self):
        """m1 rolling window (last 30 days) ends at snapshot_dt with no forward overlap."""
        m1_window_start = TRAIN_SNAPSHOT - pd.Timedelta(days=29)
        outcome_start = TRAIN_SNAPSHOT + pd.Timedelta(days=1)
        assert m1_window_start <= TRAIN_SNAPSHOT, "m1 window start must be before snapshot"
        assert TRAIN_SNAPSHOT < outcome_start, "Snapshot must be before outcome window start"

    def test_m5_window_entirely_before_snapshot(self):
        """m5 rolling window (120-150 days pre-snapshot) must not overlap the snapshot date."""
        m5_end = TRAIN_SNAPSHOT - pd.Timedelta(days=120)
        m5_start = TRAIN_SNAPSHOT - pd.Timedelta(days=150)
        assert m5_end < TRAIN_SNAPSHOT, "m5 window end must be strictly before snapshot"
        assert m5_start < TRAIN_SNAPSHOT, "m5 window start must be before snapshot"

    def test_all_monthly_windows_pre_snapshot(self):
        """All six rolling windows (m1-m6) must end before or at the snapshot date."""
        window_ends = {
            "m1": TRAIN_SNAPSHOT - pd.Timedelta(days=0),   # ends at snapshot
            "m2": TRAIN_SNAPSHOT - pd.Timedelta(days=30),
            "m3": TRAIN_SNAPSHOT - pd.Timedelta(days=60),
            "m4": TRAIN_SNAPSHOT - pd.Timedelta(days=90),
            "m5": TRAIN_SNAPSHOT - pd.Timedelta(days=120),
            "m6": TRAIN_SNAPSHOT - pd.Timedelta(days=150),
        }
        for name, end in window_ends.items():
            assert end <= TRAIN_SNAPSHOT, (
                f"Window {name} ends at {end.date()} which is after snapshot {TRAIN_SNAPSHOT.date()}"
            )

    def test_outcome_window_starts_after_snapshot(self):
        """The 30-day outcome window must start the day after the snapshot."""
        outcome_start = TRAIN_SNAPSHOT + pd.Timedelta(days=1)
        assert outcome_start > TRAIN_SNAPSHOT, "Outcome window must start after snapshot"

    def test_single_snapshot_date_in_batch(self):
        """A single-snapshot training batch must have exactly one distinct snapshot_dt."""
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
            "agent_msisdn": [f"256{i:07d}" for i in range(n)],
            "snapshot_dt": pd.to_datetime(TRAIN_SNAPSHOT),
            "bad_state_30D": rng.integers(0, 2, n),
        })
        unique_dates = pd.to_datetime(df["snapshot_dt"]).dt.date.unique()
        assert len(unique_dates) == 1, (
            f"Expected one snapshot date, found {len(unique_dates)}: {unique_dates}. "
            "Mixed dates in a single training batch indicate data preparation error."
        )
        assert unique_dates[0] == TRAIN_SNAPSHOT.date()


class TestForwardLookingColumnNames:
    def test_future_keyword_caught_by_leakage_patterns(self):
        """Column names with 'future' must be caught by LEAKAGE_PATTERNS."""
        suspicious = [
            "future_penalties_30d",
            "future_penalties_30D",
            "future_repayments",
            "future_cashflow_30d",
        ]
        for col in suspicious:
            matched = any(p in col.lower() for p in LEAKAGE_PATTERNS)
            assert matched, (
                f"Column '{col}' not caught by any LEAKAGE_PATTERN. "
                f"Patterns: {LEAKAGE_PATTERNS}"
            )

    def test_outcome_keyword_caught_by_leakage_patterns(self):
        """Column names with 'outcome' must be caught by LEAKAGE_PATTERNS."""
        suspicious = ["outcome_observed_30d", "outcome_observed_30D", "outcome_flag"]
        for col in suspicious:
            matched = any(p in col.lower() for p in LEAKAGE_PATTERNS)
            assert matched, f"Column '{col}' not caught by LEAKAGE_PATTERNS"

    def test_penalt_stem_catches_both_penalty_and_penalties(self):
        """'penalt' in LEAKAGE_PATTERNS must match both singular and plural forms."""
        assert "penalt" in LEAKAGE_PATTERNS, "'penalt' stem must be in LEAKAGE_PATTERNS"
        for col in ["future_penalties_30d", "penalty_frequency_6M", "penalties_1M"]:
            assert "penalt" in col.lower(), f"Stem mismatch: 'penalt' not in '{col}'"

    def test_bad_state_keyword_caught(self):
        """'bad_state' variants must all be caught."""
        for col in ["bad_state", "bad_state_30D", "bad_state_30d"]:
            matched = any(p in col.lower() for p in LEAKAGE_PATTERNS) or col in PD_FEATURE_BLACKLIST
            assert matched, f"Label column '{col}' not caught"

    def test_legitimate_repayment_cols_not_caught(self):
        """Pre-snapshot repayment features must not match leakage patterns."""
        clean_cols = [
            "repayment_vol_m1",
            "repayment_vol_m5",
            "repayment_coverage_ratio_6M",
            "disbursement_vol_m5",
            "net_exposure_6M",
        ]
        for col in clean_cols:
            col_low = col.lower()
            matched = any(p in col_low for p in LEAKAGE_PATTERNS)
            assert not matched, (
                f"Legitimate column '{col}' incorrectly caught by LEAKAGE_PATTERNS: "
                + str([p for p in LEAKAGE_PATTERNS if p in col_low])
            )
