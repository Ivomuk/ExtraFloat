"""
Test 7: Leakage guard completeness.

Verifies that every known forward-looking or post-snapshot column is excluded
from the model feature set by one of:
  - Explicit membership in PD_FEATURE_BLACKLIST
  - Pattern match in LEAKAGE_PATTERNS, DPD_BLOCK_PATTERNS, or ID_LIKE_PATTERNS

Critical coverage:
  - `currently_outstanding_flag` (= net_exposure_6M > 0) -- business-process
    co-definition proxy. Must be in PD_FEATURE_BLACKLIST. NOT caught by any pattern.
  - `future_penalties_30d` -- forward-looking, caught by 'future' and 'penalt' patterns.
  - `outcome_observed_30d` -- sample-selection column, caught by 'outcome' pattern.

The test for `currently_outstanding_flag` acts as a regression guard: it fails
on the pre-fix codebase and passes after the column is added to the blacklist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pd_model.config.feature_config import (
    DPD_BLOCK_PATTERNS,
    ID_LIKE_PATTERNS,
    LEAKAGE_PATTERNS,
    PD_FEATURE_BLACKLIST,
)
from pd_model.preprocessing.loan_history_features import LABEL_DIAGNOSTIC_COLUMNS
from pd_model.preprocessing.transformations import get_and_classify_pd_features


def _is_excluded_by_any_guard(col: str) -> tuple[bool, str]:
    """Return (is_excluded, reason) by simulating the guard logic of get_and_classify_pd_features."""
    c_low = col.strip().lower()

    if c_low in {c.strip().lower() for c in PD_FEATURE_BLACKLIST}:
        return True, "blacklist"
    if any(p in c_low for p in ID_LIKE_PATTERNS):
        return True, "id_like"
    if any(p in c_low for p in LEAKAGE_PATTERNS):
        return True, "leakage_pattern"
    if any(p in c_low for p in DPD_BLOCK_PATTERNS):
        return True, "dpd_block"
    return False, "not_excluded"


class TestBlacklistCompleteness:
    def test_currently_outstanding_flag_in_blacklist(self):
        """
        `currently_outstanding_flag` (= net_exposure_6M > 0) must be in PD_FEATURE_BLACKLIST.

        This is a binary proxy for net_exposure_6M encoding 'agent still owes money'.
        The penalty system triggers when outstanding debt exists, making this column
        a business-process co-definition of the label. It must be explicitly blacklisted
        because no leakage pattern catches it by name.
        """
        assert "currently_outstanding_flag" in PD_FEATURE_BLACKLIST, (
            "'currently_outstanding_flag' is NOT in PD_FEATURE_BLACKLIST. "
            "This column (loan_features.py:605, = net_exposure_6M > 0) is a binary "
            "label proxy via business-process co-definition. "
            "Add it to PD_FEATURE_BLACKLIST in pd_model/config/feature_config.py."
        )

    def test_currently_outstanding_flag_excluded_by_guard(self):
        """Verify the guard logic excludes currently_outstanding_flag."""
        excluded, reason = _is_excluded_by_any_guard("currently_outstanding_flag")
        assert excluded, (
            f"'currently_outstanding_flag' not excluded by any guard (reason: {reason}). "
            "It must be in PD_FEATURE_BLACKLIST."
        )

    def test_future_penalties_30d_excluded(self):
        """future_penalties_30d must be caught by 'future' and 'penalt' leakage patterns."""
        excluded, reason = _is_excluded_by_any_guard("future_penalties_30d")
        assert excluded, (
            f"'future_penalties_30d' not excluded (reason: {reason}). "
            "This is a post-snapshot column -- pure label leakage."
        )
        assert reason == "leakage_pattern", f"Expected exclusion via leakage_pattern, got: {reason}"

    def test_future_penalties_30D_excluded(self):
        """Case-variant of future_penalties must also be caught."""
        excluded, reason = _is_excluded_by_any_guard("future_penalties_30D")
        assert excluded, f"'future_penalties_30D' not excluded (reason: {reason})"

    def test_outcome_observed_30d_excluded(self):
        """outcome_observed_30d must be caught by the 'outcome' leakage pattern."""
        excluded, reason = _is_excluded_by_any_guard("outcome_observed_30d")
        assert excluded, (
            f"'outcome_observed_30d' not excluded (reason: {reason}). "
            "This is the sample-selection filter column -- must not enter model features."
        )

    def test_bad_state_variants_excluded(self):
        """All bad_state and label variants must be excluded."""
        for col in ["bad_state", "bad_state_30D", "bad_state_30d", "hard_bad_flag"]:
            excluded, reason = _is_excluded_by_any_guard(col)
            assert excluded, f"Label column '{col}' not excluded (reason: {reason})"

    def test_dpd_columns_excluded(self):
        """DPD leakage columns must be excluded via DPD_BLOCK_PATTERNS."""
        for col in ["max_dpd", "worst_dpd", "ever_dpd", "dpd_90", "dpd_30"]:
            excluded, _ = _is_excluded_by_any_guard(col)
            assert excluded, f"DPD column '{col}' not excluded by any guard"

    def test_agent_msisdn_excluded(self):
        """agent_msisdn must be excluded (blacklist + id_like)."""
        excluded, _ = _is_excluded_by_any_guard("agent_msisdn")
        assert excluded

    def test_penalt_stem_in_leakage_patterns(self):
        """'penalt' stem must be in LEAKAGE_PATTERNS to catch plural form 'penalties'."""
        assert "penalt" in LEAKAGE_PATTERNS, (
            "'penalt' not found in LEAKAGE_PATTERNS. "
            "Without this stem, 'future_penalties_30d' would NOT be caught because "
            "'penalty' in 'penalties' evaluates to False (substring mismatch)."
        )

    def test_future_stem_in_leakage_patterns(self):
        """'future' stem must be in LEAKAGE_PATTERNS."""
        assert "future" in LEAKAGE_PATTERNS, "'future' not in LEAKAGE_PATTERNS"

    def test_outcome_stem_in_leakage_patterns(self):
        """'outcome' stem must be in LEAKAGE_PATTERNS."""
        assert "outcome" in LEAKAGE_PATTERNS, "'outcome' not in LEAKAGE_PATTERNS"


class TestLoanLevelDiagnosticsExcluded:
    """data/loan_state_query_updated_materialized.txt's 9 label-diagnostic
    columns are future-derived (label leakage) and must never reach the
    model, regardless of whether pd_model/preprocessing/loan_history_features.py
    already stripped them upstream -- this is defense-in-depth."""

    def test_all_9_diagnostics_excluded_by_some_guard(self):
        for col in LABEL_DIAGNOSTIC_COLUMNS:
            excluded, reason = _is_excluded_by_any_guard(col)
            assert excluded, f"Label-diagnostic column '{col}' not excluded by any guard"

    def test_all_9_diagnostics_in_blacklist(self):
        """Only 4 of the 9 are caught by LEAKAGE_PATTERNS's 'outcome' substring
        (outcome_state_row_count_30d, outcome_observed_date_count_30d,
        first_outcome_state_date, last_outcome_state_date); the other 5
        (days-aging / rollover / terminal-state) rely on explicit blacklist
        membership plus the narrower max_days_aging/rollover/terminal_state
        LEAKAGE_PATTERNS entries."""
        blacklist_low = {c.strip().lower() for c in PD_FEATURE_BLACKLIST}
        for col in LABEL_DIAGNOSTIC_COLUMNS:
            assert col.lower() in blacklist_low, (
                f"Label-diagnostic column '{col}' not in PD_FEATURE_BLACKLIST"
            )

    def test_loan_level_id_columns_excluded(self):
        id_cols = [
            "msisdn",
            "disbursement_fid",
            "disbursement_uid",
            "target_loan_uid",
            "target_loan_seq",
            "same_day_disbursement_position",
            "same_day_disbursement_count",
            "scoring_state_loan_uid",
            "scoring_state_date",
            "last_closed_loan_uid",
            "last_closed_loan_closure_date",
            "loan_seq_minus_observed_prior_loan_count",
            "sales_region",
            "sales_territory",
            "district",
            "loan_date",
            "disbursement_ts",
            "label_horizon_7d_end",
            "label_horizon_30d_end",
            "bad_state_3dpd_30d",
            "bad_state_1dpd_7d",
        ]
        for col in id_cols:
            excluded, reason = _is_excluded_by_any_guard(col)
            assert excluded, f"Loan-level id/label column '{col}' not excluded (reason: {reason})"


class TestLegitimateColumnsNotExcluded:
    def test_net_exposure_6m_not_excluded(self):
        """
        net_exposure_6M is a valid pre-snapshot behavioral feature.
        Its BINARY PROXY (currently_outstanding_flag) must be excluded, but the
        continuous variable itself is a legitimate model input.
        """
        excluded, reason = _is_excluded_by_any_guard("net_exposure_6M")
        assert not excluded, (
            f"net_exposure_6M was excluded (reason: {reason}), but it is a valid "
            "pre-snapshot behavioral feature (6M aggregate of disbursements minus repayments)."
        )

    def test_repayment_coverage_ratio_not_excluded(self):
        excluded, reason = _is_excluded_by_any_guard("repayment_coverage_ratio_6M")
        assert not excluded, (
            f"repayment_coverage_ratio_6M should not be excluded: {reason}"
        )

    def test_disbursement_vol_m5_not_excluded(self):
        excluded, reason = _is_excluded_by_any_guard("disbursement_vol_m5")
        assert not excluded, f"disbursement_vol_m5 should not be excluded: {reason}"

    def test_repayment_vol_m4_not_excluded(self):
        excluded, reason = _is_excluded_by_any_guard("repayment_vol_m4")
        assert not excluded, f"repayment_vol_m4 should not be excluded: {reason}"

    def test_active_loan_days_aging_at_scoring_not_excluded(self):
        """Regression guard: LEAKAGE_PATTERNS's defense-in-depth entry for the
        max_days_aging_* diagnostics must be narrow enough ('max_days_aging',
        not the broader 'days_aging') to not also block this legitimate
        FEATURE_COLUMNS entry. Caught via a synthetic end-to-end
        run_pipeline smoke test before this fix."""
        excluded, reason = _is_excluded_by_any_guard("active_loan_days_aging_at_scoring")
        assert not excluded, (
            f"active_loan_days_aging_at_scoring was excluded (reason: {reason}), but it "
            "is a legitimate feature, not one of the 9 label-diagnostic columns."
        )

    def test_active_loan_days_aging_at_snapshot_not_excluded(self):
        excluded, reason = _is_excluded_by_any_guard("active_loan_days_aging_at_snapshot")
        assert not excluded, f"active_loan_days_aging_at_snapshot should not be excluded: {reason}"


class TestPipelineIntegration:
    """Integration tests: verify columns are actually excluded by get_and_classify_pd_features."""

    def _make_df(self, cols: list[str], n: int = 200) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        data: dict = {
            "agent_msisdn": [f"256{i:07d}" for i in range(n)],
            "snapshot_dt": pd.to_datetime("2025-09-30"),
            "bad_state": rng.integers(0, 2, n),
        }
        for col in cols:
            data[col] = rng.uniform(0, 1, n)
        return pd.DataFrame(data)

    def test_currently_outstanding_flag_excluded_from_pipeline(self):
        """currently_outstanding_flag must not appear in the pipeline feature candidates."""
        df = self._make_df(["currently_outstanding_flag", "disbursement_vol_m5"])
        pd_features, *_ = get_and_classify_pd_features(df)
        assert "currently_outstanding_flag" not in pd_features, (
            "'currently_outstanding_flag' appeared in model feature candidates. "
            "Check that it is in PD_FEATURE_BLACKLIST in feature_config.py."
        )

    def test_future_penalties_excluded_from_pipeline(self):
        df = self._make_df(["future_penalties_30d", "net_exposure_6M"])
        pd_features, *_ = get_and_classify_pd_features(df)
        assert "future_penalties_30d" not in pd_features

    def test_outcome_observed_excluded_from_pipeline(self):
        df = self._make_df(["outcome_observed_30d", "disbursement_vol_m5"])
        pd_features, *_ = get_and_classify_pd_features(df)
        assert "outcome_observed_30d" not in pd_features

    def test_legitimate_features_survive_pipeline(self):
        """Pre-snapshot behavioral features must be retained as model candidates."""
        df = self._make_df(["disbursement_vol_m5", "net_exposure_6M"])
        pd_features, *_ = get_and_classify_pd_features(df)
        surviving = [c for c in pd_features if c in {"disbursement_vol_m5", "net_exposure_6M"}]
        assert len(surviving) >= 1, (
            f"Expected at least one legitimate feature in candidates, got none. "
            f"All candidates: {pd_features}"
        )
