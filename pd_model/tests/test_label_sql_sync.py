"""
Static sync test: the materialized and non-materialized label-observability
SQL files (data/loan_state_query_updated_materialized.txt and
data/loan_state_query_updated.txt) must agree on the label-eligibility
policy's literal constants, reason-code vocabulary, and exported column set.
They intentionally differ in structure (physical tmp_loan_label_assessment
table vs. a CTE), but the business logic must stay identical -- see each
file's own header comment.

This test reads the SQL files as plain text and does not require a live
Trino/Athena connection.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MATERIALIZED_SQL = _REPO_ROOT / "data" / "loan_state_query_updated_materialized.txt"
_NON_MATERIALIZED_SQL = _REPO_ROOT / "data" / "loan_state_query_updated.txt"

_EXPECTED_REASON_CODES = (
    "KNOWN_BAD",
    "CONFIRMED_GOOD_TERMINAL",
    "CONFIRMED_GOOD_HORIZON",
    "CENSORED_NO_POST_DISBURSEMENT_STATE",
    "CENSORED_INVALID_OBSERVATION_WINDOW",
    "CENSORED_SPARSE_FOLLOW_UP",
    "CENSORED_EARLY_LAST_OBSERVATION",
    "CENSORED_OTHER",
)

# Every reason-code-shaped string literal in either file must be one of the
# ones above -- catches a typo'd or orphaned code as reliably as a missing one.
_REASON_CODE_LITERAL_RE = re.compile(r"'((?:KNOWN_BAD|CONFIRMED_GOOD_\w+|CENSORED_\w+))'")

_COVERAGE_RATIO_THRESHOLD_RE = re.compile(r"follow_up_coverage_ratio_30d\s*>=\s*([\d.]+)")
_NEAR_HORIZON_LAG_RE = re.compile(r"DATE_ADD\('day',\s*(-\d+),\s*c\.required_observation_end_date_30d\)")

# The label table's join alias is "l" in both files' final SELECT.
_L_ALIAS_COLUMN_RE = re.compile(r"\bl\.([a-z_][a-z0-9_]*)")


@pytest.fixture(scope="module")
def materialized_sql() -> str:
    return _MATERIALIZED_SQL.read_text()


@pytest.fixture(scope="module")
def non_materialized_sql() -> str:
    return _NON_MATERIALIZED_SQL.read_text()


class TestCoverageRatioThreshold:
    def test_appears_exactly_once_per_file(self, materialized_sql, non_materialized_sql):
        for name, sql in [("materialized", materialized_sql), ("non-materialized", non_materialized_sql)]:
            matches = _COVERAGE_RATIO_THRESHOLD_RE.findall(sql)
            assert len(matches) == 1, (
                f"{name} file: expected exactly one "
                f"'follow_up_coverage_ratio_30d >= <threshold>' expression (the single "
                f"source of truth for the coverage-ratio threshold), found {len(matches)}: {matches}"
            )

    def test_threshold_is_0_8_in_both_files(self, materialized_sql, non_materialized_sql):
        mat_threshold = _COVERAGE_RATIO_THRESHOLD_RE.search(materialized_sql).group(1)
        nonmat_threshold = _COVERAGE_RATIO_THRESHOLD_RE.search(non_materialized_sql).group(1)
        assert mat_threshold == "0.8"
        assert nonmat_threshold == "0.8"


class TestNearHorizonTolerance:
    def test_appears_exactly_once_per_file(self, materialized_sql, non_materialized_sql):
        for name, sql in [("materialized", materialized_sql), ("non-materialized", non_materialized_sql)]:
            matches = _NEAR_HORIZON_LAG_RE.findall(sql)
            assert len(matches) == 1, (
                f"{name} file: expected exactly one near-horizon "
                f"DATE_ADD('day', -N, c.required_observation_end_date_30d) expression, "
                f"found {len(matches)}: {matches}"
            )

    def test_tolerance_is_5_days_in_both_files(self, materialized_sql, non_materialized_sql):
        mat_lag = _NEAR_HORIZON_LAG_RE.search(materialized_sql).group(1)
        nonmat_lag = _NEAR_HORIZON_LAG_RE.search(non_materialized_sql).group(1)
        assert mat_lag == "-5"
        assert nonmat_lag == "-5"


class TestReasonCodeVocabulary:
    # Reason codes can legitimately appear more than once per file -- the
    # CASE...END assignment plus the pre-flight validation query that checks
    # CENSORED_INVALID_OBSERVATION_WINDOW both reference the literal. What
    # must hold is that every expected code is present at least once, and
    # nothing outside the canonical set appears at all.

    def test_materialized_contains_every_expected_code(self, materialized_sql):
        codes = set(_REASON_CODE_LITERAL_RE.findall(materialized_sql))
        missing = set(_EXPECTED_REASON_CODES) - codes
        assert not missing, f"materialized file is missing reason code(s): {missing}"

    def test_non_materialized_contains_every_expected_code(self, non_materialized_sql):
        codes = set(_REASON_CODE_LITERAL_RE.findall(non_materialized_sql))
        missing = set(_EXPECTED_REASON_CODES) - codes
        assert not missing, f"non-materialized file is missing reason code(s): {missing}"

    def test_no_unexpected_reason_codes_in_either_file(self, materialized_sql, non_materialized_sql):
        for name, sql in [("materialized", materialized_sql), ("non-materialized", non_materialized_sql)]:
            found = set(_REASON_CODE_LITERAL_RE.findall(sql))
            unexpected = found - set(_EXPECTED_REASON_CODES)
            assert not unexpected, f"{name} file has unexpected reason code(s) not in the canonical set: {unexpected}"


class TestFinalSelectColumnSet:
    def test_both_files_export_the_same_label_columns(self, materialized_sql, non_materialized_sql):
        mat_cols = set(_L_ALIAS_COLUMN_RE.findall(materialized_sql))
        nonmat_cols = set(_L_ALIAS_COLUMN_RE.findall(non_materialized_sql))
        assert mat_cols == nonmat_cols, (
            f"Label-table column references diverge between files: "
            f"materialized-only={mat_cols - nonmat_cols}, non-materialized-only={nonmat_cols - mat_cols}"
        )

    def test_label_eligibility_columns_are_exported(self, materialized_sql):
        mat_cols = set(_L_ALIAS_COLUMN_RE.findall(materialized_sql))
        for col in ("label_eligible_30d", "label_eligibility_reason_30d", "confirmed_good_30d"):
            assert col in mat_cols, f"'{col}' is not exported via the l. alias in the final SELECT"
