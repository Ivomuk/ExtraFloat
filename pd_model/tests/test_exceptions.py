"""Tests for the pd_model exception hierarchy and validator exception types."""

import pandas as pd
import pytest

from pd_model.exceptions import (
    ArtifactVerificationError,
    CalibrationError,
    CreditRiskError,
    DataAlignmentError,
    DataLeakageError,
    ModelInputError,
    PolicyConfigurationError,
    SchemaValidationError,
)
from pd_model.validation.schema import (
    require_binary_column,
    require_columns,
    require_index_alignment,
    require_non_empty_dataframe,
)


class TestExceptionHierarchy:
    def test_schema_error_is_credit_risk_and_value_error(self):
        exc = SchemaValidationError("bad schema")
        assert isinstance(exc, CreditRiskError)
        assert isinstance(exc, ValueError)

    def test_alignment_error_is_credit_risk_and_runtime_error(self):
        exc = DataAlignmentError("misaligned")
        assert isinstance(exc, CreditRiskError)
        assert isinstance(exc, RuntimeError)

    def test_leakage_error_is_credit_risk_and_runtime_error(self):
        exc = DataLeakageError("leaked")
        assert isinstance(exc, CreditRiskError)
        assert isinstance(exc, RuntimeError)

    def test_model_input_error_is_credit_risk_and_runtime_error(self):
        exc = ModelInputError("bad input")
        assert isinstance(exc, CreditRiskError)
        assert isinstance(exc, RuntimeError)

    def test_artifact_verification_error_is_credit_risk_and_runtime_error(self):
        exc = ArtifactVerificationError("checksum mismatch")
        assert isinstance(exc, CreditRiskError)
        assert isinstance(exc, RuntimeError)

    def test_calibration_error_is_credit_risk_and_runtime_error(self):
        exc = CalibrationError("empty cal_map")
        assert isinstance(exc, CreditRiskError)
        assert isinstance(exc, RuntimeError)

    def test_policy_configuration_error_is_credit_risk_and_value_error(self):
        exc = PolicyConfigurationError("negative floor")
        assert isinstance(exc, CreditRiskError)
        assert isinstance(exc, ValueError)

    def test_all_subclasses_catchable_as_credit_risk_error(self):
        subclasses = [
            SchemaValidationError("x"),
            DataAlignmentError("x"),
            DataLeakageError("x"),
            ModelInputError("x"),
            ArtifactVerificationError("x"),
            CalibrationError("x"),
            PolicyConfigurationError("x"),
        ]
        for exc in subclasses:
            with pytest.raises(CreditRiskError):
                raise exc

    def test_value_error_subclasses_catchable_as_builtin_value_error(self):
        for exc_cls in (SchemaValidationError, PolicyConfigurationError):
            with pytest.raises(ValueError):
                raise exc_cls("x")

    def test_runtime_error_subclasses_catchable_as_builtin_runtime_error(self):
        for exc_cls in (
            DataAlignmentError,
            DataLeakageError,
            ModelInputError,
            ArtifactVerificationError,
            CalibrationError,
        ):
            with pytest.raises(RuntimeError):
                raise exc_cls("x")


class TestValidatorExceptions:
    def test_require_columns_raises_schema_error(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(SchemaValidationError, match="Missing required columns"):
            require_columns(df, ["a", "b"], context="test")

    def test_require_columns_passes_when_all_present(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        require_columns(df, ["a", "b"], context="test")  # no exception

    def test_require_binary_column_raises_schema_error_on_nulls(self):
        df = pd.DataFrame({"flag": [0, None, 1]})
        with pytest.raises(SchemaValidationError, match="NaN"):
            require_binary_column(df, "flag", context="test")

    def test_require_binary_column_raises_schema_error_on_bad_values(self):
        df = pd.DataFrame({"flag": [0, 1, 2]})
        with pytest.raises(SchemaValidationError, match="outside"):
            require_binary_column(df, "flag", context="test")

    def test_require_binary_column_passes_on_valid_column(self):
        df = pd.DataFrame({"flag": [0, 1, 0, 1]})
        require_binary_column(df, "flag", context="test")  # no exception

    def test_require_index_alignment_raises_alignment_error(self):
        df_a = pd.DataFrame({"x": [1, 2, 3]}, index=[0, 1, 2])
        df_b = pd.DataFrame({"y": [4, 5, 6]}, index=[0, 1, 99])
        with pytest.raises(DataAlignmentError, match="Index mismatch"):
            require_index_alignment(df_a, df_b, context="test")

    def test_require_index_alignment_passes_on_matching_indices(self):
        df_a = pd.DataFrame({"x": [1, 2]}, index=[10, 20])
        df_b = pd.DataFrame({"y": [3, 4]}, index=[10, 20])
        require_index_alignment(df_a, df_b, context="test")  # no exception

    def test_require_non_empty_dataframe_raises_schema_error(self):
        df = pd.DataFrame({"a": []})
        with pytest.raises(SchemaValidationError, match="empty"):
            require_non_empty_dataframe(df, context="test")

    def test_require_non_empty_dataframe_passes_on_non_empty(self):
        df = pd.DataFrame({"a": [1]})
        require_non_empty_dataframe(df, context="test")  # no exception
