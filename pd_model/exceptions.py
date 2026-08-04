"""Domain exception hierarchy for the CreditRisk pipeline.

Using a typed hierarchy lets orchestration layers (monitoring, alerts,
retry logic) distinguish failure causes without parsing exception messages.

Inheritance keeps standard ``isinstance`` / ``except`` clauses working:
  ``except ValueError`` still catches ``SchemaValidationError``.
  ``except RuntimeError`` still catches ``DataAlignmentError``, etc.
"""

from __future__ import annotations


class CreditRiskError(Exception):
    """Base exception for all CreditRisk pipeline errors."""


class SchemaValidationError(CreditRiskError, ValueError):
    """Raised when input data violates the expected schema.

    Examples: missing required columns, null values in a binary column,
    unparseable dates, empty training set, null targets.
    """


class DataAlignmentError(CreditRiskError, RuntimeError):
    """Raised when internal DataFrames become misaligned.

    Examples: row-count mismatch after a merge, non-identical indices
    after a time-split, duplicate join keys, columns lost during a merge.
    """


class DataLeakageError(CreditRiskError, RuntimeError):
    """Raised when a leakage guard detects forward-looking or label-adjacent features.

    Examples: DPD columns in PD candidates, forbidden substring matches,
    exact NON_BEHAVIOURAL_COLS overlap, repayment behaviour present for
    thin-file agents.
    """


class ModelInputError(CreditRiskError, RuntimeError):
    """Raised when the model receives structurally incompatible input.

    Examples: train/val column order mismatch, protected columns leaked
    into the feature matrix, monotone constraint length mismatch.
    """


class ArtifactVerificationError(CreditRiskError, RuntimeError):
    """Raised when artifact integrity checks fail.

    Examples: checksum mismatch, corrupt model_metadata.json,
    missing required artifact files detected during preflight.
    """


class MissingArtifactError(ArtifactVerificationError, FileNotFoundError):
    """Raised when a required artifact file is absent.

    Inherits from both ArtifactVerificationError and FileNotFoundError so
    callers can catch all artifact problems via ArtifactVerificationError alone,
    or use FileNotFoundError for standard OS-level checks.
    """


class CalibrationError(CreditRiskError, RuntimeError):
    """Raised when the calibration map is missing, corrupt, or produces invalid output.

    Examples: empty cal_map, cal_pd values outside [0, 1], NaN calibrated scores.
    """


class PolicyConfigurationError(CreditRiskError, ValueError):
    """Raised when engine policy configuration is invalid.

    Examples: signal weights that do not sum to 1, tier thresholds out of order,
    negative floor or ceiling limits.
    """
