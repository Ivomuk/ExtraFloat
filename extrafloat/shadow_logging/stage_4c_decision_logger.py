"""
extrafloat/shadow_logging/stage_4c_decision_logger.py
=======================================================
Stage 4C Decision Log writer -- additive, read-only, prospective shadow
logging of the Capacity(F) challenger alongside the live credit limit
engine. Implements docs/stage_4c_logger_implementation_plan.md exactly,
as frozen after its review-correction round (commit 038aedc), on top of:
  - docs/stage_4c_prospective_validation_schema_design.md (frozen schema)
  - docs/stage_4c_implementation_mapping.md (five implementation-
    reconnaissance conclusions: agent x scoring_run grain,
    challenger_risk_signal naming, Gate I0 df_agent-sourced capacity
    inputs, no execution_status, explicit temporal execution attribution)

SCOPE: Decision Log only (one row per agent x scoring_run). Execution Log
and Outcome Log are explicitly out of scope -- see implementation plan
Section 1.

HARD DISCIPLINE (same as extrafloat_shadow_risk_multiplier.py): every
function here is pure/read-only. None mutate an input DataFrame in
place, and none ever write to any existing engine output column. A
failure anywhere in this module must never fail the live pipeline --
see run_decision_log_stage(), which is the single function the pipeline
integration (run_credit_risk_pipeline.py) calls, already wrapped so that
a construction/write failure degrades to a FAILED run manifest rather
than propagating.

KNOWN, FLAGGED GAP (not silently worked around): raw_model_score is not
yet wired through the live pipeline -- run_credit_risk_pipeline.py's
existing Stage 5 join only carries cal_pd/thin_file_flag from pd_scored
onto features_df, never xgb_raw_score/lgb_raw_score. This column is
therefore always NaN here, and the corresponding half of frozen-schema
invariant #3 (raw_model_score nullability tied to risk_route) is not
enforced by verify_decision_log_invariants until that join is widened --
a follow-up, not invented here.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from extrafloat.capacity.capacity_function import compute_capacity_ugx

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Frozen constants
# -----------------------------------------------------------------------------

FUNDAMENTALS_FOR_SCORE = ["float_activity_value_1m", "commission"]

K_SHADOW = 0.05
K_SENSITIVITY_LOW = 0.04
K_SENSITIVITY_HIGH = 0.06

# Bump manually if the Capacity(F) functional form itself changes (never
# auto-derived -- distinct from capacity_weights_artifact_version, which
# versions the weights alone, not the transformation).
CAPACITY_FUNCTION_VERSION = "capacity_function_v1"

# Bump manually on any DEFAULT_CAP_CONFIG change in
# extrafloat/engine/extrafloat_limit_engine_caps.py. Deliberately NOT
# sourced from that file (this module never modifies it) -- a literal,
# hand-maintained constant, same discipline as SHADOW_POLICY_VERSION in
# extrafloat_shadow_risk_multiplier.py.
CURRENT_POLICY_VERSION = "engine_policy_v1"

RISK_SCORE_TYPE_THICK = "CALIBRATED_CHAMPION_PD"
RISK_SCORE_TYPE_THIN = "THIN_SCORECARD_PD_LIKE"

AVAILABILITY_OK = "OK"
AVAILABILITY_MISSING = "MISSING"
AVAILABILITY_NONPOSITIVE = "NONPOSITIVE"

SNAPSHOT_OK = "OK"
SNAPSHOT_UNAVAILABLE = "UNAVAILABLE"

CURRENT_DECISION_STATUS_APPROVED = "APPROVED"
CURRENT_DECISION_STATUS_ZERO_LIMIT_ELIGIBLE = "ZERO_LIMIT_ELIGIBLE"
CURRENT_DECISION_STATUS_OTHER = "OTHER"

SHADOW_POPULATION_FULL = "FULL_EVALUATION_POPULATION"
SHADOW_POPULATION_DISBURSEMENT_CONDITIONED = "DISBURSEMENT_CONDITIONED_SHADOW_POPULATION"
_VALID_SHADOW_POPULATION_LABELS = {SHADOW_POPULATION_FULL, SHADOW_POPULATION_DISBURSEMENT_CONDITIONED}

MANIFEST_STATUS_SUCCESS = "SUCCESS"
MANIFEST_STATUS_FAILED = "FAILED"
MANIFEST_STATUS_PARTIAL = "PARTIAL"

_SCENARIOS = ("base", "conservative")
_K_LABELS = (
    ("shadow_k0_05", "capacity_shadow_k0_05", K_SHADOW),
    ("sensitivity_k0_04", "capacity_sensitivity_k0_04", K_SENSITIVITY_LOW),
    ("sensitivity_k0_06", "capacity_sensitivity_k0_06", K_SENSITIVITY_HIGH),
)

DECISION_LOG_COLUMNS = [
    # 4.1 Identity
    "shadow_decision_id", "run_id", "agent_msisdn", "scoring_run_timestamp",
    "current_decision_status", "decision_reason_code", "shadow_population_label",
    # 4.2 Business-state inputs
    "fundamentals_snapshot_date", "fundamentals_snapshot_id", "fundamentals_snapshot_available_at",
    "fundamentals_snapshot_age_days", "float_activity_value_1m", "commission",
    "cash_in_availability_status", "payment_availability_status",
    "float_availability_status", "commission_availability_status", "snapshot_availability_status",
    "capacity_estimable",
    # 4.3 Capacity output
    "capacity_shadow_k0_05", "capacity_sensitivity_k0_04", "capacity_sensitivity_k0_06",
    # 4.4 Risk / C3
    "risk_route", "raw_model_score", "challenger_risk_signal", "challenger_risk_signal_type", "cal_pd",
    "shadow_multiplier_base", "shadow_multiplier_conservative",
    "l_full_shadow_k0_05_base", "l_full_shadow_k0_05_conservative",
    "l_full_sensitivity_k0_04_base", "l_full_sensitivity_k0_04_conservative",
    "l_full_sensitivity_k0_06_base", "l_full_sensitivity_k0_06_conservative",
    # 4.5 Current policy
    "l_current_capacity", "current_risk_multiplier", "l_current", "current_policy_version",
    # 4.6 Disagreement
    "disagreement_abs_ugx_shadow_base", "disagreement_abs_ugx_shadow_conservative",
    "disagreement_ratio_shadow_base", "disagreement_ratio_shadow_conservative",
    # 4.7 Provenance
    "capacity_weights_artifact_version", "capacity_function_version", "fundamentals_source_version",
    "k_shadow_value", "k_sensitivity_low_value", "k_sensitivity_high_value",
    "champion_model_version", "pd_calibration_artifact_version",
    "c3_shadow_artifact_version", "c3_policy_version",
]


def _to_naive_timestamp(ts) -> pd.Timestamp:
    """Parses ts (a run_id string like '...isoformat()+Z', a Timestamp,
    etc.) to a tz-NAIVE pandas Timestamp. Real bug found while testing
    the Section 4 insertion point against run_credit_risk_pipeline.py's
    actual run_id format (datetime.utcnow().isoformat() + "Z"): parsing
    that string with pd.to_datetime() alone produces a tz-AWARE (UTC)
    Timestamp, while fundamentals_snapshot_date (parsed from the raw
    mart's own snapshot_dt/tbl_dt column) is tz-naive -- comparing the
    two directly raises TypeError. utc=True normalizes any input (naive
    or tz-aware, "Z"-suffixed or not) to UTC first, then tz_localize(None)
    drops the tz info while keeping the UTC wall-clock value, so both
    sides of every comparison in this module are consistently naive."""
    return pd.to_datetime(ts, utc=True).tz_localize(None)


def _norm_msisdn(s: pd.Series) -> pd.Series:
    """Restated verbatim from run_credit_risk_pipeline.py's own
    _norm_msisdn (one-way convention: extrafloat/ restates top-level
    script helpers rather than importing from them)."""
    out = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return out.mask(out.str.lower().isin({"", "nan", "none", "<na>"}))


def _component_availability_status(s: pd.Series) -> pd.Series:
    """OK if >0, NONPOSITIVE if <=0, MISSING if null -- used for
    component-level (pre-fill) data-quality fields only
    (cash_in_availability_status / payment_availability_status /
    commission_availability_status). NEVER used for
    float_availability_status itself, which is derived from the
    post-fill Float value (see extract_raw_capacity_inputs)."""
    return pd.Series(
        np.select(
            [s.isna(), s <= 0],
            [AVAILABILITY_MISSING, AVAILABILITY_NONPOSITIVE],
            default=AVAILABILITY_OK,
        ),
        index=s.index,
    )


def extract_raw_capacity_inputs(df_agent_raw: pd.DataFrame) -> pd.DataFrame:
    """Reads cash_in_value_1m, payment_value_1m, commission from the RAW
    df_agent snapshot DataFrame (Gate I0: these columns live on df_agent,
    never on features_df/result_df -- docs/stage_4c_implementation_
    mapping.md Section 4).

    Correction applied per review (docs/stage_4c_logger_implementation_
    plan.md Section 8, item 1): float_availability_status is derived
    from the RESULTING post-fill Float value, never from which raw
    component was null -- one missing component does not mean Float
    itself is missing under the frozen formula
    (Float = fillna0(cash_in_value_1m) + fillna0(payment_value_1m)).
    Component-level data quality is tracked separately and does NOT feed
    capacity_estimable.

    Returns one row per df_agent_raw row, keyed by agent_msisdn
    (NOT normalized here -- callers normalize once, at the join site, to
    avoid double-normalizing).
    """
    required = ["agent_msisdn", "cash_in_value_1m", "payment_value_1m", "commission"]
    missing_cols = [c for c in required if c not in df_agent_raw.columns]
    if missing_cols:
        raise ValueError(f"extract_raw_capacity_inputs: df_agent_raw is missing required column(s): {missing_cols}")

    cash_in = pd.to_numeric(df_agent_raw["cash_in_value_1m"], errors="coerce")
    payment = pd.to_numeric(df_agent_raw["payment_value_1m"], errors="coerce")
    commission = pd.to_numeric(df_agent_raw["commission"], errors="coerce")

    out = pd.DataFrame(index=df_agent_raw.index)
    out["agent_msisdn"] = df_agent_raw["agent_msisdn"]

    out["cash_in_availability_status"] = _component_availability_status(cash_in)
    out["payment_availability_status"] = _component_availability_status(payment)

    float_activity = cash_in.fillna(0) + payment.fillna(0)
    out["float_activity_value_1m"] = float_activity
    out["float_availability_status"] = np.where(float_activity > 0, AVAILABILITY_OK, AVAILABILITY_NONPOSITIVE)

    out["commission"] = commission
    out["commission_availability_status"] = _component_availability_status(commission)

    return out


def compute_snapshot_availability(df_agent_raw: pd.DataFrame, scoring_run_timestamp) -> pd.DataFrame:
    """Derives snapshot_availability_status (OK/UNAVAILABLE) and
    fundamentals_snapshot_date from df_agent_raw's own snapshot_dt/tbl_dt
    column (snapshot_dt takes precedence, matching run_credit_risk_
    pipeline.py's own --snapshot-date override). Enforces the hard PIT
    invariant fundamentals_snapshot_date < date(scoring_run_timestamp) at
    construction time -- RAISES, never silently drops the row."""
    out = pd.DataFrame(index=df_agent_raw.index)
    out["agent_msisdn"] = df_agent_raw["agent_msisdn"]

    if "snapshot_dt" in df_agent_raw.columns:
        raw_date = df_agent_raw["snapshot_dt"]
    elif "tbl_dt" in df_agent_raw.columns:
        raw_date = df_agent_raw["tbl_dt"]
    else:
        raw_date = pd.Series(pd.NaT, index=df_agent_raw.index)

    snapshot_date = pd.to_datetime(raw_date, errors="coerce").dt.normalize()
    out["fundamentals_snapshot_date"] = snapshot_date
    out["snapshot_availability_status"] = np.where(snapshot_date.notna(), SNAPSHOT_OK, SNAPSHOT_UNAVAILABLE)

    run_ts = _to_naive_timestamp(scoring_run_timestamp).normalize()
    violating = snapshot_date.notna() & (snapshot_date >= run_ts)
    if violating.any():
        bad_agents = df_agent_raw.loc[violating, "agent_msisdn"].astype(str).tolist()
        preview = bad_agents[:10]
        raise ValueError(
            "compute_snapshot_availability: PIT invariant violated -- "
            f"fundamentals_snapshot_date >= scoring_run_timestamp for {len(bad_agents)} agent(s), "
            f"e.g. {preview}{'...' if len(bad_agents) > 10 else ''}"
        )
    return out


def compute_capacity_estimable(float_status: pd.Series, commission_status: pd.Series,
                                snapshot_status: pd.Series) -> pd.Series:
    """capacity_estimable = TRUE iff all three statuses are OK. Pure
    derivation, no independent logic path (frozen schema invariant #11)."""
    return (
        (float_status == AVAILABILITY_OK)
        & (commission_status == AVAILABILITY_OK)
        & (snapshot_status == SNAPSHOT_OK)
    )


def _verify_capacity_triple_invariants(capacity_df: pd.DataFrame, inputs_df: pd.DataFrame,
                                        weights: dict, tolerance: float = 1e-6) -> None:
    """Schema invariants #2 (capacity-triple ratio check) and #10
    (independent formula-reproducibility check), each firing
    independently -- the ratio check cannot mask a shared systematic
    error across all three columns, and vice versa."""
    cols = ["capacity_shadow_k0_05", "capacity_sensitivity_k0_04", "capacity_sensitivity_k0_06"]
    all_null = capacity_df[cols].isna().all(axis=1)
    all_present = capacity_df[cols].notna().all(axis=1)
    if not (all_null | all_present).all():
        raise ValueError("invariant #2 violated: capacity triple has a partial (neither all-null nor all-present) row.")

    present = capacity_df.loc[all_present]
    if len(present) == 0:
        return

    nonzero = present["capacity_shadow_k0_05"] != 0
    if nonzero.any():
        ratio_low = present.loc[nonzero, "capacity_sensitivity_k0_04"] / present.loc[nonzero, "capacity_shadow_k0_05"]
        ratio_high = present.loc[nonzero, "capacity_sensitivity_k0_06"] / present.loc[nonzero, "capacity_shadow_k0_05"]
        bad_low = (ratio_low - (K_SENSITIVITY_LOW / K_SHADOW)).abs() > tolerance
        bad_high = (ratio_high - (K_SENSITIVITY_HIGH / K_SHADOW)).abs() > tolerance
        if bad_low.any() or bad_high.any():
            raise ValueError("invariant #2 violated: capacity-triple k-ratio mismatch.")

    for k_label, col, k_val in _K_LABELS:
        recomputed = compute_capacity_ugx(inputs_df.loc[present.index], FUNDAMENTALS_FOR_SCORE, weights, k_val)
        mismatch = (recomputed - present[col]).abs() > tolerance
        if mismatch.any():
            raise ValueError(f"invariant #10 violated: {col} does not reproduce from logged Float/Commission.")


def compute_capacity_challenger_outputs(float_activity_value_1m: pd.Series, commission: pd.Series,
                                         capacity_estimable: pd.Series, weights: dict,
                                         weights_version: str) -> pd.DataFrame:
    """Calls capacity_function.compute_capacity_ugx at k=0.05/0.04/0.06,
    masks all three to NaN where capacity_estimable=False, and
    immediately self-checks the formula-reproducibility invariant (#10)
    and the capacity-triple ratio invariant (#2) on its own output before
    returning -- a construction-time self-check, not a separate
    downstream audit step that could be skipped. weights_version is
    accepted for interface symmetry with the implementation plan's
    signature but not otherwise used here (it is logged as
    capacity_weights_artifact_version by the caller, build_decision_log_
    rows)."""
    inputs_df = pd.DataFrame({
        FUNDAMENTALS_FOR_SCORE[0]: float_activity_value_1m,
        FUNDAMENTALS_FOR_SCORE[1]: commission,
    })

    not_estimable = ~capacity_estimable.fillna(False).astype(bool)

    out = pd.DataFrame(index=float_activity_value_1m.index)
    for _k_label, col, k_val in _K_LABELS:
        series = compute_capacity_ugx(inputs_df, FUNDAMENTALS_FOR_SCORE, weights, k_val)
        out[col] = series.mask(not_estimable)

    _verify_capacity_triple_invariants(out, inputs_df, weights)
    return out


def attach_challenger_risk_signal(result_df: pd.DataFrame) -> pd.DataFrame:
    """Reads cal_pd/thin_file_flag directly off result_df (both already
    joined onto features_df in the pipeline's Stage 5, before Stage 6
    calls the engine -- requires keep_intermediate=True internally so
    thin_file_flag survives; cal_pd alone is already in
    FINAL_OUTPUT_COLUMNS regardless).

    Correction applied per review (implementation plan Section 8, item
    2): both routes read challenger_risk_signal from the engine's
    EXISTING pooled cal_pd column (already correct per route, per
    pd_model.modeling.inference.score_new_agents's existing thin/thick
    promotion logic -- never re-derived independently here). The output
    schema's own cal_pd field is a SEPARATE write: kept for thick,
    explicitly nulled for thin -- this is exactly why challenger_risk_
    signal was introduced (implementation mapping Section 5)."""
    required = ["msisdn", "cal_pd", "thin_file_flag"]
    missing_cols = [c for c in required if c not in result_df.columns]
    if missing_cols:
        raise ValueError(
            f"attach_challenger_risk_signal: result_df is missing required column(s): {missing_cols} -- "
            f"was run_extrafloat_limit_engine called with keep_intermediate=True?"
        )

    out = pd.DataFrame(index=result_df.index)
    out["msisdn"] = result_df["msisdn"]

    is_thin = pd.to_numeric(result_df["thin_file_flag"], errors="coerce").fillna(0).astype(int) == 1
    pooled_cal_pd = pd.to_numeric(result_df["cal_pd"], errors="coerce")

    out["challenger_risk_signal"] = pooled_cal_pd
    out["challenger_risk_signal_type"] = np.where(is_thin, RISK_SCORE_TYPE_THIN, RISK_SCORE_TYPE_THICK)
    out["cal_pd"] = pooled_cal_pd.mask(is_thin)
    out["risk_route"] = np.where(is_thin, "THIN_FILE_SCORECARD", "THICK_CHAMPION_SCORED")
    # Known, flagged gap (see module docstring): raw_model_score is not yet
    # wired through the live pipeline's Stage 5 join. Left NaN, not invented.
    out["raw_model_score"] = np.nan

    return out


def attach_current_policy_snapshot(result_df: pd.DataFrame) -> pd.DataFrame:
    """Pulls l_current=assigned_limit, current_risk_multiplier=
    policy_multiplier, l_current_capacity=capacity_cap (the engine's own
    six-component figure -- explicitly NOT a stand-in for Capacity(F),
    logged purely for side-by-side audit). Requires result_df to carry
    its intermediate columns (keep_intermediate=True internally) --
    policy_multiplier/capacity_cap are NOT in FINAL_OUTPUT_COLUMNS."""
    required = ["msisdn", "assigned_limit", "policy_multiplier", "capacity_cap"]
    missing_cols = [c for c in required if c not in result_df.columns]
    if missing_cols:
        raise ValueError(
            f"attach_current_policy_snapshot: result_df is missing required column(s): {missing_cols} -- "
            f"was run_extrafloat_limit_engine called with keep_intermediate=True?"
        )

    out = pd.DataFrame(index=result_df.index)
    out["msisdn"] = result_df["msisdn"]
    out["l_current"] = pd.to_numeric(result_df["assigned_limit"], errors="coerce")
    out["current_risk_multiplier"] = pd.to_numeric(result_df["policy_multiplier"], errors="coerce")
    out["l_current_capacity"] = pd.to_numeric(result_df["capacity_cap"], errors="coerce")
    out["current_policy_version"] = CURRENT_POLICY_VERSION
    return out


def _derive_current_decision_status(l_current: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [l_current.isna(), l_current > 0, l_current == 0],
            [CURRENT_DECISION_STATUS_OTHER, CURRENT_DECISION_STATUS_APPROVED, CURRENT_DECISION_STATUS_ZERO_LIMIT_ELIGIBLE],
            default=CURRENT_DECISION_STATUS_OTHER,
        ),
        index=l_current.index,
    )


def compute_disagreement(l_full_shadow_base: pd.Series, l_full_shadow_conservative: pd.Series,
                          l_current: pd.Series) -> pd.DataFrame:
    """disagreement_abs_ugx_shadow_{base,conservative} and
    disagreement_ratio_shadow_{base,conservative} only -- guarded NULL
    (never Inf) at l_current=0. No direction classification here (frozen
    schema Section 4.6, as amended -- lives in a later analytical layer,
    paired with a disagreement_rule_version, never in this module)."""
    out = pd.DataFrame(index=l_current.index)
    for suffix, l_full in (("base", l_full_shadow_base), ("conservative", l_full_shadow_conservative)):
        out[f"disagreement_abs_ugx_shadow_{suffix}"] = l_full - l_current
        out[f"disagreement_ratio_shadow_{suffix}"] = pd.Series(
            np.where(l_current.notna() & (l_current != 0), l_full / l_current, np.nan),
            index=l_current.index,
        )
    return out


def build_shadow_decision_id(run_id: str, agent_msisdn: pd.Series) -> pd.Series:
    """Deterministic surrogate key f(run_id, agent_msisdn). The REAL
    protection is a UNIQUE(run_id, agent_msisdn) constraint on the two
    source columns (always logged separately alongside this) --
    shadow_decision_id is a convenience surrogate key, never an
    independent source of uniqueness."""
    return agent_msisdn.astype(str).map(
        lambda m: hashlib.sha256(f"{run_id}|{m}".encode("utf-8")).hexdigest()
    )


def build_decision_log_rows(df_agent_raw: pd.DataFrame, result_df: pd.DataFrame, run_id: str,
                             scoring_run_timestamp, capacity_weights: dict, capacity_weights_version: str,
                             artifact_versions: dict,
                             shadow_population_label: str = SHADOW_POPULATION_FULL) -> pd.DataFrame:
    """Single orchestration entry point. One row per agent present in
    result_df (the actual scored population for this run) -- agents
    present in df_agent_raw but absent from result_df are not part of
    this run's Decision Log (they were not scored by the engine in this
    run). artifact_versions keys consumed: champion_model_version,
    pd_calibration_artifact_version (nullable), c3_shadow_artifact_version
    (nullable), c3_policy_version (nullable), fundamentals_source_version."""
    df_agent = df_agent_raw.copy()
    df_agent["agent_msisdn"] = _norm_msisdn(df_agent["agent_msisdn"])
    result = result_df.copy()
    result["msisdn"] = _norm_msisdn(result["msisdn"])

    capacity_inputs = extract_raw_capacity_inputs(df_agent).rename(columns={"agent_msisdn": "msisdn"})
    snapshot = compute_snapshot_availability(df_agent, scoring_run_timestamp).rename(columns={"agent_msisdn": "msisdn"})
    agent_inputs = capacity_inputs.merge(snapshot, on="msisdn", how="outer")

    base = result[["msisdn"]].drop_duplicates(subset=["msisdn"]).reset_index(drop=True)
    out = base.merge(agent_inputs, on="msisdn", how="left")

    # Structurally different "MISSING" case: the agent has no row in
    # df_agent_raw at all, as opposed to a present row with one or more
    # null components (which legitimately zero-fills to a real Float).
    no_match = out["cash_in_availability_status"].isna()
    for col in ("cash_in_availability_status", "payment_availability_status", "commission_availability_status"):
        out.loc[no_match, col] = AVAILABILITY_MISSING
    out.loc[no_match, "float_availability_status"] = AVAILABILITY_MISSING
    out.loc[no_match, "snapshot_availability_status"] = SNAPSHOT_UNAVAILABLE
    out.loc[no_match, "float_activity_value_1m"] = np.nan
    out.loc[no_match, "commission"] = np.nan
    out.loc[no_match, "fundamentals_snapshot_date"] = pd.NaT

    out["capacity_estimable"] = compute_capacity_estimable(
        out["float_availability_status"], out["commission_availability_status"], out["snapshot_availability_status"]
    )

    capacity_out = compute_capacity_challenger_outputs(
        out["float_activity_value_1m"], out["commission"], out["capacity_estimable"],
        capacity_weights, capacity_weights_version,
    )
    out = pd.concat([out.reset_index(drop=True), capacity_out.reset_index(drop=True)], axis=1)

    risk = attach_challenger_risk_signal(result)
    policy = attach_current_policy_snapshot(result)
    out = out.merge(risk, on="msisdn", how="left").merge(policy, on="msisdn", how="left")

    thin_mask = out["challenger_risk_signal_type"] == RISK_SCORE_TYPE_THIN

    shadow_cols = {"shadow_multiplier_base", "shadow_multiplier_conservative"}
    if shadow_cols <= set(result.columns):
        shadow_df = result[["msisdn", "shadow_multiplier_base", "shadow_multiplier_conservative"]].drop_duplicates(subset=["msisdn"])
        out = out.merge(shadow_df, on="msisdn", how="left")
    else:
        out["shadow_multiplier_base"] = np.nan
        out["shadow_multiplier_conservative"] = np.nan

    # Default-closed C3-on-thin-file (Open Item 1 of the frozen schema):
    # thin rows get no shadow multiplier / l_full, regardless of what the
    # engine's own shadow column computed -- it has no thin-file awareness
    # (it treats the pooled cal_pd column as if it were always a
    # calibrated champion PD).
    out.loc[thin_mask, "shadow_multiplier_base"] = np.nan
    out.loc[thin_mask, "shadow_multiplier_conservative"] = np.nan

    for k_label, capacity_col, _k_val in _K_LABELS:
        for scenario in _SCENARIOS:
            out[f"l_full_{k_label}_{scenario}"] = out[capacity_col] * out[f"shadow_multiplier_{scenario}"]

    disagreement = compute_disagreement(
        out["l_full_shadow_k0_05_base"], out["l_full_shadow_k0_05_conservative"], out["l_current"]
    )
    out = pd.concat([out, disagreement], axis=1)

    out["run_id"] = run_id
    out["scoring_run_timestamp"] = _to_naive_timestamp(scoring_run_timestamp)
    out["shadow_decision_id"] = build_shadow_decision_id(run_id, out["msisdn"])
    out["current_decision_status"] = _derive_current_decision_status(out["l_current"])
    out["decision_reason_code"] = result.set_index("msisdn").reindex(out["msisdn"])["final_decision_reason"].values \
        if "final_decision_reason" in result.columns else np.nan
    out["shadow_population_label"] = shadow_population_label
    out["fundamentals_snapshot_id"] = np.nan
    out["fundamentals_snapshot_available_at"] = pd.NaT
    out["fundamentals_snapshot_age_days"] = (
        out["scoring_run_timestamp"].dt.normalize() - out["fundamentals_snapshot_date"]
    ).dt.days

    out = out.rename(columns={"msisdn": "agent_msisdn"})

    out["capacity_weights_artifact_version"] = capacity_weights_version
    out["capacity_function_version"] = CAPACITY_FUNCTION_VERSION
    out["fundamentals_source_version"] = artifact_versions.get("fundamentals_source_version")
    out["k_shadow_value"] = K_SHADOW
    out["k_sensitivity_low_value"] = K_SENSITIVITY_LOW
    out["k_sensitivity_high_value"] = K_SENSITIVITY_HIGH
    out["champion_model_version"] = artifact_versions.get("champion_model_version")
    # pd.Series(...).mask(...)/.where(...) rather than np.where(cond, nan, str):
    # numpy refuses to promote a common dtype for a float NaN mixed with a
    # Python str in the same np.where call (DTypePromotionError) -- pandas'
    # object-dtype masking handles the mixed NaN/str case correctly.
    out["pd_calibration_artifact_version"] = pd.Series(
        artifact_versions.get("pd_calibration_artifact_version"), index=out.index
    ).mask(thin_mask)
    has_c3 = out["shadow_multiplier_base"].notna() | out["shadow_multiplier_conservative"].notna()
    out["c3_shadow_artifact_version"] = pd.Series(
        artifact_versions.get("c3_shadow_artifact_version"), index=out.index
    ).where(has_c3)
    out["c3_policy_version"] = pd.Series(
        artifact_versions.get("c3_policy_version"), index=out.index
    ).where(has_c3)

    return out[DECISION_LOG_COLUMNS]


def verify_decision_log_invariants(decision_log_df: pd.DataFrame, capacity_weights: dict,
                                    tolerance: float = 1e-6) -> None:
    """Enforces every Section 6 invariant applicable at Decision-Log-
    construction time (1, 2, 3 [partial -- see module docstring], 4, 5,
    8, 9, 10, 11 -- the renumbered set from the frozen schema; 6 and 7
    are Execution/Outcome Log invariants, out of scope here). Raises on
    ANY violation -- never silently drops or NULLs a violating row."""
    df = decision_log_df

    # #1: shadow_decision_id unique; (run_id, agent_msisdn) unique.
    if df["shadow_decision_id"].duplicated().any():
        raise ValueError("invariant #1 violated: duplicate shadow_decision_id.")
    if df.duplicated(subset=["run_id", "agent_msisdn"]).any():
        raise ValueError("invariant #1 violated: duplicate (run_id, agent_msisdn).")

    # #2 + #10: capacity triple + formula reproducibility, re-checked on
    # the fully assembled frame (defense in depth; already checked once
    # inside compute_capacity_challenger_outputs).
    inputs_df = df[["float_activity_value_1m", "commission"]]
    capacity_cols = ["capacity_shadow_k0_05", "capacity_sensitivity_k0_04", "capacity_sensitivity_k0_06"]
    _verify_capacity_triple_invariants(df[capacity_cols], inputs_df, capacity_weights, tolerance)

    # #3 (partial -- see module docstring re: raw_model_score gap):
    # cal_pd / challenger_risk_signal_type nullability pairing.
    thick = df["challenger_risk_signal_type"] == RISK_SCORE_TYPE_THICK
    thin = df["challenger_risk_signal_type"] == RISK_SCORE_TYPE_THIN
    if df.loc[thin, "cal_pd"].notna().any():
        raise ValueError("invariant #3 violated: cal_pd populated on a THIN_SCORECARD_PD_LIKE row.")
    if df.loc[thick, "cal_pd"].isna().any():
        raise ValueError("invariant #3 violated: cal_pd NULL on a CALIBRATED_CHAMPION_PD row.")

    # #4: l_full_* NOT NULL iff both its capacity and its multiplier are
    # NOT NULL for that (k, scenario) pair.
    for k_label, capacity_col, _k_val in _K_LABELS:
        for scenario in _SCENARIOS:
            l_full_col = f"l_full_{k_label}_{scenario}"
            mult_col = f"shadow_multiplier_{scenario}"
            expected_not_null = df[capacity_col].notna() & df[mult_col].notna()
            actual_not_null = df[l_full_col].notna()
            if not (expected_not_null == actual_not_null).all():
                raise ValueError(f"invariant #4 violated: {l_full_col} nullability mismatch.")

    # #5: k literal CHECK constraints -- the concrete mechanism preventing
    # Tier A from becoming a re-tuning exercise.
    if not (df["k_shadow_value"] == K_SHADOW).all():
        raise ValueError("invariant #5 violated: k_shadow_value drift.")
    if not (df["k_sensitivity_low_value"] == K_SENSITIVITY_LOW).all():
        raise ValueError("invariant #5 violated: k_sensitivity_low_value drift.")
    if not (df["k_sensitivity_high_value"] == K_SENSITIVITY_HIGH).all():
        raise ValueError("invariant #5 violated: k_sensitivity_high_value drift.")

    # #8: shadow_population_label is a valid enum value (the pipeline
    # itself is responsible for the pre-deployment capability
    # confirmation this label asserts -- this check only guards against
    # a typo/invalid value, not against a false confirmation).
    if not df["shadow_population_label"].isin(_VALID_SHADOW_POPULATION_LABELS).all():
        raise ValueError("invariant #8 violated: shadow_population_label has an invalid value.")

    # #9: PIT -- re-checked on the assembled frame (already enforced once
    # in compute_snapshot_availability; defense in depth).
    has_snapshot = df["fundamentals_snapshot_date"].notna()
    violating = has_snapshot & (df["fundamentals_snapshot_date"] >= df["scoring_run_timestamp"].dt.normalize())
    if violating.any():
        raise ValueError("invariant #9 violated: fundamentals_snapshot_date >= scoring_run_timestamp.")

    # #11: capacity_estimable derivation.
    recomputed_estimable = compute_capacity_estimable(
        df["float_availability_status"], df["commission_availability_status"], df["snapshot_availability_status"]
    )
    if not (recomputed_estimable == df["capacity_estimable"]).all():
        raise ValueError("invariant #11 violated: capacity_estimable does not match its derivation rule.")


def write_run_manifest(run_id: str, status: str, expected_agent_count: int, logged_agent_count: int,
                        failure_stage: str | None = None, error_code: str | None = None, sink=None) -> dict:
    """Writes the per-run completeness/failure manifest (implementation
    plan Section 3.3). Written unconditionally, with its OWN try/except
    -- logging but never raising, so a manifest-write failure cannot
    cascade into failing the pipeline either. expected_agent_count !=
    logged_agent_count is itself a signal this should not be reported as
    SUCCESS by the caller (see run_decision_log_stage)."""
    manifest = {
        "run_id": run_id,
        "shadow_logging_status": status,
        "expected_agent_count": expected_agent_count,
        "logged_agent_count": logged_agent_count,
        "failure_stage": failure_stage,
        "error_code": error_code,
    }
    try:
        if sink is not None and hasattr(sink, "write_manifest"):
            sink.write_manifest(manifest)
        else:
            logger.info("Stage 4C run manifest (no sink configured): %s", manifest)
    except Exception:
        logger.exception("Stage 4C run manifest write failed for run_id=%s -- manifest content: %s", run_id, manifest)
    return manifest


def write_decision_log(df: pd.DataFrame, sink) -> None:
    """Persists the Decision Log DataFrame to the confirmed sink.

    Correction applied per review (implementation plan Section 8, item
    4): this is NOT a NotImplementedError stub inside the pipeline's
    try/except -- that shape could make the system appear successfully
    deployed while silently failing every shadow write. No default sink
    exists yet (implementation plan Section 3.4); this function requires
    a real sink object and is only ever called when Stage 4C logging is
    explicitly enabled with one configured (see run_decision_log_stage
    and the pipeline integration's off-by-default feature flag)."""
    if sink is None:
        raise ValueError(
            "write_decision_log: no sink configured. Stage 4C logging must not be "
            "enabled until a real sink is confirmed (see docs/stage_4c_logger_"
            "implementation_plan.md Section 3.4) -- this is a configuration error, "
            "not a transient failure."
        )
    sink.write_decision_log_rows(df)


def run_decision_log_stage(df_agent_raw: pd.DataFrame, result_df: pd.DataFrame, run_id: str,
                            scoring_run_timestamp, capacity_weights: dict, capacity_weights_version: str,
                            artifact_versions: dict, sink,
                            shadow_population_label: str = SHADOW_POPULATION_FULL) -> dict:
    """The full Stage 9 orchestration run_credit_risk_pipeline.py's
    insertion point (implementation plan Section 4) wraps in a
    try/except: build, verify, write, and unconditionally write the run
    manifest. Returns the manifest dict.

    Atomicity (implementation plan Section 8, item 3 / verification item
    13): any exception anywhere in construction, verification, or the
    write itself results in logged_agent_count=0 and status=FAILED --
    never a partial write silently reported as SUCCESS. This function
    itself never raises; only write_run_manifest's own internal
    try/except can still fail (logged, not raised), by design."""
    expected = len(result_df)
    try:
        decision_log_df = build_decision_log_rows(
            df_agent_raw, result_df, run_id, scoring_run_timestamp,
            capacity_weights, capacity_weights_version, artifact_versions,
            shadow_population_label=shadow_population_label,
        )
        verify_decision_log_invariants(decision_log_df, capacity_weights)
        write_decision_log(decision_log_df, sink)
        return write_run_manifest(run_id, MANIFEST_STATUS_SUCCESS, expected, len(decision_log_df), sink=sink)
    except Exception as exc:
        logger.exception("Stage 4C Decision Log failed for run_id=%s -- live pipeline output unaffected", run_id)
        return write_run_manifest(
            run_id, MANIFEST_STATUS_FAILED, expected, 0,
            failure_stage=type(exc).__name__, error_code=str(exc)[:200], sink=sink,
        )
