"""
log_shadow_multiplier_cycle.py
=================================
Immutable, append-only decision log for the shadow continuous risk
multiplier ("C3 hybrid"). Every time the credit-risk pipeline is scored,
this script appends one row per agent to a running log file, capturing
exactly what the live tier policy decided AND what each C3 scenario would
have decided for that same agent on that same scoring run -- so months
from now, after calibration artifacts and policy versions have moved on,
this history can still be reconstructed exactly as it was at scoring time.

This is the SECOND stage of a five-stage shadow-monitoring architecture:

    Scoring Snapshot
        -> Immutable Shadow Decision Log      <-- this script
        -> Fixed Forward Outcome Windows
        -> Operational Health + Policy Migration + Outcome Monitoring
        -> Champion (live tier policy) vs Challenger (C3 base) Decision
        -> (only after enough matured cycles) Controlled Live Pilot

The later stages are deliberately NOT built here -- they need many
accumulated cycles of this log to be meaningful. This script only
accumulates that history, one scoring run at a time.

IMPORTANT (governing principles):
  * Read-only w.r.t. the live engine: this script never changes any
    existing column's value, and it has no effect whatsoever on what
    limit a borrower actually receives. It only records what the engine
    already computed.
  * Shadow failure must never fail lending, but it must never fail
    silently: optional columns that need --keep-intermediate degrade
    gracefully (NOTE, or an elevated WARNING for combined_cap -- see
    below); a genuinely broken scored_at/run_id, or a decision-key
    reproducibility problem, is loud, not swallowed.
  * A decision key is supposed to identify exactly one immutable
    decision. (run_id, msisdn) is that key, where run_id is the engine's
    own scored_at timestamp (confirmed to already be the internal run
    identifier). Re-logging the same scoring run must never create
    duplicate rows (an "exact duplicate" is silently skipped), but if the
    SAME key ever produces DIFFERENT persisted values between runs, that
    is a reproducibility problem the logger surfaces loudly as a
    CONFLICT -- it never silently overwrites history and never silently
    keeps only the first copy.
  * base is the primary C3 challenger; conservative is retained in full
    as a sensitivity/secondary scenario (prior retrospective analysis on
    this branch already showed conservative produces materially more
    migration than base).

combined_cap (capacity BEFORE both the live discrete tier multiplier and
the shadow continuous multiplier) is treated as a high-priority field,
not an ordinary optional one: without it, this log can still support
shadow-policy monitoring (migration, exposure redistribution, stability)
but NOT capacity-vs-risk decomposition -- so its absence prints an
elevated WARNING, not a routine NOTE.

Usage:
    python scripts\\log_shadow_multiplier_cycle.py ^
        --engine-output output\\engine_test_output.csv ^
        --log-file output\\shadow_multiplier_log.csv ^
        --persona-assignments-file segmentation_outputs\\persona_k8_profile\\k8_cluster_assignments.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extrafloat.engine.extrafloat_shadow_risk_multiplier import SHADOW_SCENARIOS  # noqa: E402
from segmentation.borrower_persona_clustering import digits  # noqa: E402

SCENARIOS = tuple(SHADOW_SCENARIOS.keys())  # ("base", "conservative")

# Same 1.10 / 0.90 convention as every other script this session
# (OVER_LIMIT_THRESHOLD / AGREEMENT_EDGES).
COHORT_UPLIFT_THRESHOLD = 0.10
COHORT_TIGHTEN_THRESHOLD = -0.10

# Monetary/multiplier amounts the engine computes deterministically for a given
# run -- a genuine re-run reproduces them far tighter than this; a real
# reproducibility conflict, or a real transition-cap effect, differs by orders
# of magnitude more than this tolerance. Named so neither is a magic number.
VALUE_COMPARISON_ATOL = 1e-6
VALUE_COMPARISON_RTOL = 1e-9
TRANSITION_BINDING_EPSILON = 1e-6

TRANSITION_CAP_LOWER = 0.75
TRANSITION_CAP_UPPER = 1.25

# Always present in engine output per FINAL_OUTPUT_COLUMNS / SHADOW_OUTPUT_COLUMNS
# -- the logger errors out (rather than degrading) if any of these are missing,
# since their absence means the input isn't a real pipeline scoring output.
REQUIRED_ENGINE_COLUMNS = (
    [
        "msisdn", "scored_at", "cal_pd", "shadow_calibrated_risk",
        "shadow_isotonic_model_version", "risk_tier", "live_tier_multiplier",
        "pd_decile", "assigned_limit", "assigned_limit_pre_round",
        "shadow_status", "shadow_policy_version",
        "final_decision_reason", "policy_reason", "combined_reason", "combined_top_driver",
    ]
    + [f"shadow_multiplier_{s}" for s in SCENARIOS]
    + [f"shadow_limit_pre_transition_{s}" for s in SCENARIOS]
    + [f"shadow_limit_post_transition_{s}" for s in SCENARIOS]
)

# Needs --keep-intermediate on the pipeline run. combined_cap is high-priority
# (elevated WARNING when absent); the rest degrade with an ordinary NOTE.
OPTIONAL_HIGH_PRIORITY_COLUMNS = ["combined_cap"]
OPTIONAL_COLUMNS = [
    "agent_category", "capacity_effective_ceiling", "is_thin_file",
    "regulatory_cap_applied", "policy_floor_applied", "active_floor_applied",
    "is_proven_good_borrower", "risk_cap_binding",
]

STRING_COLUMNS = {
    "run_id", "msisdn", "cycle_date", "shadow_isotonic_model_version", "shadow_policy_version",
    "risk_tier", "shadow_status", "agent_category", "persona_cluster", "persona_name",
    "final_decision_reason", "policy_reason", "combined_reason", "combined_top_driver",
} | {f"cohort_{s}" for s in SCENARIOS}

BOOL_COLUMNS = (
    {"is_thin_file", "regulatory_cap_applied", "policy_floor_applied", "active_floor_applied",
     "is_proven_good_borrower", "risk_cap_binding"}
    | {f"transition_cap_binding_{s}" for s in SCENARIOS}
    | {f"transition_threshold_exceeded_{s}" for s in SCENARIOS}
)


def _error(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


_BOOL_TRUE_STRINGS = {"true", "1", "yes"}
_BOOL_FALSE_STRINGS = {"false", "0", "no"}
# Sentinel for the string-column fillna trick below -- cannot realistically collide
# with real engine-output text (reason strings, model versions, tiers, etc).
_MISSING_STRING_SENTINEL = "\x00__MISSING__\x00"


def _normalize_bool_series(s: pd.Series) -> pd.Series:
    """Vectorized version of 'collapse to a comparable bool, NA stays NA'."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    notna = s.notna()
    strs = s[notna].astype(str).str.strip().str.lower()
    out.loc[notna] = strs.map(
        lambda v: True if v in _BOOL_TRUE_STRINGS else (False if v in _BOOL_FALSE_STRINGS else pd.NA)
    )
    return out


def _normalize_string_series(s: pd.Series) -> pd.Series:
    out = s.astype(object).where(s.notna(), None)
    notna_mask = out.notna()
    out.loc[notna_mask] = out.loc[notna_mask].astype(str).str.strip()
    return out


def _vectorized_equal(col: str, a: pd.Series, b: pd.Series) -> pd.Series:
    """Column-type-aware, fully vectorized version of the "values match" rule:
    missing equals missing, numeric compares with tolerance, string/bool compare
    exactly after normalization. See VALUE_COMPARISON_ATOL/RTOL above for why that
    tolerance is appropriate.
    """
    if col in BOOL_COLUMNS:
        na, nb = _normalize_bool_series(a), _normalize_bool_series(b)
        both_na = na.isna() & nb.isna()
        neither_na = na.notna() & nb.notna()
        return (both_na | (neither_na & (na.fillna(False) == nb.fillna(False)))).astype(bool)
    if col in STRING_COLUMNS:
        na, nb = _normalize_string_series(a), _normalize_string_series(b)
        return (na.fillna(_MISSING_STRING_SENTINEL) == nb.fillna(_MISSING_STRING_SENTINEL)).astype(bool)
    na = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    nb = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    return pd.Series(
        np.isclose(na, nb, atol=VALUE_COMPARISON_ATOL, rtol=VALUE_COMPARISON_RTOL, equal_nan=True),
        index=a.index,
    )


def build_log_rows(engine_df: pd.DataFrame, persona_path: Path) -> pd.DataFrame:
    """Build this cycle's new log rows from one scoring run's engine output.

    Exits the process (via _error) on any condition that would make the
    resulting log rows untrustworthy: missing scored_at, more than one
    scored_at value in the batch, a missing required column, or a duplicate
    (run_id, msisdn) key within the incoming batch itself.
    """
    if "scored_at" not in engine_df.columns:
        _error("--engine-output has no 'scored_at' column -- re-run the pipeline, which "
               "already stamps this column, rather than inventing a timestamp here.")

    scored_at_values = engine_df["scored_at"].dropna().unique()
    if len(scored_at_values) == 0:
        _error("--engine-output's 'scored_at' column is present but entirely empty.")
    if len(scored_at_values) > 1:
        _error(f"--engine-output contains {len(scored_at_values)} distinct scored_at/run_id "
               f"values ({sorted(str(v) for v in scored_at_values)[:5]}...) -- expected exactly "
               f"one scoring run per input file. If this file was concatenated across cycles, "
               f"split it and log each cycle separately.")
    run_id = str(scored_at_values[0])

    try:
        cycle_date = pd.to_datetime(run_id, utc=True).date().isoformat()
    except (ValueError, TypeError):
        _error(f"Could not parse scored_at value {run_id!r} as a timestamp.")

    missing_required = [c for c in REQUIRED_ENGINE_COLUMNS if c not in engine_df.columns]
    if missing_required:
        _error(f"--engine-output is missing required column(s): {missing_required} -- this "
               f"doesn't look like real pipeline scoring output.")

    df = engine_df.copy()
    df["msisdn"] = digits(df["msisdn"])
    n_unkeyable = int(df["msisdn"].isna().sum())
    if n_unkeyable:
        print(f"NOTE: {n_unkeyable} row(s) had an unusable msisdn and were dropped from this cycle's log.")
        df = df[df["msisdn"].notna()].copy()

    dup_mask = df["msisdn"].duplicated(keep=False)
    if dup_mask.any():
        n_dup_keys = df.loc[dup_mask, "msisdn"].nunique()
        _error(f"--engine-output contains {n_dup_keys} msisdn value(s) appearing more than once "
               f"for this single run_id ({run_id}) -- a data-integrity problem upstream in the "
               f"scoring output. Nothing was logged.")

    available_optional = [c for c in OPTIONAL_COLUMNS if c in df.columns]
    missing_optional = [c for c in OPTIONAL_COLUMNS if c not in df.columns]
    if missing_optional:
        print(f"NOTE: optional column(s) unavailable this cycle (needs --keep-intermediate): "
              f"{missing_optional}")
    missing_high_priority = [c for c in OPTIONAL_HIGH_PRIORITY_COLUMNS if c not in df.columns]
    for col in missing_high_priority:
        print(f"WARNING: {col} unavailable; this cycle can support shadow-policy monitoring "
              f"but not capacity-vs-risk decomposition.")
    available_high_priority = [c for c in OPTIONAL_HIGH_PRIORITY_COLUMNS if c in df.columns]

    log_df = pd.DataFrame({"msisdn": df["msisdn"].values})
    log_df["run_id"] = run_id
    log_df["cycle_date"] = cycle_date
    for col in REQUIRED_ENGINE_COLUMNS:
        if col in ("msisdn", "scored_at"):
            continue
        log_df[col] = df[col].values
    for col in OPTIONAL_COLUMNS + OPTIONAL_HIGH_PRIORITY_COLUMNS:
        log_df[col] = df[col].values if col in available_optional + available_high_priority else np.nan

    log_df["persona_cluster"] = np.nan
    log_df["persona_name"] = np.nan
    if persona_path and Path(persona_path).exists():
        persona = pd.read_csv(persona_path)
        wanted = [c for c in ("persona_cluster", "persona_name") if c in persona.columns]
        if "phonenumber" in persona.columns and wanted:
            persona = persona.copy()
            persona["_key"] = digits(persona["phonenumber"])
            # Cardinality guard: the join must never multiply logger rows, even if the
            # persona-assignments file has a duplicated phonenumber.
            persona_small = persona[["_key"] + wanted].drop_duplicates(subset="_key")
            merged = log_df[["msisdn"]].merge(
                persona_small, left_on="msisdn", right_on="_key", how="left"
            )
            n_matched = int(merged["_key"].notna().sum())
            for col in wanted:
                log_df[col] = merged[col].values
            print(f"NOTE: matched {n_matched:,} / {len(log_df):,} row(s) to a K=8 persona "
                  f"from {persona_path}.")
        else:
            print(f"NOTE: {persona_path} is missing phonenumber/persona_cluster/persona_name -- "
                  f"skipping persona join.")
    else:
        print(f"NOTE: persona-assignments file not found ({persona_path}) -- skipping persona join.")

    for scenario in SCENARIOS:
        pre = log_df[f"shadow_limit_pre_transition_{scenario}"]
        post = log_df[f"shadow_limit_post_transition_{scenario}"]
        assigned = log_df["assigned_limit"]
        impact = post - pre
        log_df[f"transition_impact_{scenario}"] = impact

        pre_post_known = pre.notna() & post.notna()
        binding = (impact.abs() > TRANSITION_BINDING_EPSILON).astype("boolean")
        log_df[f"transition_cap_binding_{scenario}"] = binding.where(pre_post_known, pd.NA)

        assigned_known = assigned.notna() & (assigned > 0)
        lower = assigned * TRANSITION_CAP_LOWER
        upper = assigned * TRANSITION_CAP_UPPER
        exceeded = ((pre < lower) | (pre > upper)).astype("boolean")
        log_df[f"transition_threshold_exceeded_{scenario}"] = exceeded.where(
            pre_post_known & assigned_known, pd.NA
        )

        pct_change = pd.Series(
            np.where(assigned > 0, (post - assigned) / assigned, np.nan), index=log_df.index
        )
        log_df[f"pct_change_{scenario}"] = pct_change

        # Frozen-at-scoring-time cohort label, fully vectorized. Eligible only when
        # the shadow result is real (status ok, live limit positive, post-transition
        # value known); otherwise "unclassified", never defaulted into "neutral".
        eligible = (log_df["shadow_status"] == "ok") & (assigned > 0) & post.notna() & pct_change.notna()
        log_df[f"cohort_{scenario}"] = np.select(
            [eligible & (pct_change > COHORT_UPLIFT_THRESHOLD), eligible & (pct_change < COHORT_TIGHTEN_THRESHOLD), eligible],
            ["uplift", "tighten", "neutral"],
            default="unclassified",
        )

    return log_df


def _load_existing_log(log_path: Path, expected_columns: list[str]) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame(columns=expected_columns)
    # dtype=str on the key columns: round-tripping a purely-numeric msisdn through CSV
    # otherwise gets auto-cast to int64 on reload, which would silently break key
    # equality against the in-memory (string) msisdn built by build_log_rows().
    existing = pd.read_csv(log_path, dtype={"run_id": str, "msisdn": str}, low_memory=False)
    dup_mask = existing.duplicated(subset=["run_id", "msisdn"], keep=False)
    if dup_mask.any():
        n_dup = existing.loc[dup_mask, ["run_id", "msisdn"]].drop_duplicates().shape[0]
        _error(f"{log_path} already contains {n_dup} duplicate (run_id, msisdn) key(s) -- the "
               f"supposedly-immutable history is already inconsistent. Refusing to append until "
               f"this is resolved by hand.")
    return existing


def append_to_log(new_rows: pd.DataFrame, log_path: Path) -> int:
    """Append new_rows to log_path using the (run_id, msisdn) idempotency rule.

    Returns the process exit code (0 normally, 1 if any conflicting duplicate
    keys were found and withheld).
    """
    existing = _load_existing_log(log_path, list(new_rows.columns))
    # Match _load_existing_log's key dtype so (run_id, msisdn) tuples compare equal
    # regardless of whether a value came from this run's in-memory DataFrame or a
    # freshly-reloaded CSV.
    new_rows = new_rows.copy()
    new_rows["run_id"] = new_rows["run_id"].astype(str)
    new_rows["msisdn"] = new_rows["msisdn"].astype(str)

    if existing.empty:
        combined = new_rows
        n_new, n_skipped_exact, n_conflict = len(new_rows), 0, 0
    else:
        # Fully vectorized (no per-row Python loop -- this runs at full-pipeline
        # scale, ~140k+ rows per cycle): a single left merge finds which incoming
        # keys already exist in the log, then each compare_col's equality is a
        # whole-column vectorized comparison via _vectorized_equal.
        compare_cols = [c for c in new_rows.columns if c not in ("run_id", "msisdn")
                        and c in existing.columns]
        right = existing[["run_id", "msisdn"] + compare_cols]
        merged = new_rows.merge(right, on=["run_id", "msisdn"], how="left",
                                 suffixes=("_new", "_exist"), indicator=True)

        is_both = (merged["_merge"] == "both").to_numpy()
        is_new_key = (merged["_merge"] == "left_only").to_numpy()

        match = np.ones(len(merged), dtype=bool)
        for col in compare_cols:
            match &= _vectorized_equal(col, merged[f"{col}_new"], merged[f"{col}_exist"]).to_numpy()

        is_exact = is_both & match
        is_conflict = is_both & ~match

        n_new = int(is_new_key.sum())
        n_skipped_exact = int(is_exact.sum())
        n_conflict = int(is_conflict.sum())

        if n_conflict:
            conflicting_keys = list(zip(
                merged.loc[is_conflict, "run_id"].head(10), merged.loc[is_conflict, "msisdn"].head(10)
            ))
            print(f"CONFLICT: {n_conflict} existing decision key(s) have different incoming "
                  f"values; existing immutable records retained, conflicting incoming rows NOT "
                  f"logged. Affected keys (first 10): {conflicting_keys}")

        to_append_df = new_rows.loc[is_new_key]
        combined = pd.concat([existing, to_append_df], ignore_index=True)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(log_path, index=False)

    print(f"\nRun summary: {len(new_rows)} row(s) read for this cycle -> "
          f"{n_new} newly appended, {n_skipped_exact} skipped as exact duplicates, "
          f"{n_conflict} withheld as conflicting duplicates.")
    print(f"Log file now has {len(combined):,} row(s) total: {log_path}")

    return 1 if n_conflict else 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine-output", default="output/engine_test_output.csv")
    ap.add_argument("--log-file", default="output/shadow_multiplier_log.csv")
    ap.add_argument("--persona-assignments-file",
                     default="segmentation_outputs/persona_k8_profile/k8_cluster_assignments.csv",
                     help="optional -- K=8 persona assignments (phonenumber, persona_cluster, "
                          "persona_name); skipped with a NOTE if not found")
    args = ap.parse_args(argv)

    engine_path = Path(args.engine_output)
    if not engine_path.exists():
        _error(f"--engine-output not found: {engine_path}")
    engine_df = pd.read_csv(engine_path)

    new_rows = build_log_rows(engine_df, Path(args.persona_assignments_file))
    exit_code = append_to_log(new_rows, Path(args.log_file))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
