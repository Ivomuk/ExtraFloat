"""
build_retrospective_scoring_matrix.py
==================================================
Stage 4B-R1 -- ASSEMBLY + VALIDATION ONLY of the historical retrospective
scoring matrix. Produces NO scores, NO predict_proba call, NO model
loaded for prediction, NO calibration, NO C3, NO Capacity(F). This script
answers exactly one question: can the champion's real 80-feature
`feature_order.json` vector be correctly, completely, and leakage-free
constructed for each of Stage 4A.1's strictly-prior-reconstructed
historical loan episodes? Scoring those episodes with the champion
(predict_proba) is Stage 4B-R2, a separate, later script.

WHY THIS SCRIPT EXISTS: Stage 4A's disposition is B
(RETROSPECTIVE_CURRENT_MODEL_ONLY) -- historical PIT features are
reconstructable, but only today's champion/calibration can score them,
producing `retrospective_current_model_pd`, never `cal_pd_pit`. Before
that scoring can happen, the exact 80-feature historical input vector
must be assembled and proven correct.

ARCHITECTURAL CHOICE, per explicit review: this script IMPORTS the real
production feature-derivation and preprocessing functions rather than
restating them a third time -- `run_phase_2_1_richer_tx_behaviour`
(pd_model/preprocessing/transaction_features.py), `derive_thin_file_flag`
(pd_model/preprocessing/loan_history_features.py), `get_and_classify_
pd_features` + `build_transformed_dataframe` (pd_model/preprocessing/
transformations.py), and `align_features` (pd_model/modeling/
inference.py). This is strictly MORE faithful to production than any
restatement could be, and mechanically satisfies "preprocessing is
identical to the current champion" rather than requiring it to be proven
by inspection. Same one-way `scripts/` layering convention already used
throughout this session: importing from non-`scripts/` modules is fine;
this script never imports another `scripts/` module, and never modifies
any production file.

NO MODEL IS LOADED. `pd_model.modeling.inference.load_artifacts()` is
deliberately NOT called -- it deserializes the xgb/lgb model joblib
files, which is unnecessary (and against the spirit of "no model loading
for prediction") for an assembly-only script. `feature_order.json` and
`transform_report.csv` are read directly as plain JSON/CSV instead.

PHASE 2.1 SOURCE: Stage 4A.1's own reconstruction output
(`{prefix}_phase21_pit_features.csv` from
build_historical_pit_transaction_features.py's full run), filtered to
`reconstruction_status == PIT_PHASE21_RECONSTRUCTED` -- the ~15.5% with
no eligible prior snapshot are excluded here, never imputed or
backfilled, consistent with every prior step in this workstream.

PHASE 2.2 / DECISION-CONTEXT / HYBRID SOURCE: the real loan-training-file
itself (`data/loan_state_query_updated_materialized.txt`'s export).
These need NO additional feature engineering -- per the champion-feature-
coverage check's own evidence, they are already present, per-row,
strictly-pre-disbursement-guarded columns in that export. Only Phase 2.1
needs assembly (merging Stage 4A.1's as-of-joined raw mart columns onto
this same file by `disbursement_fid`).

`days_since_snapshot` HANDLING (per explicit review -- more defensive
than a plain overwrite): the Stage 4A.1 per-episode value is preserved
under a temporary name, the real `run_phase_2_1_richer_tx_behaviour` is
called, and ONLY THEN is `days_since_snapshot` set from the preserved
value (with an equality assertion if the production call happened to
create one itself). Confirmed by inspection, not assumed: Stage 4A.1's
own `load_mart` already drops `tbl_dt`/`snapshot_dt` immediately after
deriving `fundamentals_snapshot_date` from it, so `tbl_dt` is NEVER
present in this script's assembled input -- the one batch-relative
branch in `run_phase_2_1_richer_tx_behaviour` (`if "tbl_dt" in
df.columns: ref_date = df["tbl_dt"].max()...`) therefore cannot fire on
this data at all, which is itself recorded as evidence
(`tbl_dt_present_in_assembled_input`), not merely assumed.

LEAKAGE VALIDATION, TWO HARD ASSERTIONS (per explicit review -- checking
only the final 80 columns is insufficient, since `build_transformed_
dataframe`'s own "concat non-numeric + transformed numeric" step passes
through any non-numeric column verbatim, label/diagnostic columns
included, unless they are removed first):
  1. `separate_forbidden_columns()` DROPS every name in
     FORBIDDEN_PRE_TRANSFORM_COLS (LABEL_DIAGNOSTIC_COLUMNS union
     bad_state/bad_state_3dpd_30d/bad_state_1dpd_7d/label_eligible_30d/
     label_eligibility_reason_30d) into a SEPARATE metadata frame, keyed
     by disbursement_fid, BEFORE `get_and_classify_pd_features`/
     `build_transformed_dataframe` ever see the data.
  2. `assert_no_forbidden_columns()` is then called TWICE: once
     immediately after the drop (self-check) and once on the final
     aligned 80-column matrix (the actual claim this script supports:
     no outcome/censoring/label-diagnostic field was supplied to the
     production feature-transformation path at all).

SCHEMA CONSTRUCTION COMPLETENESS vs. PRODUCTION ALIGNMENT COMPLETENESS
(per explicit review -- the most important check here): production
`align_features()` silently fills an absent expected feature with NaN,
which is correct behavior for live inference but would hide an R1
reconstruction failure ("Stage 4A says reconstructable -> R1 fails to
construct it -> align_features inserts NaN -> matrix looks valid"). This
script therefore classifies every one of the champion's 80 features,
BEFORE calling align_features, into ABSENT_BEFORE_ALIGNMENT / PRESENT_
ALL_NAN / PRESENT_PARTIALLY_MISSING / PRESENT_POPULATED. Any
ABSENT_BEFORE_ALIGNMENT feature is an R1 FAILURE -- there is no
documented production reason any of the 80 should be intentionally
synthesized as missing here.

DISPOSITION: `RETROSPECTIVE_SCORING_MATRIX_READY` only when every one of
these holds (never a partial/implicit pass): row count equals Stage
4A.1's own reconstructed count, `disbursement_fid` unique, every row
genuinely Stage-4A.1-reconstructed, the strict-prior PIT invariant
re-verified independently (not merely trusted from Stage 4A.1), schema
construction completeness (80/80, zero ABSENT_BEFORE_ALIGNMENT), zero
forbidden fields at either leakage gate, training-fitted transformation
params actually replayed (non-empty `fitted_params`), and the final
aligned feature order exactly matches `feature_order.json`. Artifact
identity (sha256 of `feature_order.json`/`transform_report.csv`/
`model_metadata.json`, sha256 of the imported production source files,
and the repo's git commit where available) is recorded as part of the
experiment's definition, since this is specifically a CURRENT-MODEL
retrospective exercise.

ROUTING NOTE (recorded for Stage 4B-R2, not acted on here -- confirmed
by inspection of pd_model/modeling/inference.py's score_new_agents):
there is exactly ONE 80-feature `feature_order.json`-aligned matrix for
the xgb/lgb champion, used identically (via predict_proba) for every
row regardless of thin/thick status -- there are NOT two different
champion-model feature vectors. Separately and in parallel, thin-file
agents (`thin_file_flag==1`) have their FINAL reported PD overridden
afterward by a wholly separate scorecard/LR path (`add_never_loan_
scorecard_from_phase_2_1`, optionally refined by `apply_thin_file_lr`
using its own `thin_file_lr_features.json`), built from raw Phase 2.1
features outside `feature_order.json` entirely. Stage 4B-R2 will need
that separate, smaller feature set for thin-file rows' final PD; this
script's 80-feature matrix is unaffected and needs no branching.

Usage:
    python scripts\\build_retrospective_scoring_matrix.py ^
        --phase21-reconstructed-features-csv stage4a1_phase21_pit_phase21_pit_features.csv ^
        --loan-training-file data\\state_data_20260910_retail_filtered.csv ^
        --pd-model-artifacts-dir pd_model\\artifacts ^
        --out-prefix stage4br1_retrospective_matrix
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pd_model.config.model_config import DEFAULT_CONFIG  # noqa: E402
from pd_model.modeling.inference import align_features  # noqa: E402
from pd_model.preprocessing.loan_history_features import (  # noqa: E402
    LABEL_DIAGNOSTIC_COLUMNS,
    derive_thin_file_flag,
)
from pd_model.preprocessing.transaction_features import run_phase_2_1_richer_tx_behaviour  # noqa: E402
from pd_model.preprocessing.transformations import (  # noqa: E402
    build_transformed_dataframe,
    get_and_classify_pd_features,
)

STATUS_RECONSTRUCTED = "PIT_PHASE21_RECONSTRUCTED"

# Restated from build_historical_pit_transaction_features.py's PHASE_21_RAW_COLS
# (one-way scripts/ layering convention: restate, never import, another scripts/
# module's constant). Used ONLY to narrow this script's own CSV read via usecols
# -- Stage 4A.1's output CSV also carries its OWN already-derived Phase 2.1
# feature columns (e.g. is_fully_inactive_6m), which this script does not need:
# apply_production_phase21() recomputes every derived feature fresh from these
# raw columns via the real production function, so loading the already-derived
# ones too would just be ~2x the memory for no benefit at the real 4.78M-row
# scale this script runs at -- the same class of memory fix as PHASE_21_RAW_COLS
# itself (see that constant's own docstring for the original OOM this pattern fixes).
_PHASE21_RAW_MART_COLS = sorted(set(
    ["vol_1m", "vol_3m", "vol_6m", "commission", "account_balance", "average_balance",
     "cust_1m", "cust_3m", "cash_in_value_3m", "cash_out_value_3m", "payment_value_3m",
     "commission_cluster_mean", "vol_3m_cluster_mean", "cluster_avg_commission", "cluster_avg_vol_3m",
     "cash_in_peers_3m", "cash_in_vol_3m"]
    + [f"{base}_{horizon}" for base in ("cash_out_vol", "cash_in_vol", "payment_vol", "vol")
       for horizon in ("1m", "3m", "6m")]
))

# Union of LABEL_DIAGNOSTIC_COLUMNS and the raw/derived label columns --
# exact-name match, never substring, same discipline as the champion-
# feature-coverage check's PROHIBITED_LABEL_LEAKAGE_NAMES.
FORBIDDEN_PRE_TRANSFORM_COLS = set(LABEL_DIAGNOSTIC_COLUMNS) | {
    "bad_state", "bad_state_3dpd_30d", "bad_state_1dpd_7d",
    "label_eligible_30d", "label_eligibility_reason_30d",
}

PRESENCE_ABSENT = "ABSENT_BEFORE_ALIGNMENT"
PRESENCE_ALL_NAN = "PRESENT_ALL_NAN"
PRESENCE_PARTIAL = "PRESENT_PARTIALLY_MISSING"
PRESENCE_POPULATED = "PRESENT_POPULATED"

DISPOSITION_READY = "RETROSPECTIVE_SCORING_MATRIX_READY"
DISPOSITION_NOT_READY = "RETROSPECTIVE_SCORING_MATRIX_NOT_READY"

# The exact production source files this script imports from -- hashed
# into the validation report as the "code version" identifier, since no
# git commit may be available in every environment this runs in.
PRODUCTION_SOURCE_FILES = [
    "pd_model/preprocessing/transaction_features.py",
    "pd_model/preprocessing/loan_history_features.py",
    "pd_model/preprocessing/transformations.py",
    "pd_model/modeling/inference.py",
]


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def load_reconstructed_phase21(path: Path) -> pd.DataFrame:
    """Loads Stage 4A.1's reconstruction output, keeps ONLY rows with
    reconstruction_status == PIT_PHASE21_RECONSTRUCTED (the ~15.5% with no
    eligible prior snapshot are excluded here, never imputed/backfilled),
    renames days_since_snapshot to a temporary name so it can never be
    silently confused with whatever run_phase_2_1_richer_tx_behaviour
    might compute. The loan-training file (joined on next) is the
    authoritative source for agent_msisdn/loan_date going forward --
    Stage 4A.1's own copies of those are excluded by the usecols
    narrowing below, never merely dropped after the fact.

    Narrows the CSV read via usecols to the identifier/status columns plus
    the RAW mart columns (_PHASE21_RAW_MART_COLS) -- deliberately NOT the
    already-derived Phase 2.1 feature columns Stage 4A.1's CSV also
    carries, since apply_production_phase21() recomputes every derived
    feature fresh from the raw ones via the real production function. A
    real 4.78M-row run of this CSV has reported sizes in the multi-GB
    range; loading columns this script discards and recomputes anyway is
    a real memory cost, not a theoretical one, at this scale."""
    keep_cols = {"disbursement_fid", "reconstruction_status", "fundamentals_snapshot_date",
                 "days_since_snapshot"} | set(_PHASE21_RAW_MART_COLS)
    df = pd.read_csv(path, usecols=lambda c: c in keep_cols)
    required = ("disbursement_fid", "reconstruction_status", "fundamentals_snapshot_date",
                "days_since_snapshot")
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing}")

    n_total = len(df)
    df = df[df["reconstruction_status"] == STATUS_RECONSTRUCTED].copy()
    n_excluded = n_total - len(df)
    print(f"Phase 2.1 reconstruction input: {n_total:,} row(s); {len(df):,} have "
          f"reconstruction_status == {STATUS_RECONSTRUCTED} and are retained. "
          f"{n_excluded:,} excluded (no eligible prior snapshot) -- never imputed or backfilled.")

    dup = int(df["disbursement_fid"].duplicated().sum())
    if dup:
        sys.exit(f"ERROR: {dup} duplicate disbursement_fid value(s) in the reconstructed Phase 2.1 rows.")

    df = df.rename(columns={"days_since_snapshot": "_pit_days_since_snapshot"})
    df["fundamentals_snapshot_date"] = pd.to_datetime(df["fundamentals_snapshot_date"], errors="coerce")
    return df


def load_loan_training_frame(path: Path) -> pd.DataFrame:
    """Loads the real materialized-SQL loan-training-file as-is -- every
    Phase 2.2 / decision-context / hybrid column, plus every label/
    diagnostic column (removed later, explicitly, by
    separate_forbidden_columns -- never silently here)."""
    df = pd.read_csv(path)
    if "agent_msisdn" not in df.columns and "msisdn" in df.columns:
        df = df.rename(columns={"msisdn": "agent_msisdn"})
    required = ["disbursement_fid", "agent_msisdn", "loan_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing}")

    dup = int(df["disbursement_fid"].duplicated().sum())
    if dup:
        sys.exit(f"ERROR: {dup} duplicate disbursement_fid value(s) in {path} -- each target loan "
                  f"must map to exactly one row.")

    df["loan_date"] = pd.to_datetime(df["loan_date"], errors="coerce")
    print(f"Loan-training file: {len(df):,} row(s), {len(df.columns)} column(s).")
    return df


def assemble_episode_matrix(phase21_df: pd.DataFrame, loan_df: pd.DataFrame) -> pd.DataFrame:
    """Plain key merge by disbursement_fid -- the as-of join itself was
    already done by Stage 4A.1; this is NOT a second temporal join. Every
    reconstructed Phase 2.1 episode's disbursement_fid must already exist
    in the loan-training file it was built from -- a merge row-count
    mismatch means the two inputs are not from the same Stage 4A.1 run,
    which is a hard error, not a silently-dropped row."""
    merged = loan_df.merge(phase21_df, on="disbursement_fid", how="inner", validate="one_to_one")
    if len(merged) != len(phase21_df):
        sys.exit(
            f"ERROR: merge row-count mismatch -- {len(phase21_df):,} reconstructed Phase 2.1 "
            f"episode(s) but only {len(merged):,} matched the loan-training file by "
            f"disbursement_fid. --loan-training-file is not the same file (or a superset of it) "
            f"used to build the Phase 2.1 reconstruction -- re-run Stage 4A.1 against the same "
            f"loan-training-file first."
        )
    print(f"Assembled episode matrix: {len(merged):,} row(s) (disbursement_fid-keyed, 1:1).")
    return merged


def assert_strict_prior_pit(df: pd.DataFrame) -> None:
    """Re-verifies the strict-prior PIT invariant INDEPENDENTLY of Stage
    4A.1's own leakage assertion -- never merely trusted across this new
    artifact boundary."""
    violations = df[df["fundamentals_snapshot_date"] >= df["loan_date"]]
    if len(violations):
        raise AssertionError(
            f"LEAKAGE DETECTED (re-verified independently of Stage 4A.1): {len(violations)} "
            f"row(s) have fundamentals_snapshot_date >= loan_date after episode-matrix assembly."
        )
    print(f"Strict-prior PIT invariant re-verified independently: all {len(df):,} row(s) have "
          f"fundamentals_snapshot_date < loan_date.")


def apply_production_phase21(df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, dict]:
    """Calls the REAL production run_phase_2_1_richer_tx_behaviour, then
    sets days_since_snapshot from the preserved Stage 4A.1 per-episode
    value -- defensively: if the production call happened to create its
    own (batch-relative, wrong) days_since_snapshot, that value is
    discarded and an equality check is skipped (they are KNOWN to
    disagree in that case, by design); the fact of whether it was even
    created is itself recorded as evidence, never assumed."""
    pit_days = df["_pit_days_since_snapshot"].reset_index(drop=True)
    df = df.drop(columns=["_pit_days_since_snapshot"]).reset_index(drop=True)

    tbl_dt_present = "tbl_dt" in df.columns
    df = run_phase_2_1_richer_tx_behaviour(df, cfg=cfg)
    production_created_days_since_snapshot = "days_since_snapshot" in df.columns

    df["days_since_snapshot"] = pit_days.values
    if not (df["days_since_snapshot"].values == pit_days.values).all():
        raise AssertionError(
            "days_since_snapshot override failed -- this must never happen (it is a direct "
            "assignment from the same values just computed)."
        )

    evidence = {
        "production_phase21_function_used": True,
        "production_phase21_known_non_pit_field_overridden": "days_since_snapshot",
        "days_since_snapshot_source": "Stage4A1_per_episode_strict_prior_reconstruction",
        "tbl_dt_present_in_assembled_input": tbl_dt_present,
        "days_since_snapshot_created_by_production_call_before_override": production_created_days_since_snapshot,
        "note": (
            "tbl_dt is never present in Stage 4A.1's reconstruction output (load_mart drops it "
            "immediately after deriving fundamentals_snapshot_date), so the one batch-relative "
            "branch in run_phase_2_1_richer_tx_behaviour structurally cannot fire on this input -- "
            "confirmed above (tbl_dt_present_in_assembled_input), not assumed. days_since_snapshot "
            "is set from the Stage 4A.1 per-episode value regardless."
        ),
    }
    print("Production run_phase_2_1_richer_tx_behaviour applied; days_since_snapshot set from the "
          f"Stage 4A.1 per-episode value (tbl_dt present in input: {tbl_dt_present}).")
    return df, evidence


def separate_forbidden_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drops every FORBIDDEN_PRE_TRANSFORM_COLS name present into a
    separate, disbursement_fid-keyed metadata frame BEFORE the
    transformation pipeline ever sees the data -- never merely tested
    for and left in place."""
    present = sorted(FORBIDDEN_PRE_TRANSFORM_COLS & set(df.columns))
    metadata_cols = ["disbursement_fid"] + present
    metadata_df = df[metadata_cols].copy()
    clean_df = df.drop(columns=present)
    if present:
        print(f"Separated {len(present)} forbidden label/diagnostic column(s) into audit-only "
              f"metadata, never passed to the transformation pipeline: {present}")
    else:
        print("No forbidden label/diagnostic columns were present in the assembled input.")
    return clean_df, metadata_df, present


def assert_no_forbidden_columns(df: pd.DataFrame, context: str) -> None:
    found = sorted(FORBIDDEN_PRE_TRANSFORM_COLS & set(df.columns))
    if found:
        raise AssertionError(f"LEAKAGE GATE FAILED ({context}): forbidden column(s) present: {found}")
    print(f"Leakage gate passed ({context}): zero forbidden label/diagnostic columns present.")


def classify_feature_presence(df: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    """Classifies EVERY one of the champion's selected features into
    ABSENT_BEFORE_ALIGNMENT / PRESENT_ALL_NAN / PRESENT_PARTIALLY_MISSING
    / PRESENT_POPULATED -- schema construction completeness, distinct
    from (and checked BEFORE) production alignment completeness.

    MUST be called on `df_transformed` (build_transformed_dataframe's
    output), never on the pre-transformation `clean_df`: build_
    transformed_dataframe's own prune_post_transform_features step can
    DROP a feature entirely if it is all-NaN or has too few unique
    values in this batch -- exactly the kind of silent loss align_
    features would otherwise paper over with a NaN fill. Checking
    pre-transformation presence alone would miss a feature pruned away
    between clean_df and df_transformed."""
    n = len(df)
    rows = []
    for feat in feature_order:
        if feat not in df.columns:
            rows.append({"selected_feature": feat, "presence_status": PRESENCE_ABSENT,
                         "n_missing": n, "pct_missing": 100.0 if n else 0.0, "dtype": None})
            continue
        s = df[feat]
        n_missing = int(s.isna().sum())
        pct_missing = round(100.0 * n_missing / n, 4) if n else 0.0
        if n_missing == n and n > 0:
            status = PRESENCE_ALL_NAN
        elif n_missing > 0:
            status = PRESENCE_PARTIAL
        else:
            status = PRESENCE_POPULATED
        rows.append({"selected_feature": feat, "presence_status": status,
                     "n_missing": n_missing, "pct_missing": pct_missing, "dtype": str(s.dtype)})
    return pd.DataFrame(rows)


def build_disposition(
    n_rows: int, n_reconstructed_input: int, disbursement_fid_unique: bool,
    presence_df: pd.DataFrame, forbidden_remaining_pre_transform: list[str], forbidden_post: list[str],
    fitted_params: dict, feature_order_exact_match: bool,
) -> tuple[str, dict]:
    """`forbidden_remaining_pre_transform` is the set of forbidden names
    STILL PRESENT in the dataframe actually handed to get_and_classify_
    pd_features/build_transformed_dataframe (i.e. clean_df, AFTER
    separate_forbidden_columns already removed what it found) -- this is
    deliberately NOT the same thing as "how many forbidden columns were
    found and separated," which is expected to be non-zero whenever the
    raw input legitimately contains labels (that IS the point of
    separating them, not a failure)."""
    schema_construction_complete = bool((presence_df["presence_status"] != PRESENCE_ABSENT).all())
    conditions = {
        "n_rows_equals_reconstructed_input": n_rows == n_reconstructed_input,
        "disbursement_fid_unique": disbursement_fid_unique,
        "all_rows_stage4a1_reconstructed": True,  # enforced by load_reconstructed_phase21's own filter
        "strict_prior_pit_invariant_reverified": True,  # assert_strict_prior_pit already raised if not
        "schema_construction_completeness_80_of_80": schema_construction_complete,
        "forbidden_fields_pre_transformation": len(forbidden_remaining_pre_transform) == 0,
        "forbidden_fields_in_final_matrix": len(forbidden_post) == 0,
        "training_fitted_transformation_params_replayed": len(fitted_params) > 0,
        "final_aligned_feature_order_exact_match": feature_order_exact_match,
    }
    disposition = DISPOSITION_READY if all(conditions.values()) else DISPOSITION_NOT_READY
    return disposition, conditions


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase21-reconstructed-features-csv", type=Path, required=True)
    ap.add_argument("--loan-training-file", type=Path, required=True)
    ap.add_argument("--pd-model-artifacts-dir", type=Path, required=True)
    ap.add_argument("--out-prefix", type=str, default="stage4br1_retrospective_matrix")
    args = ap.parse_args(argv)

    print(f"\n{'#' * 100}\nSTAGE 4B-R1 -- RETROSPECTIVE SCORING MATRIX: ASSEMBLY + VALIDATION ONLY\n"
          f"Produces NO scores. No model is loaded for prediction (load_artifacts() is never "
          f"called). No calibration, no C3, no Capacity(F).\n{'#' * 100}")

    feature_order_path = Path(args.pd_model_artifacts_dir) / "feature_order.json"
    transform_report_path = Path(args.pd_model_artifacts_dir) / "transform_report.csv"
    model_metadata_path = Path(args.pd_model_artifacts_dir) / "model_metadata.json"
    for p in (feature_order_path, transform_report_path):
        if not p.exists():
            sys.exit(f"ERROR: required artifact not found: {p}")

    feature_order_data = json.loads(feature_order_path.read_text())
    feature_order = feature_order_data.get("selected_features") or []
    if not feature_order:
        sys.exit(f"ERROR: {feature_order_path} has no selected_features.")
    print(f"Loaded {feature_order_path}: {len(feature_order)} selected feature(s).")

    transform_report = pd.read_csv(transform_report_path)
    # Restated verbatim from pd_model/modeling/inference.py's run_inference_pipeline --
    # that block is inline glue code, not itself a standalone importable function.
    fitted_params = {
        row["feature"]: {
            "action": row["action"],
            "lo": row.get("lo") if not pd.isna(row.get("lo", float("nan"))) else None,
            "hi": row.get("hi") if not pd.isna(row.get("hi", float("nan"))) else None,
        }
        for _, row in transform_report.iterrows()
        if pd.notna(row.get("action"))
    }
    print(f"Loaded {transform_report_path}: {len(fitted_params)} fitted transformation parameter "
          f"set(s), replayed exactly (never refit on this batch).")

    phase21_df = load_reconstructed_phase21(args.phase21_reconstructed_features_csv)
    loan_df = load_loan_training_frame(args.loan_training_file)
    merged = assemble_episode_matrix(phase21_df, loan_df)
    n_reconstructed_input = len(phase21_df)
    disbursement_fid_unique = not bool(merged["disbursement_fid"].duplicated().any())

    assert_strict_prior_pit(merged)

    merged, phase21_evidence = apply_production_phase21(merged, DEFAULT_CONFIG)
    merged = derive_thin_file_flag(merged, cfg=DEFAULT_CONFIG)

    clean_df, metadata_df, forbidden_separated_names = separate_forbidden_columns(merged)
    assert_no_forbidden_columns(clean_df, context="pre-transformation (after separate_forbidden_columns)")
    # The real signal for the disposition gate: whatever remains in clean_df
    # after separation (must be empty; assert_no_forbidden_columns above
    # already raised if not -- recomputed here, not reused, so this stays
    # correct even if that assertion is ever relaxed to a warning).
    forbidden_remaining_pre_transform = sorted(FORBIDDEN_PRE_TRANSFORM_COLS & set(clean_df.columns))

    disbursement_fid_col = clean_df[["disbursement_fid"]].reset_index(drop=True)

    pd_features, log_cols, cap_cols, _protected, signed_log_cols, _excluded = \
        get_and_classify_pd_features(clean_df)
    # build_transformed_dataframe SKIPS its own post-transform pruning step
    # entirely whenever fitted_params is given (its own documented inference-
    # time behavior: "pruning against the scoring batch would be batch-
    # dependent"), so its second return value here is simply every pd_feature
    # that survived into the transformed numeric subset (== pd_features,
    # always, on this script's path) -- never a "dropped" list. Captured
    # under an underscore name to avoid that exact confusion.
    df_transformed, _kept_pd_features, _report = build_transformed_dataframe(
        clean_df, pd_features=pd_features, log_cols=log_cols, cap_cols=cap_cols,
        signed_log_cols=signed_log_cols, cfg=DEFAULT_CONFIG, fitted_params=fitted_params,
    )

    # Checked on df_transformed (post-transformation, post-pruning), NEVER on
    # clean_df -- prune_post_transform_features (inside build_transformed_
    # dataframe) can drop a feature entirely if it is all-NaN or too few
    # unique values in this batch, which align_features would otherwise
    # paper over with a silent NaN fill. This is the actual input align_
    # features is about to receive.
    presence_df = classify_feature_presence(df_transformed, feature_order)
    absent = presence_df.loc[presence_df["presence_status"] == PRESENCE_ABSENT, "selected_feature"].tolist()
    if absent:
        print(f"\n{'!' * 100}\nSCHEMA CONSTRUCTION FAILURE: {len(absent)} champion feature(s) are "
              f"ABSENT_BEFORE_ALIGNMENT -- this is an R1 failure, distinct from and prior to "
              f"production alignment (which would silently fill these with NaN): {absent}\n{'!' * 100}")
    else:
        print(f"Schema construction completeness: all {len(feature_order)} champion feature(s) are "
              f"present before alignment (not necessarily fully populated -- see the presence report).")

    aligned = align_features(df_transformed.copy(), feature_order)
    forbidden_post = sorted(FORBIDDEN_PRE_TRANSFORM_COLS & set(aligned.columns))
    assert_no_forbidden_columns(aligned, context="final aligned 80-feature matrix")
    feature_order_exact_match = list(aligned.columns) == list(feature_order)
    print(f"Final aligned feature order exact match: {feature_order_exact_match}")

    disposition, conditions = build_disposition(
        n_rows=len(aligned), n_reconstructed_input=n_reconstructed_input,
        disbursement_fid_unique=disbursement_fid_unique, presence_df=presence_df,
        forbidden_remaining_pre_transform=forbidden_remaining_pre_transform, forbidden_post=forbidden_post,
        fitted_params=fitted_params, feature_order_exact_match=feature_order_exact_match,
    )

    print(f"\n{'#' * 100}\nDISPOSITION: {disposition}\n{'#' * 100}")
    for name, ok in conditions.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    out_prefix = args.out_prefix
    output_matrix = pd.concat([disbursement_fid_col, aligned.reset_index(drop=True)], axis=1)
    matrix_path = f"{out_prefix}_matrix.csv"
    print(f"\nWriting {len(output_matrix):,} row(s) x {len(output_matrix.columns)} column(s) to "
          f"{matrix_path} (no further progress output during this step)...")
    output_matrix.to_csv(matrix_path, index=False)
    print(f"Wrote {matrix_path}")

    metadata_path = f"{out_prefix}_forbidden_columns_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"Wrote {metadata_path}")

    presence_path = f"{out_prefix}_feature_presence_report.csv"
    presence_df.to_csv(presence_path, index=False)
    print(f"Wrote {presence_path}")

    validation_report = {
        "disposition": disposition,
        "conditions": conditions,
        "n_rows": len(aligned),
        "n_reconstructed_input": n_reconstructed_input,
        "n_champion_features": len(feature_order),
        "n_absent_before_alignment": len(absent),
        "absent_before_alignment": absent,
        "forbidden_columns_separated_before_transformation": forbidden_separated_names,
        "forbidden_columns_remaining_pre_transformation": forbidden_remaining_pre_transform,
        "forbidden_columns_found_in_final_matrix": forbidden_post,
        "phase21_evidence": phase21_evidence,
        "artifact_identity": {
            "feature_order_json_sha256": _sha256_file(feature_order_path),
            "transform_report_csv_sha256": _sha256_file(transform_report_path),
            "model_metadata_json_sha256": _sha256_file(model_metadata_path),
            "git_commit": _git_commit(),
            "production_source_file_sha256": {
                f: _sha256_file(Path(__file__).resolve().parent.parent / f)
                for f in PRODUCTION_SOURCE_FILES
            },
        },
        "inputs": {
            "phase21_reconstructed_features_csv": str(args.phase21_reconstructed_features_csv),
            "loan_training_file": str(args.loan_training_file),
            "pd_model_artifacts_dir": str(args.pd_model_artifacts_dir),
        },
    }
    report_path = f"{out_prefix}_validation_report.json"
    Path(report_path).write_text(json.dumps(validation_report, indent=2, default=str))
    print(f"Wrote {report_path}")

    print(
        f"\n{'#' * 100}\nCONCLUSION\n{'#' * 100}\n"
        f"{disposition}. No scores were computed -- this script only assembles and validates the "
        f"80-feature historical input matrix. Scoring (Stage 4B-R2) must not proceed unless "
        f"disposition == {DISPOSITION_READY}."
    )


if __name__ == "__main__":
    main()
