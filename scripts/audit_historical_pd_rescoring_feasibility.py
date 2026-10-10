"""
audit_historical_pd_rescoring_feasibility.py
=================================================
Stage 4A -- a provenance/feasibility AUDIT, not a modeling or scoring
script. Produces NO cal_pd, NO per-loan scores, NO risk dataframe.

THE QUESTION THIS SCRIPT ANSWERS: can a genuinely point-in-time (PIT)
`cal_pd` be reconstructed for historical loan episodes (the same loans in
`loan_episode_capacity_dataset.csv`), without leakage, with sufficient
model/version provenance -- or would any attempt have to smuggle in future
information or a non-comparable risk measure? This mirrors Gate 0's own
discipline: the rigor that killed the original monthly-snapshot capacity
dataset now applies to any historical PD reconstruction.

THREE POSSIBLE DISPOSITIONS, feature-reconstructability gates FIRST --
model/calibration provenance only decides between A and B once features
clear that gate:

  EXACT_PIT_RECONSTRUCTION_SUPPORTED
      Every required feature group can be computed strictly as of each
      historical loan's own decision date, AND an identifiable
      historical/frozen champion exists whose provenance establishes it
      is the exact model specification/version actually in effect during
      that historical decision regime. A model trained now, after the
      fact, does NOT qualify, however complete its own metadata is.

  RETROSPECTIVE_CURRENT_MODEL_ONLY
      Historical PIT features ARE reconstructable, but no historical-
      era-matched model/calibration provenance exists -- only the CURRENT
      champion/calibration is available to score the genuinely-
      reconstructed historical features. The resulting object must be
      named `retrospective_current_model_pd` (current model AND current
      calibration both usable) or a plain `raw_model_score` (calibration
      linkage itself is broken) -- NEVER `cal_pd_pit`.

  PIT_RECONSTRUCTION_NOT_SUPPORTED
      A FEATURE-LEVEL blocker, independent of and prior to any model/
      calibration question: if the required historical features
      themselves cannot be legitimately reconstructed as of each
      historical decision date, there is nothing legitimate to feed to
      ANY model, today's or historical. Scoring "latest/frozen state"
      attached to an old loan is exactly the temporal mismatch this audit
      exists to catch -- it is never reported as a form of "retrospective
      current model" scoring. Also applies if no model exists at all.

GATING ORDER (the single most important design decision here): Gate 1
(feature reconstructability) is checked FIRST and can by itself force
PIT_RECONSTRUCTION_NOT_SUPPORTED regardless of how healthy the model/
calibration state is. Only once Gate 1 passes does Gate 2 (model/
calibration provenance) decide between A and B. A missing/unknown check
NEVER silently becomes a pass.

EXACT WORDING REQUIREMENT for the Phase 2.1 (transaction/commission/
balance) finding: report "PIT reconstruction is not supported by the
CURRENT PIPELINE" -- never phrasing that implies the underlying historical
transaction data itself could never support PIT reconstruction. The
blocker is that `pd_model/preprocessing/transaction_features.py` has no
historical as-of-date feature-computation path (it computes recency
relative to the max date in whatever batch happens to be loaded); a
future, properly redesigned pipeline could change this without changing
the conceptual framework here. Supplying a multi-date mart file does NOT
by itself flip this verdict -- the function would still need a code
change to use it safely; this script reports the distinct-snapshot-date
count only as a separate, informational data-level fact.

EXPLICITLY PROHIBITED as substitutes if this audit returns
PIT_RECONSTRUCTION_NOT_SUPPORTED (recorded here, not acted on by this
script): joining current/latest PD backward onto historical periods,
using an agent's current risk tier as historical risk, interpolating PD,
training a new proxy PD model on the same historical outcomes merely to
enable Stage 4 (this would contaminate the independence the whole
Capacity/Risk separation exists to preserve), deriving risk from
`bad_state_3dpd_30d`, or using historical assigned exposure or post-
disbursement repayment behavior as a risk proxy. If this audit returns
PIT_RECONSTRUCTION_NOT_SUPPORTED, the next, separate design task is a
PROSPECTIVE shadow-logging mechanism (contemporaneous F_t, Capacity(F_t),
cal_pd_t, M_C3,t, L_current,t, L_capacity,t, L_full,t, plus eventual
matured outcomes) -- not designed or implemented here.

FORWARD-LOOKING DESIGN NOTE (recorded here only, not acted on by this
script): if a future run ever reaches EXACT_PIT_RECONSTRUCTION_SUPPORTED
or RETROSPECTIVE_CURRENT_MODEL_ONLY and a historical scoring pipeline is
subsequently built, it must reconstruct risk at the LOAN-EPISODE grain
(PD_{i,j,t-} per disbursement_fid), never fabricated directly at the
slower agent-period/business-state grain Capacity(F) uses -- aggregation
to a RiskSummary_{i,s} happens only afterward, as a distinct step,
preserving the three different clocks in play: F_{i,s} (business state),
PD_{i,s,j} (loan decision), Y_{i,s,j} (same-loan outcome).

NOT A CONTINUATION of scripts/fit_capacity_challenger_model.py or
scripts/analyze_capacity_dimension_redundancy.py (the abandoned, pre-
Gate-0 lineage). This script reads no episode-dataset column at all --
it inspects pd_model/ code structure, configuration, and (optionally)
real local artifact paths, never an episode-grain CSV.

One-way `scripts/` layering convention: never imports another `scripts/`
module. Importing from `extrafloat.engine.extrafloat_shadow_risk_
multiplier` (not a `scripts/` module) to read its own published
constants is fine and is done defensively (falls back to "unknown" if
the import fails, never silently assumes success).

Usage:
    python scripts\\audit_historical_pd_rescoring_feasibility.py ^
        --pd-model-artifacts-dir pd_model\\artifacts ^
        --transaction-mart-file data\\mfs_daily_agent_mart_20251115.csv ^
        --loan-training-file data\\loan_training_file.csv ^
        --shadow-artifacts-dir pd_model\\artifacts ^
        --out-prefix stage4a_audit
"""

import argparse
import importlib
import json
import re
from pathlib import Path

import pandas as pd

DEFAULT_LOAN_STATE_QUERY_FILE = "data/loan_state_query_updated_materialized.txt"
DEFAULT_TRANSACTION_FEATURES_FILE = "pd_model/preprocessing/transaction_features.py"
CALIBRATION_JOBLIB_NAME = "pd_isotonic_calibrated_risk.joblib"
CALIBRATION_METADATA_NAME = "pd_isotonic_calibrated_risk_metadata.json"

NOT_AVAILABLE = "NOT_AVAILABLE"
AVAILABLE_CURRENT_ONLY = "AVAILABLE_CURRENT_ONLY"
AVAILABLE_HISTORICAL_MATCHED = "AVAILABLE_HISTORICAL_MATCHED"

DISPOSITION_EXACT_PIT = "EXACT_PIT_RECONSTRUCTION_SUPPORTED"
DISPOSITION_RETRO_CURRENT = "RETROSPECTIVE_CURRENT_MODEL_ONLY"
DISPOSITION_NOT_SUPPORTED = "PIT_RECONSTRUCTION_NOT_SUPPORTED"


def _row(component, available, pit_reconstructable, provenance, blocking, notes=""):
    return {
        "component": component,
        "available": available,
        "pit_reconstructable": pit_reconstructable,
        "provenance": provenance,
        "blocking": bool(blocking),
        "notes": notes,
    }


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def check_champion_model(artifacts_dir):
    """Tri-state: NOT_AVAILABLE / AVAILABLE_CURRENT_ONLY /
    AVAILABLE_HISTORICAL_MATCHED. A model's metadata must carry an
    explicit, non-empty `historical_regime_provenance` block (with at
    least `regime_start`/`regime_end`) to count as AVAILABLE_HISTORICAL_
    MATCHED -- the real pd_model/run_pipeline.py metadata schema does not
    currently write this field, so a real run today will correctly report
    AVAILABLE_CURRENT_ONLY at best, never AVAILABLE_HISTORICAL_MATCHED."""
    if artifacts_dir is None:
        return _row("Champion model", NOT_AVAILABLE, "N/A",
                     "no --pd-model-artifacts-dir given", True,
                     "cannot assess without an artifacts directory")
    meta_path = Path(artifacts_dir) / "model_metadata.json"
    meta = _read_json(meta_path)
    if not meta or not meta.get("training_completed_at"):
        return _row("Champion model", NOT_AVAILABLE, "N/A",
                     f"{meta_path} missing or placeholder/unpopulated", True,
                     "no trained champion model exists in this location")
    regime = meta.get("historical_regime_provenance")
    if isinstance(regime, dict) and regime.get("regime_start") and regime.get("regime_end"):
        return _row("Champion model", AVAILABLE_HISTORICAL_MATCHED, "N/A",
                     f"{meta_path}: historical_regime_provenance={regime}", False,
                     "model version provably in effect for the stated historical regime")
    return _row("Champion model", AVAILABLE_CURRENT_ONLY, "N/A",
                 f"{meta_path}: real metadata present "
                 f"(training_completed_at={meta.get('training_completed_at')}), "
                 f"but no historical_regime_provenance establishing it as the "
                 f"historical-era-effective model", True,
                 "a model trained now cannot retroactively become the historical champion")


def check_feature_schema(artifacts_dir):
    if artifacts_dir is None:
        return _row("Feature schema", False, "N/A", "no --pd-model-artifacts-dir given", True)
    path = Path(artifacts_dir) / "feature_order.json"
    data = _read_json(path)
    features = (data or {}).get("selected_features") or []
    if features:
        return _row("Feature schema", True, "N/A", f"{path}: {len(features)} feature(s)", False)
    return _row("Feature schema", False, "N/A", f"{path} missing or empty", True)


def check_preprocessing(artifacts_dir):
    if artifacts_dir is None:
        return _row("Preprocessing", False, "N/A", "no --pd-model-artifacts-dir given", True)
    meta = _read_json(Path(artifacts_dir) / "model_metadata.json") or {}
    version = meta.get("preprocessor_version")
    schema_version = meta.get("feature_schema_version")
    if version is None or schema_version is None:
        return _row("Preprocessing", False, "N/A", "preprocessor_version/feature_schema_version missing",
                     True)
    note = ("hardcoded static int, not content-addressed -- cannot strongly verify preprocessing "
            "hasn't changed between training runs" if isinstance(version, int) and isinstance(schema_version, int)
            else "")
    return _row("Preprocessing", True, "N/A",
                f"preprocessor_version={version}, feature_schema_version={schema_version}",
                False, note)


def _count_distinct_mart_dates(transaction_mart_file):
    try:
        df = pd.read_csv(transaction_mart_file, nrows=None)
    except Exception as e:
        return None, f"could not read {transaction_mart_file}: {e}"
    date_col = next((c for c in df.columns if c.lower() in
                      ("tbl_dt", "snapshot_date", "date", "snapshot_dt")), None)
    if date_col is None:
        return None, f"no recognizable date column in {transaction_mart_file}"
    n = pd.to_datetime(df[date_col], errors="coerce").dt.date.nunique()
    return n, f"{n} distinct snapshot date(s) in column '{date_col}'"


def check_historical_input_features(transaction_features_file, transaction_mart_file,
                                     loan_state_query_file):
    """Returns TWO rows: Phase 2.1 (transaction mart) and Phase 2.2
    (loan-history). Phase 2.1's verdict is a CODE-LEVEL finding, never
    flipped by data alone -- a multi-date mart is reported separately as
    an informational fact, never as fixing the verdict."""
    rows = []

    # Phase 2.1 -- transaction/commission/balance features.
    tf_path = Path(transaction_features_file) if transaction_features_file else \
        Path(DEFAULT_TRANSACTION_FEATURES_FILE)
    as_of_present = False
    if tf_path.exists():
        text = tf_path.read_text(errors="ignore")
        as_of_present = bool(re.search(r"\bas_of_date\b", text))
    notes = ("PIT reconstruction is not supported by the current pipeline -- "
             "transaction_features.py computes recency relative to the max date "
             "in whatever batch is loaded, with no as_of_date parameter. This is a "
             "pipeline/implementation finding, not a claim that the underlying "
             "historical transaction data could never support PIT reconstruction; "
             "a redesigned Phase 2.1 feature builder could change this disposition.")
    pit_phase21 = "No (not supported by the current pipeline)"
    blocking_phase21 = True
    if as_of_present:
        notes = (f"{tf_path} now contains an 'as_of_date' reference -- manual review required "
                 "before treating Phase 2.1 as PIT-reconstructable; not auto-upgraded by this check.")
        pit_phase21 = "Possibly -- as_of_date parameter present, manual review required"
    if transaction_mart_file:
        n_dates, date_note = _count_distinct_mart_dates(transaction_mart_file)
        notes += f" [data-level fact, does NOT change the verdict above: {date_note}]"
    rows.append(_row("Historical input features -- Phase 2.1 (transaction/commission/balance)",
                      True, pit_phase21,
                      f"{tf_path}" if tf_path.exists() else "transaction_features.py not found",
                      blocking_phase21, notes))

    # Phase 2.2 -- loan-history features (PIT-safe by construction, verified against the SQL guard).
    lsq_path = Path(loan_state_query_file) if loan_state_query_file else \
        Path(DEFAULT_LOAN_STATE_QUERY_FILE)
    if lsq_path.exists():
        text = lsq_path.read_text(errors="ignore")
        guard_found = bool(re.search(r"state_date\s*<\s*loan_date", text))
        if guard_found:
            rows.append(_row("Historical input features -- Phase 2.2 (loan-history)",
                              True, "Yes", f"{lsq_path}: 'state_date < loan_date' guard found", False,
                              "PIT-safe by construction (event-log source, strict pre-disbursement guard)"))
        else:
            rows.append(_row("Historical input features -- Phase 2.2 (loan-history)",
                              True, "Unknown", f"{lsq_path}: guard pattern not found", True,
                              "could not verify the point-in-time guard textually; treat as unverified"))
    else:
        rows.append(_row("Historical input features -- Phase 2.2 (loan-history)",
                          False, "Unknown", f"{lsq_path} not found", True,
                          "cannot verify point-in-time guard without the query file"))
    return rows


def check_calibration(shadow_artifacts_dir):
    if shadow_artifacts_dir is None:
        return _row("Calibration", False, "N/A", "no --shadow-artifacts-dir given", True)
    joblib_path = Path(shadow_artifacts_dir) / CALIBRATION_JOBLIB_NAME
    meta_path = Path(shadow_artifacts_dir) / CALIBRATION_METADATA_NAME
    meta = _read_json(meta_path)
    if not joblib_path.exists() or not meta:
        return _row("Calibration", False, "N/A",
                     f"{joblib_path} and/or {meta_path} missing", True,
                     "no calibration artifact -- any reconstructed score could at most be a "
                     "raw_model_score, never cal_pd")
    required = ("cal_pd_plateau", "r_plateau", "version")
    missing = [k for k in required if meta.get(k) is None]
    if missing:
        return _row("Calibration", False, "N/A",
                     f"{meta_path} missing field(s): {missing}", True,
                     "incomplete calibration metadata -- cannot legitimately call a score cal_pd")
    return _row("Calibration", True, "N/A",
                f"{meta_path}: version={meta.get('version')}, "
                f"cal_pd_plateau={meta.get('cal_pd_plateau')}, r_plateau={meta.get('r_plateau')}",
                False, "calibration artifact present and self-describing")


def check_model_effective_dates(artifacts_dir):
    if artifacts_dir is None:
        return _row("Model effective dates", False, "N/A", "no --pd-model-artifacts-dir given", True)
    meta = _read_json(Path(artifacts_dir) / "model_metadata.json") or {}
    fields = ("train_snapshot_date", "val_snapshot_date", "train_cutoff")
    present = {k: meta.get(k) for k in fields if meta.get(k) is not None}
    if len(present) == len(fields):
        return _row("Model effective dates", True, "N/A", f"recorded: {present}", False)
    return _row("Model effective dates", False, "N/A",
                f"missing field(s): {[k for k in fields if k not in present]}", True,
                "cannot establish which historical period this champion version covers")


def check_c3_mapping(shadow_artifacts_dir):
    try:
        mod = importlib.import_module("extrafloat.engine.extrafloat_shadow_risk_multiplier")
        scenarios = getattr(mod, "SHADOW_SCENARIOS", None)
        policy_version = getattr(mod, "SHADOW_POLICY_VERSION", None)
    except Exception as e:
        return _row("C3 mapping", False, "N/A", f"could not import extrafloat_shadow_risk_multiplier: {e}",
                     True, "treat as unknown -- never silently assumed consistent")
    if not scenarios or not policy_version:
        return _row("C3 mapping", False, "N/A", "SHADOW_SCENARIOS/SHADOW_POLICY_VERSION not found", True)
    if shadow_artifacts_dir is None:
        return _row("C3 mapping", False, "N/A",
                     f"config present (policy_version={policy_version}) but no "
                     "--shadow-artifacts-dir given to check self-consistency", True)
    meta = _read_json(Path(shadow_artifacts_dir) / CALIBRATION_METADATA_NAME)
    if not meta:
        return _row("C3 mapping", False, "N/A",
                     f"config present (policy_version={policy_version}) but calibration metadata missing",
                     True)
    return _row("C3 mapping", True, "N/A",
                f"policy_version={policy_version}, scenarios={list(scenarios.keys())}, "
                f"calibration_version={meta.get('version')}", False,
                "config present; full numeric self-consistency check (tolerance 1e-4) deferred to "
                "compute_shadow_risk_multiplier itself")


def determine_disposition(rows):
    """Gate 1 (feature reconstructability) decides PIT_RECONSTRUCTION_NOT_
    SUPPORTED outright, overriding anything else. Gate 2 (model/
    calibration provenance) only runs if Gate 1 passes."""
    by_component = {r["component"]: r for r in rows}
    phase21 = next(r for r in rows if r["component"].startswith("Historical input features -- Phase 2.1"))
    phase22 = next(r for r in rows if r["component"].startswith("Historical input features -- Phase 2.2"))

    features_ok = (not phase21["blocking"]) and (not phase22["blocking"])
    if not features_ok:
        return DISPOSITION_NOT_SUPPORTED, (
            "Gate 1 (feature reconstructability) failed -- "
            f"Phase 2.1 blocking={phase21['blocking']}, Phase 2.2 blocking={phase22['blocking']}. "
            "Model/calibration provenance is moot: there is no legitimate historical feature "
            "vector to feed to any model, today's or historical."
        )

    champion = by_component["Champion model"]
    if champion["available"] == NOT_AVAILABLE:
        return DISPOSITION_NOT_SUPPORTED, (
            "Gate 1 passed, but no champion model exists at all (current or historical) -- "
            "there is nothing to score the reconstructed features with."
        )

    calibration = by_component["Calibration"]
    schema = by_component["Feature schema"]
    preprocessing = by_component["Preprocessing"]
    effective_dates = by_component["Model effective dates"]
    c3 = by_component["C3 mapping"]

    if (champion["available"] == AVAILABLE_HISTORICAL_MATCHED and schema["available"]
            and preprocessing["available"] and calibration["available"]
            and effective_dates["available"] and c3["available"]):
        return DISPOSITION_EXACT_PIT, (
            "Gate 1 passed, and the champion model is historically matched with full "
            "provenance (feature schema, preprocessing, calibration, effective dates, C3 "
            "mapping all available). Historical cal_pd_pit rescoring is defensible."
        )

    if champion["available"] in (AVAILABLE_CURRENT_ONLY, AVAILABLE_HISTORICAL_MATCHED):
        produced_object = "retrospective_current_model_pd" if calibration["available"] else "raw_model_score"
        return DISPOSITION_RETRO_CURRENT, (
            "Gate 1 passed (features legitimately reconstructed), but historical-era-matched "
            "model/calibration provenance is not fully established. Only today's model can "
            f"score the reconstructed features, producing a '{produced_object}', never "
            "cal_pd_pit."
        )

    return DISPOSITION_NOT_SUPPORTED, "Unresolved/unknown champion-model state -- defaults to blocked."


def run_audit(pd_model_artifacts_dir, transaction_mart_file, loan_training_file,
              shadow_artifacts_dir, transaction_features_file=None, loan_state_query_file=None):
    rows = []
    rows.append(check_champion_model(pd_model_artifacts_dir))
    rows.append(check_feature_schema(pd_model_artifacts_dir))
    rows.append(check_preprocessing(pd_model_artifacts_dir))
    rows.extend(check_historical_input_features(transaction_features_file, transaction_mart_file,
                                                 loan_state_query_file))
    rows.append(check_calibration(shadow_artifacts_dir))
    rows.append(check_model_effective_dates(pd_model_artifacts_dir))
    rows.append(check_c3_mapping(shadow_artifacts_dir))

    if loan_training_file:
        try:
            df = pd.read_csv(loan_training_file, usecols=lambda c: c in {"agent_msisdn", "loan_date"})
            n_dates = pd.to_datetime(df["loan_date"], errors="coerce").dt.date.nunique() \
                if "loan_date" in df.columns else None
            rows.append(_row("Loan-training-file sanity check", True, "N/A",
                              f"{loan_training_file}: {n_dates} distinct loan date(s)", False))
        except Exception as e:
            rows.append(_row("Loan-training-file sanity check", False, "N/A", str(e), False))

    disposition, rationale = determine_disposition(rows)
    return rows, disposition, rationale


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pd-model-artifacts-dir", type=Path, default=None)
    ap.add_argument("--transaction-mart-file", type=Path, default=None)
    ap.add_argument("--loan-training-file", type=Path, default=None)
    ap.add_argument("--shadow-artifacts-dir", type=Path, default=None)
    ap.add_argument("--transaction-features-file", type=Path, default=None,
                     help=f"default: {DEFAULT_TRANSACTION_FEATURES_FILE}")
    ap.add_argument("--loan-state-query-file", type=Path, default=None,
                     help=f"default: {DEFAULT_LOAN_STATE_QUERY_FILE}")
    ap.add_argument("--out-prefix", type=str, default="stage4a_pd_rescoring_audit")
    args = ap.parse_args(argv)

    print(f"\n{'#' * 100}\nSTAGE 4A -- PIT PD RESCORING FEASIBILITY AUDIT\n"
          f"Produces NO cal_pd, NO per-loan scores, NO risk dataframe -- a provenance/feasibility "
          f"check only.\n{'#' * 100}")

    rows, disposition, rationale = run_audit(
        args.pd_model_artifacts_dir, args.transaction_mart_file, args.loan_training_file,
        args.shadow_artifacts_dir, args.transaction_features_file, args.loan_state_query_file,
    )

    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print("\n" + table.to_string(index=False))

    print(f"\n{'#' * 100}\nDISPOSITION: {disposition}\n{'#' * 100}")
    print(rationale)

    component_csv = f"{args.out_prefix}_component_table.csv"
    table.to_csv(component_csv, index=False)
    print(f"\nWrote {component_csv}")

    disposition_json = f"{args.out_prefix}_disposition.json"
    Path(disposition_json).write_text(json.dumps({
        "disposition": disposition,
        "rationale": rationale,
        "components": rows,
    }, indent=2))
    print(f"Wrote {disposition_json}")

    print(
        "\nThis disposition determines what happens next, and nothing else does:\n"
        "  EXACT_PIT_RECONSTRUCTION_SUPPORTED   -> design historical Stage 4B using cal_pd_pit.\n"
        "  RETROSPECTIVE_CURRENT_MODEL_ONLY     -> decide separately whether a retrospective-\n"
        "                                          current-model analysis is useful.\n"
        "  PIT_RECONSTRUCTION_NOT_SUPPORTED     -> stop the historical Stage 4 branch; design\n"
        "                                          prospective shadow logging instead.\n"
        "No PD model is trained or scored by this script. No Stage 4B design proceeds from this\n"
        "run alone."
    )


if __name__ == "__main__":
    main()
