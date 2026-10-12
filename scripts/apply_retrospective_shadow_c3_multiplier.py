"""
apply_retrospective_shadow_c3_multiplier.py
==================================================
Stage 4B-R4 -- MECHANICAL C3 POLICY REPLAY ONLY. Applies the existing,
frozen shadow C3 multiplier policy (extrafloat/engine/
extrafloat_shadow_risk_multiplier.py) to Stage 4B-R3's thick-file
retrospective_current_cal_pd, producing a continuous risk multiplier
per frozen scenario. No Capacity(F), no k selection, no frontier
analysis, no tuning -- this script answers nothing about capacity; it
only mechanically replays retrospective_current_cal_pd -> M_C3^retro
using the policy exactly as frozen.

SCOPE, confirmed by reading compute_shadow_risk_multiplier() in full
(not assumed): the full production function also computes a DOLLAR
shadow limit, which additionally requires combined_cap/assigned_limit
-- live capacity-engine outputs that do not exist for historical
episodes (no Capacity(F) has been computed for them, by design, this
stage). But the MULTIPLIER itself (shadow_multiplier_<scenario>) is a
pure function of cal_pd alone, via two independently-callable pieces
of the SAME production module: `load_isotonic_calibration()` (the
frozen shadow isotonic model, SEPARATE from the PD model's own
pd_calibration_map.csv used in Stage 4B-R3) and `_policy_3_hybrid()`
(the frozen, non-increasing-in-risk multiplier curve). This script
calls those two real functions directly rather than the full wrapper,
and never reads, requires, or fabricates combined_cap/assigned_limit
anywhere. This is exactly why the requested minimal output schema
(a multiplier, not a dollar limit) is the natural boundary here, not
merely a convenient one -- confirmed by code, not assumed.

BOTH frozen scenarios (SHADOW_SCENARIOS: "base", "conservative") are
reported as separate columns -- collapsing to one would discard real,
already-frozen policy information the user did not ask to drop.

TERMINOLOGY: retrospective_current_c3_multiplier_<scenario>. Thin-file
rows remain NaN for every scenario -- the provenance gap from Stage
4B-R2 does not disappear merely because R3 and R4 succeeded.

MECHANICAL-CONSISTENCY VERIFICATION, never assumed:
  1. Artifact self-consistency: load_isotonic_calibration() itself
     already checks model.predict(cal_pd_plateau) against the stored
     r_plateau (tolerance 1e-4) -- this script hard-exits if that
     check fails (STATUS_OK required), rather than silently proceeding
     on a broken artifact.
  2. Intrinsic monotonicity: model.predict() evaluated on a synthetic,
     ascending [0,1] grid must be non-decreasing -- a property of the
     artifact alone, independent of this run's data.
  3. Applied monotonicity: on the REAL thick population's own cal_pd
     values, sorted ascending, shadow_calibrated_risk must be
     non-decreasing AND each scenario's multiplier must be
     non-increasing (per _policy_3_hybrid's own documented contract:
     "Non-increasing in r"). Never assumed from (2) alone.
  4. Exact agreement with the production C3 mapping: this script also
     calls the REAL, full compute_shadow_risk_multiplier() on a SAMPLE
     of real cal_pd values (with throwaway placeholder combined_cap=
     assigned_limit=1.0 -- values that provably do not affect the
     multiplier computation, confirmed by reading the function; used
     ONLY for this cross-check, never for the main 4.78M-row run) and
     asserts its shadow_multiplier_<scenario> output matches this
     script's own restated computation on the identical cal_pd values,
     to floating-point tolerance. This is a real, run-time structural
     check, not an assumption from importing the same functions.

Usage:
    python scripts\\apply_retrospective_shadow_c3_multiplier.py ^
        --retrospective-cal-scores-csv stage4br3_retrospective_cal_scores_scores.csv ^
        --shadow-artifacts-dir pd_model\\shadow_artifacts ^
        --out-prefix stage4br4_retrospective_c3_multiplier
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extrafloat.engine.extrafloat_shadow_risk_multiplier import (  # noqa: E402
    SHADOW_POLICY_VERSION,
    SHADOW_SCENARIOS,
    _policy_3_hybrid,
    compute_shadow_risk_multiplier,
    load_isotonic_calibration,
)

ROUTE_THICK = "THICK_CURRENT_CHAMPION_SCORED"
ROUTE_THIN = "THIN_SCORE_NOT_PIT_REPRODUCIBLE"

CAL_COL = "retrospective_current_cal_pd"
RAW_COL = "retrospective_current_model_pd"
RISK_COL = "retrospective_current_shadow_calibrated_risk"

DISPOSITION_READY = "R4_C3_REPLAY_READY"
DISPOSITION_NOT_READY = "R4_C3_REPLAY_NOT_READY"

MONOTONICITY_TOLERANCE = 1e-9
CROSS_CHECK_TOLERANCE = 1e-9
_SCENARIO_NAMES = tuple(SHADOW_SCENARIOS.keys())


def multiplier_col(name: str) -> str:
    return f"retrospective_current_c3_multiplier_{name}"


def load_retrospective_cal_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["disbursement_fid", "thin_file_flag", "route_status", RAW_COL, CAL_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing}")
    dup = int(df["disbursement_fid"].duplicated().sum())
    if dup:
        sys.exit(f"ERROR: {dup} duplicate disbursement_fid value(s) in {path}.")
    print(f"Loaded {len(df):,} row(s) from {path}.")
    return df


def check_intrinsic_monotonicity(model, n_points: int = 1000) -> dict:
    """Evaluates model.predict() on a synthetic, ascending [0,1] grid --
    intrinsic to the artifact alone, independent of this run's data."""
    grid = np.linspace(0.0, 1.0, n_points)
    predicted = model.predict(grid)
    diffs = np.diff(predicted)
    violations = int((diffs < -MONOTONICITY_TOLERANCE).sum())
    return {
        "isotonic_model_intrinsically_monotonic": violations == 0,
        "n_intrinsic_monotonicity_violations": violations,
    }


def check_applied_monotonicity(cal_pd: pd.Series, calibrated_risk: pd.Series,
                                multipliers: dict[str, pd.Series]) -> dict:
    """On the REAL thick population, sorted by cal_pd ascending: risk must
    be non-decreasing, and each scenario's multiplier must be
    non-increasing (per _policy_3_hybrid's own documented contract)."""
    order = np.argsort(cal_pd.to_numpy(), kind="stable")
    risk_sorted = calibrated_risk.to_numpy()[order]
    risk_diffs = np.diff(risk_sorted)
    risk_violations = int((risk_diffs < -MONOTONICITY_TOLERANCE).sum())

    per_scenario = {}
    for name, series in multipliers.items():
        mult_sorted = series.to_numpy()[order]
        mult_diffs = np.diff(mult_sorted)
        mult_violations = int((mult_diffs > MONOTONICITY_TOLERANCE).sum())  # must be NON-increasing
        per_scenario[name] = {
            "multiplier_applied_non_increasing": mult_violations == 0,
            "n_multiplier_monotonicity_violations": mult_violations,
        }
    return {
        "risk_applied_non_decreasing": risk_violations == 0,
        "n_risk_monotonicity_violations": risk_violations,
        "per_scenario": per_scenario,
    }


def verify_exact_agreement_with_production_function(
    cal_pd_sample: np.ndarray, shadow_artifacts_dir: Path, restated_multipliers: dict[str, np.ndarray],
) -> dict:
    """Calls the REAL, full compute_shadow_risk_multiplier() on a sample of
    real cal_pd values, with throwaway placeholder combined_cap=
    assigned_limit=1.0 (provably irrelevant to the multiplier itself --
    confirmed by reading the function; used ONLY here, never in the main
    run), and compares its shadow_multiplier_<scenario> output against
    this script's own restated computation on the IDENTICAL cal_pd
    values. A real, run-time check, not an assumption from shared code."""
    sample_df = pd.DataFrame({
        "cal_pd": cal_pd_sample,
        "combined_cap": 1.0,
        "assigned_limit": 1.0,
    })
    production_out = compute_shadow_risk_multiplier(
        sample_df, artifacts_dir=shadow_artifacts_dir, config=None, run_id="r4_cross_check")

    per_scenario = {}
    all_exact = True
    for name in _SCENARIO_NAMES:
        prod_col = f"shadow_multiplier_{name}"
        prod_vals = production_out[prod_col].to_numpy()
        restated_vals = restated_multipliers[name]
        max_abs_diff = float(np.nanmax(np.abs(prod_vals - restated_vals))) if len(prod_vals) else 0.0
        exact = bool(max_abs_diff <= CROSS_CHECK_TOLERANCE)
        all_exact = all_exact and exact
        per_scenario[name] = {"max_abs_diff": max_abs_diff, "exact": exact}
    return {"n_sampled": len(cal_pd_sample), "all_scenarios_exact": all_exact, "per_scenario": per_scenario}


def build_plateau_stats(s: pd.Series) -> dict:
    if len(s) == 0:
        return {"n_unique_values": 0, "largest_plateau_count": 0, "largest_plateau_value": None}
    vc = s.value_counts()
    return {
        "n_unique_values": int(vc.shape[0]),
        "largest_plateau_count": int(vc.max()),
        "largest_plateau_value": float(vc.idxmax()),
    }


def build_quantile_dict(s: pd.Series, quantile_points: list[float]) -> dict:
    if len(s) == 0:
        return {}
    return {str(q): float(v) for q, v in s.quantile(quantile_points).items()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrospective-cal-scores-csv", type=Path, required=True,
                     help="Stage 4B-R3's own *_scores.csv output.")
    ap.add_argument("--shadow-artifacts-dir", type=Path, required=True,
                     help="Directory containing pd_isotonic_calibrated_risk.joblib + its metadata "
                          "JSON -- the frozen SHADOW C3 isotonic artifact. SEPARATE from "
                          "--pd-model-artifacts-dir used in R1-R3.")
    ap.add_argument("--out-prefix", type=str, default="stage4br4_retrospective_c3_multiplier")
    ap.add_argument("--cross-check-sample-size", type=int, default=5000,
                     help="Number of real cal_pd values to run through the FULL production "
                          "compute_shadow_risk_multiplier() for the exact-agreement check.")
    args = ap.parse_args(argv)

    print(f"\n{'#' * 100}\nSTAGE 4B-R4 -- MECHANICAL C3 POLICY REPLAY (THICK-FILE ROUTE ONLY)\n"
          f"No Capacity(F), no k selection, no frontier analysis, no tuning -- the frozen policy "
          f"is replayed exactly as-is.\n{'#' * 100}")

    # ------------------------------------------------------------------ #
    # 1) Load R3's scores + the real frozen shadow isotonic artifact
    # ------------------------------------------------------------------ #
    scores_df = load_retrospective_cal_scores(args.retrospective_cal_scores_csv)
    n_total = len(scores_df)
    n_thick = int((scores_df["route_status"] == ROUTE_THICK).sum())
    n_thin = int((scores_df["route_status"] == ROUTE_THIN).sum())

    model, metadata, status = load_isotonic_calibration(args.shadow_artifacts_dir)
    if status != "ok":
        sys.exit(f"ERROR: load_isotonic_calibration failed with status={status} at "
                  f"{args.shadow_artifacts_dir} -- cannot replay the C3 policy on a broken or "
                  f"missing shadow artifact.")
    model_version = metadata.get("version", "unknown")
    r_plateau = float(metadata["r_plateau"])
    print(f"Loaded shadow isotonic artifact: version={model_version}, r_plateau={r_plateau}, "
          f"policy_version={SHADOW_POLICY_VERSION}, scenarios={_SCENARIO_NAMES}.")

    # ------------------------------------------------------------------ #
    # 2) Apply the frozen policy to the THICK subset only
    # ------------------------------------------------------------------ #
    thick_df = scores_df.loc[scores_df["route_status"] == ROUTE_THICK].copy()
    thin_df = scores_df.loc[scores_df["route_status"] == ROUTE_THIN].copy()

    cal_pd_thick = pd.to_numeric(thick_df[CAL_COL], errors="coerce")
    valid_cal_pd = cal_pd_thick.notna() & np.isfinite(cal_pd_thick)
    n_thick_valid_cal_pd = int(valid_cal_pd.sum())

    calibrated_risk = pd.Series(np.nan, index=thick_df.index, dtype="float64")
    calibrated_risk[valid_cal_pd] = model.predict(cal_pd_thick[valid_cal_pd].to_numpy())
    thick_df[RISK_COL] = calibrated_risk

    multipliers: dict[str, pd.Series] = {}
    for name, params in SHADOW_SCENARIOS.items():
        mult = pd.Series(np.nan, index=thick_df.index, dtype="float64")
        mult[valid_cal_pd] = _policy_3_hybrid(
            calibrated_risk[valid_cal_pd].to_numpy(), r_plateau,
            params["r_floor"], params["m_max"], params["m_min"],
        )
        multipliers[name] = mult
        thick_df[multiplier_col(name)] = mult

    for name in _SCENARIO_NAMES:
        thin_df[multiplier_col(name)] = np.nan
    thin_df[RISK_COL] = np.nan

    output_cols = (["disbursement_fid", "thin_file_flag", "route_status", RAW_COL, CAL_COL, RISK_COL]
                    + [multiplier_col(name) for name in _SCENARIO_NAMES])
    all_output = pd.concat([thick_df, thin_df], ignore_index=True)[output_cols]
    out_prefix = args.out_prefix
    scores_path = f"{out_prefix}_scores.csv"
    all_output.to_csv(scores_path, index=False)
    print(f"Wrote {scores_path} ({len(all_output):,} row(s) total).")

    # ------------------------------------------------------------------ #
    # 3) Mechanical-consistency checks
    # ------------------------------------------------------------------ #
    intrinsic = check_intrinsic_monotonicity(model)
    applied = check_applied_monotonicity(cal_pd_thick[valid_cal_pd], calibrated_risk[valid_cal_pd],
                                          {name: multipliers[name][valid_cal_pd] for name in _SCENARIO_NAMES})

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(np.flatnonzero(valid_cal_pd.to_numpy()),
                             size=min(args.cross_check_sample_size, n_thick_valid_cal_pd), replace=False)
    cal_pd_sample = cal_pd_thick.to_numpy()[sample_idx]
    restated_sample_multipliers = {name: multipliers[name].to_numpy()[sample_idx] for name in _SCENARIO_NAMES}
    exact_agreement = verify_exact_agreement_with_production_function(
        cal_pd_sample, args.shadow_artifacts_dir, restated_sample_multipliers)

    # ------------------------------------------------------------------ #
    # 4) Structural validation
    # ------------------------------------------------------------------ #
    n_thick_multiplier_complete = all(
        int(multipliers[name][valid_cal_pd].notna().sum()) == n_thick_valid_cal_pd
        for name in _SCENARIO_NAMES
    )
    n_thin_multiplier_nonnull = sum(
        int(thin_df[multiplier_col(name)].notna().sum()) for name in _SCENARIO_NAMES
    )
    multiplier_valid_range = all(
        bool(
            multipliers[name][valid_cal_pd].between(
                SHADOW_SCENARIOS[name]["m_min"] - MONOTONICITY_TOLERANCE,
                SHADOW_SCENARIOS[name]["m_max"] + MONOTONICITY_TOLERANCE,
            ).all()
        )
        for name in _SCENARIO_NAMES
    )
    structural = {
        "row_preservation": len(all_output) == n_total,
        "disbursement_fid_unique": not all_output["disbursement_fid"].duplicated().any(),
        "thick_multiplier_complete": n_thick_multiplier_complete,
        "thin_multiplier_is_nan": n_thin_multiplier_nonnull == 0,
        "isotonic_model_intrinsically_monotonic": intrinsic["isotonic_model_intrinsically_monotonic"],
        "risk_applied_monotonic": applied["risk_applied_non_decreasing"],
        "multiplier_applied_monotonic_non_increasing": all(
            applied["per_scenario"][name]["multiplier_applied_non_increasing"] for name in _SCENARIO_NAMES
        ),
        "multiplier_values_valid_range": multiplier_valid_range,
        "exact_agreement_with_production_function": exact_agreement["all_scenarios_exact"],
    }
    disposition = DISPOSITION_READY if all(structural.values()) else DISPOSITION_NOT_READY

    print(f"\n{'#' * 100}\nDISPOSITION: {disposition}\n{'#' * 100}")
    for name, ok in structural.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    # ------------------------------------------------------------------ #
    # 5) Analytical diagnostics (never gate disposition)
    # ------------------------------------------------------------------ #
    quantile_points = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    per_scenario_diagnostics = {}
    for name in _SCENARIO_NAMES:
        s = multipliers[name][valid_cal_pd]
        per_scenario_diagnostics[name] = {
            "multiplier_quantiles": build_quantile_dict(s, quantile_points),
            "plateau_stats": build_plateau_stats(s),
        }
    risk_quantiles = build_quantile_dict(calibrated_risk[valid_cal_pd], quantile_points)

    validation_report = {
        "disposition": disposition,
        "structural_validation": structural,
        "counts": {
            "n_total": n_total, "n_thick": n_thick, "n_thin": n_thin,
            "n_thick_valid_cal_pd": n_thick_valid_cal_pd,
        },
        "mechanical_consistency": {
            "intrinsic_monotonicity": intrinsic,
            "applied_monotonicity": applied,
            "exact_agreement_with_production_function": exact_agreement,
        },
        "analytical_diagnostics": {
            "shadow_calibrated_risk_quantiles": risk_quantiles,
            "per_scenario": per_scenario_diagnostics,
        },
        "shadow_artifact_identity": {
            "model_version": model_version, "r_plateau": r_plateau,
            "policy_version": SHADOW_POLICY_VERSION, "scenarios": SHADOW_SCENARIOS,
            "metadata": metadata,
        },
        "terminology_note": (
            f"retrospective_current_c3_multiplier_<scenario> replays the existing FROZEN C3 "
            f"policy (calibration, breakpoints, multiplier curve, and scenarios all unchanged) "
            f"against {CAL_COL} for thick-file episodes only. No Capacity(F), no k selection, no "
            f"frontier analysis -- this is a mechanical policy replay, nothing else. Thin-file "
            f"rows remain {ROUTE_THIN} / NaN for every scenario, unchanged from Stage 4B-R2/R3."
        ),
        "inputs": {
            "retrospective_cal_scores_csv": str(args.retrospective_cal_scores_csv),
            "shadow_artifacts_dir": str(args.shadow_artifacts_dir),
        },
    }
    report_path = f"{out_prefix}_validation_report.json"
    Path(report_path).write_text(json.dumps(validation_report, indent=2, default=str))
    print(f"Wrote {report_path}")

    print(
        f"\n{'#' * 100}\nCONCLUSION\n{'#' * 100}\n"
        f"{disposition}. {n_thick_valid_cal_pd:,} thick-file episode(s) received a C3 multiplier "
        f"for every frozen scenario ({', '.join(_SCENARIO_NAMES)}); {n_thin:,} thin-file "
        f"episode(s) remain NaN ({ROUTE_THIN}). No Capacity(F), no k selection, no frontier "
        f"analysis performed. Stage 4 capacity-frontier analysis must not proceed unless "
        f"disposition == {DISPOSITION_READY}."
    )


if __name__ == "__main__":
    main()
