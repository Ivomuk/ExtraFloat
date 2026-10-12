"""
apply_retrospective_current_isotonic_calibration.py
==================================================
Stage 4B-R3 -- FROZEN CURRENT ISOTONIC CALIBRATION of Stage 4B-R2's
thick-file `retrospective_current_model_pd`, producing
`retrospective_current_cal_pd`. THICK-FILE ONLY -- thin-file rows
remain `THIN_SCORE_NOT_PIT_REPRODUCIBLE` / NaN, exactly as R2 left
them; there is no raw score to calibrate for them. No C3, no
Capacity(F), no `k` selection -- that is Stage 4B-R4/R3B, deliberately
kept separate so each layer (historical PIT information -> current
model -> current calibration -> current C3 policy) carries its own,
independently inspectable provenance.

TERMINOLOGY: the output field is `retrospective_current_cal_pd`, NEVER
bare `cal_pd` (which could be confused with a genuine historical
production value) and NEVER `cal_pd_pit`.

ARTIFACT FIDELITY: calls the REAL production `attach_cal_pd`
(pd_model/modeling/calibration.py) against the frozen
`artifacts.cal_map` loaded by the REAL `load_artifacts()` -- the
isotonic regression is NEVER refit here; this script only replays the
existing bin-lookup map. Reproduces `score_new_agents`'s own exact call
shape (pd_model/modeling/inference.py:286-291): rename the raw score
column to `feature_config.RAW_SCORE_COL`, add a placeholder
`bad_state=0` column (not available at this stage, same placeholder
production itself uses at inference time -- `_standardize_scored_df`
requires a "y"/"bad_state" column to build its schema, but that column
plays no role in the actual bin-lookup calibration logic), then call
`attach_cal_pd(..., artifacts.cal_map, champion, cfg=cfg)`.

MECHANICAL CONSISTENCY, verified (never assumed): isotonic calibration
can create plateaus (`s_i < s_j` does not require `p_i < p_j`), but
must never invert ordering (`s_i < s_j` must always imply
`p_i <= p_j`). Checked two independent ways:
  1. The loaded `cal_map_tbl`'s own `pd` column is non-decreasing when
     sorted by `score_min` -- an intrinsic property of the artifact
     itself, independent of this run's data.
  2. The applied `retrospective_current_cal_pd` column, with the thick
     population sorted by its raw `retrospective_current_model_pd`, is
     itself non-decreasing -- never assumed from (1) alone, since a
     bug in THIS script's own replay could still misapply a correct
     map.
Plateau structure (number of unique calibrated values, largest
plateau) is reported alongside, since isotonic maps are inherently
piecewise-constant and a large plateau is expected behavior, not a
defect.

HISTORICAL DIAGNOSTIC: repeats R2's matured (`label_eligible_30d==1`)
bad-rate-by-band analysis against `retrospective_current_cal_pd`
instead of the raw score -- still a retrospective diagnostic, never a
calibration-fitting signal. The calibration map is NEVER adjusted
because historical bad rates don't match the calibrated values -- a
mismatch there, if present, is a reportable finding, not a defect to
fix by refitting.

No chunking: unlike R1/R2's wide (80+ column) matrix operations that
triggered real OOMs, this script's input is narrow (disbursement_fid +
3-4 scalar columns, thick-only subset) and `attach_cal_pd`'s own
internals operate on that same handful of columns, never the full
feature matrix -- confirmed by reading it (pd_model/modeling/
calibration.py), not assumed safe by analogy. If a real run still OOMs
at this narrower width, chunking can be added the same way R1/R2's
was, but there is no a priori reason to expect it here.

Usage:
    python scripts\\apply_retrospective_current_isotonic_calibration.py ^
        --retrospective-scores-csv stage4br2_retrospective_scores_scores.csv ^
        --pd-model-artifacts-dir pd_model\\artifacts ^
        --forbidden-columns-metadata-csv stage4br1_retrospective_matrix_forbidden_columns_metadata.csv ^
        --champion xgb ^
        --out-prefix stage4br3_retrospective_cal_scores
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pd_model.config import feature_config  # noqa: E402
from pd_model.config.model_config import DEFAULT_CONFIG  # noqa: E402
from pd_model.modeling.calibration import attach_cal_pd  # noqa: E402
from pd_model.modeling.inference import load_artifacts  # noqa: E402

ROUTE_THICK = "THICK_CURRENT_CHAMPION_SCORED"
ROUTE_THIN = "THIN_SCORE_NOT_PIT_REPRODUCIBLE"

CAL_COL = "retrospective_current_cal_pd"
RAW_COL = "retrospective_current_model_pd"

DISPOSITION_READY = "R3_CALIBRATION_READY"
DISPOSITION_NOT_READY = "R3_CALIBRATION_NOT_READY"

MONOTONICITY_TOLERANCE = 1e-9

PRODUCTION_SOURCE_FILES = [
    "pd_model/modeling/calibration.py",
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


def load_retrospective_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["disbursement_fid", "thin_file_flag", "route_status", RAW_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing}")
    dup = int(df["disbursement_fid"].duplicated().sum())
    if dup:
        sys.exit(f"ERROR: {dup} duplicate disbursement_fid value(s) in {path}.")
    print(f"Loaded {len(df):,} row(s) from {path}.")
    return df


def check_cal_map_intrinsic_monotonicity(cal_map_tbl: pd.DataFrame, champion: str) -> dict:
    """Checks the FROZEN artifact's own pd column is non-decreasing when
    sorted by score_min -- intrinsic to the artifact, independent of
    this script's data or logic. A failure here means the artifact
    itself is broken, not that this script mis-replayed it."""
    sub = cal_map_tbl[cal_map_tbl["model"].astype(str) == str(champion)].copy()
    sub = sub.sort_values(["score_min", "score_max"]).reset_index(drop=True)
    diffs = sub["pd"].diff().dropna()
    is_monotonic = bool((diffs >= -MONOTONICITY_TOLERANCE).all())
    return {
        "cal_map_rows_for_champion": len(sub),
        "cal_map_intrinsically_monotonic": is_monotonic,
        "cal_map_min_pd": float(sub["pd"].min()) if len(sub) else None,
        "cal_map_max_pd": float(sub["pd"].max()) if len(sub) else None,
    }


def apply_calibration_to_thick(thick_df: pd.DataFrame, cal_map: pd.DataFrame, champion: str) -> pd.Series:
    """Reproduces score_new_agents' exact call shape (pd_model/modeling/
    inference.py:286-291): rename the raw score column to RAW_SCORE_COL,
    add a placeholder bad_state=0 column, call the REAL attach_cal_pd.
    Index is set to disbursement_fid first so the result can be
    reattached unambiguously regardless of any internal reordering/
    dropna attach_cal_pd's own _standardize_scored_df performs."""
    work = thick_df[["disbursement_fid", RAW_COL]].copy()
    work = work.set_index("disbursement_fid", drop=False)
    work = work.rename(columns={RAW_COL: feature_config.RAW_SCORE_COL})
    work["bad_state"] = 0  # placeholder, matching production's own inference-time convention
    calibrated = attach_cal_pd(work, cal_map, champion, cfg=DEFAULT_CONFIG)
    # _standardize_scored_df may rebuild AGENT_KEY from the positional
    # index if agent_msisdn is absent (it is, here) -- but disbursement_fid
    # was ALSO carried as the DataFrame index before the call, and pandas
    # preserves index across the internal .copy()/column-selection/dropna
    # chain in _standardize_scored_df, so reattaching by that same index
    # is safe and independently verified by the row-count check in main().
    return calibrated[feature_config.CAL_PD_COL]


def check_applied_monotonicity(raw_scores: pd.Series, cal_scores: pd.Series) -> dict:
    """Sorts the THICK population by its raw score and checks the
    calibrated score is non-decreasing -- never assumed from the
    artifact's own intrinsic monotonicity alone (check_cal_map_
    intrinsic_monotonicity above), since a bug in this script's own
    replay could still misapply a correct map."""
    order = np.argsort(raw_scores.to_numpy(), kind="stable")
    cal_sorted = cal_scores.to_numpy()[order]
    diffs = np.diff(cal_sorted)
    violations = int((diffs < -MONOTONICITY_TOLERANCE).sum())
    return {
        "applied_calibration_monotonic": violations == 0,
        "n_monotonicity_violations": violations,
    }


def build_plateau_stats(cal_scores: pd.Series) -> dict:
    if len(cal_scores) == 0:
        return {"n_unique_calibrated_values": 0, "largest_plateau_count": 0, "largest_plateau_value": None}
    vc = cal_scores.value_counts()
    return {
        "n_unique_calibrated_values": int(vc.shape[0]),
        "largest_plateau_count": int(vc.max()),
        "largest_plateau_value": float(vc.idxmax()),
    }


def build_quantile_dict(s: pd.Series, quantile_points: list[float]) -> dict:
    if len(s) == 0:
        return {}
    return {str(q): float(v) for q, v in s.quantile(quantile_points).items()}


def build_bad_rate_by_band(
    all_output: pd.DataFrame, score_col: str, forbidden_columns_metadata_csv: Path | None, n_bands: int,
) -> dict:
    """Restated from Stage 4B-R2's build_bad_rate_by_band, parameterized
    by score_col so it can be called identically against the raw and
    calibrated score -- same censoring discipline (label_eligible_30d
    == 1), same strictly-after-scoring join."""
    if forbidden_columns_metadata_csv is None:
        return {"available": False, "note": "--forbidden-columns-metadata-csv not provided."}
    path = Path(forbidden_columns_metadata_csv)
    if not path.exists():
        return {"available": False, "note": f"{path} does not exist."}

    meta = pd.read_csv(path)
    required_meta_cols = {"disbursement_fid", "bad_state_3dpd_30d", "label_eligible_30d"}
    missing_meta = required_meta_cols - set(meta.columns)
    if missing_meta:
        return {"available": False,
                "note": f"{path} is missing required column(s): {sorted(missing_meta)}"}

    thick_out = all_output.loc[all_output["route_status"] == ROUTE_THICK, ["disbursement_fid", score_col]]
    joined = thick_out.merge(
        meta[["disbursement_fid", "bad_state_3dpd_30d", "label_eligible_30d"]],
        on="disbursement_fid", how="left", validate="one_to_one",
    )
    eligible = joined[pd.to_numeric(joined["label_eligible_30d"], errors="coerce") == 1].copy()
    n_matured = len(eligible)
    if n_matured < n_bands:
        return {"available": False,
                "note": f"only {n_matured} matured/eligible (label_eligible_30d==1) thick-file "
                        f"row(s) -- too few to band at n_bands={n_bands}."}

    eligible["score_band"] = pd.qcut(eligible[score_col], q=n_bands, duplicates="drop")
    band_tbl = eligible.groupby("score_band", observed=True).agg(
        n=("bad_state_3dpd_30d", "size"),
        n_bad=("bad_state_3dpd_30d", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
    ).reset_index()
    band_tbl["bad_rate"] = band_tbl["n_bad"] / band_tbl["n"]
    band_tbl["score_band"] = band_tbl["score_band"].astype(str)
    return {
        "available": True,
        "n_matured_eligible_thick": n_matured,
        "bands": band_tbl.to_dict(orient="records"),
        "caveat": (
            f"Among historically matured (label_eligible_30d==1) thick-file episodes, observed "
            f"subsequent same-loan bad rate by {score_col} band. Descriptive retrospective "
            f"diagnostic only -- never used to refit or adjust the calibration map. A mismatch "
            f"between calibrated values and observed historical bad rates is a reportable "
            f"finding, not a defect."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrospective-scores-csv", type=Path, required=True,
                     help="Stage 4B-R2's own *_scores.csv output.")
    ap.add_argument("--pd-model-artifacts-dir", type=Path, required=True)
    ap.add_argument("--forbidden-columns-metadata-csv", type=Path, default=None)
    ap.add_argument("--champion", choices=["xgb", "lgb"], default="xgb",
                     help="Must match the --champion Stage 4B-R2 was run with -- the calibration "
                          "map is model-specific.")
    ap.add_argument("--r2-validation-report-json", type=Path, default=None,
                     help="Optional. Stage 4B-R2's own validation report -- if given, cross-checks "
                          "--champion against the champion R2 actually used (warns, does not block, "
                          "on a mismatch) and carries R2's audit trail forward into this report.")
    ap.add_argument("--out-prefix", type=str, default="stage4br3_retrospective_cal_scores")
    ap.add_argument("--n-bands", type=int, default=10)
    args = ap.parse_args(argv)

    print(f"\n{'#' * 100}\nSTAGE 4B-R3 -- FROZEN CURRENT ISOTONIC CALIBRATION (THICK-FILE ROUTE ONLY)\n"
          f"No isotonic regression is refit here -- only the frozen artifact's existing bin-lookup "
          f"map is replayed. No C3, no Capacity(F), no k selection.\n{'#' * 100}")

    r2_champion_cross_check: dict = {"r2_validation_report_provided": False}
    if args.r2_validation_report_json is not None:
        r2_report_path = Path(args.r2_validation_report_json)
        if r2_report_path.exists():
            r2_report = json.loads(r2_report_path.read_text())
            r2_champion = r2_report.get("champion_artifact_identity", {}).get("champion")
            match = r2_champion == args.champion
            r2_champion_cross_check = {
                "r2_validation_report_provided": True, "r2_champion": r2_champion,
                "r3_champion": args.champion, "match": match,
                "r2_model_file_sha256": r2_report.get("champion_artifact_identity", {}).get("model_file_sha256"),
            }
            if not match:
                print(f"\n{'!' * 100}\nWARNING: --champion={args.champion} does not match the champion "
                      f"R2 was run with ({r2_champion}) -- proceeding anyway (not a hard block), but "
                      f"this is almost certainly a mistake.\n{'!' * 100}")
        else:
            print(f"NOTE: --r2-validation-report-json given but {r2_report_path} does not exist -- skipping cross-check.")

    # ------------------------------------------------------------------ #
    # 1) Load R2's scores + the real champion artifact (frozen cal_map)
    # ------------------------------------------------------------------ #
    scores_df = load_retrospective_scores(args.retrospective_scores_csv)
    n_total = len(scores_df)
    n_thick = int((scores_df["route_status"] == ROUTE_THICK).sum())
    n_thin = int((scores_df["route_status"] == ROUTE_THIN).sum())

    artifacts = load_artifacts(args.pd_model_artifacts_dir)
    cal_map_fidelity = check_cal_map_intrinsic_monotonicity(artifacts.cal_map, args.champion)
    print(f"Loaded cal_map: {cal_map_fidelity['cal_map_rows_for_champion']} row(s) for champion="
          f"{args.champion}, intrinsically monotonic={cal_map_fidelity['cal_map_intrinsically_monotonic']}.")

    # ------------------------------------------------------------------ #
    # 2) Apply calibration to the THICK subset only
    # ------------------------------------------------------------------ #
    thick_df = scores_df.loc[scores_df["route_status"] == ROUTE_THICK].copy()
    thin_df = scores_df.loc[scores_df["route_status"] == ROUTE_THIN].copy()

    cal_series = apply_calibration_to_thick(thick_df, artifacts.cal_map, args.champion)
    # Reattach by disbursement_fid (the index apply_calibration_to_thick's
    # own call chain preserves) -- never assumed positionally aligned.
    thick_df = thick_df.set_index("disbursement_fid", drop=False)
    thick_df[CAL_COL] = cal_series.reindex(thick_df.index)
    thick_df = thick_df.reset_index(drop=True)
    thin_df[CAL_COL] = np.nan

    all_output = pd.concat([thick_df, thin_df], ignore_index=True)
    out_prefix = args.out_prefix
    scores_path = f"{out_prefix}_scores.csv"
    all_output[["disbursement_fid", "thin_file_flag", "route_status", RAW_COL, CAL_COL]].to_csv(
        scores_path, index=False)
    print(f"Wrote {scores_path} ({len(all_output):,} row(s) total).")

    thick_cal = thick_df[CAL_COL]
    thick_raw = thick_df[RAW_COL]
    n_thick_cal_nonnull = int(thick_cal.notna().sum())
    n_thin_cal_nonnull = int(thin_df[CAL_COL].notna().sum())

    # ------------------------------------------------------------------ #
    # 3) Mechanical-consistency checks
    # ------------------------------------------------------------------ #
    applied_monotonicity = check_applied_monotonicity(thick_raw, thick_cal)
    plateau_stats = build_plateau_stats(thick_cal)

    # ------------------------------------------------------------------ #
    # 4) Structural validation
    # ------------------------------------------------------------------ #
    cal_values_valid_range = bool(
        n_thick_cal_nonnull == 0
        or (thick_cal.notna().all() and np.isfinite(thick_cal.to_numpy()).all()
            and thick_cal.between(0.0, 1.0).all())
    )
    structural = {
        "row_preservation": len(all_output) == n_total,
        "thick_calibration_complete": n_thick_cal_nonnull == n_thick,
        "thin_calibration_is_nan": n_thin_cal_nonnull == 0,
        "cal_map_intrinsically_monotonic": cal_map_fidelity["cal_map_intrinsically_monotonic"],
        "applied_calibration_monotonic": applied_monotonicity["applied_calibration_monotonic"],
        "calibrated_values_valid_range": cal_values_valid_range,
    }
    disposition = DISPOSITION_READY if all(structural.values()) else DISPOSITION_NOT_READY

    print(f"\n{'#' * 100}\nDISPOSITION: {disposition}\n{'#' * 100}")
    for name, ok in structural.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    # ------------------------------------------------------------------ #
    # 5) Analytical diagnostics (never gate disposition)
    # ------------------------------------------------------------------ #
    quantile_points = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    raw_bad_rate = build_bad_rate_by_band(all_output, RAW_COL, args.forbidden_columns_metadata_csv, args.n_bands)
    cal_bad_rate = build_bad_rate_by_band(all_output, CAL_COL, args.forbidden_columns_metadata_csv, args.n_bands)

    validation_report = {
        "disposition": disposition,
        "structural_validation": structural,
        "counts": {
            "n_total": n_total, "n_thick": n_thick, "n_thin": n_thin,
            "n_thick_calibrated": n_thick_cal_nonnull, "n_thin_calibrated_nonnull": n_thin_cal_nonnull,
        },
        "mechanical_consistency": {**cal_map_fidelity, **applied_monotonicity, **plateau_stats},
        "analytical_diagnostics": {
            "score_quantiles_raw_thick": build_quantile_dict(thick_raw, quantile_points),
            "score_quantiles_calibrated_thick": build_quantile_dict(thick_cal, quantile_points),
            "bad_rate_by_band_raw_score": raw_bad_rate,
            "bad_rate_by_band_calibrated_score": cal_bad_rate,
        },
        "champion": args.champion,
        "r2_cross_check": r2_champion_cross_check,
        "terminology_note": (
            f"{CAL_COL} is the frozen current isotonic calibration applied to {RAW_COL} for "
            f"thick-file episodes only -- never bare 'cal_pd' (which could be confused with a "
            f"genuine historical production value) and never 'cal_pd_pit'. The isotonic "
            f"regression itself is never refit on these historical episodes; only the existing "
            f"bin-lookup map is replayed. Thin-file rows remain {ROUTE_THIN} / NaN, unchanged "
            f"from Stage 4B-R2."
        ),
        "artifact_identity": {
            "pd_calibration_map_csv_sha256": _sha256_file(
                Path(args.pd_model_artifacts_dir) / "pd_calibration_map.csv"),
            "git_commit": _git_commit(),
            "production_source_file_sha256": {
                f: _sha256_file(Path(__file__).resolve().parent.parent / f)
                for f in PRODUCTION_SOURCE_FILES
            },
        },
        "inputs": {
            "retrospective_scores_csv": str(args.retrospective_scores_csv),
            "pd_model_artifacts_dir": str(args.pd_model_artifacts_dir),
            "forbidden_columns_metadata_csv": str(args.forbidden_columns_metadata_csv)
            if args.forbidden_columns_metadata_csv else None,
            "champion": args.champion,
        },
    }
    report_path = f"{out_prefix}_validation_report.json"
    Path(report_path).write_text(json.dumps(validation_report, indent=2, default=str))
    print(f"Wrote {report_path}")

    print(
        f"\n{'#' * 100}\nCONCLUSION\n{'#' * 100}\n"
        f"{disposition}. {n_thick_cal_nonnull:,} thick-file episode(s) calibrated with "
        f"{CAL_COL}; {n_thin:,} thin-file episode(s) remain NaN ({ROUTE_THIN}). No C3, no "
        f"Capacity(F), no k selection performed. Stage 4B-R4/R3B must not proceed unless "
        f"disposition == {DISPOSITION_READY}."
    )


if __name__ == "__main__":
    main()
