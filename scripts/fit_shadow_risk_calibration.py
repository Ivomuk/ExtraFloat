"""
fit_shadow_risk_calibration.py
================================
MANUAL-ONLY, one-time/periodic calibration step for the shadow continuous
risk multiplier (C3 hybrid). Fits the same isotonic calibration used
throughout the prior analysis (cal_pd -> realized closed-loan bad rate,
loan-count weighted -- identical approach to
calibrate_pd_isotonic_risk_curve.py and
backtest_limit_multiplier_policies.py's _fit_isotonic()), then persists it
as a frozen artifact the live scoring path only ever reads with
.predict() -- it never calls .fit() itself.

This script is the ONLY place that writes
extrafloat.engine.extrafloat_shadow_risk_multiplier.ISOTONIC_MODEL_FILENAME /
ISOTONIC_METADATA_FILENAME. Re-run it by hand to recalibrate; the live
pipeline never refits automatically.

The metadata JSON is deliberately over-documented (target/window/weighting/
population/join-key-normalization/isotonic-parameters/plateau/schema
version, plus sha256 of both source files) so the artifact is
self-describing: a reader with no access to this script or the original
analysis notebooks should be able to tell exactly what the calibration
represents and reproduce it.

Usage:
    python scripts\\fit_shadow_risk_calibration.py \\
        --engine-output-file output\\engine_test_output.csv \\
        --forward-outcomes-file data\\persona_k8_forward_outcomes.csv \\
        --artifacts-dir pd_model\\artifacts \\
        --version-tag iso_v1_2026_10_05
"""

import argparse
import datetime as _dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

from extrafloat.engine.extrafloat_shadow_risk_multiplier import (  # noqa: E402
    ISOTONIC_METADATA_FILENAME,
    ISOTONIC_MODEL_FILENAME,
    SHADOW_SCENARIOS,
    _policy_3_hybrid,
)
from segmentation.borrower_persona_clustering import digits  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ENGINE_OUTPUT_PATH = REPO / "output" / "engine_test_output.csv"
CAL_PD_PLATEAU = 0.25  # the adopted policy plateau boundary -- see calibrate_pd_isotonic_risk_curve.py
SCHEMA_VERSION = 1
MSISDN_NORMALIZATION = "digits-only (segmentation.borrower_persona_clustering.digits())"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine-output-file", type=Path, default=ENGINE_OUTPUT_PATH, help=f"default: {ENGINE_OUTPUT_PATH}")
    p.add_argument("--forward-outcomes-file", type=Path, required=True,
                   help="CSV exported from data/persona_k8_forward_outcomes_query.sql")
    p.add_argument("--artifacts-dir", type=Path, required=True,
                   help="directory to write the shadow calibration artifact into "
                        "(typically the same artifacts_dir as the PD model's own joblib files -- "
                        "safe to co-locate, _check_artifacts() only checks a fixed named list)")
    p.add_argument("--version-tag", type=str, required=True,
                   help="required, no default -- identifies this calibration artifact "
                        "(e.g. iso_v1_2026_10_05), separate from the shadow POLICY version")
    p.add_argument("--min-loans-per-borrower", type=int, default=0,
                   help="drop borrowers with fewer than this many closed loans in the window (default 0 = keep all with >=1)")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing artifact (the previous version is archived first, not deleted)")
    args = p.parse_args(argv)

    if not args.engine_output_file.exists():
        print(f"ERROR: {args.engine_output_file} not found -- nothing to calibrate.")
        return
    if not args.forward_outcomes_file.exists():
        print(f"ERROR: {args.forward_outcomes_file} not found -- nothing to calibrate.")
        return

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / ISOTONIC_MODEL_FILENAME
    meta_path = args.artifacts_dir / ISOTONIC_METADATA_FILENAME

    if model_path.exists() or meta_path.exists():
        if not args.force:
            print(
                f"ERROR: an artifact already exists at {args.artifacts_dir} "
                f"({ISOTONIC_MODEL_FILENAME} / {ISOTONIC_METADATA_FILENAME}). "
                "Pass --force to overwrite (the previous version will be archived first, not deleted)."
            )
            return
        old_version = "unknown"
        if meta_path.exists():
            try:
                old_version = json.loads(meta_path.read_text()).get("version", "unknown")
            except Exception:
                pass
        archive_dir = args.artifacts_dir / "shadow_calibration_history" / str(old_version)
        archive_dir.mkdir(parents=True, exist_ok=True)
        print(f"--force: archiving previous artifact (version={old_version}) to {archive_dir}")
        if model_path.exists():
            shutil.copy2(model_path, archive_dir / ISOTONIC_MODEL_FILENAME)
        if meta_path.exists():
            shutil.copy2(meta_path, archive_dir / ISOTONIC_METADATA_FILENAME)

    print("=== Load engine output (cal_pd) ===")
    eng = pd.read_csv(args.engine_output_file, usecols=lambda c: c in {"msisdn", "cal_pd"})
    eng["_id"] = digits(eng["msisdn"])
    eng = eng.drop(columns=["msisdn"]).dropna(subset=["cal_pd"])
    print(f"  {len(eng):,} borrowers with a cal_pd score")

    print("\n=== Load forward-window closed-loan outcomes ===")
    fwd = pd.read_csv(args.forward_outcomes_file)
    fwd["_id"] = digits(fwd["customer_msisdn"])
    needed = {"fwd_new_loans_closed_good_count", "fwd_new_loans_closed_bad_count"}
    if not needed <= set(fwd.columns):
        print(f"  ERROR: {args.forward_outcomes_file} is missing {needed - set(fwd.columns)} -- cannot calibrate.")
        return

    merged = eng.merge(fwd[["_id"] + list(needed)], on="_id", how="inner")
    merged["_closed_n"] = merged["fwd_new_loans_closed_good_count"] + merged["fwd_new_loans_closed_bad_count"]
    merged = merged[merged["_closed_n"] >= max(1, args.min_loans_per_borrower)]
    merged["_bad_rate"] = merged["fwd_new_loans_closed_bad_count"] / merged["_closed_n"]
    n_borrowers = len(merged)
    n_closed_loans = int(merged["_closed_n"].sum())
    print(f"  {n_borrowers:,} borrowers with >=1 closed loan in the forward window "
          f"({n_closed_loans:,} closed loans total)")
    if n_borrowers < 50:
        print("  WARNING: very small sample -- the fitted calibration below will be unstable.")

    print("\n=== Fit isotonic regression: cal_pd -> calibrated bad rate (loan-count weighted) ===")
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(merged["cal_pd"], merged["_bad_rate"], sample_weight=merged["_closed_n"])

    r_plateau = float(iso.predict([CAL_PD_PLATEAU])[0])
    observed_range = [float(merged["cal_pd"].min()), float(merged["cal_pd"].max())]
    calibrated_risk_range = [float(iso.predict([observed_range[0]])[0]), float(iso.predict([observed_range[1]])[0])]
    print(f"  cal_pd_plateau={CAL_PD_PLATEAU} -> r_plateau={r_plateau:.4f}")
    print(f"  observed cal_pd range: {observed_range}")
    print(f"  observed calibrated-risk range: {calibrated_risk_range}")

    print("\n=== Sanity table: multiplier by cal_pd decile, both scenarios (review before committing) ===")
    deciles = merged["cal_pd"].quantile(np.arange(0, 1.01, 0.1))
    rows = []
    for q, cutoff in deciles.items():
        r = float(iso.predict([cutoff])[0])
        row = {"pd_decile": round(q, 2), "cal_pd_cutoff": round(cutoff, 4), "calibrated_risk": round(r, 4)}
        for name, params in SHADOW_SCENARIOS.items():
            m = _policy_3_hybrid(np.array([r]), r_plateau, params["r_floor"], params["m_max"], params["m_min"])[0]
            row[f"multiplier_{name}"] = round(float(m), 4)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    metadata = {
        "version": args.version_tag,
        "fit_timestamp_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "n_borrowers": n_borrowers,
        "n_closed_loans": n_closed_loans,
        "source_engine_output_file": str(args.engine_output_file),
        "source_engine_output_sha256": _sha256(args.engine_output_file),
        "source_forward_outcomes_file": str(args.forward_outcomes_file),
        "source_forward_outcomes_sha256": _sha256(args.forward_outcomes_file),
        "sklearn_version": sklearn.__version__,
        "target_definition": "closed_loan_bad",
        "forward_window_definition": "forward outcomes window as defined in "
                                      "data/persona_k8_forward_outcomes_query.sql at the time this artifact was fit",
        "weighting_method": "closed_loan_count",
        "population_definition": (
            f"borrowers with >=1 eligible closed loan in the defined forward window "
            f"(min_loans_per_borrower={max(1, args.min_loans_per_borrower)}), weighted by number of closed loans"
        ),
        "msisdn_normalization": MSISDN_NORMALIZATION,
        "isotonic_parameters": {"increasing": True, "out_of_bounds": "clip", "y_min": 0.0, "y_max": 1.0},
        "cal_pd_plateau": CAL_PD_PLATEAU,
        "r_plateau": r_plateau,
        "observed_cal_pd_range": observed_range,
        "observed_calibrated_risk_range": calibrated_risk_range,
        "calibration_artifact_schema_version": SCHEMA_VERSION,
    }

    joblib.dump(iso, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nWrote {model_path}")
    print(f"Wrote {meta_path}")
    print(
        "\nThis script is MANUAL ONLY. The live scoring path "
        "(extrafloat_shadow_risk_multiplier.compute_shadow_risk_multiplier) never calls .fit(), "
        "only joblib.load() + .predict(). Re-run this script by hand to recalibrate -- "
        "it will refuse to overwrite the existing artifact without --force, and archives the "
        "previous version first when you do."
    )


if __name__ == "__main__":
    main()
