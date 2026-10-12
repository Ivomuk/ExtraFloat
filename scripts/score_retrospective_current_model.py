"""
score_retrospective_current_model.py
==================================================
Stage 4B-R2 -- Retrospective Current-Model Scoring, THICK-FILE ROUTE ONLY.

Scores Stage 4B-R1's validated 80-feature retrospective matrix with the
REAL current champion artifact (`load_artifacts()` + `predict_proba`),
producing `retrospective_current_model_pd`. Produces NO isotonic
calibration, NO C3 multiplier, NO Capacity(F) combination, NO `k`
selection -- all of that is later, separately-gated work (Stage 4B-R3
onward).

TERMINOLOGY, enforced everywhere in this script's output: the resulting
field is `retrospective_current_model_pd`, never `cal_pd_pit` and never
"historical production PD". Even for thick-file rows it answers a
narrower question: how the CURRENT champion artifact would rank/assess
this historical episode using information available at that decision
time -- not what the lending system's risk signal actually was at that
time (that is Outcome A's question, already closed as infeasible by
Stage 4A/4A.1).

WHY THIN-FILE ROWS ARE DELIBERATELY LEFT UNSCORED (NaN), not a gap to
be filled later: investigating production's actual thin-file routing
(`pd_model/modeling/inference.py`'s `score_new_agents`,
`add_never_loan_scorecard_from_phase_2_1`,
`apply_thin_file_lr` -- `pd_model/modeling/scorecard.py`) found that
`add_never_loan_scorecard_from_phase_2_1` is NOT row-local or
artifact-backed: it has no fitted/frozen parameters at all, and its own
code and comments confirm it recomputes winsorization quantiles AND
thin-file-subset median/IQR statistics fresh from WHATEVER BATCH OF
ROWS is passed into one call. There is no recorded historical batch
population to replay -- reconstructing it would require either (a)
pooling all ~4.78M historical episodes into one batch (which would make
an August decision's score depend on agents appearing in later months --
temporal population leakage, not label leakage, but leakage all the
same), or (b) inventing a per-date/per-cohort batching scheme with no
basis in the code or any artifact. Both were rejected as manufacturing
provenance that does not exist, rather than reporting the coverage gap
honestly. Thin-file rows are therefore explicitly classified
`THIN_SCORE_NOT_PIT_REPRODUCIBLE` with `retrospective_current_model_pd
= NaN` -- a provenance limitation, recorded as evidence, never hidden
by picking an arbitrary reference population.

This is consistent with, and confirmed by, this script's own trace of
calibration lineage: `pd_model/run_pipeline.py` restricts BOTH XGBoost
and LightGBM training to `thick_train_mask`/`thick_val_mask` (thin_file
rows are excluded from training entirely -- the code's own comment
explains why: including them "teaches the model to use missingness as a
proxy for thin-file status rather than genuine credit risk"), and
`run_locked_policy_pipeline(xgb_val_cal, lgb_val_cal, ...)` --
`pd_model/run_pipeline.py:904` -- receives only this thick-only-derived
data. The isotonic `cal_map` artifact was fit EXCLUSIVELY on thick-file
champion scores; there is no combined pre-calibration field spanning
both routes. The population for which the strongest retrospective
reconstruction exists (thick-file, row-local, artifact-backed scoring +
calibration) is exactly the population this script scores.

ROUTING SOURCE: `thin_file_flag` is RECOMPUTED here via the real
`derive_thin_file_flag` (pd_model/preprocessing/loan_history_features.py),
called on the SAME assembled input Stage 4B-R1 built (restated, never
imported, from R1's own assembly steps -- one-way `scripts/` layering
convention) -- NOT consumed from R1's frozen matrix, which never
promised to carry it. This is recorded explicitly in the validation
report as `thin_file_routing_source =
"production_derive_thin_file_flag_recomputed_from_same_R1_inputs"`,
never silently treated as part of the frozen R1 artifact. None of the
restated assembly steps are part of the OOM class found in R1 (that was
specifically `get_and_classify_pd_features`/`build_transformed_
dataframe`/`select_dtypes` on the full 80-feature matrix; none of those
are called here).

FAIL-CLOSED ROUTING (never defaults an ambiguous episode to either
route): every row's recomputed `thin_file_flag` must be exactly 0 or 1
(no NaN, no other value) and every `disbursement_fid` must be unique in
the recomputed routing table and in R1's matrix -- any violation is a
hard exit, not a silent default.

PREDICT_PROBA CONVENTION: reproduces `score_new_agents`'s own exact call
(`pd_model/modeling/inference.py:260-261`):
`artifacts.<champion>_model.predict_proba(X)[:, 1]` -- never assumed to
be the "obvious" convention, cited from the production call site.
`X` is built via the REAL `align_features()` on R1's already-aligned
matrix columns (idempotent here, but cheap and exactly matches
production's own call shape).

STRUCTURAL VALIDATION vs. ANALYTICAL DIAGNOSTICS (kept as two separate
sections in the output, per explicit review): structural PASS concerns
ONLY scoring integrity (row preservation, routing partition, champion
artifact loaded, thick-scoring completeness, thin-NaN enforcement,
score validity, feature-order match) -- it is NEVER gated on model
performance. Score quantiles, repeated-value concentration, and the
observed-bad-rate-by-score-band diagnostic are reported separately and
NEVER determine the disposition; monotonicity of bad rate across bands
is explicitly NOT required for `R2_SCORING_READY`.

CENSORING DISCIPLINE IN THE BAD-RATE DIAGNOSTIC: computed only over
rows where `label_eligible_30d == 1` (the same censoring flag this
whole workstream has used throughout -- 0 means inadequate follow-up,
explicitly excluded, never defaulted to "good"), read from R1's own
`_forbidden_columns_metadata.csv` side output, joined back by
`disbursement_fid` STRICTLY AFTER scoring is complete. The label never
touches feature construction or the model call at any point.

Usage:
    python scripts\\score_retrospective_current_model.py ^
        --phase21-reconstructed-features-csv stage4a1_phase21_pit_phase21_pit_features.csv ^
        --loan-training-file data\\state_data_20260910_retail_filtered.csv ^
        --pd-model-artifacts-dir pd_model\\artifacts ^
        --retrospective-matrix-csv stage4br1_retrospective_matrix_matrix.csv ^
        --forbidden-columns-metadata-csv stage4br1_retrospective_matrix_forbidden_columns_metadata.csv ^
        --champion xgb ^
        --out-prefix stage4br2_retrospective_scores
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

from pd_model.config.model_config import DEFAULT_CONFIG  # noqa: E402
from pd_model.modeling.inference import align_features, load_artifacts  # noqa: E402
from pd_model.preprocessing.loan_history_features import derive_thin_file_flag  # noqa: E402
from pd_model.preprocessing.transaction_features import run_phase_2_1_richer_tx_behaviour  # noqa: E402

STATUS_RECONSTRUCTED = "PIT_PHASE21_RECONSTRUCTED"

ROUTE_THICK = "THICK_CURRENT_CHAMPION_SCORED"
ROUTE_THIN = "THIN_SCORE_NOT_PIT_REPRODUCIBLE"

DISPOSITION_READY = "R2_SCORING_READY"
DISPOSITION_NOT_READY = "R2_SCORING_NOT_READY"

THIN_FILE_ROUTING_SOURCE = "production_derive_thin_file_flag_recomputed_from_same_R1_inputs"

PREDICT_PROBA_CONVENTION = (
    "artifacts.<champion>_model.predict_proba(X)[:, 1] -- identical to score_new_agents "
    "(pd_model/modeling/inference.py:260-261), not assumed."
)

# Restated verbatim from build_retrospective_scoring_matrix.py -- see that
# file's own comment for why this must stay the FULL raw mart schema.
_PHASE21_RAW_MART_COLS = sorted(set(
    ["account_balance", "average_balance", "commission",
     "commission_cluster_mean", "vol_3m_cluster_mean", "cluster_avg_commission", "cluster_avg_vol_3m",
     "cash_in_peers_3m", "cash_in_vol_3m"]
    + ["cust_1m", "cust_3m", "cust_6m", "vol_1m", "vol_3m", "vol_6m"]
    + [
        f"{txn_type}_{metric}_{horizon}"
        for txn_type in ("cash_out", "cash_in", "payment")
        for metric in ("vol", "value", "peers", "comm", "cust")
        for horizon in ("1m", "3m", "6m")
    ]
))

# The exact production source files this script imports from / restates
# against -- hashed into the validation report as the "code version"
# identifier, same convention as build_retrospective_scoring_matrix.py.
PRODUCTION_SOURCE_FILES = [
    "pd_model/preprocessing/transaction_features.py",
    "pd_model/preprocessing/loan_history_features.py",
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


# ============================================================================
# Restated from build_retrospective_scoring_matrix.py (Stage 4B-R1) -- one-way
# scripts/ layering convention: restated, never imported, so this script
# never depends on R1's module internals changing underneath it. Needed
# ONLY to recompute thin_file_flag; R1's own matrix CSV supplies the 80
# scoring features.
# ============================================================================


def load_reconstructed_phase21(path: Path) -> pd.DataFrame:
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
    print(f"Phase 2.1 reconstruction input (routing recomputation): {n_total:,} row(s); "
          f"{len(df):,} have reconstruction_status == {STATUS_RECONSTRUCTED} and are retained. "
          f"{n_excluded:,} excluded (no eligible prior snapshot).")

    dup = int(df["disbursement_fid"].duplicated().sum())
    if dup:
        sys.exit(f"ERROR: {dup} duplicate disbursement_fid value(s) in the reconstructed Phase 2.1 rows.")

    df = df.rename(columns={"days_since_snapshot": "_pit_days_since_snapshot"})
    df["fundamentals_snapshot_date"] = pd.to_datetime(df["fundamentals_snapshot_date"], errors="coerce")
    return df


def load_loan_training_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "agent_msisdn" not in df.columns and "msisdn" in df.columns:
        df = df.rename(columns={"msisdn": "agent_msisdn"})
    required = ["disbursement_fid", "agent_msisdn", "loan_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing}")

    dup = int(df["disbursement_fid"].duplicated().sum())
    if dup:
        sys.exit(f"ERROR: {dup} duplicate disbursement_fid value(s) in {path}.")

    df["loan_date"] = pd.to_datetime(df["loan_date"], errors="coerce")
    print(f"Loan-training file (routing recomputation): {len(df):,} row(s), {len(df.columns)} column(s).")
    return df


def assemble_episode_matrix(phase21_df: pd.DataFrame, loan_df: pd.DataFrame) -> pd.DataFrame:
    merged = loan_df.merge(phase21_df, on="disbursement_fid", how="inner", validate="one_to_one")
    if len(merged) != len(phase21_df):
        sys.exit(
            f"ERROR: merge row-count mismatch -- {len(phase21_df):,} reconstructed Phase 2.1 "
            f"episode(s) but only {len(merged):,} matched the loan-training file. "
            f"--loan-training-file must be the SAME file used to build Stage 4B-R1's matrix."
        )
    print(f"Assembled episode matrix (routing recomputation): {len(merged):,} row(s).")
    return merged


def assert_strict_prior_pit(df: pd.DataFrame) -> None:
    violations = df[df["fundamentals_snapshot_date"] >= df["loan_date"]]
    if len(violations):
        raise AssertionError(
            f"LEAKAGE DETECTED (routing recomputation): {len(violations)} row(s) have "
            f"fundamentals_snapshot_date >= loan_date."
        )


def apply_production_phase21(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Restated from R1 -- only days_since_snapshot handling is kept (needed
    for run_phase_2_1_richer_tx_behaviour to run at all); the full evidence
    dict R1 builds is not needed here since this call exists solely to feed
    derive_thin_file_flag's inputs, not to be re-validated for leakage
    (R1 already owns that claim for the feature matrix itself)."""
    pit_days = df["_pit_days_since_snapshot"].reset_index(drop=True)
    df = df.drop(columns=["_pit_days_since_snapshot"]).reset_index(drop=True)
    df = run_phase_2_1_richer_tx_behaviour(df, cfg=cfg)
    df["days_since_snapshot"] = pit_days.values
    return df


def compute_routing(phase21_csv: Path, loan_training_file: Path) -> tuple[pd.DataFrame, dict]:
    """Recomputes (disbursement_fid, thin_file_flag) via the identical
    restated assembly steps Stage 4B-R1 used, calling the REAL
    derive_thin_file_flag on the same assembled input -- recorded as
    evidence, never assumed identical to R1's own (R1 never persisted
    thin_file_flag at all, so there is nothing to compare against;
    THIN_FILE_ROUTING_SOURCE documents how this value was obtained)."""
    phase21_df = load_reconstructed_phase21(phase21_csv)
    loan_df = load_loan_training_frame(loan_training_file)
    merged = assemble_episode_matrix(phase21_df, loan_df)
    assert_strict_prior_pit(merged)
    merged = apply_production_phase21(merged, DEFAULT_CONFIG)
    merged = derive_thin_file_flag(merged, cfg=DEFAULT_CONFIG)
    routing_df = merged[["disbursement_fid", "thin_file_flag"]].copy()
    evidence = {
        "thin_file_routing_source": THIN_FILE_ROUTING_SOURCE,
        "routing_function": "pd_model.preprocessing.loan_history_features.derive_thin_file_flag",
        "n_routing_rows": len(routing_df),
    }
    return routing_df, evidence


def validate_routing(routing_df: pd.DataFrame) -> dict:
    """Fail-closed: thin_file_flag must be exactly 0 or 1 for every row (no
    NaN, no other value), and disbursement_fid must be unique. Hard exits
    on any violation -- never defaults an ambiguous row to either route."""
    dup = int(routing_df["disbursement_fid"].duplicated().sum())
    if dup:
        sys.exit(f"ERROR: {dup} duplicate disbursement_fid value(s) in recomputed routing -- "
                  f"cannot assign exactly one route per episode.")

    flag_num = pd.to_numeric(routing_df["thin_file_flag"], errors="coerce")
    bad_mask = flag_num.isna() | ~flag_num.isin([0, 1])
    n_bad = int(bad_mask.sum())
    if n_bad:
        sample = routing_df.loc[bad_mask, "disbursement_fid"].head(10).tolist()
        sys.exit(f"ERROR: {n_bad} row(s) have thin_file_flag outside {{0,1}} (including NaN) -- "
                  f"routing must fail closed, never default. Sample disbursement_fid: {sample}")

    n_total = len(routing_df)
    n_thick = int((flag_num == 0).sum())
    n_thin = int((flag_num == 1).sum())
    if n_thick + n_thin != n_total:
        sys.exit("ERROR: n_thick + n_thin != n_total after routing validation -- must never happen.")

    print(f"Routing validated: {n_total:,} row(s); thin_file_flag in {{0,1}} for every row; "
          f"disbursement_fid unique. n_thick={n_thick:,}, n_thin={n_thin:,}.")
    return {"n_total": n_total, "n_thick": n_thick, "n_thin": n_thin}


def score_chunk_thick_route(
    chunk_features: pd.DataFrame, feature_order: list[str], model,
) -> np.ndarray:
    """Scores a chunk of THICK-FILE rows with the real champion model, using
    the identical predict_proba convention production uses in
    score_new_agents -- see PREDICT_PROBA_CONVENTION above."""
    X = align_features(chunk_features.copy(), feature_order)
    return model.predict_proba(X)[:, 1]


def build_structural_validation(
    n_total_matrix_rows: int, n_dup_within_chunk_total: int, route_counts: dict,
    n_thick_total: int, n_thin_total: int, n_thick_scored: int, n_thin_scored_nonnull: int,
    thick_scores: pd.Series, champion_artifact_loaded: bool, feature_order_match: bool,
) -> tuple[str, dict]:
    disbursement_fid_unique = n_dup_within_chunk_total == 0 and n_total_matrix_rows == route_counts["n_total"]
    score_values_valid_range = bool(
        len(thick_scores) == 0
        or (thick_scores.notna().all() and np.isfinite(thick_scores.to_numpy()).all()
            and thick_scores.between(0.0, 1.0).all())
    )
    conditions = {
        "row_preservation": n_total_matrix_rows == route_counts["n_total"],
        "disbursement_fid_unique": disbursement_fid_unique,
        "routing_partition_exact": (
            (n_thick_total + n_thin_total == n_total_matrix_rows)
            and n_thick_total == route_counts["n_thick"]
            and n_thin_total == route_counts["n_thin"]
        ),
        "thick_scoring_complete": n_thick_scored == n_thick_total,
        "thin_scores_are_nan": n_thin_scored_nonnull == 0,
        "champion_artifact_loaded": champion_artifact_loaded,
        "score_values_valid_range": score_values_valid_range,
        "feature_order_match": feature_order_match,
    }
    disposition = DISPOSITION_READY if all(conditions.values()) else DISPOSITION_NOT_READY
    return disposition, conditions


def build_bad_rate_by_band(
    all_output: pd.DataFrame, forbidden_columns_metadata_csv: Path | None, n_bands: int,
) -> dict:
    """Censoring-aware diagnostic: restricted to label_eligible_30d == 1
    (matured) THICK-file rows, joined back by disbursement_fid STRICTLY
    AFTER scoring -- the label never touches feature construction or the
    model call. Never gates the structural disposition."""
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

    thick_out = all_output.loc[all_output["route_status"] == ROUTE_THICK,
                                ["disbursement_fid", "retrospective_current_model_pd"]]
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

    eligible["score_band"] = pd.qcut(eligible["retrospective_current_model_pd"], q=n_bands, duplicates="drop")
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
            "Among historically matured (label_eligible_30d==1) thick-file episodes, observed "
            "subsequent same-loan bad rate by retrospective-current-model score band. Descriptive "
            "diagnostic only -- does not imply calibration or historical production PD. "
            "Monotonicity across bands is NOT a structural PASS condition."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase21-reconstructed-features-csv", type=Path, required=True)
    ap.add_argument("--loan-training-file", type=Path, required=True)
    ap.add_argument("--pd-model-artifacts-dir", type=Path, required=True)
    ap.add_argument("--retrospective-matrix-csv", type=Path, required=True)
    ap.add_argument("--forbidden-columns-metadata-csv", type=Path, default=None,
                     help="Optional. R1's _forbidden_columns_metadata.csv side output -- enables the "
                          "censoring-aware observed-bad-rate-by-score-band diagnostic. Read-only, "
                          "joined strictly after scoring; never touches feature construction.")
    ap.add_argument("--champion", choices=["xgb", "lgb"], default="xgb")
    ap.add_argument("--out-prefix", type=str, default="stage4br2_retrospective_scores")
    ap.add_argument("--chunk-size", type=int, default=250_000,
                     help="Row-chunk size for the predict_proba step (default 250,000). Purely a "
                          "memory safeguard -- predict_proba is row-local, so chunk boundaries cannot "
                          "change any scored value, unlike R1's thin-file-scorecard batch-dependence "
                          "risk this script deliberately does not attempt to replay.")
    ap.add_argument("--n-bands", type=int, default=10)
    args = ap.parse_args(argv)

    print(f"\n{'#' * 100}\nSTAGE 4B-R2 -- RETROSPECTIVE CURRENT-MODEL SCORING (THICK-FILE ROUTE ONLY)\n"
          f"No isotonic calibration, no C3 multiplier, no Capacity(F), no k selection. Thin-file "
          f"rows are classified {ROUTE_THIN} with retrospective_current_model_pd = NaN -- a "
          f"provenance limitation (no recorded historical scoring-batch population for the "
          f"whole-batch-dependent thin-file scorecard), never a manufactured score.\n{'#' * 100}")

    # ------------------------------------------------------------------ #
    # 1) Routing (recomputed, fail-closed)
    # ------------------------------------------------------------------ #
    routing_df, routing_evidence = compute_routing(
        args.phase21_reconstructed_features_csv, args.loan_training_file)
    route_counts = validate_routing(routing_df)
    thin_lookup = routing_df.set_index("disbursement_fid")["thin_file_flag"].astype(int)

    # ------------------------------------------------------------------ #
    # 2) Real champion artifact
    # ------------------------------------------------------------------ #
    artifacts = load_artifacts(args.pd_model_artifacts_dir)
    model = artifacts.xgb_model if args.champion == "xgb" else artifacts.lgb_model
    model_filename = "xgb_model.joblib" if args.champion == "xgb" else "lgbm_model.joblib"
    model_path = Path(args.pd_model_artifacts_dir) / model_filename
    model_sha256 = _sha256_file(model_path)
    champion_artifact_loaded = model is not None and model_sha256 is not None
    print(f"Loaded champion artifact: {args.champion} ({model_filename}, sha256={model_sha256}). "
          f"{len(artifacts.feature_order)} feature(s) in feature_order.json.")

    model_metadata: dict = {}
    meta_path = Path(args.pd_model_artifacts_dir) / "model_metadata.json"
    if meta_path.exists():
        try:
            model_metadata = json.loads(meta_path.read_text())
        except Exception as exc:
            model_metadata = {"_error": f"could not parse model_metadata.json: {exc}"}

    # ------------------------------------------------------------------ #
    # 3) Chunked scoring over R1's validated matrix
    # ------------------------------------------------------------------ #
    out_prefix = args.out_prefix
    scores_path = f"{out_prefix}_scores.csv"
    n_total_matrix_rows = 0
    n_dup_within_chunk_total = 0
    feature_order_match: bool | None = None
    output_chunks: list[pd.DataFrame] = []

    print(f"\nScoring thick-file rows in chunks of {args.chunk_size:,} from "
          f"{args.retrospective_matrix_csv}, streaming to {scores_path}.")
    reader = pd.read_csv(args.retrospective_matrix_csv, chunksize=args.chunk_size)
    for chunk_idx, chunk in enumerate(reader):
        n_dup_within_chunk_total += int(chunk["disbursement_fid"].duplicated().sum())

        if chunk_idx == 0:
            feature_cols = [c for c in chunk.columns if c != "disbursement_fid"]
            feature_order_match = feature_cols == list(artifacts.feature_order)
        else:
            feature_cols = [c for c in chunk.columns if c != "disbursement_fid"]

        thin_flags = thin_lookup.reindex(chunk["disbursement_fid"])
        unmatched = thin_flags.isna()
        if unmatched.any():
            sample = chunk.loc[unmatched.values, "disbursement_fid"].head(10).tolist()
            sys.exit(f"ERROR: {int(unmatched.sum())} matrix row(s) in chunk {chunk_idx + 1} have a "
                      f"disbursement_fid not present in the recomputed routing table. Sample: {sample}")

        thin_flags_arr = thin_flags.to_numpy().astype(int)
        thick_mask = thin_flags_arr == 0
        thin_mask = ~thick_mask

        scores = np.full(len(chunk), np.nan, dtype="float64")
        if thick_mask.any():
            scores[thick_mask] = score_chunk_thick_route(
                chunk.loc[thick_mask, feature_cols], artifacts.feature_order, model)

        route_status = np.where(thick_mask, ROUTE_THICK, ROUTE_THIN)
        out_chunk = pd.DataFrame({
            "disbursement_fid": chunk["disbursement_fid"].to_numpy(),
            "thin_file_flag": thin_flags_arr,
            "route_status": route_status,
            "retrospective_current_model_pd": scores,
        })
        out_chunk.to_csv(scores_path, mode="w" if chunk_idx == 0 else "a",
                          header=(chunk_idx == 0), index=False)
        output_chunks.append(out_chunk)
        n_total_matrix_rows += len(chunk)
        print(f"  chunk {chunk_idx + 1}: {len(chunk):,} row(s) scored/routed "
              f"({int(thick_mask.sum()):,} thick, {int(thin_mask.sum()):,} thin).")

    print(f"Wrote {scores_path} ({n_total_matrix_rows:,} row(s) total).")

    all_output = pd.concat(output_chunks, ignore_index=True) if output_chunks else pd.DataFrame(
        columns=["disbursement_fid", "thin_file_flag", "route_status", "retrospective_current_model_pd"])

    n_thick_total = int((all_output["route_status"] == ROUTE_THICK).sum())
    n_thin_total = int((all_output["route_status"] == ROUTE_THIN).sum())
    thick_scores = all_output.loc[all_output["route_status"] == ROUTE_THICK, "retrospective_current_model_pd"]
    thin_scores = all_output.loc[all_output["route_status"] == ROUTE_THIN, "retrospective_current_model_pd"]
    n_thick_scored = int(thick_scores.notna().sum())
    n_thin_scored_nonnull = int(thin_scores.notna().sum())

    # ------------------------------------------------------------------ #
    # 4) Structural validation (gates disposition)
    # ------------------------------------------------------------------ #
    disposition, structural = build_structural_validation(
        n_total_matrix_rows=n_total_matrix_rows, n_dup_within_chunk_total=n_dup_within_chunk_total,
        route_counts=route_counts, n_thick_total=n_thick_total, n_thin_total=n_thin_total,
        n_thick_scored=n_thick_scored, n_thin_scored_nonnull=n_thin_scored_nonnull,
        thick_scores=thick_scores, champion_artifact_loaded=champion_artifact_loaded,
        feature_order_match=bool(feature_order_match),
    )

    print(f"\n{'#' * 100}\nDISPOSITION: {disposition}\n{'#' * 100}")
    for name, ok in structural.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    # ------------------------------------------------------------------ #
    # 5) Analytical diagnostics (never gate disposition)
    # ------------------------------------------------------------------ #
    quantile_points = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    score_quantiles = (
        {str(q): float(v) for q, v in thick_scores.quantile(quantile_points).items()}
        if len(thick_scores) else {}
    )
    value_counts = thick_scores.value_counts().head(10)
    repeated_score_concentration = [
        {"value": float(v), "count": int(c),
         "pct": round(100.0 * c / len(thick_scores), 4) if len(thick_scores) else 0.0}
        for v, c in value_counts.items()
    ]
    bad_rate_section = build_bad_rate_by_band(
        all_output, args.forbidden_columns_metadata_csv, args.n_bands)

    # ------------------------------------------------------------------ #
    # 6) Write validation report
    # ------------------------------------------------------------------ #
    validation_report = {
        "disposition": disposition,
        "structural_validation": structural,
        "counts": {
            "n_total": n_total_matrix_rows,
            "n_thick": n_thick_total, "n_thin": n_thin_total,
            "n_thick_scored": n_thick_scored, "n_thin_scored_nonnull": n_thin_scored_nonnull,
        },
        "thin_file_routing_source": THIN_FILE_ROUTING_SOURCE,
        "routing_evidence": routing_evidence,
        "champion_artifact_identity": {
            "champion": args.champion,
            "model_file": model_filename,
            "model_file_sha256": model_sha256,
            "predict_proba_convention": PREDICT_PROBA_CONVENTION,
            "model_metadata": model_metadata,
        },
        "analytical_diagnostics": {
            "score_quantiles_thick": score_quantiles,
            "repeated_score_concentration_thick_top10": repeated_score_concentration,
            "bad_rate_by_score_band": bad_rate_section,
        },
        "terminology_note": (
            "retrospective_current_model_pd reflects how the CURRENT champion artifact would "
            "rank/assess this historical thick-file episode using information available at that "
            "decision time -- it is NOT historical production PD, and NEVER cal_pd_pit. Thin-file "
            f"rows are deliberately left NaN ({ROUTE_THIN}) rather than assigned a manufactured "
            "score, because the thin-file scorecard is whole-batch-dependent with no recorded "
            "historical reference population to replay."
        ),
        "artifact_identity": {
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
            "retrospective_matrix_csv": str(args.retrospective_matrix_csv),
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
        f"{disposition}. {n_thick_total:,} thick-file episode(s) scored with "
        f"retrospective_current_model_pd; {n_thin_total:,} thin-file episode(s) left NaN "
        f"({ROUTE_THIN}). No calibration, no C3, no Capacity(F), no k selection performed. "
        f"Stage 4B-R3 must not proceed unless disposition == {DISPOSITION_READY}."
    )


if __name__ == "__main__":
    main()
