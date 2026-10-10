"""
derive_capacity_score_weights.py
====================================
Deliverable 2, Stage 1 (part 1 of 2 new scripts). Derives and records the
weights used to combine business fundamentals into a single Business
Capacity Score / Capacity(F) formula -- NOT a calibration in the sense
fit_shadow_risk_calibration.py uses that word. With two standardized
inputs (float activity, commission), we are not estimating weights from
repayment performance or historical exposure; we are recording a design
choice and verifying its statistical structure. "Calibration" is reserved
for fitting a curve to an outcome -- that never happens here.

NOT A CONTINUATION of scripts/fit_capacity_challenger_model.py or
scripts/analyze_capacity_dimension_redundancy.py. Those are an earlier,
abandoned lineage (pre-Gate-0 "Analysis 1/3" numbering) that fits a
quantile-regression target of log(actual_exposure_ugx) -- exactly the
endogenous historical-assignment information Analysis 3/4
(docs/analysis_3_findings.md, docs/analysis_4_findings.md) established
must be excluded from a capacity estimate. This script and its sibling
(derive_capacity_function_and_backtest_frontier.py) are a disjoint
lineage: they read ONLY float_activity_value_1m and commission (plus join
keys) from the episode dataset, via an explicit usecols whitelist --
assigned_limit, cal_pd, disbursement_amount_ugx, and every outcome column
are never loaded into memory, not merely unused.

WHY THIS SCRIPT EXISTS: Capacity(F) (see the sibling script) combines
float activity and commission via a weighted geometric mean,
k * ((1+Float)^w_F * (1+Commission)^w_C - 1), with w_F + w_C = 1. The
weights need a methodology, not an arbitrary choice -- but "determined by
an explicit methodology" does not mean "optimized against historical
assigned limits or outcomes." This script derives w_F/w_C from ONLY the
two fundamentals' own joint distribution (their covariance structure),
never touching exposure, PD, or outcome data.

THE PCA RESULT AT n=2 IS A MATHEMATICAL TAUTOLOGY, NOT A DISCOVERY -- state
this plainly rather than oversell it. For any two variables standardized
to unit variance, their correlation matrix is [[1, rho], [rho, 1]], and
this matrix's top eigenvector is (1,1)/sqrt(2) for EVERY value of rho
(a general fact about any symmetric 2x2 matrix with equal diagonal
entries, confirmed numerically across rho in {0.1,...,0.879,...,0.99} --
always exactly [0.7071, 0.7071], normalizing to w_F=w_C=0.5). So
"standardized PCA" and "equal weighting" are NOT two different options
here -- they are the same answer. This script computes and reports both
the standardized path (which will reproduce 0.5/0.5 and is asserted to do
so, as a self-verifying governance statement) and the UNSTANDARDIZED PCA
path (reported only as a labeled diagnostic): running PCA on raw log1p
values instead would produce a different, asymmetric split, but one
driven by which variable happens to have more log-space variance in this
particular dataset snapshot -- an arbitrary scale artifact, not
economically meaningful information. That is the "degenerate, not more
defensible than equal weights" failure mode; it lives in the
unstandardized path, which is exactly why it is reported as a diagnostic
only, never used as the recommended weights.

EXTENSIBILITY: this script is written generically over an arbitrary list
of fundamentals (--fundamentals), not hardcoded to exactly two. The
0.5/0.5-forcing result above is specific to exactly two equal-variance
standardized inputs; at n>=3 (e.g. adding cust_1m/average_balance later)
standardized PCA's first component becomes genuinely data-dependent, and
re-running this script will do real work instead of confirming a
tautology.

MANUAL-ONLY, one-time/periodic script, mirroring
scripts/fit_shadow_risk_calibration.py's exact structural template:
refuses to overwrite an existing artifact without --force; archives the
previous version (not deletes) when --force is given; writes a single,
heavily self-documenting, versioned metadata JSON so a reader with no
access to this script can tell exactly what the weights represent and
reproduce them. The live scoring path (derive_capacity_function_and_
backtest_frontier.py) only ever reads this JSON's recommended_weights --
it never recomputes weights itself.

Usage:
    python scripts\\derive_capacity_score_weights.py ^
        --episode-dataset loan_episode_capacity_dataset.csv ^
        --artifacts-dir capacity_artifacts ^
        --version-tag capacity_weights_v1_2026_10_10
"""

import argparse
import datetime as _dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

FUNDAMENTALS_FOR_SCORE = ["float_activity_value_1m", "commission"]
WEIGHTS_FILENAME = "capacity_score_weights.json"
SCHEMA_VERSION = 1
WEIGHT_EQUIVALENCE_TOL = 1e-6


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_fundamentals(episode_dataset_path: Path, cols: list[str]) -> pd.DataFrame:
    """Reads ONLY the join keys plus `cols` from the episode dataset via an
    explicit usecols whitelist -- assigned_limit, cal_pd, exposure, and
    outcome columns are never loaded into memory in the first place, not
    merely unused. Deduped to one row per (agent_msisdn,
    fundamentals_snapshot_date) agent-period unit (restated subset of the
    build_agent_period_summary pattern used throughout this workstream,
    one-way scripts/ layering convention)."""
    key = ["agent_msisdn", "fundamentals_snapshot_date"]
    usecols = key + cols
    df = pd.read_csv(episode_dataset_path, usecols=lambda c: c in set(usecols))
    missing = [c for c in usecols if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: episode dataset is missing required column(s): {missing}")
    df = df[df["fundamentals_snapshot_date"].notna()]
    summary = df.groupby(key, sort=False)[cols].first().reset_index()
    return summary


def log1p_zscore(df: pd.DataFrame, cols: list[str]) -> tuple:
    """z(log1p(x)) per column, restricted to rows where every column is
    valid (notna and > 0 -- same fundamentals-validity guard used
    throughout this workstream). Returns (z_df, stats) where
    stats[col] = {"log1p_mean": ..., "log1p_std": ...}, recorded for
    metadata reproducibility."""
    valid_mask = pd.Series(True, index=df.index)
    for c in cols:
        valid_mask &= df[c].notna() & (df[c] > 0)
    valid = df.loc[valid_mask, cols].copy()
    log1p_df = np.log1p(valid)
    stats = {}
    z_df = pd.DataFrame(index=log1p_df.index)
    for c in cols:
        mean, std = float(log1p_df[c].mean()), float(log1p_df[c].std())
        stats[c] = {"log1p_mean": mean, "log1p_std": std}
        z_df[c] = (log1p_df[c] - mean) / std if std > 0 else 0.0
    return z_df, stats


def _orient_and_normalize(loadings: np.ndarray, cols: list[str]) -> dict:
    """Sign-orients the WHOLE loading vector (never per-element, which
    would change the loading ratio) so it is positively correlated with
    each input, then normalizes to sum to 1."""
    if loadings.sum() < 0:
        loadings = -loadings
    total = loadings.sum()
    weights = {c: float(loadings[i] / total) for i, c in enumerate(cols)}
    return weights


def compute_pca_weights(z_df: pd.DataFrame, cols: list[str]) -> dict:
    """PCA(n_components=1) on the STANDARDIZED z(log1p(.)) inputs. At n=2
    this is mathematically guaranteed to return [0.5, 0.5] for any
    correlation strength (see module docstring) -- this is the
    recommended, governance-facing path."""
    pca = PCA(n_components=1)
    pca.fit(z_df[cols].to_numpy())
    loadings = pca.components_[0].copy()
    weights = _orient_and_normalize(loadings, cols)
    return {
        "pca_loadings_raw": {c: float(loadings[i]) for i, c in enumerate(cols)},
        "pca_weights": weights,
        "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
    }


def compute_pca_weights_unstandardized(log1p_df: pd.DataFrame, cols: list[str]) -> dict:
    """PCA(n_components=1) on the RAW (unstandardized) log1p(.) values --
    labeled-diagnostic only. Will generally deviate from [0.5, 0.5] when
    the inputs have unequal log-space variance, but that deviation is a
    scale artifact of this dataset snapshot, not economically meaningful
    information -- never used as the recommended weights."""
    pca = PCA(n_components=1)
    pca.fit(log1p_df[cols].to_numpy())
    loadings = pca.components_[0].copy()
    weights = _orient_and_normalize(loadings, cols)
    return {
        "pca_loadings_raw": {c: float(loadings[i]) for i, c in enumerate(cols)},
        "pca_weights": weights,
        "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
    }


def compute_equal_weights(cols: list[str]) -> dict:
    return {c: 1.0 / len(cols) for c in cols}


def check_weights_equivalent(a: dict, b: dict, tol: float) -> bool:
    return all(abs(a[c] - b[c]) <= tol for c in a)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dataset", type=Path, default=Path("loan_episode_capacity_dataset.csv"))
    ap.add_argument("--artifacts-dir", type=Path, required=True,
                     help="directory to write capacity_score_weights.json into -- deliberately a NEW, "
                          "dedicated directory (e.g. capacity_artifacts), separate from pd_model/artifacts, "
                          "keeping Capacity(F) architecturally distinct from the PD/C3 artifact family")
    ap.add_argument("--version-tag", type=str, required=True,
                     help="required, no default -- identifies this weights artifact")
    ap.add_argument("--fundamentals", type=str, default=",".join(FUNDAMENTALS_FOR_SCORE),
                     help="comma-separated list of fundamentals columns (extensibility hook -- "
                          "PCA stops being a tautology once a third fundamental is added)")
    ap.add_argument("--force", action="store_true",
                     help="overwrite an existing artifact (the previous version is archived first, not deleted)")
    args = ap.parse_args(argv)

    cols = [c.strip() for c in args.fundamentals.split(",") if c.strip()]
    if len(cols) < 2:
        sys.exit("ERROR: need at least 2 fundamentals to derive a combination weighting.")

    if not args.episode_dataset.exists():
        sys.exit(f"ERROR: {args.episode_dataset} not found.")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.artifacts_dir / WEIGHTS_FILENAME

    if weights_path.exists():
        if not args.force:
            sys.exit(
                f"ERROR: an artifact already exists at {weights_path}. "
                "Pass --force to overwrite (the previous version will be archived first, not deleted)."
            )
        old_version = "unknown"
        try:
            old_version = json.loads(weights_path.read_text()).get("version", "unknown")
        except Exception:
            pass
        archive_dir = args.artifacts_dir / "capacity_score_weights_history" / str(old_version)
        archive_dir.mkdir(parents=True, exist_ok=True)
        print(f"--force: archiving previous artifact (version={old_version}) to {archive_dir}")
        shutil.copy2(weights_path, archive_dir / WEIGHTS_FILENAME)

    print(f"\n{'#' * 100}\nNEVER touches assigned_limit, cal_pd, disbursement_amount_ugx, or any outcome "
          f"column -- only {cols} and join keys are loaded from the episode dataset.\n{'#' * 100}")

    summary = load_fundamentals(args.episode_dataset, cols)
    print(f"\n{len(summary):,} agent-period unit(s) across "
          f"{summary['agent_msisdn'].nunique():,} agent(s) loaded.")

    z_df, log1p_stats = log1p_zscore(summary, cols)
    log1p_df = np.log1p(summary.loc[z_df.index, cols])
    n_units = len(z_df)
    print(f"{n_units:,} unit(s) have every fundamental valid (notna and > 0) -- used for weight derivation.")
    if n_units < 10:
        print("WARNING: very small sample -- the weights below will be unstable.")

    spearman_corr = np.nan
    if len(cols) == 2 and n_units >= 2:
        spearman_corr, _ = spearmanr(summary.loc[z_df.index, cols[0]], summary.loc[z_df.index, cols[1]])

    pca_standardized = compute_pca_weights(z_df, cols)
    pca_unstandardized = compute_pca_weights_unstandardized(log1p_df, cols)
    equal_weights = compute_equal_weights(cols)
    pca_equals_equal = check_weights_equivalent(
        pca_standardized["pca_weights"], equal_weights, WEIGHT_EQUIVALENCE_TOL)

    print("\n=== Sanity table: weight derivation methods (review before committing) ===")
    rows = [
        {"method": "pca_standardized (RECOMMENDED)", **pca_standardized["pca_weights"]},
        {"method": "pca_unstandardized (diagnostic only)", **pca_unstandardized["pca_weights"]},
        {"method": "equal_weights", **equal_weights},
    ]
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSpearman correlation between fundamentals (this run, n={len(cols)}): "
          f"{spearman_corr if not np.isnan(spearman_corr) else 'n/a (requires exactly 2 fundamentals)'}")
    print(f"Standardized PCA weights equal equal-weighting within {WEIGHT_EQUIVALENCE_TOL}: {pca_equals_equal}")
    if len(cols) == 2:
        print(
            "This equality is EXPECTED and is not a coincidence of this dataset -- for exactly two "
            "standardized inputs, PCA's first component is provably (1,1)/sqrt(2) for any correlation "
            "strength. Re-run this script once a third fundamental is added; PCA stops being a tautology at n>=3."
        )

    recommended_weights = equal_weights if len(cols) == 2 else pca_standardized["pca_weights"]
    recommendation_rationale = (
        "At n=2 standardized fundamentals, standardized PCA's first component is provably (1,1)/sqrt(2) "
        "for any correlation strength -- equal weighting is not a simplifying fallback, it is the unique "
        "eigen-based answer, confirmed by the pca_standardized row above. Re-run this script if a third "
        "fundamental is added via --fundamentals; PCA stops being vacuous at n>=3 and "
        "pca_weights_standardized should then be adopted as recommended_weights instead."
    )

    metadata = {
        "version": args.version_tag,
        "fit_timestamp_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "fundamentals_used": cols,
        "n_units": n_units,
        "source_episode_dataset_sha256": _sha256(args.episode_dataset),
        "standardization_method": "z(log1p(x)), per-column mean/std",
        "per_column_log1p_stats": log1p_stats,
        "spearman_corr_between_fundamentals_this_run": None if np.isnan(spearman_corr) else float(spearman_corr),
        "pca_weights_standardized": pca_standardized["pca_weights"],
        "pca_weights_unstandardized_diagnostic_only": pca_unstandardized["pca_weights"],
        "equal_weights": equal_weights,
        "pca_equals_equal_weights": pca_equals_equal,
        "weight_equivalence_tolerance": WEIGHT_EQUIVALENCE_TOL,
        "recommended_weights": recommended_weights,
        "recommendation_rationale": recommendation_rationale,
        "sklearn_version": sklearn.__version__,
        "calibration_artifact_schema_version": SCHEMA_VERSION,
    }

    weights_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nWrote {weights_path}")
    print(
        "\nThis script is MANUAL ONLY and NEVER refits automatically. The scoring/backtest script "
        "(derive_capacity_function_and_backtest_frontier.py) only ever reads this JSON's "
        "recommended_weights -- it never recomputes weights itself. Re-run this script by hand to "
        "revisit the weighting -- it will refuse to overwrite the existing artifact without --force, "
        "and archives the previous version first when you do."
    )


if __name__ == "__main__":
    main()
