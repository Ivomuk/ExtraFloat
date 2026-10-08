"""
fit_capacity_challenger_model.py
====================================
Analysis 3: the first continuous capacity-only challenger, per this
session's confirmed v1 specification. This is NOT yet a capacity
formula to deploy -- it estimates a quantity this session's design
discussion deliberately named:

    L_demonstrated_capacity_pXX  (NOT "safe capacity", NOT "maximum capacity")

because current policy already censors observed exposure: even the 90th
percentile of realized exposure among low-risk agents only tells us the
upper end of what HAS been serviced under the existing engine, not a
latent true ceiling.

TARGET AND POPULATION (confirmed, do not change without re-confirming):
  Population for FITTING: cal_pd < --low-risk-cal-pd (default 0.15, the
  two PD bands this session's robustness check found cleanest) AND
  actual_exposure_ugx > 0. Deliberately NO outcome-based filtering --
  agents who later deteriorated are KEPT, not deleted. Dropping them
  would mean the model never sees exposure levels that proved excessive,
  biasing the estimated upper quantile upward (the exact failure mode
  this session flagged: "Agent B borrowed 750K and went bad" must stay
  in the training data, not be filtered out because it is bad news).
  Target: log(actual_exposure_ugx) -- a genuine quantile regression
  target, not a classification label.

FEATURES (confirmed v1 set -- exactly four, all monotonic increasing):
  log1p(float_activity_value_1m), log1p(commission), log1p(cust_1m),
  log1p(average_balance). Business-persistence is DELIBERATELY EXCLUDED
  from v1: its underlying recency columns have a known, still-unfixed
  bug (median_days_since_any_activity = -20,666 in every cell of every
  table from an earlier analysis in this session) -- an unconstrained
  tree model is particularly capable of exploiting a broken feature in
  ways that are hard to detect after the fact, so it is excluded
  entirely rather than included "unconstrained but uncertain."

build_feature_matrix() is an explicit WHITELIST of exactly these four
raw columns -- combined_cap, assigned_limit, cal_pd, risk_tier,
pd_decile, every capacity_* engine component, and every outcome column
are never read by it, so they cannot accidentally enter training no
matter what else the source dataframe carries.

THREE QUANTILE MODELS (P85/P90/P95), not one: P90 is the primary
shadow candidate, P85 the conservative read, P95 the aggressive read.
Separately-fit quantile models CAN cross (P85 > P90 at some rows) --
this script reports the crossing frequency as a model-health
diagnostic and does NOT silently sort/clip predictions to force
agreement, per this session's explicit instruction.

monotonic_cst IS AN EXACT GUARANTEE for HistGradientBoostingRegressor
with loss="squared_error", but confirmed directly (not assumed) to NOT
be exact for loss="quantile" -- small violations can occur. Per this
session's re-framing: these are directional constraints with
empirically verified near-monotonic behavior for this estimator, not a
hard mathematical guarantee for the quantile fits. monotonicity_diagnostic()
sweeps each feature at THREE representative profiles of the other three
(their marginal p10/p50/p90, not only the median) and reports, in UGX:
pct of steps that go backward, the single worst downward step, that
step as a % of the sweep's UGX range, and the total negative variation
-- so a rare but economically large reversal cannot hide behind a low
average violation rate. min_samples_leaf trades off this smoothness
against model flexibility, so it is chosen via the hyperparameter
comparison (pinball loss + capacity-distribution stability +
monotonicity across candidates), not by minimizing violations alone.

VALIDATION SPLIT: out-of-time if --oot-research-dataset (an earlier
snapshot, built the same way via build_capacity_research_dataset.py) is
given; otherwise a plain random split within --research-dataset, which
this script explicitly labels as NOT out-of-time rather than silently
treating it as equivalent -- a random split can look excellent while
masking temporal instability in a relationship that needs to survive
changes in transaction activity and borrowing behavior over time.

VALIDATION LAYERS, run AFTER fitting (this script does not treat the
model as validated just because it fits):
  0. Out-of-sample pinball loss (log-space) and quantile calibration
     (what fraction of held-out actual exposures fall at/below their
     predicted quantile -- should be near 85/90/95%, with deviation
     itself informative given censoring and generalization).
  1. CapacityUtilization = actual_exposure_ugx / demonstrated_capacity_p90,
     banded, with borrower-only 3+DPD / bad-closure / shortfall / N / PD
     mix reported per band -- the falsification test: if performance
     does not deteriorate above ~1.0x utilization, P90 historical
     exposure is not identifying an economically meaningful boundary.
  2. Capacity-gap check (run for BOTH "All agents" and "Diamond A"):
     CapacityGap = demonstrated_capacity_p90 - combined_cap,
     CapacityUpliftRatio = demonstrated_capacity_p90 / combined_cap,
     banded, reporting whether agents with LARGE predicted gaps
     (particularly >1.5x/>2.0x, where the challenger makes its
     strongest claim) actually show the fundamentals and historical
     performance consistent with the gap -- not just a large number.

A printed "V1 ACCEPTANCE CRITERIA" section lists what to check across
all of the above -- deliberately NOT a single hard-coded cutoff (e.g.
not "reject if >2% violations"); the diagnostics are meant to show the
empirical scale before any cutoff is encoded.

Usage:
    python scripts\\fit_capacity_challenger_model.py ^
        --research-dataset capacity_research_dataset.csv
"""

import argparse
import hashlib
import json
import sys
import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

SCHEMA_VERSION = 1

# Explicit whitelist -- see module docstring. Nothing outside this list is
# ever read by build_feature_matrix().
FEATURE_COLUMNS = ["float_activity_value_1m", "commission", "cust_1m", "average_balance"]
MONOTONIC_CST = [1, 1, 1, 1]  # all four: increasing

DEFAULT_QUANTILES = [0.85, 0.90, 0.95]
DEFAULT_PRIMARY_QUANTILE = 0.90
DEFAULT_LOW_RISK_CAL_PD = 0.15
DEFAULT_DIAMOND_HIGH_RISK_CAL_PD = 0.30
DEFAULT_RANDOM_STATE = 42
DEFAULT_MIN_SAMPLES_LEAF = 100  # the FINAL/chosen value -- see the printed hyperparameter comparison
                                 # table (pinball loss + capacity-distribution stability + monotonicity
                                 # across candidates) for why. monotonic_cst is an exact guarantee for
                                 # HistGradientBoostingRegressor with loss="squared_error", but NOT for
                                 # loss="quantile" (confirmed directly) -- min_samples_leaf trades off
                                 # monotonicity smoothness against model flexibility, so it must not be
                                 # picked on monotonicity alone.
DEFAULT_MIN_SAMPLES_LEAF_CANDIDATES = [50, 100, 200]
DEFAULT_VALIDATION_FRACTION = 0.2
MONOTONICITY_PROFILE_PERCENTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}

UTILIZATION_EDGES = [0.0, 0.50, 0.75, 1.00, 1.25, np.inf]
UTILIZATION_LABELS = ["<=0.50", "0.50-0.75", "0.75-1.00", "1.00-1.25", ">1.25"]

UPLIFT_EDGES = [-np.inf, 1.0, 1.5, 2.0, np.inf]
UPLIFT_LABELS = ["<=1.0x", "1.0-1.5x", "1.5-2.0x", ">2.0x"]


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _capacity_col(q: float) -> str:
    return f"demonstrated_capacity_p{int(round(q * 100))}"


def build_feature_matrix(df: pd.DataFrame) -> tuple:
    """Explicit whitelist of FEATURE_COLUMNS only -- see module docstring.
    Returns (X, valid_mask). A row's feature values are only used if ALL
    four are present and >= 0 (zero is a legitimate value and survives
    log1p correctly; a negative or missing value makes the WHOLE row
    invalid -- never silently zero-filled). X holds NaN for every column
    of an invalid row, never a substituted value."""
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        sys.exit(f"ERROR: research dataset missing required feature column(s): {missing_cols}")

    valid = pd.Series(True, index=df.index)
    numeric = {}
    for c in FEATURE_COLUMNS:
        col = pd.to_numeric(df[c], errors="coerce")
        numeric[c] = col
        valid &= col.notna() & (col >= 0)

    X = pd.DataFrame(index=df.index)
    for c in FEATURE_COLUMNS:
        X[f"log1p_{c}"] = np.log1p(numeric[c].clip(lower=0))
    X.loc[~valid, :] = np.nan
    return X, valid


def training_mask(df: pd.DataFrame, low_risk_cal_pd: float) -> pd.Series:
    """Population for FITTING: low risk AND currently carrying exposure.
    Deliberately NOT filtered on outcome -- see module docstring."""
    return (
        df["cal_pd"].notna() & (df["cal_pd"] < low_risk_cal_pd)
        & df["actual_exposure_ugx"].notna() & (df["actual_exposure_ugx"] > 0)
    )


def fit_quantile_models(X: pd.DataFrame, y: pd.Series, quantiles: list,
                         monotonic_cst: list, random_state: int,
                         min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF) -> dict:
    models = {}
    X_arr = X.to_numpy()
    y_arr = y.to_numpy()
    for q in quantiles:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, monotonic_cst=monotonic_cst,
            random_state=random_state, max_iter=200, min_samples_leaf=min_samples_leaf,
        )
        m.fit(X_arr, y_arr)
        models[q] = m
    return models


def feature_percentiles(df: pd.DataFrame, mask: pd.Series) -> dict:
    """Per-feature {p10, p50, p90} marginal values over the given population
    -- the 'several representative profiles' used by monotonicity_diagnostic
    to hold the other three features at, instead of only their median."""
    out = {}
    for c in FEATURE_COLUMNS:
        s = df.loc[mask, c]
        out[c] = {name: float(s.quantile(p)) for name, p in MONOTONICITY_PROFILE_PERCENTILES.items()}
    return out


def make_random_split(train_mask: pd.Series, validation_fraction: float, random_state: int) -> tuple:
    """Random split WITHIN the training population. NOT an out-of-time
    split -- see --oot-research-dataset for that. Returns (fit_mask, val_mask)."""
    idx = train_mask[train_mask].index.to_numpy()
    rng = np.random.RandomState(random_state)
    shuffled = rng.permutation(idx)
    n_val = max(1, int(len(shuffled) * validation_fraction))
    val_idx = shuffled[:n_val]
    val_mask = pd.Series(False, index=train_mask.index)
    val_mask.loc[val_idx] = True
    fit_mask = train_mask & ~val_mask
    return fit_mask, val_mask


def pinball_loss_by_quantile(models: dict, X_val: pd.DataFrame, y_val_log: pd.Series) -> dict:
    """Pinball (quantile) loss in LOG space -- the same units/scale the
    models were actually fit to minimize, so hyperparameter choices are
    compared on an apples-to-apples basis."""
    X_arr = X_val.to_numpy()
    y_arr = y_val_log.to_numpy()
    out = {}
    for q, m in models.items():
        pred = m.predict(X_arr)
        out[q] = round(float(mean_pinball_loss(y_arr, pred, alpha=q)), 5)
    return out


def quantile_calibration(models: dict, X_val: pd.DataFrame, y_val_log: pd.Series) -> pd.DataFrame:
    """Out-of-sample calibration: what fraction of actual (held-out, log)
    exposures fall at or below their predicted quantile? Should be roughly
    85/90/95% -- not necessarily exact, because of censoring (current
    policy bounds observed exposure) and ordinary generalization error,
    but the deviation itself is informative, per this session's
    instruction. Reported, not forced to match."""
    X_arr = X_val.to_numpy()
    y_arr = y_val_log.to_numpy()
    rows = []
    for q, m in models.items():
        pred = m.predict(X_arr)
        empirical_coverage = float((y_arr <= pred).mean()) * 100
        rows.append({"quantile": q, "nominal_coverage_pct": q * 100,
                     "empirical_coverage_pct": round(empirical_coverage, 2),
                     "deviation_pp": round(empirical_coverage - q * 100, 2),
                     "n_validation": len(y_arr)})
    return pd.DataFrame(rows)


def monotonicity_diagnostic(models: dict, feat_pctiles: dict, feature_ranges: dict, n_grid: int = 60) -> pd.DataFrame:
    """For each feature, sweep it across its observed training range while
    holding the OTHER THREE jointly at each of several representative
    profiles (their marginal p10/p50/p90), and report, IN UGX (not log)
    space, the step-to-step behavior of predicted demonstrated capacity:
    pct of adjacent steps that go backward, the single worst downward
    step in UGX, that worst step as a % of the sweep's total UGX range,
    and the total negative variation (sum of every backward step's size)
    -- so a rare but economically large reversal cannot hide behind a
    small average violation rate. monotonic_cst guarantees this is
    exactly >= 0 for loss="squared_error" but NOT for loss="quantile"
    (confirmed directly) -- reported as a model-health diagnostic, never
    silently hidden or corrected by re-sorting."""
    rows = []
    for feat in FEATURE_COLUMNS:
        lo, hi = feature_ranges[feat]
        other_feats = [c for c in FEATURE_COLUMNS if c != feat]
        for profile_name in MONOTONICITY_PROFILE_PERCENTILES:
            grid = pd.DataFrame({c: np.full(n_grid, feat_pctiles[c][profile_name]) for c in other_feats})
            grid[feat] = np.linspace(lo, hi, n_grid)
            X_grid, valid_grid = build_feature_matrix(grid)
            X_arr = X_grid.to_numpy()
            for q, m in models.items():
                preds_ugx = np.exp(m.predict(X_arr))
                diffs = np.diff(preds_ugx)
                total_range = preds_ugx.max() - preds_ugx.min()
                neg = diffs[diffs < 0]
                n_steps = len(diffs)
                rows.append({
                    "feature": feat, "profile": profile_name, "quantile": q,
                    "n_steps": n_steps, "n_violations": int(len(neg)),
                    "pct_violations": round(len(neg) / n_steps * 100, 2) if n_steps else 0.0,
                    "max_downward_violation_ugx": round(float(neg.min()), 2) if len(neg) else 0.0,
                    "max_violation_pct_of_range": round(float(-neg.min()) / total_range * 100, 2) if total_range > 0 and len(neg) else 0.0,
                    "total_negative_variation_ugx": round(float(-neg.sum()), 2) if len(neg) else 0.0,
                })
    return pd.DataFrame(rows)


def score(X: pd.DataFrame, valid_mask: pd.Series, models: dict) -> pd.DataFrame:
    """Predict log(exposure) for every row with valid features, exponentiate
    back to UGX. Invalid rows get NaN (never a model prediction on
    fabricated/zero-filled inputs)."""
    out = pd.DataFrame(index=X.index)
    X_filled = X.fillna(0.0).to_numpy()  # placeholder only -- overwritten by NaN below for invalid rows
    valid_arr = valid_mask.to_numpy()
    for q, m in models.items():
        pred_log = m.predict(X_filled)
        out[_capacity_col(q)] = np.where(valid_arr, np.exp(pred_log), np.nan)
    return out


def crossing_diagnostic(scored: pd.DataFrame, quantiles: list) -> dict:
    """Separately-fit quantile models can cross -- report frequency, never
    silently sort/clip to force P85<=P90<=P95."""
    cols = [_capacity_col(q) for q in sorted(quantiles)]
    valid = scored[cols].notna().all(axis=1)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {"n_valid": 0, "n_crossing": 0, "pct_crossing": float("nan")}
    sub = scored.loc[valid, cols]
    ok = pd.Series(True, index=sub.index)
    for a, b in zip(cols[:-1], cols[1:]):
        ok &= sub[a] <= sub[b]
    n_crossing = int((~ok).sum())
    return {"n_valid": n_valid, "n_crossing": n_crossing,
            "pct_crossing": round(n_crossing / n_valid * 100, 2)}


def add_capacity_gap(df: pd.DataFrame, scored: pd.DataFrame, primary_col: str) -> pd.DataFrame:
    out = scored.copy()
    has_cap = df["combined_cap"].notna() & (df["combined_cap"] > 0) if "combined_cap" in df.columns else pd.Series(False, index=df.index)
    eligible = has_cap & out[primary_col].notna()
    out["capacity_gap"] = np.where(eligible, out[primary_col] - df["combined_cap"], np.nan)
    out["capacity_uplift_ratio"] = np.where(eligible, out[primary_col] / df["combined_cap"], np.nan)
    return out


def _performance_block(cell: pd.DataFrame) -> dict:
    row = {}
    if "fwd_new_loan_count" in cell.columns:
        borrowers = cell[cell["fwd_new_loan_count"] > 0]
        row["n_borrowers_fwd"] = len(borrowers)
        if "fwd_any_bad_3dpd" in cell.columns:
            row["fwd_any_bad_3dpd_rate_among_borrowers_pct"] = (
                round(borrowers["fwd_any_bad_3dpd"].mean() * 100, 1) if len(borrowers) else float("nan")
            )
    if {"fwd_new_loans_closed_good_count", "fwd_new_loans_closed_bad_count"} <= set(cell.columns):
        n_good = int(cell["fwd_new_loans_closed_good_count"].sum())
        n_bad = int(cell["fwd_new_loans_closed_bad_count"].sum())
        n_closed = n_good + n_bad
        row["n_closed_new_loans"] = n_closed
        row["bad_closure_rate_pct"] = round(n_bad / n_closed * 100, 2) if n_closed else float("nan")
    if {"fwd_new_loans_disbursed_ugx", "fwd_new_loans_repaid_ugx"} <= set(cell.columns):
        disbursed = cell["fwd_new_loans_disbursed_ugx"].sum()
        repaid = cell["fwd_new_loans_repaid_ugx"].sum()
        row["shortfall_pct_of_forward_disbursed"] = round((disbursed - repaid) / disbursed * 100, 2) if disbursed else float("nan")
    return row


def diamond_a_mask(df: pd.DataFrame, diamond_high_risk_cal_pd: float) -> pd.Series:
    diamond = (
        df["agent_category"].astype(str).str.strip().str.lower() == "diamond"
        if "agent_category" in df.columns else pd.Series(False, index=df.index)
    )
    return diamond & (df["cal_pd"] < diamond_high_risk_cal_pd)


def utilization_validation(df: pd.DataFrame, scored: pd.DataFrame, primary_col: str) -> pd.DataFrame:
    """Validation layer 1: does performance deteriorate above ~1.0x
    utilization? Run across the WHOLE scored population (not just the
    low-risk training population) -- a stronger test of whether the
    boundary generalizes out of its training PD range."""
    working = df.join(scored[[primary_col]], how="left")
    mask = working["actual_exposure_ugx"].notna() & (working["actual_exposure_ugx"] > 0) & working[primary_col].notna()
    working = working[mask].copy()
    if working.empty:
        return pd.DataFrame()
    working["_utilization"] = working["actual_exposure_ugx"] / working[primary_col]
    working["_band"] = pd.cut(working["_utilization"], bins=UTILIZATION_EDGES, labels=UTILIZATION_LABELS)

    rows = []
    for band in UTILIZATION_LABELS:
        cell = working[working["_band"] == band]
        row = {"utilization_band": band, "n_agents": len(cell)}
        if len(cell):
            if "cal_pd" in cell.columns:
                row["median_cal_pd"] = round(cell["cal_pd"].median(), 4)
                row["pct_cal_pd_lt_15pct"] = round((cell["cal_pd"] < DEFAULT_LOW_RISK_CAL_PD).mean() * 100, 1)
            row.update(_performance_block(cell))
        rows.append(row)
    return pd.DataFrame(rows)


def capacity_gap_check(df: pd.DataFrame, scored: pd.DataFrame, primary_col: str,
                        population_mask: pd.Series) -> pd.DataFrame:
    """Validation layer 2 (population-agnostic): do agents with LARGE
    predicted capacity gaps actually show the fundamentals and performance
    consistent with having supported exposure above their current
    combined_cap -- not just a large number from the model? Called once
    for 'All agents' and once for 'Diamond A' specifically: per this
    session's point #6, the agents where the challenger makes its
    strongest claim (largest uplift ratio) deserve direct interrogation
    regardless of which population they fall in."""
    working = df[population_mask].join(
        scored[[primary_col, "capacity_gap", "capacity_uplift_ratio"]], how="left"
    )
    working = working[working["capacity_uplift_ratio"].notna()].copy()
    if working.empty:
        return pd.DataFrame()
    working["_band"] = pd.cut(working["capacity_uplift_ratio"], bins=UPLIFT_EDGES, labels=UPLIFT_LABELS)

    rows = []
    for band in UPLIFT_LABELS:
        cell = working[working["_band"] == band]
        row = {"uplift_band": band, "n_agents": len(cell)}
        if len(cell):
            for c in FEATURE_COLUMNS:
                if c in cell.columns:
                    row[f"median_{c}"] = cell[c].median()
            row["median_actual_exposure_ugx"] = cell["actual_exposure_ugx"].median() if "actual_exposure_ugx" in cell.columns else np.nan
            row["median_combined_cap"] = cell["combined_cap"].median() if "combined_cap" in cell.columns else np.nan
            row[f"median_{primary_col}"] = cell[primary_col].median()
            row.update(_performance_block(cell))
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--research-dataset", default="capacity_research_dataset.csv")
    ap.add_argument("--oot-research-dataset", default=None,
                     help="optional: a capacity_research_dataset.csv built from an EARLIER snapshot, used "
                          "ONLY for out-of-time validation (pinball loss + calibration), never for fitting "
                          "the final model. Without this, the script falls back to a random split within "
                          "--research-dataset and says so explicitly -- a random split is NOT out-of-time.")
    ap.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    ap.add_argument("--low-risk-cal-pd", type=float, default=DEFAULT_LOW_RISK_CAL_PD)
    ap.add_argument("--diamond-high-risk-cal-pd", type=float, default=DEFAULT_DIAMOND_HIGH_RISK_CAL_PD)
    ap.add_argument("--quantiles", default=",".join(str(q) for q in DEFAULT_QUANTILES))
    ap.add_argument("--primary-quantile", type=float, default=DEFAULT_PRIMARY_QUANTILE)
    ap.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    ap.add_argument("--min-samples-leaf", type=int, default=DEFAULT_MIN_SAMPLES_LEAF,
                     help="the FINAL/chosen value used for the production model.")
    ap.add_argument("--min-samples-leaf-candidates", default=",".join(str(v) for v in DEFAULT_MIN_SAMPLES_LEAF_CANDIDATES),
                     help="comma-separated values compared via pinball loss, capacity-distribution "
                          "stability, and monotonicity BEFORE committing to --min-samples-leaf.")
    ap.add_argument("--model-out-dir", default="capacity_challenger_artifacts")
    ap.add_argument("--out-prefix", default="capacity_challenger")
    ap.add_argument("--no-save-model", action="store_true")
    args = ap.parse_args(argv)
    quantiles = [float(q) for q in args.quantiles.split(",")]
    primary_col = _capacity_col(args.primary_quantile)
    leaf_candidates = sorted(set(int(v) for v in args.min_samples_leaf_candidates.split(",")) | {args.min_samples_leaf})

    path = Path(args.research_dataset)
    if not path.exists():
        sys.exit(f"ERROR: {path} not found -- run build_capacity_research_dataset.py first.")
    df = pd.read_csv(path, low_memory=False)

    required_other = ["cal_pd", "actual_exposure_ugx"]
    missing_other = [c for c in required_other if c not in df.columns]
    if missing_other:
        sys.exit(f"ERROR: {path} is missing required column(s): {missing_other}")

    X, feature_valid = build_feature_matrix(df)
    train_pop = training_mask(df, args.low_risk_cal_pd)
    train_mask = train_pop & feature_valid
    n_dropped_invalid_features = int((train_pop & ~feature_valid).sum())
    if n_dropped_invalid_features:
        print(f"NOTE: {n_dropped_invalid_features} agent(s) in the low-risk/exposed training population "
              f"dropped for missing/negative feature value(s).")

    n_train = int(train_mask.sum())
    print(f"Training population (cal_pd < {args.low_risk_cal_pd}, actual_exposure_ugx > 0, valid features): "
          f"{n_train:,} agent(s). NO outcome-based filtering applied.")
    if n_train < 50:
        sys.exit(f"ERROR: only {n_train} training rows -- too few to fit. Check --research-dataset / --low-risk-cal-pd.")

    # -- Validation split: out-of-time if an earlier snapshot is provided, otherwise a DISCLOSED random split. --
    if args.oot_research_dataset:
        oot_path = Path(args.oot_research_dataset)
        if not oot_path.exists():
            sys.exit(f"ERROR: --oot-research-dataset {oot_path} not found.")
        oot_df = pd.read_csv(oot_path, low_memory=False)
        oot_missing = [c for c in required_other if c not in oot_df.columns]
        if oot_missing:
            sys.exit(f"ERROR: {oot_path} is missing required column(s): {oot_missing}")
        oot_X, oot_feature_valid = build_feature_matrix(oot_df)
        oot_train_pop = training_mask(oot_df, args.low_risk_cal_pd)
        val_mask = oot_train_pop & oot_feature_valid
        val_df, val_X = oot_df, oot_X
        fit_mask = train_mask
        split_description = f"out-of-time (--oot-research-dataset {oot_path}, {int(val_mask.sum()):,} validation agents)"
    else:
        fit_mask, val_mask = make_random_split(train_mask, args.validation_fraction, args.random_state)
        val_df, val_X = df, X
        split_description = (f"RANDOM split within --research-dataset (validation_fraction="
                              f"{args.validation_fraction}) -- this is NOT out-of-time. Pass "
                              f"--oot-research-dataset (built from an earlier snapshot via "
                              f"build_capacity_research_dataset.py) for a genuine temporal holdout.")
    print(f"Validation split method: {split_description}")
    print(f"  Fit rows: {int(fit_mask.sum()):,}  |  Validation rows: {int(val_mask.sum()):,}")

    y_fit = np.log(df.loc[fit_mask, "actual_exposure_ugx"])
    X_fit = X.loc[fit_mask]
    y_val = np.log(val_df.loc[val_mask, "actual_exposure_ugx"])
    X_val = val_X.loc[val_mask]

    # -- Hyperparameter comparison: pinball loss + capacity-distribution stability + monotonicity, --
    # -- across candidates. min_samples_leaf is NOT auto-selected from this -- a human reviews it. --
    print(f"\n{'=' * 100}\nHyperparameter comparison: min_samples_leaf in {leaf_candidates}\n{'=' * 100}")
    feat_pctiles = feature_percentiles(df, fit_mask)
    feature_ranges = {c: (float(df.loc[fit_mask, c].min()), float(df.loc[fit_mask, c].max())) for c in FEATURE_COLUMNS}
    comparison_rows = []
    diagnostic_models_by_leaf = {}
    for leaf in leaf_candidates:
        diag_models = fit_quantile_models(X_fit, y_fit, quantiles, MONOTONIC_CST, args.random_state, leaf)
        diagnostic_models_by_leaf[leaf] = diag_models
        pinball = pinball_loss_by_quantile(diag_models, X_val, y_val)
        scored_full_tmp = score(X, feature_valid, diag_models)
        mono_tmp = monotonicity_diagnostic(diag_models, feat_pctiles, feature_ranges, n_grid=40)
        cross_tmp = crossing_diagnostic(scored_full_tmp, quantiles)
        row = {"min_samples_leaf": leaf}
        for q in quantiles:
            row[f"pinball_loss_p{int(round(q*100))}"] = pinball[q]
            row[f"median_demonstrated_capacity_p{int(round(q*100))}"] = round(float(np.nanmedian(scored_full_tmp[_capacity_col(q)])), 0)
            iqr = np.nanpercentile(scored_full_tmp[_capacity_col(q)].dropna(), 75) - np.nanpercentile(scored_full_tmp[_capacity_col(q)].dropna(), 25)
            row[f"iqr_demonstrated_capacity_p{int(round(q*100))}"] = round(float(iqr), 0)
        row["pct_quantile_crossing"] = cross_tmp["pct_crossing"]
        row["worst_monotonicity_violation_pct_of_range"] = mono_tmp["max_violation_pct_of_range"].max()
        comparison_rows.append(row)
    comparison_df = pd.DataFrame(comparison_rows)
    with pd.option_context("display.float_format", "{:,.4f}".format, "display.max_columns", None, "display.width", 240):
        print(comparison_df.to_string(index=False))
    comparison_df.to_csv(f"{args.out_prefix}_hyperparameter_comparison.csv", index=False)
    print(f"-> hyperparameter comparison written: {args.out_prefix}_hyperparameter_comparison.csv")
    print(f"Proceeding with --min-samples-leaf={args.min_samples_leaf} (chosen by the operator, "
          f"not auto-selected from the table above).")

    # -- Out-of-sample pinball loss + calibration for the CHOSEN min_samples_leaf specifically. --
    chosen_diag_models = diagnostic_models_by_leaf[args.min_samples_leaf]
    chosen_pinball = pinball_loss_by_quantile(chosen_diag_models, X_val, y_val)
    calibration = quantile_calibration(chosen_diag_models, X_val, y_val)
    print(f"\n{'=' * 100}\nOut-of-sample pinball loss and quantile calibration (min_samples_leaf={args.min_samples_leaf})\n{'=' * 100}")
    print(f"Pinball loss (log-space) by quantile: {chosen_pinball}")
    with pd.option_context("display.float_format", "{:,.2f}".format):
        print(calibration.to_string(index=False))
    calibration.to_csv(f"{args.out_prefix}_quantile_calibration.csv", index=False)

    # -- Production fit: full training population, chosen hyperparameter. This is the model that --
    # -- gets scored, validated, and (optionally) persisted. --
    print(f"\nFitting PRODUCTION quantile models {quantiles} (primary={args.primary_quantile}), "
          f"monotonic_cst={MONOTONIC_CST} on features {FEATURE_COLUMNS} on the FULL training population "
          f"(sklearn {sklearn.__version__}, random_state={args.random_state}, "
          f"min_samples_leaf={args.min_samples_leaf})...")
    y_train = np.log(df.loc[train_mask, "actual_exposure_ugx"])
    X_train = X.loc[train_mask]
    models = fit_quantile_models(X_train, y_train, quantiles, MONOTONIC_CST, args.random_state, args.min_samples_leaf)

    scored = score(X, feature_valid, models)
    scored = add_capacity_gap(df, scored, primary_col)

    diag = crossing_diagnostic(scored, quantiles)
    print(f"\nCrossing diagnostic (quantile models fit independently -- NOT forced monotonic across q): "
          f"{diag['n_crossing']:,} of {diag['n_valid']:,} valid rows ({diag['pct_crossing']}%) have "
          f"P85/P90/P95 out of order. Not corrected -- reported as a model-health signal.")

    mono_diag = monotonicity_diagnostic(models, feat_pctiles, feature_ranges)
    worst = mono_diag["max_violation_pct_of_range"].max()
    worst_row = mono_diag.loc[mono_diag["max_violation_pct_of_range"].idxmax()]
    print(f"\nMonotonicity diagnostic (sweeping each feature at p10/p50/p90 profiles of the other three; "
          f"monotonic_cst is an EXACT guarantee for loss='squared_error' but NOT for loss='quantile' -- "
          f"confirmed directly, not assumed): worst observed violation is {worst:.2f}% of that sweep's "
          f"UGX output range (feature={worst_row['feature']}, profile={worst_row['profile']}, "
          f"quantile={worst_row['quantile']}, max_downward_violation_ugx={worst_row['max_downward_violation_ugx']:,.0f}). "
          f"Reported, not corrected -- see {args.out_prefix}_monotonicity_diagnostic.csv for the full breakdown.")
    mono_diag.to_csv(f"{args.out_prefix}_monotonicity_diagnostic.csv", index=False)

    medians = {q: float(np.nanmedian(scored[_capacity_col(q)])) for q in quantiles}
    print(f"Median demonstrated capacity by quantile (whole scored population): "
          f"{ {f'p{int(round(q*100))}': round(v, 0) for q, v in medians.items()} }")

    scored_out = f"{args.out_prefix}_scored.csv"
    pd.concat([df[["msisdn"]] if "msisdn" in df.columns else pd.DataFrame(index=df.index), scored], axis=1).to_csv(scored_out, index=False)
    print(f"-> scored dataset written: {scored_out}")

    print(f"\n{'=' * 100}\nValidation 1: CapacityUtilization = actual_exposure_ugx / {primary_col}\n{'=' * 100}")
    util = utilization_validation(df, scored, primary_col)
    with pd.option_context("display.float_format", "{:,.2f}".format, "display.max_columns", None, "display.width", 240):
        print(util.to_string(index=False))
    util.to_csv(f"{args.out_prefix}_utilization_validation.csv", index=False)

    diamond_mask = diamond_a_mask(df, args.diamond_high_risk_cal_pd)
    for pop_label, pop_mask, out_suffix in [
        ("All agents", pd.Series(True, index=df.index), "all"),
        (f"Diamond A (cal_pd < {args.diamond_high_risk_cal_pd * 100:.0f}%)", diamond_mask, "diamond_a"),
    ]:
        print(f"\n{'=' * 100}\nValidation 2: capacity-gap check -- {pop_label}\n{'=' * 100}")
        gap_check = capacity_gap_check(df, scored, primary_col, pop_mask)
        with pd.option_context("display.float_format", "{:,.2f}".format, "display.max_columns", None, "display.width", 240):
            print(gap_check.to_string(index=False))
        gap_check.to_csv(f"{args.out_prefix}_gap_check_{out_suffix}.csv", index=False)

    if not args.no_save_model:
        out_dir = Path(args.model_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for q, m in models.items():
            joblib.dump(m, out_dir / f"capacity_challenger_p{int(round(q * 100))}.joblib")
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "fit_timestamp_utc": _dt.datetime.utcnow().isoformat() + "Z",
            "n_train": n_train,
            "source_research_dataset_file": str(path),
            "source_research_dataset_sha256": _sha256(path),
            "sklearn_version": sklearn.__version__,
            "feature_columns": FEATURE_COLUMNS,
            "monotonic_cst": MONOTONIC_CST,
            "target_definition": "log(actual_exposure_ugx), no outcome-based filtering",
            "population_definition": f"cal_pd < {args.low_risk_cal_pd} AND actual_exposure_ugx > 0",
            "quantiles": quantiles,
            "primary_quantile": args.primary_quantile,
            "random_state": args.random_state,
            "min_samples_leaf": args.min_samples_leaf,
            "min_samples_leaf_candidates_compared": leaf_candidates,
            "validation_split_method": split_description,
            "out_of_sample_pinball_loss_log_space": chosen_pinball,
            "out_of_sample_calibration_pct": calibration.set_index("quantile")["empirical_coverage_pct"].to_dict(),
            "monotonicity_note": "monotonic_cst settings are DIRECTIONAL CONSTRAINTS with empirically "
                                  "verified near-monotonic behavior for this estimator/loss combination, "
                                  "NOT an exact mathematical guarantee for the quantile fits (confirmed by "
                                  "direct testing against loss='squared_error', where it IS exact) -- see "
                                  f"{args.out_prefix}_monotonicity_diagnostic.csv from this fit for the "
                                  "observed magnitude at multiple feature profiles.",
            "output_naming": "demonstrated_capacity_pXX -- the upper end of historically demonstrated "
                              "exposure under the EXISTING (censoring) engine, not a latent maximum capacity.",
            "excluded_features_note": "business persistence excluded from v1 pending the recency-column bug fix "
                                       "flagged earlier in this session (median_days_since_any_activity = -20,666).",
        }
        (out_dir / "capacity_challenger_metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"\nWrote model artifact(s) + metadata to {out_dir}/")

    print(f"\n{'#' * 100}")
    print("V1 ACCEPTANCE CRITERIA (for manual review -- NOT auto-enforced, no hard cutoff coded)")
    print(f"{'#' * 100}")
    print("Accept this quantile model for shadow analysis only if, taken together:\n"
          "  1. Monotonic violations are infrequent AND economically immaterial -- see the monotonicity\n"
          "     diagnostic above (pct_violations, max_downward_violation_ugx, max_violation_pct_of_range,\n"
          "     total_negative_variation_ugx, across p10/p50/p90 profiles -- a rare but large reversal\n"
          "     should not hide behind a low average rate).\n"
          "  2. Quantile crossing (P85/P90/P95 out of order) is limited -- see the crossing diagnostic.\n"
          "  3. Out-of-sample pinball loss is stable across the min_samples_leaf candidates compared above\n"
          "     (a candidate that wins on monotonicity alone but loses materially on pinball loss is not\n"
          "     a free improvement).\n"
          "  4. Predicted capacity distributions (median/IQR per quantile) are stable across those same\n"
          "     candidates -- the chosen hyperparameter should not be perched on a knife-edge.\n"
          "  5. Out-of-sample quantile calibration (above) is in the right neighborhood of its nominal\n"
          "     85/90/95% -- deviation is expected (censoring, generalization) but should be inspected,\n"
          "     not ignored.\n"
          "No single number here is a pass/fail gate by itself -- that is a deliberate choice: let the\n"
          "diagnostics show the empirical scale before encoding a cutoff.")

    print(f"\n{'#' * 100}")
    print("What this does and does not establish")
    print(f"{'#' * 100}")
    print("demonstrated_capacity_pXX estimates the upper end of exposure HISTORICALLY SERVICED by low-risk\n"
          "agents with similar business fundamentals under the existing (censoring) engine -- not a latent\n"
          "maximum capacity, and not yet a formula to deploy. combined_cap was never read by the feature\n"
          "matrix. Validation 1 tests whether performance actually deteriorates above ~1.0x utilization\n"
          "(the falsification test); Validation 2 (run for both All agents and Diamond A) tests whether\n"
          "agents with large predicted gaps show fundamentals/performance consistent with the gap, not just\n"
          "a large model output -- pay particular attention to the >1.5x and >2.0x uplift bands, where the\n"
          "challenger makes its strongest claim that the existing engine understates business scale. Only\n"
          "after both pass would this be combined with C3 (L_recommended = L_capacity_challenger x M_C3).")


if __name__ == "__main__":
    main()
