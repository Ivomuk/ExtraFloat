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

TWO VALIDATION LAYERS, run AFTER fitting (this script does not treat
the model as validated just because it fits):
  1. CapacityUtilization = actual_exposure_ugx / demonstrated_capacity_p90,
     banded, with borrower-only 3+DPD / bad-closure / shortfall / N / PD
     mix reported per band -- the falsification test: if performance
     does not deteriorate above ~1.0x utilization, P90 historical
     exposure is not identifying an economically meaningful boundary.
  2. Diamond A capacity-gap check: CapacityGap = demonstrated_capacity_p90
     - combined_cap, CapacityUpliftRatio = demonstrated_capacity_p90 /
     combined_cap, banded, reporting whether agents with LARGE predicted
     gaps actually show the fundamentals and historical performance
     consistent with having supported exposure above their current
     combined_cap -- not just a large number from the model.

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
DEFAULT_MIN_SAMPLES_LEAF = 100  # tuned to reduce (not eliminate -- see monotonicity_diagnostic) the
                                 # near-monotonicity noise that sklearn's HistGradientBoostingRegressor
                                 # exhibits with loss="quantile" + monotonic_cst (confirmed directly:
                                 # monotonic_cst is an exact guarantee for loss="squared_error", but
                                 # NOT for loss="quantile" -- small violations of a few percent of the
                                 # output range can occur even with the constraint set. This is a known
                                 # property of this estimator, not a bug in this script.

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


def monotonicity_diagnostic(models: dict, feature_medians: dict, feature_ranges: dict, n_grid: int = 60) -> pd.DataFrame:
    """For each feature, sweep it across its observed training range while
    holding the other three at their training median, and report the worst
    (most negative) step-to-step change in predicted log-capacity for each
    quantile model. monotonic_cst guarantees this is exactly >= 0 for
    loss="squared_error" but NOT for loss="quantile" (confirmed directly --
    see module-level comment) -- this is reported as a model-health
    diagnostic, never silently hidden or corrected by re-sorting."""
    rows = []
    for feat in FEATURE_COLUMNS:
        lo, hi = feature_ranges[feat]
        grid = pd.DataFrame({c: np.full(n_grid, feature_medians[c]) for c in FEATURE_COLUMNS})
        grid[feat] = np.linspace(lo, hi, n_grid)
        X_grid, valid_grid = build_feature_matrix(grid)
        X_arr = X_grid.to_numpy()
        for q, m in models.items():
            preds = m.predict(X_arr)
            diffs = np.diff(preds)
            total_range = preds.max() - preds.min()
            rows.append({
                "feature": feat, "quantile": q,
                "n_steps": len(diffs), "n_violations": int((diffs < 0).sum()),
                "worst_violation": round(float(diffs.min()), 5) if len(diffs) else 0.0,
                "worst_violation_pct_of_range": round(float(-diffs.min()) / total_range * 100, 2) if total_range > 0 and diffs.min() < 0 else 0.0,
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


def diamond_capacity_gap_check(df: pd.DataFrame, scored: pd.DataFrame, primary_col: str,
                                diamond_high_risk_cal_pd: float) -> pd.DataFrame:
    """Validation layer 2: for Diamond A, do agents with LARGE predicted
    capacity gaps actually show the fundamentals and performance consistent
    with having supported exposure above their current combined_cap?"""
    diamond_mask = (
        df["agent_category"].astype(str).str.strip().str.lower() == "diamond"
        if "agent_category" in df.columns else pd.Series(False, index=df.index)
    )
    low_risk_mask = df["cal_pd"] < diamond_high_risk_cal_pd
    working = df[diamond_mask & low_risk_mask].join(
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
    ap.add_argument("--low-risk-cal-pd", type=float, default=DEFAULT_LOW_RISK_CAL_PD)
    ap.add_argument("--diamond-high-risk-cal-pd", type=float, default=DEFAULT_DIAMOND_HIGH_RISK_CAL_PD)
    ap.add_argument("--quantiles", default=",".join(str(q) for q in DEFAULT_QUANTILES))
    ap.add_argument("--primary-quantile", type=float, default=DEFAULT_PRIMARY_QUANTILE)
    ap.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    ap.add_argument("--min-samples-leaf", type=int, default=DEFAULT_MIN_SAMPLES_LEAF)
    ap.add_argument("--model-out-dir", default="capacity_challenger_artifacts")
    ap.add_argument("--out-prefix", default="capacity_challenger")
    ap.add_argument("--no-save-model", action="store_true")
    args = ap.parse_args(argv)
    quantiles = [float(q) for q in args.quantiles.split(",")]
    primary_col = _capacity_col(args.primary_quantile)

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

    y_train = np.log(df.loc[train_mask, "actual_exposure_ugx"])
    X_train = X.loc[train_mask]

    print(f"Fitting quantile models {quantiles} (primary={args.primary_quantile}), "
          f"monotonic_cst={MONOTONIC_CST} on features {FEATURE_COLUMNS} "
          f"(sklearn {sklearn.__version__}, random_state={args.random_state}, "
          f"min_samples_leaf={args.min_samples_leaf})...")
    models = fit_quantile_models(X_train, y_train, quantiles, MONOTONIC_CST, args.random_state, args.min_samples_leaf)

    scored = score(X, feature_valid, models)
    scored = add_capacity_gap(df, scored, primary_col)

    diag = crossing_diagnostic(scored, quantiles)
    print(f"\nCrossing diagnostic (quantile models fit independently -- NOT forced monotonic across q): "
          f"{diag['n_crossing']:,} of {diag['n_valid']:,} valid rows ({diag['pct_crossing']}%) have "
          f"P85/P90/P95 out of order. Not corrected -- reported as a model-health signal.")

    feature_medians = {c: float(df.loc[train_mask, c].median()) for c in FEATURE_COLUMNS}
    feature_ranges = {c: (float(df.loc[train_mask, c].min()), float(df.loc[train_mask, c].max())) for c in FEATURE_COLUMNS}
    mono_diag = monotonicity_diagnostic(models, feature_medians, feature_ranges)
    worst = mono_diag["worst_violation_pct_of_range"].max()
    print(f"\nMonotonicity diagnostic (monotonic_cst is an EXACT guarantee for loss='squared_error' but "
          f"NOT for loss='quantile' -- confirmed directly, not assumed): worst observed violation is "
          f"{worst:.2f}% of that sweep's output range. Reported, not corrected -- see "
          f"{args.out_prefix}_monotonicity_diagnostic.csv for the full per-feature/per-quantile breakdown.")
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

    print(f"\n{'=' * 100}\nValidation 2: Diamond A ({'cal_pd < %.0f%%' % (args.diamond_high_risk_cal_pd * 100)}) "
          f"capacity-gap check\n{'=' * 100}")
    diamond_check = diamond_capacity_gap_check(df, scored, primary_col, args.diamond_high_risk_cal_pd)
    with pd.option_context("display.float_format", "{:,.2f}".format, "display.max_columns", None, "display.width", 240):
        print(diamond_check.to_string(index=False))
    diamond_check.to_csv(f"{args.out_prefix}_diamond_a_gap_check.csv", index=False)

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
            "monotonicity_note": "monotonic_cst is an exact guarantee for HistGradientBoostingRegressor "
                                  "with loss='squared_error', but NOT for loss='quantile' (confirmed by "
                                  "direct testing, not assumed) -- small violations can occur; see "
                                  f"{args.out_prefix}_monotonicity_diagnostic.csv from this fit for the "
                                  "observed magnitude.",
            "output_naming": "demonstrated_capacity_pXX -- the upper end of historically demonstrated "
                              "exposure under the EXISTING (censoring) engine, not a latent maximum capacity.",
            "excluded_features_note": "business persistence excluded from v1 pending the recency-column bug fix "
                                       "flagged earlier in this session (median_days_since_any_activity = -20,666).",
        }
        (out_dir / "capacity_challenger_metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"\nWrote model artifact(s) + metadata to {out_dir}/")

    print(f"\n{'#' * 100}")
    print("What this does and does not establish")
    print(f"{'#' * 100}")
    print("demonstrated_capacity_pXX estimates the upper end of exposure HISTORICALLY SERVICED by low-risk\n"
          "agents with similar business fundamentals under the existing (censoring) engine -- not a latent\n"
          "maximum capacity, and not yet a formula to deploy. combined_cap was never read by the feature\n"
          "matrix. Validation 1 tests whether performance actually deteriorates above ~1.0x utilization\n"
          "(the falsification test); Validation 2 tests whether Diamond A agents with large predicted gaps\n"
          "show fundamentals/performance consistent with the gap, not just a large model output. Only after\n"
          "both pass would this be combined with C3 (L_recommended = L_capacity_challenger x M_C3).")


if __name__ == "__main__":
    main()
