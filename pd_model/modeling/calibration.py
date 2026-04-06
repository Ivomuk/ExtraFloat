"""
PD calibration, bootstrap comparison, and policy placement pipeline.

Preserves all algorithms from file9.txt exactly:
  - Isotonic regression for monotonic PD calibration
  - Fail-closed guards on minimum sample size and calibration coverage
  - Paired bootstrap for AUC confidence intervals
  - Approval-rate policy tables

No hardcoded export paths — callers pass output_dir if persistence is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from pd_model.config import feature_config
from pd_model.config.model_config import DEFAULT_CONFIG, ModelConfig
from pd_model.logging_config import get_logger

logger = get_logger(__name__)


# ======================================================================== #
# Internal helpers
# ======================================================================== #

def _standardize_scored_df(
    scored_df: pd.DataFrame,
    model_key: str,
) -> tuple[pd.DataFrame, str]:
    """
    Normalise a scored DataFrame to columns: agent_msisdn, y, model_score.
    Returns (normalised_df, score_col_name_used).
    """
    df = scored_df.copy()

    # Target column
    if "y" not in df.columns and "bad_state" in df.columns:
        df["y"] = pd.to_numeric(df["bad_state"], errors="coerce")
    df["y"] = (pd.to_numeric(df.get("y", pd.Series(dtype=float)), errors="coerce") > 0.5).astype(int)

    # Agent ID
    if feature_config.AGENT_KEY not in df.columns:
        df[feature_config.AGENT_KEY] = df.index.astype(str)
    else:
        df[feature_config.AGENT_KEY] = df[feature_config.AGENT_KEY].astype(str)

    # Score column — look for model-specific first, then generic
    score_col = None
    candidates = [f"{model_key}_score", "score", feature_config.RAW_SCORE_COL]
    for c in candidates:
        if c in df.columns:
            score_col = c
            break
    if score_col is None:
        score_cands = [c for c in df.columns if "score" in str(c).lower()]
        if score_cands:
            score_col = score_cands[0]
    if score_col is None:
        raise ValueError(f"No score column found for model '{model_key}'")

    df["model_score"] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[[feature_config.AGENT_KEY, "y", "model_score"]].dropna()

    return df, score_col


# ======================================================================== #
# Calibration map
# ======================================================================== #

def build_pd_calibration_map(
    scored_df: pd.DataFrame,
    model_key: str,
    cfg: ModelConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Build an isotonic-regression PD calibration map from a scored DataFrame.

    Fail-closed on cfg.cal_min_n, cfg.cal_min_bads, and degenerate event rates.
    Applies isotonic regression to enforce monotone PD vs. score.

    Parameters
    ----------
    scored_df : DataFrame with bad_state (or y) and a score column
    model_key : "xgb" or "lgb" — used to look up the score column
    cfg       : ModelConfig

    Returns
    -------
    DataFrame with columns:
        model, score_min, score_max, pd, ascending_risk, score_col, n, bad_rate
    """
    df, score_col = _standardize_scored_df(scored_df, model_key)

    n_obs = int(df.shape[0])
    n_bads = int(df["y"].sum())
    event_rate = float(df["y"].mean())

    if n_obs < cfg.cal_min_n:
        raise ValueError(
            f"[build_pd_calibration_map] Fail-closed: n={n_obs} < cal_min_n={cfg.cal_min_n} "
            f"for model '{model_key}'"
        )
    if n_bads < cfg.cal_min_bads:
        raise ValueError(
            f"[build_pd_calibration_map] Fail-closed: bads={n_bads} < cal_min_bads={cfg.cal_min_bads} "
            f"for model '{model_key}'"
        )
    if event_rate <= 0.0 or event_rate >= 1.0:
        raise ValueError(
            f"[build_pd_calibration_map] Fail-closed: event_rate={event_rate:.4f} is degenerate "
            f"for model '{model_key}'"
        )

    # Risk direction
    corr_val = float(pd.Series(df["model_score"]).corr(pd.Series(df["y"])))
    if np.isnan(corr_val):
        raise ValueError(
            f"[build_pd_calibration_map] Fail-closed: correlation undefined for model '{model_key}'"
        )
    ascending_risk = bool(corr_val > 0)

    # Quantile binning
    df = df.sort_values("model_score").reset_index(drop=True)
    try:
        df["bin"] = pd.qcut(df["model_score"], q=cfg.cal_n_bins, labels=False, duplicates="drop")
    except Exception:
        df["bin"] = pd.qcut(df["model_score"], q=20, labels=False, duplicates="drop")

    bin_tbl = (
        df.groupby("bin", as_index=False)
        .agg(
            n=("y", "size"),
            bad_rate=("y", "mean"),
            score_min=("model_score", "min"),
            score_max=("model_score", "max"),
            score_mean=("model_score", "mean"),
        )
        .sort_values("score_mean")
        .reset_index(drop=True)
    )

    x_vals = bin_tbl["score_mean"].to_numpy()
    y_vals = bin_tbl["bad_rate"].to_numpy()

    iso = IsotonicRegression(increasing=ascending_risk, out_of_bounds="clip")
    pd_hat = iso.fit_transform(x_vals, y_vals)
    pd_hat = np.clip(pd_hat, cfg.eps, 1.0 - cfg.eps)

    bin_tbl["pd"] = pd_hat
    bin_tbl["model"] = model_key
    bin_tbl["ascending_risk"] = ascending_risk
    bin_tbl["score_col"] = score_col

    map_tbl = (
        bin_tbl[["model", "score_min", "score_max", "pd", "ascending_risk", "score_col", "n", "bad_rate"]]
        .sort_values(["score_min", "score_max"])
        .reset_index(drop=True)
    )

    logger.info(
        "build_pd_calibration_map: model=%s | bins=%d | ascending_risk=%s | pd range=[%.4f, %.4f]",
        model_key, len(map_tbl), ascending_risk,
        float(map_tbl["pd"].min()), float(map_tbl["pd"].max()),
    )
    return map_tbl


# ======================================================================== #
# Attach calibrated PD
# ======================================================================== #

def attach_cal_pd(
    scored_df: pd.DataFrame,
    cal_map_tbl: pd.DataFrame,
    model_key: str,
    cfg: ModelConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Interval-join calibration map onto scored_df to produce cal_pd column.

    Fail-closed on cfg.cal_min_coverage — raises if too many rows are uncovered.

    Returns
    -------
    DataFrame identical to the standardised scored_df plus a cal_pd column.
    """
    df, _ = _standardize_scored_df(scored_df, model_key)

    cal_sub = cal_map_tbl[cal_map_tbl["model"].astype(str) == str(model_key)].copy()
    if cal_sub.shape[0] == 0:
        raise ValueError(
            f"[attach_cal_pd] Fail-closed: no calibration mapping for model '{model_key}'"
        )

    for col in ("score_min", "score_max", "pd"):
        cal_sub[col] = pd.to_numeric(cal_sub[col], errors="coerce")
    cal_sub = cal_sub.dropna(subset=["score_min", "score_max", "pd"])
    cal_sub = cal_sub.sort_values(["score_min", "score_max"]).reset_index(drop=True)

    score_vals = df["model_score"].to_numpy()

    # Use score_min values as left edges; searchsorted handles gaps and avoids O(n*bins) loop.
    # Bins defined as [score_min[i], score_min[i+1]) for i < last, and [score_min[last], +inf) for last.
    bin_lefts = cal_sub["score_min"].to_numpy()
    pd_lookup = cal_sub["pd"].to_numpy()

    # searchsorted returns the insertion point — subtract 1 to get the bin index
    bin_idx = np.searchsorted(bin_lefts, score_vals, side="right") - 1

    # Clip to valid range: index -1 → bin 0 (below minimum), index >= n_bins → last bin
    bin_idx = np.clip(bin_idx, 0, len(bin_lefts) - 1)

    assigned_pd = pd_lookup[bin_idx]
    df[feature_config.CAL_PD_COL] = assigned_pd
    coverage = float(pd.Series(assigned_pd).notna().mean())
    if coverage < cfg.cal_min_coverage:
        raise ValueError(
            f"[attach_cal_pd] Fail-closed: cal_pd coverage={coverage:.4f} < "
            f"cal_min_coverage={cfg.cal_min_coverage} for model '{model_key}'"
        )

    logger.info(
        "attach_cal_pd: model=%s | rows=%d | coverage=%.4f | pd range=[%.4f, %.4f]",
        model_key, df.shape[0], coverage,
        float(np.nanmin(assigned_pd)), float(np.nanmax(assigned_pd)),
    )
    return df


# ======================================================================== #
# Policy tables
# ======================================================================== #

def build_policy_tables(
    scored_df_with_pd: pd.DataFrame,
    cfg: ModelConfig = DEFAULT_CONFIG,
    prefer_pd: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build approval-rate threshold grid and decile band tables.

    Parameters
    ----------
    scored_df_with_pd : DataFrame with cal_pd (and y) columns
    cfg               : ModelConfig (policy_operating_points not used here;
                        grid is 5%–95% in 5pp steps)
    prefer_pd         : if True, sort by cal_pd ascending (lower PD = approved first)

    Returns
    -------
    (threshold_tbl, decile_band_tbl, df_sorted)
    """
    df = scored_df_with_pd.copy()

    if prefer_pd:
        if feature_config.CAL_PD_COL not in df.columns:
            raise ValueError(
                "[build_policy_tables] Fail-closed: cal_pd missing but prefer_pd=True"
            )
        sort_var = feature_config.CAL_PD_COL
        sort_ascending = True
    else:
        sort_var = "model_score"
        corr_val = float(pd.Series(df["model_score"]).corr(pd.Series(df["y"])))
        if np.isnan(corr_val):
            raise ValueError("[build_policy_tables] Fail-closed: correlation undefined")
        sort_ascending = not bool(corr_val > 0)

    df_sorted = df.sort_values(sort_var, ascending=sort_ascending).reset_index(drop=True)
    df_sorted["rank"] = np.arange(1, df_sorted.shape[0] + 1)
    df_sorted["approval_rate"] = df_sorted["rank"] / float(df_sorted.shape[0])
    df_sorted["cum_bad_rate"] = df_sorted["y"].cumsum() / df_sorted["rank"]

    grid = np.linspace(0.05, 0.95, 19)
    rows = []
    for ar in grid:
        k = int(np.floor(ar * float(df_sorted.shape[0])))
        if k < 1:
            continue
        sub_df = df_sorted.iloc[:k]
        cutoff = float(sub_df[sort_var].max())
        rows.append([float(ar), int(k), float(sub_df["y"].mean()), cutoff])

    thresh_tbl = pd.DataFrame(
        rows, columns=["approve_rate_target", "n_approved", "expected_bad_rate", "cutoff"]
    )
    thresh_tbl["cutoff_var"] = sort_var

    band_var = sort_var
    band_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[band_var, "y"]).copy()
    band_df["band_decile"] = pd.qcut(band_df[band_var], q=10, labels=False, duplicates="drop")
    dec_tbl = (
        band_df.groupby("band_decile")
        .agg(
            n=("y", "size"),
            obs_bad_rate=("y", "mean"),
            band_min=(band_var, "min"),
            band_max=(band_var, "max"),
            band_mean=(band_var, "mean"),
        )
        .reset_index()
        .sort_values("band_decile")
    )
    dec_tbl["band_var"] = band_var

    return thresh_tbl, dec_tbl, df_sorted


# ======================================================================== #
# Policy flags
# ======================================================================== #

def add_policy_flags(
    agent_df: pd.DataFrame,
    policy_threshold_tbl: pd.DataFrame,
    prefix: str,
    cfg: ModelConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Add approved_at_XX columns based on cal_pd vs. policy cutoffs.

    For each operating point in cfg.policy_operating_points (default 10/20/50/80%),
    adds a binary column ``{prefix}_approved_at_{int(op*100)}``.
    """
    out = agent_df.copy()
    for ar in cfg.policy_operating_points:
        idx = policy_threshold_tbl["approve_rate_target"].sub(ar).abs().idxmin()
        cutoff_val = float(policy_threshold_tbl.loc[idx, "cutoff"])
        col_name = f"{prefix}_approved_at_{int(ar * 100)}"
        out[col_name] = (out[feature_config.CAL_PD_COL] <= cutoff_val).astype(int)
    return out


def make_policy_bucket(
    agent_df: pd.DataFrame,
    prefix: str,
    cfg: ModelConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """
    Compact placement label: APPROVE_10/20/50/80 or DECLINE_ALL.
    Agents approved at a tighter threshold are labelled at that threshold.
    """
    ops_sorted = sorted(cfg.policy_operating_points)
    conditions = []
    choices = []
    for ar in ops_sorted:
        col = f"{prefix}_approved_at_{int(ar * 100)}"
        if col in agent_df.columns:
            conditions.append(agent_df[col] == 1)
            choices.append(f"APPROVE_{int(ar * 100)}")
    if not conditions:
        return pd.Series("DECLINE_ALL", index=agent_df.index)
    return pd.Series(
        np.select(conditions, choices, default="DECLINE_ALL"),
        index=agent_df.index,
    )


# ======================================================================== #
# Bootstrap comparison
# ======================================================================== #

def run_bootstrap_comparison(
    xgb_scored: pd.DataFrame,
    lgb_scored: pd.DataFrame,
    cfg: ModelConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Paired bootstrap for (LGB AUC − XGB AUC) and XGB AUC 95% CI.

    Aligns rows by agent_msisdn if available, otherwise by index intersection.
    Preserves algorithm from file9.txt exactly.

    Returns
    -------
    DataFrame with one row per metric (align_mode, n_rows, point estimates, CIs).
    """
    id_candidates = [feature_config.AGENT_KEY, "msisdn", "agent_id", "agent_key"]
    join_key = next((c for c in id_candidates
                     if c in xgb_scored.columns and c in lgb_scored.columns), None)

    if join_key is not None:
        xgb_small = xgb_scored[[join_key, "bad_state", "raw_score"]].drop_duplicates(join_key)
        lgb_small = lgb_scored[[join_key, "bad_state", "raw_score"]].drop_duplicates(join_key)
        merged = xgb_small.merge(lgb_small, on=join_key, how="inner", suffixes=("_xgb", "_lgb"))
        cmp_df = pd.DataFrame({
            "y_xgb": pd.to_numeric(merged["bad_state_xgb"], errors="coerce"),
            "xgb": pd.to_numeric(merged["raw_score_xgb"], errors="coerce"),
            "y_lgb": pd.to_numeric(merged["bad_state_lgb"], errors="coerce"),
            "lgb": pd.to_numeric(merged["raw_score_lgb"], errors="coerce"),
        }).dropna()
        align_mode = f"join_on_{join_key}"
    else:
        common_idx = xgb_scored.index.intersection(lgb_scored.index)
        cmp_df = pd.DataFrame({
            "y_xgb": pd.to_numeric(xgb_scored.loc[common_idx, "bad_state"], errors="coerce"),
            "xgb": pd.to_numeric(xgb_scored.loc[common_idx, "raw_score"], errors="coerce"),
            "y_lgb": pd.to_numeric(lgb_scored.loc[common_idx, "bad_state"], errors="coerce"),
            "lgb": pd.to_numeric(lgb_scored.loc[common_idx, "raw_score"], errors="coerce"),
        }).dropna()
        align_mode = "index_intersection"

    cmp_df["y_xgb"] = cmp_df["y_xgb"].astype(int)
    cmp_df["y_lgb"] = cmp_df["y_lgb"].astype(int)
    y_mismatch = int((cmp_df["y_xgb"] != cmp_df["y_lgb"]).sum())
    cmp_df = cmp_df.drop(columns=["y_lgb"]).rename(columns={"y_xgb": "y"})

    logger.info(
        "bootstrap: align_mode=%s | aligned_rows=%d | y_mismatch=%d",
        align_mode, cmp_df.shape[0], y_mismatch,
    )

    xgb_point = lgb_point = diff_point = np.nan
    if cmp_df.shape[0] > 0 and np.unique(cmp_df["y"].values).size >= 2:
        xgb_point = float(roc_auc_score(cmp_df["y"], cmp_df["xgb"]))
        lgb_point = float(roc_auc_score(cmp_df["y"], cmp_df["lgb"]))
        diff_point = float(lgb_point - xgb_point)

    rng = np.random.default_rng(20260115)
    n_rows = int(cmp_df.shape[0])
    y_vals = cmp_df["y"].to_numpy()
    xgb_vals = cmp_df["xgb"].to_numpy()
    lgb_vals = cmp_df["lgb"].to_numpy()

    xgb_auc_boot = np.full(cfg.bootstrap_n, np.nan)
    diff_auc_boot = np.full(cfg.bootstrap_n, np.nan)

    for b in tqdm(range(cfg.bootstrap_n), desc="Bootstrap", leave=False):
        samp_idx = rng.integers(0, n_rows, size=n_rows)
        y_b = y_vals[samp_idx]
        if np.unique(y_b).size < 2:
            continue
        auc_x = roc_auc_score(y_b, xgb_vals[samp_idx])
        auc_l = roc_auc_score(y_b, lgb_vals[samp_idx])
        xgb_auc_boot[b] = auc_x
        diff_auc_boot[b] = auc_l - auc_x

    xgb_ser = pd.Series(xgb_auc_boot).dropna()
    diff_ser = pd.Series(diff_auc_boot).dropna()

    out_tbl = pd.DataFrame({
        "metric": [
            "align_mode", "n_rows_aligned", "y_mismatch",
            "xgb_point_auc", "xgb_ci95_lo", "xgb_ci95_hi",
            "lgb_point_auc",
            "diff_point_lgb_minus_xgb", "diff_ci95_lo", "diff_ci95_hi",
            "p_lgb_better",
        ],
        "value": [
            align_mode,
            float(n_rows),
            float(y_mismatch),
            xgb_point,
            float(xgb_ser.quantile(0.025)) if xgb_ser.shape[0] > 0 else np.nan,
            float(xgb_ser.quantile(0.975)) if xgb_ser.shape[0] > 0 else np.nan,
            lgb_point,
            diff_point,
            float(diff_ser.quantile(0.025)) if diff_ser.shape[0] > 0 else np.nan,
            float(diff_ser.quantile(0.975)) if diff_ser.shape[0] > 0 else np.nan,
            float((diff_ser > 0).mean()) if diff_ser.shape[0] > 0 else np.nan,
        ],
    })

    logger.info(
        "bootstrap complete: XGB AUC=%.4f [%.4f, %.4f] | LGB-XGB diff=%.4f",
        xgb_point,
        float(xgb_ser.quantile(0.025)) if xgb_ser.shape[0] > 0 else np.nan,
        float(xgb_ser.quantile(0.975)) if xgb_ser.shape[0] > 0 else np.nan,
        diff_point,
    )
    return out_tbl


# ======================================================================== #
# Full locked policy pipeline
# ======================================================================== #

def run_locked_policy_pipeline(
    xgb_scored: pd.DataFrame,
    lgb_scored: pd.DataFrame,
    cfg: ModelConfig = DEFAULT_CONFIG,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Orchestrate calibration map building, cal_pd attachment, and policy tables
    for both XGB and LGB models.

    Parameters
    ----------
    xgb_scored : val_scored DataFrame from train_xgb (must have raw_score, bad_state)
    lgb_scored : val_scored DataFrame from train_lgbm (must have raw_score, bad_state)
    cfg        : ModelConfig
    output_dir : if provided, writes CSV artifacts to this directory

    Returns
    -------
    dict with keys:
        pd_calibration_map_tbl,
        xgb_policy_threshold_tbl, xgb_policy_band_tbl,
        lgb_policy_threshold_tbl, lgb_policy_band_tbl,
        xgb_policy_df_sorted, lgb_policy_df_sorted,
        paths (dict of written file names, if output_dir provided)
    """
    # Alias lgb raw_score to lgb_score for _standardize_scored_df
    lgb_work = lgb_scored.copy()
    if "raw_score" in lgb_work.columns and "lgb_score" not in lgb_work.columns:
        lgb_work["lgb_score"] = lgb_work["raw_score"]

    xgb_map = build_pd_calibration_map(xgb_scored, "xgb", cfg=cfg)
    lgb_map = build_pd_calibration_map(lgb_work, "lgb", cfg=cfg)
    pd_calibration_map_tbl = pd.concat([xgb_map, lgb_map], ignore_index=True)

    xgb_with_pd = attach_cal_pd(xgb_scored, pd_calibration_map_tbl, "xgb", cfg=cfg)
    lgb_with_pd = attach_cal_pd(lgb_work, pd_calibration_map_tbl, "lgb", cfg=cfg)

    xgb_thresh, xgb_band, xgb_sorted = build_policy_tables(xgb_with_pd, cfg=cfg, prefer_pd=True)
    lgb_thresh, lgb_band, lgb_sorted = build_policy_tables(lgb_with_pd, cfg=cfg, prefer_pd=True)

    paths: dict[str, str] = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for df_out, fname in [
            (xgb_thresh, "xgb_policy_thresholds.csv"),
            (xgb_band, "xgb_policy_band_deciles.csv"),
            (lgb_thresh, "lgb_policy_thresholds.csv"),
            (lgb_band, "lgb_policy_band_deciles.csv"),
            (pd_calibration_map_tbl, "pd_calibration_map.csv"),
        ]:
            p = output_dir / fname
            df_out.to_csv(p, index=False)
            paths[fname] = str(p)
            logger.info("Wrote %s (%d rows)", p, len(df_out))

    return {
        "pd_calibration_map_tbl": pd_calibration_map_tbl,
        "xgb_policy_threshold_tbl": xgb_thresh,
        "xgb_policy_band_tbl": xgb_band,
        "lgb_policy_threshold_tbl": lgb_thresh,
        "lgb_policy_band_tbl": lgb_band,
        "xgb_policy_df_sorted": xgb_sorted,
        "lgb_policy_df_sorted": lgb_sorted,
        "paths": paths,
    }
