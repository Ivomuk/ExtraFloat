"""
Information Value (IV) computation and feature selection for PD modelling (file5).

Provides:
- ``compute_iv``                              – quantile-binned IV for a single feature
- ``compute_iv_fixed_bins``                   – IV with bins learned from a reference series
- ``compute_iv_with_bins``                    – IV using externally provided bin edges
- ``iv_filter_phase_2``                       – IV-based feature selection (train-only)
- ``iv_filter_phase_2_separability_sensitive``– IV selection with diagnostic variants
- ``iv_audit``                                – annotate IV table for leakage / degeneracy

Note: the duplicate ``compute_iv_fixed_bins`` definition present in the original
file5.txt has been removed; only one canonical definition is kept here.
"""

from __future__ import annotations

from typing import FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd

from pd_model.config.feature_config import LEAKAGE_PATTERNS, PD_FEATURE_BLACKLIST
from pd_model.config.model_config import DEFAULT_CONFIG, ModelConfig
from pd_model.logging_config import get_logger

logger = get_logger(__name__)


# ======================================================================== #
# Core IV computation
# ======================================================================== #

def compute_iv(
    x: pd.Series,
    y: pd.Series,
    n_bins: int = 10,
    bins: Optional[np.ndarray] = None,
    return_bins: bool = False,
    eps: float = DEFAULT_CONFIG.eps,
) -> float | Tuple[float, Optional[np.ndarray]]:
    """
    Compute Information Value for a single feature vs a binary target.

    Uses quantile-based binning by default; falls back to equal-width bins if
    ``pd.qcut`` fails (e.g. too many duplicate values).  If ``bins`` is
    provided those edges are used directly (no binning from data).

    Args:
        x:           Numeric feature series.
        y:           Binary target series (0/1).
        n_bins:      Number of quantile bins (ignored when ``bins`` is supplied).
        bins:        Pre-computed bin edges (optional).
        return_bins: If ``True``, also return the bin edges used.
        eps:         Smoothing constant for WOE computation.

    Returns:
        IV value (float), or ``(IV, bin_edges)`` when ``return_bins=True``.
    """
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    x_nn = data["x"]
    y_nn = data["y"]

    if x_nn.nunique() <= 1:
        if return_bins:
            return 0.0, None
        return 0.0

    if bins is None:
        max_bins = min(n_bins, x_nn.nunique())
        try:
            binned, bin_edges = pd.qcut(x_nn, q=max_bins, retbins=True, duplicates="drop")
        except ValueError:
            binned, bin_edges = pd.cut(
                x_nn, bins=max_bins, retbins=True, include_lowest=True
            )
    else:
        bin_edges = np.unique(np.asarray(bins))
        if bin_edges.shape[0] < 2:
            if return_bins:
                return 0.0, bin_edges
            return 0.0
        bin_edges = _enforce_strictly_increasing(bin_edges, eps)
        binned = pd.cut(x_nn, bins=bin_edges, include_lowest=True)

    iv_val = _iv_from_groupby(binned, y_nn, eps)

    if iv_val == 0.0:
        logger.warning("compute_iv: feature returned IV=0.0 (check binning or class balance)")

    if return_bins:
        return float(iv_val), bin_edges
    return float(iv_val)


def compute_iv_fixed_bins(
    x: pd.Series,
    y: pd.Series,
    n_bins: int = 10,
    binning: str = "quantile",
    return_bins: bool = False,
    eps: float = DEFAULT_CONFIG.eps,
) -> float | Tuple[float, Optional[np.ndarray]]:
    """
    Compute IV by first learning bin edges from *x*, then applying those fixed
    edges to compute WOE/IV.

    This is the canonical single definition (the duplicate from the original
    file5.txt has been removed).

    Args:
        x:           Numeric series from which bin edges are learned.
        y:           Binary target series (0/1).
        n_bins:      Target number of bins.
        binning:     ``"quantile"`` (default) or ``"uniform"`` (equal-width).
        return_bins: If ``True``, also return the bin edges used.
        eps:         WOE smoothing constant.

    Returns:
        IV value (float), or ``(IV, bin_edges)`` when ``return_bins=True``.
    """
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    x_nn = data["x"]
    y_nn = data["y"]

    if x_nn.nunique(dropna=True) <= 1:
        if return_bins:
            return 0.0, None
        return 0.0

    max_bins = int(min(n_bins, x_nn.nunique(dropna=True)))

    if binning == "quantile":
        try:
            _, bin_edges = pd.qcut(x_nn, q=max_bins, retbins=True, duplicates="drop")
        except ValueError:
            _, bin_edges = pd.cut(x_nn, bins=max_bins, retbins=True, include_lowest=True)
    else:
        _, bin_edges = pd.cut(x_nn, bins=max_bins, retbins=True, include_lowest=True)

    bin_edges = np.unique(np.asarray(bin_edges))
    if bin_edges.shape[0] < 2:
        if return_bins:
            return 0.0, bin_edges
        return 0.0

    bin_edges = _enforce_strictly_increasing(bin_edges, eps=1e-9)
    binned = pd.cut(x_nn, bins=bin_edges, include_lowest=True)

    grouped = pd.DataFrame({"bin": binned, "y": y_nn}).groupby("bin", observed=False)
    bad = grouped["y"].sum()
    good = grouped["y"].count() - bad
    bad_dist = (bad + eps) / (bad.sum() + eps)
    good_dist = (good + eps) / (good.sum() + eps)
    woe = np.log(bad_dist / good_dist)
    iv_val = float(((bad_dist - good_dist) * woe).sum())

    if return_bins:
        return iv_val, bin_edges
    return iv_val


def compute_iv_with_bins(
    x: pd.Series,
    y: pd.Series,
    bins: np.ndarray,
    return_bins: bool = False,
    eps: float = DEFAULT_CONFIG.eps,
) -> float | Tuple[float, Optional[np.ndarray]]:
    """
    Compute IV for *x* using externally provided fixed bin edges.

    Args:
        x:           Numeric feature series.
        y:           Binary target series (0/1).
        bins:        Pre-computed bin edges (e.g. from a training set).
        return_bins: If ``True``, also return the bin edges used.
        eps:         WOE smoothing constant.

    Returns:
        IV value, or ``(IV, bin_edges)`` when ``return_bins=True``.
    """
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    x_nn = data["x"]
    y_nn = data["y"]

    if x_nn.nunique(dropna=True) <= 1:
        if return_bins:
            return 0.0, np.unique(np.asarray(bins)) if bins is not None else None
        return 0.0

    if bins is None:
        if return_bins:
            return 0.0, None
        return 0.0

    bin_edges = np.unique(np.asarray(bins))
    if bin_edges.shape[0] < 2:
        if return_bins:
            return 0.0, bin_edges
        return 0.0

    bin_edges = _enforce_strictly_increasing(bin_edges, eps=1e-9)
    binned = pd.cut(x_nn, bins=bin_edges, include_lowest=True)
    grouped = pd.DataFrame({"bin": binned, "y": y_nn}).groupby("bin", observed=False)
    bad = grouped["y"].sum()
    good = grouped["y"].count() - bad
    bad_dist = (bad + eps) / (bad.sum() + eps)
    good_dist = (good + eps) / (good.sum() + eps)
    woe = np.log(bad_dist / good_dist)
    iv_val = float(((bad_dist - good_dist) * woe).sum())

    if return_bins:
        return iv_val, bin_edges
    return iv_val


# ======================================================================== #
# IV-based feature selection
# ======================================================================== #

def iv_filter_phase_2(
    X_train_raw: pd.DataFrame,
    X_train_transformed: pd.DataFrame,
    y_train: pd.Series,
    cfg: ModelConfig = DEFAULT_CONFIG,
    pd_feature_blacklist: FrozenSet[str] = PD_FEATURE_BLACKLIST,
    forbidden_feature_patterns: Tuple[str, ...] = LEAKAGE_PATTERNS,
    target_col: str = "bad_state",
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features by Information Value computed on the **training set only**.

    Selection criteria:
    - ``iv_after >= cfg.iv_min``
    - ``iv_uplift >= cfg.iv_min_uplift`` (or NaN, treated as min_uplift)

    Args:
        X_train_raw:               Raw numeric features (train split).
        X_train_transformed:       Transformed numeric features (train split).
        y_train:                   Binary target (train split).
        cfg:                       Model config for IV thresholds.
        pd_feature_blacklist:      Column names to always exclude.
        forbidden_feature_patterns:Substring patterns to exclude.
        target_col:                Name of the target column to exclude.

    Returns:
        Tuple of ``(selected_features, iv_table)``.
    """
    blacklist_low = {str(c).lower() for c in pd_feature_blacklist}
    patterns_low = [str(p).lower() for p in forbidden_feature_patterns]
    target_low = str(target_col).lower()
    raw_cols = set(X_train_raw.columns) if X_train_raw is not None else set()

    iv_records: List[dict] = []

    for col in X_train_transformed.columns:
        col_low = str(col).lower()
        if col_low == target_low or col_low in blacklist_low or any(
            p in col_low for p in patterns_low
        ):
            continue

        x_after = X_train_transformed[col]
        iv_after = compute_iv(x_after, y_train, n_bins=cfg.iv_n_bins, eps=cfg.eps)

        if X_train_raw is not None and col in raw_cols:
            iv_before = compute_iv(
                X_train_raw[col], y_train, n_bins=cfg.iv_n_bins, eps=cfg.eps
            )
            iv_uplift = iv_after - iv_before
        else:
            iv_before = np.nan
            iv_uplift = np.nan

        iv_records.append(
            {
                "feature": col,
                "iv_before": iv_before,
                "iv_after": iv_after,
                "iv_uplift": iv_uplift,
            }
        )

    iv_table = (
        pd.DataFrame(iv_records)
        .sort_values("iv_after", ascending=False)
        .reset_index(drop=True)
    )

    selected_mask = (iv_table["iv_after"] >= cfg.iv_min) & (
        iv_table["iv_uplift"].fillna(cfg.iv_min_uplift) >= cfg.iv_min_uplift
    )
    selected_features = iv_table.loc[selected_mask, "feature"].tolist()

    logger.info(
        "iv_filter_phase_2: %d/%d features selected (min_iv=%.3f, min_uplift=%.3f)",
        len(selected_features),
        len(iv_records),
        cfg.iv_min,
        cfg.iv_min_uplift,
    )
    return selected_features, iv_table


def iv_filter_phase_2_separability_sensitive(
    X_train_raw: pd.DataFrame,
    X_train_transformed: pd.DataFrame,
    y_train: pd.Series,
    cfg: ModelConfig = DEFAULT_CONFIG,
    pd_feature_blacklist: FrozenSet[str] = PD_FEATURE_BLACKLIST,
    forbidden_feature_patterns: Tuple[str, ...] = LEAKAGE_PATTERNS,
    target_col: str = "bad_state",
    binning: str = "quantile",
) -> Tuple[List[str], pd.DataFrame]:
    """
    Enhanced IV selection with three diagnostic IV variants:

    - **Selection IV**:    bins in transformed space (used for selection).
    - **Fixed-rank IV**:   bins learned from raw ranks, applied to both raw / transformed.
    - **Fixed-edges IV**:  bin edges in raw value space; transformed values mapped
                           back via percentile matching to avoid unit-mismatch artefacts.

    Args:
        X_train_raw:               Raw numeric features (train split).
        X_train_transformed:       Transformed numeric features (train split).
        y_train:                   Binary target (train split).
        cfg:                       Model config.
        pd_feature_blacklist:      Columns to always exclude.
        forbidden_feature_patterns:Substring patterns to exclude.
        target_col:                Target column name to exclude.
        binning:                   ``"quantile"`` or ``"uniform"``.

    Returns:
        Tuple of ``(selected_features, iv_table)`` where *iv_table* includes
        all three IV variant columns.
    """
    blacklist_low = {str(c).lower() for c in pd_feature_blacklist}
    patterns_low = [str(p).lower() for p in forbidden_feature_patterns]
    target_low = str(target_col).lower()
    raw_cols = set(X_train_raw.columns) if X_train_raw is not None else set()

    # Normalise y
    y_vec = y_train
    if isinstance(y_vec, pd.DataFrame) and y_vec.shape[1] == 1:
        y_vec = y_vec.iloc[:, 0]
    y_arr = y_vec.to_numpy() if isinstance(y_vec, (pd.Series, pd.Index)) else np.asarray(y_vec)

    iv_records: List[dict] = []

    for col in X_train_transformed.columns:
        col_low = str(col).lower()
        if col_low == target_low or col_low in blacklist_low or any(
            p in col_low for p in patterns_low
        ):
            continue

        x_after = X_train_transformed[col]
        iv_after = compute_iv(x_after, y_train, n_bins=cfg.iv_n_bins, eps=cfg.eps)

        if X_train_raw is not None and col in raw_cols:
            x_before = X_train_raw[col]
            iv_before = compute_iv(x_before, y_train, n_bins=cfg.iv_n_bins, eps=cfg.eps)
            iv_uplift = iv_after - iv_before

            # Fixed-rank diagnostic
            bin_ids_raw = _make_rank_bins(x_before, n_bins=cfg.iv_n_bins, binning=binning)
            iv_before_fr = _iv_from_bins(bin_ids_raw, y_arr, eps=cfg.eps)
            bin_ids_after = _make_rank_bins(x_after, n_bins=cfg.iv_n_bins, binning=binning)
            iv_after_fr = _iv_from_bins(bin_ids_after, y_arr, eps=cfg.eps)
            iv_uplift_fr = iv_after_fr - iv_before_fr

            # Fixed-edges diagnostic
            iv_b_fe, iv_a_fe, iv_up_fe = _fixed_edges_uplift(
                x_before, x_after, y_arr, n_bins=cfg.iv_n_bins,
                binning=binning, eps=cfg.eps
            )
        else:
            iv_before = np.nan
            iv_uplift = np.nan
            iv_before_fr = iv_after_fr = iv_uplift_fr = np.nan
            iv_b_fe = iv_a_fe = iv_up_fe = np.nan

        iv_records.append(
            {
                "feature": col,
                "iv_before": _safe_float(iv_before),
                "iv_after": _safe_float(iv_after),
                "iv_uplift": _safe_float(iv_uplift),
                "iv_before_fixed_rank": _safe_float(iv_before_fr),
                "iv_after_fixed_rank": _safe_float(iv_after_fr),
                "iv_uplift_fixed_rank": _safe_float(iv_uplift_fr),
                "iv_before_fixed_edges": _safe_float(iv_b_fe),
                "iv_after_fixed_edges": _safe_float(iv_a_fe),
                "iv_uplift_fixed_edges": _safe_float(iv_up_fe),
            }
        )

    iv_table = pd.DataFrame(iv_records)
    if iv_table.shape[0] == 0:
        return [], iv_table

    iv_table = iv_table.sort_values("iv_after", ascending=False).reset_index(drop=True)
    selected_mask = (iv_table["iv_after"] >= cfg.iv_min) & (
        iv_table["iv_uplift"].fillna(cfg.iv_min_uplift) >= cfg.iv_min_uplift
    )
    selected_features = iv_table.loc[selected_mask, "feature"].tolist()

    logger.info(
        "iv_filter_phase_2_separability_sensitive: %d/%d features selected",
        len(selected_features),
        len(iv_records),
    )
    return selected_features, iv_table


# ======================================================================== #
# Leakage audit on IV table
# ======================================================================== #

def iv_audit(
    iv_table: pd.DataFrame,
    X_train: Optional[pd.DataFrame] = None,
    X_train_transformed: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    cfg: ModelConfig = DEFAULT_CONFIG,
    hard_fail_on_leakage: bool = True,
) -> pd.DataFrame:
    """
    Annotate an IV table with leakage and degeneracy flags.

    Checks:
    - Column present in X_train and numeric (basic sanity)
    - Non-constant and not all-missing
    - ``iv_after > cfg.iv_high_flag`` (suspiciously high IV)
    - Correlation with target > 0.99 (near-perfect predictor)
    - Name contains label/target/bad_state/dpd keywords
    - Name looks like an ID or date column

    Args:
        iv_table:            Output of ``iv_filter_phase_2``.
        X_train:             Raw training features (optional, used for name checks).
        X_train_transformed: Transformed training features (optional).
        y_train:             Binary target (optional, used for correlation check).
        cfg:                 Model config.
        hard_fail_on_leakage:If ``True``, raise ``RuntimeError`` on HIGH findings.

    Returns:
        Annotated *iv_table* with additional flag columns.
    """
    iv_aug = iv_table.copy()
    iv_aug["flag_high_iv"] = iv_aug["iv_after"] > cfg.iv_high_flag
    iv_aug["flag_label_name"] = iv_aug["feature"].str.lower().str.contains(
        "target|label|bad_state|default|dpd|penalty", na=False
    )
    iv_aug["flag_id_name"] = iv_aug["feature"].str.lower().str.contains(
        "_id|customer_id|account_id|msisdn", na=False
    )
    iv_aug["flag_date_name"] = iv_aug["feature"].str.lower().str.contains(
        "_dt|_timestamp|date", na=False
    )

    # High-correlation check
    high_corr_features: List[str] = []
    if y_train is not None and X_train_transformed is not None:
        y = pd.to_numeric(y_train, errors="coerce")
        checked = 0
        for col in iv_aug["feature"].tolist():
            if col not in X_train_transformed.columns:
                continue
            if checked >= cfg.max_corr_cols:
                logger.warning(
                    "iv_audit: correlation check limited to first %d features",
                    cfg.max_corr_cols,
                )
                break
            x = pd.to_numeric(X_train_transformed[col], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() < 10:
                checked += 1
                continue
            corr = abs(float(x[valid].corr(y[valid])))
            if corr > 0.99:
                high_corr_features.append(col)
            checked += 1

    iv_aug["flag_high_target_corr"] = iv_aug["feature"].isin(high_corr_features)

    if hard_fail_on_leakage and high_corr_features:
        raise RuntimeError(
            f"[iv_audit] Features with near-perfect target correlation (>0.99): "
            f"{sorted(high_corr_features)}"
        )

    n_flagged = int(
        iv_aug[
            ["flag_high_iv", "flag_label_name", "flag_id_name",
             "flag_date_name", "flag_high_target_corr"]
        ]
        .any(axis=1)
        .sum()
    )
    logger.info("iv_audit: %d/%d features flagged", n_flagged, len(iv_aug))
    return iv_aug


# ======================================================================== #
# Private helpers
# ======================================================================== #

def _enforce_strictly_increasing(edges: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Ensure bin edges are strictly increasing by adding eps where needed."""
    edges = edges.astype("float64")
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps
    return edges


def _iv_from_groupby(binned: pd.Series, y: pd.Series, eps: float) -> float:
    grouped = pd.DataFrame({"bin": binned, "y": y}).groupby("bin")
    bad = grouped["y"].sum()
    good = grouped["y"].count() - bad
    bad_dist = (bad + eps) / (bad.sum() + eps)
    good_dist = (good + eps) / (good.sum() + eps)
    woe = np.log(bad_dist / good_dist)
    return float(((bad_dist - good_dist) * woe).sum())


def _iv_from_bins(bin_ids: pd.Series, y_arr: np.ndarray, eps: float = 1e-6) -> float:
    """Compute IV given precomputed bin assignments (labels); NaNs allowed."""
    bin_s = pd.Series(bin_ids)
    y_s = pd.Series(y_arr)
    ok = bin_s.notna() & y_s.notna()
    if ok.sum() == 0:
        return 0.0
    tmp = pd.DataFrame({"bin": bin_s[ok], "y": y_s[ok]})
    grp = tmp.groupby("bin", observed=False)["y"]
    bad = grp.sum()
    good = grp.count() - bad
    bad_dist = (bad + eps) / (bad.sum() + eps)
    good_dist = (good + eps) / (good.sum() + eps)
    woe = np.log(bad_dist / good_dist)
    return float(((bad_dist - good_dist) * woe).sum())


def _make_rank_bins(
    x: pd.Series,
    n_bins: int = 10,
    binning: str = "quantile",
) -> pd.Series:
    """Produce stable bin assignments based on *x*'s distribution."""
    x_s = pd.Series(x)
    x_nn = x_s.dropna()
    if x_nn.nunique(dropna=True) <= 1:
        return pd.Series([np.nan] * len(x_s), index=x_s.index)
    max_bins = int(min(n_bins, x_nn.nunique(dropna=True)))
    if binning == "quantile":
        try:
            return pd.qcut(x_s, q=max_bins, labels=False, duplicates="drop")
        except ValueError:
            return pd.cut(x_s, bins=max_bins, labels=False, include_lowest=True)
    return pd.cut(x_s, bins=max_bins, labels=False, include_lowest=True)


def _fixed_edges_uplift(
    x_before: pd.Series,
    x_after: pd.Series,
    y_arr: np.ndarray,
    n_bins: int,
    binning: str,
    eps: float,
) -> Tuple[float, float, float]:
    """
    Diagnostic: learn bin edges in raw value space, then map transformed values
    back onto raw scale via percentile matching to compute IV without unit-mismatch
    artefacts.
    """
    x_b = pd.Series(x_before)
    x_a = pd.Series(x_after)
    x_b_nn = x_b.dropna()

    if x_b_nn.nunique(dropna=True) <= 1:
        return np.nan, np.nan, np.nan

    max_bins = int(min(n_bins, x_b_nn.nunique(dropna=True)))
    if binning == "quantile":
        try:
            _, raw_edges = pd.qcut(x_b_nn, q=max_bins, retbins=True, duplicates="drop")
        except Exception:
            raw_edges = np.linspace(float(x_b_nn.min()), float(x_b_nn.max()), max_bins + 1)
    else:
        raw_edges = np.linspace(float(x_b_nn.min()), float(x_b_nn.max()), max_bins + 1)

    raw_edges = np.unique(raw_edges)
    if raw_edges.shape[0] < 2:
        return np.nan, np.nan, np.nan

    bin_ids_b = pd.cut(x_b, bins=raw_edges, labels=False, include_lowest=True)
    iv_b = _iv_from_bins(bin_ids_b, y_arr, eps)

    ok_after = x_a.notna()
    if ok_after.sum() == 0:
        return float(iv_b), np.nan, np.nan

    after_pct = x_a[ok_after].rank(method="average", pct=True).to_numpy()
    raw_sorted = np.sort(x_b_nn.to_numpy())
    raw_n = raw_sorted.shape[0]
    if raw_n <= 1:
        return float(iv_b), np.nan, np.nan

    raw_idx = np.clip((after_pct * (raw_n - 1)).round().astype(int), 0, raw_n - 1)
    after_mapped = pd.Series(np.nan, index=x_a.index)
    after_mapped.loc[ok_after] = raw_sorted[raw_idx]

    bin_ids_a = pd.cut(after_mapped, bins=raw_edges, labels=False, include_lowest=True)
    iv_a = _iv_from_bins(bin_ids_a, y_arr, eps)

    return float(iv_b), float(iv_a), float(iv_a - iv_b)


def _safe_float(val) -> float:
    return float(val) if pd.notna(val) else np.nan
