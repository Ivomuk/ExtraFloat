"""
extrafloat_segmentation_features.py
=====================================
Feature engineering pipeline for the MTN MoMo Uganda agent segmentation model.

Consolidates raw KPI mart data into a clean, scaled, PCA-reduced feature matrix
suitable for K-Means or GMM clustering.

Refactors cl_file1.txt + cl_file2.txt.

Market: Uganda (UG) — MTN Mobile Money.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# COMMISSION / SUPER-AGENT MSISDN EXCLUSION LIST
# ─────────────────────────────────────────────────────────────────────────────

COMMISSION_AGENTS_MSISDN: frozenset[int] = frozenset(
    {256789760000, 256783891111, 256781872222, 256772453333}
)

# ─────────────────────────────────────────────────────────────────────────────
# DATE COLUMN NAMES
# ─────────────────────────────────────────────────────────────────────────────

DATE_COLUMNS: tuple[str, ...] = (
    "tbl_dt",
    "activation_dt",
    "date_of_birth",
    "payment_last",
    "cash_in_last",
    "cash_out_last",
)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMNS EXCLUDED FROM THE NUMERIC FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

COLUMNS_TO_EXCLUDE_FROM_FEATURES: frozenset[str] = frozenset(
    {
        "account_name",
        "account_number",
        "currency",
        "firstname",
        "lastname",
        "ncr_number",
        "address",
        "agent_profile",
        "gender",
        "region_simple",
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMNS THAT ALWAYS RECEIVE log1p TRANSFORM (known right-skewed financials)
# ─────────────────────────────────────────────────────────────────────────────

LOG1P_COLS: tuple[str, ...] = (
    "account_balance",
    "average_balance",
    "commission",
    "cash_in_value_1m",
    "cash_out_value_1m",
    "payment_value_1m",
    "total_peers_6m",
    "total_customers_6m",
)

# ─────────────────────────────────────────────────────────────────────────────
# MINIMUM REQUIRED COLUMNS
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS: tuple[str, ...] = (
    "agent_msisdn",
    "commission",
    "cash_out_vol_1m",
    "cash_out_value_1m",
    "cash_in_value_1m",
)

# Share-like columns excluded from log/winsorize (already in [0,1])
_SHARE_LIKE_COLS: frozenset[str] = frozenset(
    {
        "share_cash_in_value_1m",
        "share_cash_out_value_1m",
        "share_voucher_value_1m",
        "share_payment_value_1m",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _get_features_config(config: dict | None) -> dict:
    """Merge caller config with production defaults."""
    defaults: dict = {
        "date_format": "%Y%m%d",
        "recency_cap_days": 3650,
        "skew_threshold": 1.0,
        "winsorize_lower_pct": 0.005,
        "winsorize_upper_pct": 0.995,
        "corr_threshold": 0.97,
        "target_pca_variance": 0.90,
        "random_state": 42,
        "max_interaction_cols": 10,
    }
    if config:
        defaults.update(config)
    return defaults


def _validate_input(df: pd.DataFrame, cfg: dict) -> None:  # noqa: ARG001
    """Raise ValueError if required columns are absent."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"prepare_features: input DataFrame is missing required columns: "
            f"{', '.join(missing)}"
        )


def _basic_prep(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Cast IDs, parse dates, clean strings, derive region_simple, filter commission agents.

    Parameters
    ----------
    df  : Raw agent DataFrame.
    cfg : Feature configuration dict.

    Returns
    -------
    Cleaned DataFrame (copy).
    """
    out = df.copy()
    date_fmt: str = cfg["date_format"]

    # Cast ID columns to str
    for c in ("agent_msisdn", "pos_msisdn"):
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip().str.split(".").str[0]

    # Filter commission / super-agent rows
    if "agent_msisdn" in out.columns:
        numeric_msisdn = pd.to_numeric(out["agent_msisdn"], errors="coerce")
        mask_commission = numeric_msisdn.isin(COMMISSION_AGENTS_MSISDN)
        n_dropped = int(mask_commission.sum())
        if n_dropped:
            logger.info("_basic_prep: dropping %d commission-agent rows", n_dropped)
        out = out[~mask_commission].reset_index(drop=True)

    # Parse tbl_dt
    if "tbl_dt" in out.columns:
        out["tbl_dt"] = pd.to_datetime(out["tbl_dt"].astype(str), errors="coerce")

    # Parse activation_dt (strip ".0" artefacts first)
    if "activation_dt" in out.columns:
        out["activation_dt"] = (
            out["activation_dt"]
            .astype(str)
            .str.split(".")
            .str[0]
            .replace({"": pd.NA, "None": pd.NA, "nan": pd.NA})
        )
        out["activation_dt"] = pd.to_datetime(
            out["activation_dt"].astype(str),
            format=date_fmt,
            errors="coerce",
        )

    # Parse remaining date columns
    for c in ("date_of_birth", "payment_last", "cash_in_last", "cash_out_last"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c].astype(str), errors="coerce")

    # Strip whitespace / blank -> NaN for object columns
    obj_cols = out.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        out[c] = (
            out[c]
            .astype(str)
            .str.strip()
            .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        )

    # Derive simple region label from address
    if "address" in out.columns:
        addr = out["address"].astype(str).str.upper()
        addr = addr.replace({"": np.nan, "-": np.nan, "NAN": np.nan})
        region = np.where(addr.str.contains("KAMPALA", na=False), "KAMPALA", addr)
        region = np.where(
            pd.Series(region).str.contains("UGANDA", na=False), "UGANDA", region
        )
        out["region_simple"] = pd.Series(region, index=out.index).fillna("OTHERS")

    logger.info("_basic_prep: %d rows after commission-agent filter", len(out))
    return out


def _engineer_date_features(
    df: pd.DataFrame,
    ref_date: pd.Timestamp,
    cfg: dict,
) -> pd.DataFrame:
    """Compute age, tenure, recency columns, one-hot encode categorical fields.

    Parameters
    ----------
    df       : DataFrame after _basic_prep.
    ref_date : Reference date for calculations.
    cfg      : Feature configuration dict.

    Returns
    -------
    DataFrame with all date-derived features added (copy).
    """
    out = df.copy()
    cap: int = cfg["recency_cap_days"]
    old_date = pd.Timestamp("1900-01-01")

    # Age
    if "date_of_birth" in out.columns:
        out["age"] = ((ref_date - out["date_of_birth"]).dt.days / 365.25).clip(lower=0)
        out["age"] = out["age"].fillna(out["age"].median())

    # NCR flag
    if "ncr_number" in out.columns:
        out["has_ncr"] = out["ncr_number"].notna().astype(int)
    else:
        out["has_ncr"] = 0

    # Activation missing flag + tenure
    if "activation_dt" in out.columns:
        out["activation_missing"] = out["activation_dt"].isna().astype(int)
        median_activation = out["activation_dt"].median()
        if pd.isna(median_activation):
            median_activation = pd.Timestamp("2015-01-01")
        out["activation_dt"] = out["activation_dt"].fillna(median_activation)
        out["tenure_years"] = (
            (ref_date - out["activation_dt"]).dt.days / 365.25
        ).clip(lower=0)
    else:
        out["activation_missing"] = 1

    # Clean profile / gender for one-hot encoding
    if "agent_profile" in out.columns:
        out["agent_profile_clean"] = (
            out["agent_profile"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": np.nan})
        )
    if "gender" in out.columns:
        out["gender_clean"] = (
            out["gender"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": np.nan})
        )

    # Days since last activity
    recency_computed: list[str] = []
    for c in ("payment_last", "cash_in_last", "cash_out_last"):
        ds_col = f"days_since_{c}"
        if c in out.columns:
            out[ds_col] = (
                (ref_date - out[c].fillna(old_date)).dt.days.fillna(cap).clip(upper=cap)
            )
            out[f"{c}_ever"] = out[c].notna().astype(int)
        else:
            out[ds_col] = float(cap)
            out[f"{c}_ever"] = 0
        recency_computed.append(ds_col)

    # Recency min
    out["recency_min"] = (
        out[recency_computed].min(axis=1).fillna(cap).clip(upper=cap)
    )

    # Most recent action one-hot
    _last_cols = [c for c in ("payment_last", "cash_in_last", "cash_out_last") if c in out.columns]
    if _last_cols:
        temp = out[_last_cols].fillna(old_date)
        out["most_recent_action"] = temp.idxmax(axis=1)
        all_missing = out[_last_cols].isna().all(axis=1)
        out.loc[all_missing, "most_recent_action"] = "none"
        out = pd.get_dummies(
            out, columns=["most_recent_action"], prefix="last_action", dtype=int
        )

    # Gap features
    def _safe_gap(col_a: str, col_b: str) -> pd.Series:
        if col_a not in out.columns or col_b not in out.columns:
            return pd.Series(float(cap), index=out.index, dtype="float64")
        return (out[col_a] - out[col_b]).dt.days.abs().fillna(cap).clip(upper=cap)

    out["gap_cash_in_payment"] = _safe_gap("payment_last", "cash_in_last")
    out["gap_cash_out_payment"] = _safe_gap("payment_last", "cash_out_last")

    # One-hot encode gender_clean / agent_profile_clean
    for col in ("gender_clean", "agent_profile_clean"):
        if col in out.columns:
            out[col] = out[col].fillna("unknown")
            out = pd.get_dummies(out, columns=[col], prefix=col, dtype=int)

    # Drop raw date and excluded columns
    drop_raw = list(DATE_COLUMNS) + list(COLUMNS_TO_EXCLUDE_FROM_FEATURES)
    out = out.drop(columns=[c for c in drop_raw if c in out.columns], errors="ignore")

    # Drop fully-NA and constant columns (preserve ID columns)
    protected = {"agent_msisdn", "pos_msisdn"}
    na_full_cols = out.columns[out.isna().all()].tolist()
    constant_cols = [
        c
        for c in out.columns
        if out[c].nunique(dropna=True) <= 1 and c not in protected
    ]
    cols_to_drop = sorted(set(na_full_cols + constant_cols))
    if cols_to_drop:
        logger.debug(
            "_engineer_date_features: dropping %d all-NA/constant cols",
            len(cols_to_drop),
        )
    out = out.drop(columns=cols_to_drop, errors="ignore")

    return out


def _add_premium_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived premium features: value aggregates, commission efficiency,
    momentum ratios, product-mix shares, and network density.

    Parameters
    ----------
    df : DataFrame after date feature engineering.

    Returns
    -------
    DataFrame with premium features appended (copy).
    """
    out = df.copy()
    eps = 1e-9

    def _col(name: str) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=out.index, dtype="float64")

    # Total value 3m / 6m
    out["total_value_3m"] = (
        _col("cash_in_value_3m") + _col("cash_out_value_3m")
        + _col("voucher_value_3m") + _col("payment_value_3m")
    )
    out["total_value_6m"] = (
        _col("cash_in_value_6m") + _col("cash_out_value_6m")
        + _col("voucher_value_6m") + _col("payment_value_6m")
    )

    # Commission per value (profitability)
    total_comm_3m = (
        _col("cash_in_comm_3m") + _col("cash_out_comm_3m")
        + _col("voucher_comm_3m") + _col("payment_comm_3m")
    )
    total_comm_6m = (
        _col("cash_in_comm_6m") + _col("cash_out_comm_6m")
        + _col("voucher_comm_6m") + _col("payment_comm_6m")
    )
    out["commission_per_value_3m"] = total_comm_3m / (out["total_value_3m"] + eps)
    out["commission_per_value_6m"] = total_comm_6m / (out["total_value_6m"] + eps)

    # Momentum: 1m vs 3m
    out["vol_1m_to_3m_ratio"] = _col("vol_1m") / (_col("vol_3m") + eps)
    out["cash_out_vol_1m_to_3m_ratio"] = _col("cash_out_vol_1m") / (_col("cash_out_vol_3m") + eps)
    out["payment_vol_1m_to_3m_ratio"] = _col("payment_vol_1m") / (_col("payment_vol_3m") + eps)

    # Product-mix shares (1m value)
    total_value_1m = (
        _col("cash_in_value_1m") + _col("cash_out_value_1m")
        + _col("voucher_value_1m") + _col("payment_value_1m")
    )
    out["share_cash_in_value_1m"] = _col("cash_in_value_1m") / (total_value_1m + eps)
    out["share_cash_out_value_1m"] = _col("cash_out_value_1m") / (total_value_1m + eps)
    out["share_voucher_value_1m"] = _col("voucher_value_1m") / (total_value_1m + eps)
    out["share_payment_value_1m"] = _col("payment_value_1m") / (total_value_1m + eps)

    # Network richness 6m
    out["total_peers_6m"] = (
        _col("cash_in_peers_6m") + _col("cash_out_peers_6m") + _col("payment_peers_6m")
    )
    out["total_customers_6m"] = (
        _col("cash_in_cust_6m") + _col("cash_out_cust_6m") + _col("payment_cust_6m")
    )
    out["peers_per_value_6m"] = out["total_peers_6m"] / (out["total_value_6m"] + eps)
    out["customers_per_value_6m"] = out["total_customers_6m"] / (out["total_value_6m"] + eps)

    return out


def _add_enhanced_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add ratio features, recency bins, and pairwise interaction terms.

    Parameters
    ----------
    df  : DataFrame after _add_premium_features.
    cfg : Feature configuration dict.

    Returns
    -------
    DataFrame with enhanced features appended (copy).
    """
    out = df.copy()
    max_interactions: int = cfg["max_interaction_cols"]
    horizons = ("7", "30", "90", "1m", "3m", "6m")

    num_cols_all = out.select_dtypes(include="number").columns.tolist()

    # A1: Ratio features (value/vol per count within matching horizon)
    vol_cols = [c for c in num_cols_all if "value" in c.lower() or "vol" in c.lower()]
    cnt_cols = [c for c in num_cols_all if "count" in c.lower() or "txn" in c.lower()]
    for vcol in vol_cols:
        for ccol in cnt_cols:
            if any(h in vcol.lower() and h in ccol.lower() for h in horizons):
                denom = out[ccol].replace({0: np.nan})
                out[f"{vcol}_per_{ccol}"] = (
                    (out[vcol] / denom)
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )

    # A2: Recency bins
    recency_cols = [c for c in out.columns if c.startswith("days_since_")]
    for rc in recency_cols:
        out[f"{rc}_bin_0_7"] = (out[rc] <= 7).astype(int)
        out[f"{rc}_bin_8_30"] = ((out[rc] > 7) & (out[rc] <= 30)).astype(int)
        out[f"{rc}_bin_31_90"] = ((out[rc] > 30) & (out[rc] <= 90)).astype(int)
        out[f"{rc}_bin_gt_90"] = (out[rc] > 90).astype(int)

    # A3: Interaction terms (limited)
    num_cols_current = out.select_dtypes(include="number").columns.tolist()
    key_num_cols = [
        c for c in num_cols_current
        if any(k in c.lower() for k in ("payment", "cash_in", "cash_out", "commission"))
        and not c.startswith("days_since_")
    ][:max_interactions]

    for i in range(len(key_num_cols)):
        for j in range(i + 1, len(key_num_cols)):
            c1, c2 = key_num_cols[i], key_num_cols[j]
            out[f"{c1}_times_{c2}"] = out[c1] * out[c2]

    logger.info(
        "_add_enhanced_features: feature shape after enhancement = %s", out.shape
    )
    return out


def _apply_log_winsorize(X_out: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply log1p transform then winsorize right-skewed columns.

    Parameters
    ----------
    X_out : Numeric feature DataFrame.
    cfg   : Feature configuration dict.

    Returns
    -------
    Transformed DataFrame (copy).
    """
    out = X_out.copy()
    skew_threshold: float = cfg["skew_threshold"]
    lower_pct: float = cfg["winsorize_lower_pct"]
    upper_pct: float = cfg["winsorize_upper_pct"]

    skewness = out.skew(numeric_only=True)
    auto_skewed = [
        col for col in out.columns
        if col not in _SHARE_LIKE_COLS
        and skewness.get(col, 0) > skew_threshold
        and (out[col] >= 0).all()
    ]
    forced_cols = [
        c for c in LOG1P_COLS if c in out.columns and (out[c] >= 0).all()
    ]
    transform_cols = sorted(set(auto_skewed) | set(forced_cols))

    logger.info(
        "_apply_log_winsorize: transforming %d columns", len(transform_cols)
    )

    for col in transform_cols:
        out[col] = np.log1p(out[col])
        out[col] = out[col].clip(
            lower=out[col].quantile(lower_pct),
            upper=out[col].quantile(upper_pct),
        )

    return out


def _prune_correlated_features(
    X_out: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, list[str]]:
    """Remove highly correlated features and noisy one-hot groups.

    Parameters
    ----------
    X_out : Numeric feature DataFrame.
    cfg   : Feature configuration dict.

    Returns
    -------
    (pruned_df, selected_feature_names)
    """
    corr_threshold: float = cfg["corr_threshold"]

    corr_matrix = X_out.corr().abs()
    to_drop: set[str] = set()
    visited: set[str] = set()

    for col in corr_matrix.columns:
        if col in visited:
            continue
        high_corr = corr_matrix.index[
            (corr_matrix[col] >= corr_threshold) & (corr_matrix.index != col)
        ].tolist()
        visited.add(col)
        if high_corr:
            visited.update(high_corr)
            to_drop.update(high_corr)

    # Drop noisy one-hot groups and age
    for col in list(X_out.columns):
        if (
            col.startswith("agent_profile_clean_")
            or col.startswith("gender_clean_")
            or col == "age"
        ):
            to_drop.add(col)

    selected = [c for c in X_out.columns if c not in to_drop]
    logger.info(
        "_prune_correlated_features: %d -> %d features (%d dropped)",
        len(X_out.columns),
        len(selected),
        len(to_drop),
    )
    return X_out[selected].copy(), selected


def _scale_and_reduce(
    X_out: pd.DataFrame, cfg: dict
) -> tuple[np.ndarray, np.ndarray]:
    """RobustScaler + PCA with variance-explained stopping criterion.

    Parameters
    ----------
    X_out : Pruned numeric feature DataFrame (no missing values).
    cfg   : Feature configuration dict.

    Returns
    -------
    (X_scaled, X_pca)
    """
    target_variance: float = cfg["target_pca_variance"]
    random_state: int = cfg["random_state"]

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_out.astype(float))

    pca_full = PCA(random_state=random_state)
    pca_full.fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cum_var >= target_variance) + 1)

    logger.info(
        "_scale_and_reduce: PCA n_components=%d (%.0f%% variance target)",
        n_components,
        target_variance * 100,
    )

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_state,
    )
    X_pca = pca.fit_transform(X_scaled)

    return X_scaled, X_pca


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────


def prepare_features(
    df: pd.DataFrame,
    config: dict | None = None,
    ref_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Full feature engineering pipeline for agent segmentation.

    Stages:
    1. Config resolution and input validation
    2. Basic prep: ID casting, date parsing, string cleaning, commission-agent filter
    3. Date/recency features: age, tenure, days-since, one-hots
    4. Premium features: value aggregates, commission efficiency, network density
    5. Enhanced features: ratios, recency bins, interaction terms
    6. Numeric matrix extraction with median imputation
    7. Log1p transform + winsorization for skewed columns
    8. Correlation-based feature pruning (drops agent_profile/gender one-hots)
    9. RobustScaler + PCA

    Parameters
    ----------
    df       : Raw agent DataFrame from the KPI mart.
    config   : Feature configuration dict. Missing keys fall back to defaults.
    ref_date : Reference date for tenure/recency. Defaults to max(tbl_dt).

    Returns
    -------
    (features_df_raw, X_scaled, X_pca, selected_feature_names)
        features_df_raw        : DataFrame after all feature engineering (pre-scaling).
        X_scaled               : RobustScaler-transformed array (n_agents, n_features).
        X_pca                  : PCA-reduced array (n_agents, n_components).
        selected_feature_names : Feature column names in X_scaled / X_pca column order.
    """
    cfg = _get_features_config(config)
    _validate_input(df, cfg)

    logger.info("prepare_features: starting pipeline on %d rows", len(df))

    # Stage 1: basic prep (includes commission-agent filter)
    prepped = _basic_prep(df, cfg)

    # Resolve reference date
    if ref_date is None:
        if "tbl_dt" in prepped.columns:
            ref_date = pd.to_datetime(prepped["tbl_dt"].max())
        else:
            ref_date = pd.Timestamp.now().normalize()
    logger.info("prepare_features: ref_date = %s", ref_date.date())

    # Stage 2: date + recency features
    with_dates = _engineer_date_features(prepped, ref_date, cfg)

    # Stage 3: premium features
    with_premium = _add_premium_features(with_dates)

    # Stage 4: enhanced features
    features_df_raw = _add_enhanced_features(with_premium, cfg)

    # Stage 5: build numeric matrix
    id_cols = ["agent_msisdn", "pos_msisdn"]
    numeric_cols = features_df_raw.select_dtypes(include="number").columns.tolist()
    for c in id_cols:
        if c in numeric_cols:
            numeric_cols.remove(c)

    X_raw = features_df_raw[numeric_cols].copy()
    medians = X_raw.median()
    X_imputed = X_raw.fillna(medians).fillna(0.0)

    # Stage 6: log1p + winsorize
    X_transformed = _apply_log_winsorize(X_imputed, cfg)

    # Stage 7: correlation pruning
    X_pruned, selected_feature_names = _prune_correlated_features(X_transformed, cfg)

    # Safety re-impute after pruning
    X_pruned = X_pruned.fillna(X_pruned.median()).fillna(0.0)

    # Stage 8: scale + PCA
    X_scaled, X_pca = _scale_and_reduce(X_pruned, cfg)

    logger.info(
        "prepare_features: done — %d agents, %d features, %d PCA components",
        X_scaled.shape[0],
        X_scaled.shape[1],
        X_pca.shape[1],
    )

    return features_df_raw, X_scaled, X_pca, selected_feature_names
