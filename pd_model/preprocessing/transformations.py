"""
Feature classification and transformation for PD modelling (Phase 2 / file4).

Provides:
- ``get_and_classify_pd_features``  – extract numeric candidates and classify into
                                      LOG, SIGNED_LOG, CAP, or PROTECTED buckets.
- ``apply_pd_transformations``       – apply bucket-appropriate transforms with
                                      automatic reversion on failure.
- ``prune_post_transform_features``  – drop degenerate (all-NaN / near-constant)
                                      features after transformation.
"""

from __future__ import annotations

from typing import FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd

from pd_model.config.feature_config import (
    CAP_ONLY_PATTERNS,
    COUNT_PATTERNS,
    DPD_ALLOW_PATTERNS,
    DPD_BLOCK_PATTERNS,
    ID_LIKE_PATTERNS,
    LEAKAGE_PATTERNS,
    LOG_PATTERNS,
    PD_FEATURE_BLACKLIST,
    PROTECTED_PATTERNS,
    SIGNED_AMOUNT_PATTERNS,
)
from pd_model.config.model_config import DEFAULT_CONFIG, ModelConfig
from pd_model.logging_config import get_logger
from pd_model.validation.schema import assert_index_aligned

logger = get_logger(__name__)


# ======================================================================== #
# Feature classification
# ======================================================================== #

def get_and_classify_pd_features(
    df_pd: pd.DataFrame,
    blacklist: FrozenSet[str] = PD_FEATURE_BLACKLIST,
    debug: bool = False,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], pd.DataFrame]:
    """
    Extract numeric PD candidate features and classify them into transformation
    buckets.

    Classification precedence (first match wins):
    1. **Blacklist** / **ID-like** / **Leakage** / **DPD-ambiguous** → excluded
    2. ``PROTECTED``     – flags, indicators, binary/ordinal; left raw
    3. ``CAP ONLY``      – ratios, shares, growth rates; winsorize only
    4. ``COUNT``         – volume counts; winsorize only (mapped to cap)
    5. ``SIGNED LOG``    – net flows, deltas; sign(x)*log1p(|x|) + winsorize
    6. ``LOG``           – amounts, balances, commissions; log1p + winsorize
    7. Default          – PROTECTED (if none of the above matched)

    Args:
        df_pd:     Modelling DataFrame.
        blacklist: Set of column names to always exclude.
        debug:     If ``True``, logs which pattern matched each column.

    Returns:
        Tuple of:
        - ``pd_features``     – full list of accepted numeric candidates
        - ``log_cols``        – LOG + CAP columns
        - ``cap_cols``        – CAP ONLY columns
        - ``protected_cols``  – PROTECTED (left raw) columns
        - ``signed_log_cols`` – SIGNED LOG + CAP columns
        - ``excluded_df``     – DataFrame of excluded columns with reasons
    """
    blacklist_lower = {str(c).strip().lower() for c in blacklist}

    pd_features: List[str] = []
    excluded: List[Tuple[str, str]] = []

    for c in df_pd.select_dtypes(include=["number"]).columns:
        c_clean = str(c).strip()
        c_low = c_clean.lower()

        if c_low in blacklist_lower:
            excluded.append((c_clean, "blacklist"))
            continue

        if any(p in c_low for p in ID_LIKE_PATTERNS):
            excluded.append((c_clean, "id_like"))
            continue

        if "dpd" in c_low:
            if any(p in c_low for p in DPD_ALLOW_PATTERNS):
                pd_features.append(c_clean)
                if debug:
                    logger.debug("%s -> ALLOWED DPD", c_clean)
                continue
            if any(p in c_low for p in DPD_BLOCK_PATTERNS):
                excluded.append((c_clean, "dpd_leakage_like"))
                continue
            excluded.append((c_clean, "dpd_ambiguous"))
            continue

        if any(p in c_low for p in LEAKAGE_PATTERNS):
            excluded.append((c_clean, "leakage_like"))
            continue

        pd_features.append(c_clean)

    # Hard DPD guard
    leaked_dpd = [
        c
        for c in pd_features
        if "dpd" in c.lower()
        and not any(p in c.lower() for p in DPD_ALLOW_PATTERNS)
    ]
    assert not leaked_dpd, (
        "[get_and_classify_pd_features] DPD-like columns in PD candidates: "
        + str(sorted(leaked_dpd))
    )

    # ------------------------------------------------------------------ #
    # Classify into transformation buckets
    # ------------------------------------------------------------------ #
    log_cols: set = set()
    signed_log_cols: set = set()
    cap_cols: set = set()
    protected_cols: set = set()

    for c in pd_features:
        c_low = c.lower()

        if any(p in c_low for p in PROTECTED_PATTERNS):
            protected_cols.add(c)
            matched = "PROTECTED (leave raw)"
        elif any(p in c_low for p in CAP_ONLY_PATTERNS):
            cap_cols.add(c)
            matched = "CAP ONLY (ratio/shape)"
        elif any(p in c_low for p in COUNT_PATTERNS):
            cap_cols.add(c)
            matched = "CAP ONLY (count/gated)"
        elif any(p in c_low for p in SIGNED_AMOUNT_PATTERNS):
            signed_log_cols.add(c)
            matched = "SIGNED LOG + CAP (signed amount)"
        elif any(p in c_low for p in LOG_PATTERNS):
            log_cols.add(c)
            matched = "LOG + CAP (amount)"
        elif c_low.startswith("cash_in") or c_low.startswith("cash_out"):
            cap_cols.add(c)
            matched = "CAP ONLY (auto cash_in/out)"
        else:
            protected_cols.add(c)
            matched = "DEFAULT PROTECTED"

        if debug:
            logger.debug("%-40s -> %s", c, matched)

    logger.info(
        "get_and_classify_pd_features: %d candidates → LOG=%d, SIGNED_LOG=%d, "
        "CAP=%d, PROTECTED=%d, excluded=%d",
        len(pd_features),
        len(log_cols),
        len(signed_log_cols),
        len(cap_cols),
        len(protected_cols),
        len(excluded),
    )

    return (
        pd_features,
        sorted(log_cols),
        sorted(cap_cols),
        sorted(protected_cols),
        sorted(signed_log_cols),
        pd.DataFrame(excluded, columns=["feature", "reason"]),
    )


# ======================================================================== #
# Transformations
# ======================================================================== #

def apply_pd_transformations(
    df: pd.DataFrame,
    pd_features: List[str],
    log_cols: List[str],
    cap_cols: List[str],
    signed_log_cols: Optional[List[str]] = None,
    cfg: ModelConfig = DEFAULT_CONFIG,
    neg_policy: str = "signed_log1p",
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Apply PD-safe feature transformations with automatic reversion on failure.

    Transformation rules
    --------------------
    - **SIGNED LOG + CAP** (``signed_log_cols``):
      ``sign(x) * log1p(|x|)``, then winsorize.
    - **LOG + CAP** (``log_cols``):
      If ``neg_frac > cfg.neg_frac_threshold``, apply signed log (or cap only
      if ``neg_policy == "cap_only"``); else ``log1p(clip(x, 0))``.
    - **CAP ONLY** (``cap_cols``):
      Winsorize without any log transform.
    - **PROTECTED** (columns in ``pd_features`` but none of the above):
      Left unchanged.

    If a transformed column fails the ``_is_broken`` check (< ``cfg.finite_frac_min``
    finite values, or <= 1 unique value), it is reverted to raw cap and a WARNING
    is emitted.

    All numeric outputs are cast to ``float32`` for memory efficiency.

    Args:
        df:                   Full modelling DataFrame.
        pd_features:          Candidate numeric features to transform.
        log_cols:             Columns to log-transform + winsorize.
        cap_cols:             Columns to winsorize only.
        signed_log_cols:      Columns forced into signed-log + winsorize.
        cfg:                  Model config for winsorization quantiles, eps, thresholds.
        neg_policy:           ``"signed_log1p"`` (default) or ``"cap_only"`` for
                              log columns with >``neg_frac_threshold`` negatives.

    Returns:
        Tuple of:
        - transformed ``df_out``
        - list of unclassified (left raw) feature names
        - ``transform_report`` DataFrame
    """
    q_low = cfg.winsor_q_low
    q_high = cfg.winsor_q_high

    df_out = df.copy()
    pd_features_use = [c for c in pd_features if c in df_out.columns]

    signed_log_cols_use: List[str] = []
    if signed_log_cols:
        signed_log_cols_use = [
            c for c in signed_log_cols if c in df_out.columns and c in pd_features_use
        ]

    log_cols_use = [
        c for c in log_cols if c in df_out.columns and c in pd_features_use
    ]
    cap_cols_use = [
        c for c in cap_cols if c in df_out.columns and c in pd_features_use
    ]

    # Ensure disjoint processing order
    signed_set = set(signed_log_cols_use)
    log_cols_use = [c for c in log_cols_use if c not in signed_set]
    cap_cols_use = [
        c for c in cap_cols_use if c not in set(log_cols_use) and c not in signed_set
    ]

    report_rows: List[dict] = []

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _winsorize(s: pd.Series) -> Tuple[pd.Series, bool, float, float]:
        lo_val, hi_val = s.quantile([q_low, q_high])
        if pd.isna(lo_val) or pd.isna(hi_val) or float(hi_val) <= float(lo_val):
            return s, True, float(lo_val), float(hi_val)
        return s.clip(lo_val, hi_val), False, float(lo_val), float(hi_val)

    def _is_broken(s: pd.Series) -> Tuple[bool, str]:
        vals = pd.to_numeric(s, errors="coerce")
        finite_frac = float(np.isfinite(vals.to_numpy()).mean())
        nun = int(vals.dropna().nunique())
        if finite_frac < cfg.finite_frac_min:
            return True, f"low_finite_frac={finite_frac:.3f}"
        if nun <= 1:
            return True, f"nunique={nun}"
        return False, "ok"

    def _signed_log1p(s: pd.Series) -> pd.Series:
        s_num = pd.to_numeric(s, errors="coerce")
        return np.sign(s_num) * np.log1p(np.abs(s_num))

    # ------------------------------------------------------------------ #
    # 0) FORCED SIGNED LOG + CAP
    # ------------------------------------------------------------------ #
    for col in signed_log_cols_use:
        s_raw = pd.to_numeric(df_out[col], errors="coerce")
        n_nonnull = int(s_raw.notna().sum())
        if n_nonnull == 0:
            df_out[col] = s_raw.astype("float32")
            report_rows.append({"feature": col, "action": "forced_signed_log_skip_all_nan"})
            continue

        s_nonnull = s_raw.dropna()
        neg_frac = float((s_nonnull < 0).mean()) if len(s_nonnull) > 0 else 0.0

        s_work = _signed_log1p(s_raw)
        s_work, skipped, lo, hi = _winsorize(s_work)
        broken, why = _is_broken(s_work)
        if broken:
            s_fb, _, _, _ = _winsorize(s_raw)
            df_out[col] = s_fb.astype("float32")
            logger.warning("Feature '%s' forced-signed-log FAILED (%s); reverted to raw cap", col, why)
            report_rows.append({"feature": col, "action": "forced_signed_log_reverted_raw_cap", "reason": why, "neg_frac": neg_frac})
            continue

        df_out[col] = s_work.astype("float32")
        report_rows.append({"feature": col, "action": "forced_signed_log_ok", "neg_frac": neg_frac})

    # ------------------------------------------------------------------ #
    # 1) LOG + CAP
    # ------------------------------------------------------------------ #
    for col in log_cols_use:
        s_raw = pd.to_numeric(df_out[col], errors="coerce")
        n_nonnull = int(s_raw.notna().sum())
        if n_nonnull == 0:
            df_out[col] = s_raw.astype("float32")
            report_rows.append({"feature": col, "action": "log_cap_skip_all_nan"})
            continue

        s_nonnull = s_raw.dropna()
        neg_frac = float((s_nonnull < 0).mean()) if len(s_nonnull) > 0 else 0.0

        if neg_frac > cfg.neg_frac_threshold:
            if neg_policy == "cap_only":
                s_work, skipped, lo, hi = _winsorize(s_raw)
                df_out[col] = s_work.astype("float32")
                report_rows.append({"feature": col, "action": "cap_only_due_to_negatives", "neg_frac": neg_frac})
                continue

            s_work = _signed_log1p(s_raw)
            s_work, skipped, lo, hi = _winsorize(s_work)
            broken, why = _is_broken(s_work)
            if broken:
                s_fb, _, _, _ = _winsorize(s_raw)
                df_out[col] = s_fb.astype("float32")
                logger.warning("Feature '%s' signed-log FAILED (%s); reverted to raw cap", col, why)
                report_rows.append({"feature": col, "action": "signed_log_reverted_raw_cap", "reason": why, "neg_frac": neg_frac})
                continue

            df_out[col] = s_work.astype("float32")
            report_rows.append({"feature": col, "action": "signed_log_ok", "neg_frac": neg_frac})
            continue

        s_work = np.log1p(s_raw.clip(lower=0))
        s_work, skipped, lo, hi = _winsorize(s_work)
        broken, why = _is_broken(s_work)
        if broken:
            s_fb, _, _, _ = _winsorize(s_raw)
            df_out[col] = s_fb.astype("float32")
            logger.warning("Feature '%s' log-cap FAILED (%s); reverted to raw cap", col, why)
            report_rows.append({"feature": col, "action": "log_cap_reverted_raw_cap", "reason": why, "neg_frac": neg_frac})
            continue

        df_out[col] = s_work.astype("float32")
        report_rows.append({"feature": col, "action": "log_cap_ok", "neg_frac": neg_frac})

    # ------------------------------------------------------------------ #
    # 2) CAP ONLY
    # ------------------------------------------------------------------ #
    for col in cap_cols_use:
        s_raw = pd.to_numeric(df_out[col], errors="coerce")
        n_nonnull = int(s_raw.notna().sum())
        if n_nonnull == 0:
            df_out[col] = s_raw.astype("float32")
            report_rows.append({"feature": col, "action": "cap_skip_all_nan"})
            continue

        s_work, skipped, lo, hi = _winsorize(s_raw)
        broken, why = _is_broken(s_work)
        if broken:
            df_out[col] = s_raw.astype("float32")
            logger.warning("Feature '%s' cap FAILED (%s); left raw", col, why)
            report_rows.append({"feature": col, "action": "cap_reverted_raw", "reason": why})
            continue

        df_out[col] = s_work.astype("float32")
        report_rows.append({"feature": col, "action": "cap_ok"})

    classified = signed_set | set(log_cols_use) | set(cap_cols_use)
    unclassified = [c for c in pd_features_use if c not in classified]
    transform_report = pd.DataFrame(report_rows)

    logger.info(
        "apply_pd_transformations: %d features processed "
        "(signed_log=%d, log=%d, cap=%d, unclassified=%d)",
        len(pd_features_use),
        len(signed_log_cols_use),
        len(log_cols_use),
        len(cap_cols_use),
        len(unclassified),
    )
    if transform_report.shape[0] > 0:
        action_counts = transform_report["action"].value_counts(dropna=False)
        reverted = action_counts[action_counts.index.str.contains("reverted", na=False)].sum()
        if reverted > 0:
            logger.warning(
                "apply_pd_transformations: %d feature(s) reverted to raw cap", reverted
            )

    return df_out, unclassified, transform_report


# ======================================================================== #
# Post-transform pruning
# ======================================================================== #

def prune_post_transform_features(
    df: pd.DataFrame,
    pd_features: List[str],
    cfg: ModelConfig = DEFAULT_CONFIG,
    drop_all_nan: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Drop degenerate features after transformation.

    A feature is dropped if:
    - ``drop_all_nan=True`` and it is entirely missing, OR
    - It has fewer than ``cfg.min_unique_post_transform`` distinct non-null values.

    Args:
        df:           Transformed DataFrame.
        pd_features:  Feature names to inspect.
        cfg:          Model config supplying ``min_unique_post_transform``.
        drop_all_nan: Drop all-NaN features (default ``True``).

    Returns:
        Tuple of:
        - pruned ``df``
        - ``kept_features`` list
        - ``drop_cols`` list
    """
    present = [c for c in pd_features if c in df.columns]
    drop_cols: List[str] = []

    for c in present:
        s = df[c]
        if drop_all_nan and s.notna().sum() == 0:
            drop_cols.append(c)
            continue
        if s.nunique(dropna=True) < cfg.min_unique_post_transform:
            drop_cols.append(c)

    if drop_cols:
        df = df.drop(columns=drop_cols)

    kept_features = [c for c in pd_features if c in df.columns]

    logger.info(
        "prune_post_transform_features: dropped %d degenerate features, "
        "kept %d",
        len(drop_cols),
        len(kept_features),
    )
    return df, kept_features, drop_cols


# ======================================================================== #
# Full transformation pipeline (orchestration helper)
# ======================================================================== #

def build_transformed_dataframe(
    df_pd: pd.DataFrame,
    pd_features: List[str],
    log_cols: List[str],
    cap_cols: List[str],
    signed_log_cols: List[str],
    cfg: ModelConfig = DEFAULT_CONFIG,
    neg_policy: str = "signed_log1p",
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Orchestrate the full numeric transformation + pruning pipeline.

    Steps:
    1. Downcast numeric PD features to float32.
    2. Apply transformations (``apply_pd_transformations``).
    3. Prune degenerate post-transform features.
    4. Assert index alignment then concat non-numeric + transformed numeric.

    Args:
        df_pd:           Full modelling DataFrame.
        pd_features:     Candidate numeric PD features.
        log_cols:        LOG + CAP columns.
        cap_cols:        CAP ONLY columns.
        signed_log_cols: SIGNED LOG + CAP columns.
        cfg:             Model config.
        neg_policy:      Negative-value handling policy for LOG columns.

    Returns:
        Tuple of:
        - ``df_pd_transformed`` – DataFrame with transformed numeric features
        - ``pd_features_pruned`` – kept feature names after pruning
        - ``transform_report`` – per-feature transformation action log
    """
    # 1) Downcast numeric features
    for col in [c for c in pd_features if c in df_pd.columns]:
        df_pd[col] = pd.to_numeric(df_pd[col], errors="coerce").astype("float32")

    # 2) Apply transformations
    df_numeric_transformed, _unclassified, transform_report = apply_pd_transformations(
        df_pd[pd_features].copy() if all(c in df_pd.columns for c in pd_features)
        else df_pd[[c for c in pd_features if c in df_pd.columns]].copy(),
        pd_features=pd_features,
        log_cols=log_cols,
        cap_cols=cap_cols,
        signed_log_cols=signed_log_cols,
        cfg=cfg,
        neg_policy=neg_policy,
    )

    # 3) Prune
    df_numeric_transformed, pd_features_pruned, dropped_cols = prune_post_transform_features(
        df_numeric_transformed, pd_features, cfg=cfg
    )

    # 4) Index alignment guard (BEFORE concat)
    assert_index_aligned(df_pd, df_numeric_transformed, context="build_transformed_dataframe")

    # 5) Concat: non-numeric cols + transformed numeric
    cols_to_replace = sorted(set(pd_features))
    df_pd_transformed = pd.concat(
        [
            df_pd.drop(columns=cols_to_replace, errors="ignore"),
            df_numeric_transformed,
        ],
        axis=1,
    )

    assert df_pd_transformed.columns.is_unique, (
        "[build_transformed_dataframe] Duplicate columns after concat"
    )
    leaked = sorted(set(dropped_cols) & set(df_pd_transformed.columns))
    assert not leaked, (
        "[build_transformed_dataframe] Dropped numeric cols leaked into output: "
        + ", ".join(leaked[:20])
    )
    assert "agent_msisdn" in df_pd_transformed.columns, (
        "[build_transformed_dataframe] agent_msisdn missing after transform join"
    )
    assert df_pd_transformed["agent_msisdn"].notna().all(), (
        "[build_transformed_dataframe] agent_msisdn has nulls after transform join"
    )

    return df_pd_transformed, pd_features_pruned, transform_report
