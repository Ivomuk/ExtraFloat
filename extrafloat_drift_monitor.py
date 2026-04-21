from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional scipy — PSI works without it; KS p-value and chi-squared require it
try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _scipy_stats = None
    _SCIPY_AVAILABLE = False
    logger.info(
        "extrafloat_drift_monitor: scipy not available — "
        "KS p-values and chi-squared tests will be skipped; PSI runs normally."
    )

# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_STABLE  = "stable"
SEVERITY_MONITOR = "monitor"
SEVERITY_ALERT   = "alert"

# Features with fewer unique values than this are routed through categorical
# PSI (each distinct value = one bin) to prevent quantile-bin collapse from
# muting real drift on integer counts and binary flags.
_LOW_CARDINALITY_THRESHOLD = 15

_SEVERITY_RANK: Dict[str, int] = {
    SEVERITY_STABLE: 0,
    SEVERITY_MONITOR: 1,
    SEVERITY_ALERT: 2,
}
_RANK_SEVERITY: Dict[int, str] = {v: k for k, v in _SEVERITY_RANK.items()}

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DRIFT_CONFIG: Dict[str, Any] = {
    # ── PSI thresholds ────────────────────────────────────────────────────────
    "psi": {
        "stable_threshold": 0.10,
        "alert_threshold":  0.25,
        "n_bins":           10,
        "epsilon":          1e-6,
    },

    # ── KS test ───────────────────────────────────────────────────────────────
    "ks": {
        "alpha": 0.05,
    },

    # ── Percentile shift ──────────────────────────────────────────────────────
    "percentile_shift": {
        "percentiles":        [25, 50, 75, 95],
        "monitor_pct_change": 0.10,   # 10% relative change → monitor
        "alert_pct_change":   0.20,   # 20% relative change → alert
    },

    # ── Composition / chi-squared ─────────────────────────────────────────────
    "composition": {
        "chi2_alpha":               0.05,
        "min_expected_freq":        5,
        "alert_fraction_delta":     0.05,   # fallback when scipy absent
        # Cramér's V thresholds — applied when chi-sq is significant to prevent
        # large-n datasets from inflating everything to alert.
        "cramers_v_alert_threshold":   0.30,  # V >= 0.30 → alert
        "cramers_v_monitor_threshold": 0.10,  # 0.10 <= V < 0.30 → monitor
    },

    # ── Policy calibration health ─────────────────────────────────────────────
    "policy_health": {
        "regulatory_cap_rate_alert":          0.20,
        "thin_file_abs_change_alert":         0.05,
        "fallback_inputs_rate_alert":         0.10,
        "missing_inputs_rate_alert":          0.05,
        "proven_good_relative_monitor":       0.20,
        "active_floor_relative_monitor":      0.20,
        "kyc_block_relative_monitor":         0.20,
        "usage_inactive_relative_monitor":    0.20,
        # Relative-change checks are suppressed when ref_rate < this floor
        # to prevent spurious alerts from tiny baselines (e.g. 0.1% → 0.3%
        # is a 200% relative change but both are operationally negligible).
        "min_relative_baseline":              0.02,
    },

    # ── Cap driver composition ─────────────────────────────────────────────────
    "cap_driver": {
        "chi2_alpha":               0.05,
        "dominant_shift_alert":     0.15,   # 15 pp shift in top driver → alert
    },

    # ── Input features monitored per category ─────────────────────────────────
    "input_features": {
        "capacity": [
            "avg_daily_balance_30d",
            "avg_daily_balance_90d",
            "avg_monthly_revenue_30d",
            "avg_monthly_revenue_90d",
            "avg_monthly_txn_count_30d",
            "avg_monthly_txn_count_90d",
            "avg_monthly_payments_30d",
            "avg_monthly_payments_90d",
            "active_customer_count_30d",
            "active_customer_count_90d",
            "avg_monthly_txn_volume_30d",
            "avg_monthly_txn_volume_90d",
        ],
        "repayment": [
            "on_time_repayment_rate",
            "lifetime_default_rate",
            "default_rate_last_10_loans",
            "avg_cure_time_hours",
            "repayment_stability_score",
        ],
        "usage": [
            "recent_disbursement_amount_1m",
            "recent_repayment_amount_1m",
            "recent_repayment_coverage_1m",
            "recent_penalty_events_1m",
        ],
    },

    # ── Output features monitored ─────────────────────────────────────────────
    "output_features": {
        "limits": [
            "assigned_limit",
            "capacity_cap",
            "recent_usage_cap",
            "prior_exposure_cap",
            "risk_cap",
            "combined_cap",
            "policy_cap",
        ],
        "risk": ["risk_score"],
    },

    # ── Categorical columns for composition monitoring ─────────────────────────
    "composition_features": [
        "risk_tier",
        "combined_top_driver",
        "capacity_top_driver",
        "policy_reason",
        "prior_exposure_top_driver",
    ],
}


def _get_drift_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return config if config is not None else DEFAULT_DRIFT_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureDriftResult:
    feature:          str
    psi:              Optional[float]
    psi_severity:     str
    ks_statistic:     Optional[float]
    ks_pvalue:        Optional[float]
    ks_significant:   Optional[bool]
    percentile_shift: Optional[Dict[str, float]]
    severity:         str   # max(psi_severity, ks_severity, percentile_severity)


@dataclass
class CompositionDriftResult:
    feature:            str
    ref_distribution:   Dict[str, float]   # category → proportion
    cur_distribution:   Dict[str, float]
    chi2_statistic:     Optional[float]
    chi2_pvalue:        Optional[float]
    chi2_significant:   Optional[bool]
    cramers_v:          Optional[float]    # bias-corrected; None when scipy absent
    max_absolute_shift: float              # largest single-category pp change
    severity:           str


@dataclass
class PolicyHealthResult:
    metric:          str
    ref_rate:        float
    cur_rate:        float
    absolute_change: float
    relative_change: float
    threshold:       float
    severity:        str


@dataclass
class DriftReport:
    input_drift:       List[FeatureDriftResult]
    output_drift:      List[FeatureDriftResult]
    composition_drift: List[CompositionDriftResult]
    policy_health:     List[PolicyHealthResult]
    cap_driver_drift:  List[CompositionDriftResult]
    overall_severity:  str
    scipy_available:   bool
    ref_row_count:     int
    cur_row_count:     int
    skipped_features:  List[str]

    def summary_dict(self) -> Dict[str, Any]:
        def _counts(results, attr="severity"):
            alert   = sum(1 for r in results if getattr(r, attr) == SEVERITY_ALERT)
            monitor = sum(1 for r in results if getattr(r, attr) == SEVERITY_MONITOR)
            stable  = sum(1 for r in results if getattr(r, attr) == SEVERITY_STABLE)
            return alert, monitor, stable

        in_a, in_m, in_s   = _counts(self.input_drift)
        out_a, out_m, out_s = _counts(self.output_drift)
        comp_a, comp_m, _  = _counts(self.composition_drift)
        pol_a, pol_m, _    = _counts(self.policy_health)
        cap_a, cap_m, _    = _counts(self.cap_driver_drift)

        return {
            "overall_severity":              self.overall_severity,
            "ref_row_count":                 self.ref_row_count,
            "cur_row_count":                 self.cur_row_count,
            "scipy_available":               self.scipy_available,
            "input_features_alert_count":    in_a,
            "input_features_monitor_count":  in_m,
            "input_features_stable_count":   in_s,
            "output_features_alert_count":   out_a,
            "output_features_monitor_count": out_m,
            "output_features_stable_count":  out_s,
            "composition_alert_count":       comp_a,
            "composition_monitor_count":     comp_m,
            "policy_health_alert_count":     pol_a,
            "policy_health_monitor_count":   pol_m,
            "cap_driver_alert_count":        cap_a,
            "cap_driver_monitor_count":      cap_m,
            "top_input_alerts":  [r.feature for r in self.input_drift  if r.severity == SEVERITY_ALERT],
            "top_output_alerts": [r.feature for r in self.output_drift if r.severity == SEVERITY_ALERT],
            "top_composition_alerts": [r.feature for r in self.composition_drift if r.severity == SEVERITY_ALERT],
            "top_policy_alerts":      [r.metric  for r in self.policy_health      if r.severity == SEVERITY_ALERT],
            "skipped_features_count": len(self.skipped_features),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CORE STATISTICAL PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def _psi_severity(psi: float, cfg: Dict[str, Any]) -> str:
    psi_cfg = cfg["psi"]
    if psi >= psi_cfg["alert_threshold"]:
        return SEVERITY_ALERT
    if psi >= psi_cfg["stable_threshold"]:
        return SEVERITY_MONITOR
    return SEVERITY_STABLE


def _compute_psi(
    ref: np.ndarray,
    cur: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    if np.std(ref) < epsilon:
        return 0.0  # constant column — no distributional information

    # Quantile-based bin edges from reference population (robust to skew)
    quantiles  = np.linspace(0, 100, n_bins + 1)
    bin_edges  = np.unique(np.percentile(ref, quantiles))

    if len(bin_edges) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    ref_pct = np.clip(ref_pct, epsilon, 1.0)
    cur_pct = np.clip(cur_pct, epsilon, 1.0)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _compute_psi_categorical(
    ref: np.ndarray,
    cur: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """PSI for low-cardinality columns: each unique value is its own bin.

    Avoids the quantile-edge collapse that makes standard PSI return ~0
    for binary flags or small integer counts even when the distribution
    has clearly shifted.
    """
    ref_clean = ref[~np.isnan(ref)]
    cur_clean = cur[~np.isnan(cur)]

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    all_vals = np.unique(np.concatenate([ref_clean, cur_clean]))

    ref_counts = np.array([np.sum(ref_clean == v) for v in all_vals], dtype=float)
    cur_counts = np.array([np.sum(cur_clean == v) for v in all_vals], dtype=float)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    ref_pct = np.clip(ref_pct, epsilon, 1.0)
    cur_pct = np.clip(cur_pct, epsilon, 1.0)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _compute_ks(
    ref: np.ndarray,
    cur: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float], Optional[bool]]:
    if not _SCIPY_AVAILABLE:
        return None, None, None

    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return None, None, None

    result = _scipy_stats.ks_2samp(ref, cur)
    stat   = float(result.statistic)
    pval   = float(result.pvalue)
    return stat, pval, pval < alpha


def _compute_percentile_shift(
    ref: np.ndarray,
    cur: np.ndarray,
    percentiles: List[int],
    monitor_pct: float,
    alert_pct: float,
) -> Tuple[Dict[str, float], str]:
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return {}, SEVERITY_STABLE

    shift: Dict[str, float] = {}
    max_severity = SEVERITY_STABLE

    for p in percentiles:
        ref_val = float(np.percentile(ref, p))
        cur_val = float(np.percentile(cur, p))
        denom   = abs(ref_val) if abs(ref_val) > 1e-9 else 1e-9
        pct_chg = abs(cur_val - ref_val) / denom

        shift[f"p{p}_ref"]        = ref_val
        shift[f"p{p}_cur"]        = cur_val
        shift[f"p{p}_pct_change"] = pct_chg

        if pct_chg >= alert_pct:
            sev = SEVERITY_ALERT
        elif pct_chg >= monitor_pct:
            sev = SEVERITY_MONITOR
        else:
            sev = SEVERITY_STABLE

        if _SEVERITY_RANK[sev] > _SEVERITY_RANK[max_severity]:
            max_severity = sev

    return shift, max_severity


def _compute_chi2(
    ref_counts: Dict[str, int],
    cur_counts: Dict[str, int],
    alpha: float = 0.05,
    min_expected: int = 5,
) -> Tuple[Optional[float], Optional[float], Optional[bool]]:
    if not _SCIPY_AVAILABLE:
        return None, None, None

    all_cats = sorted(set(ref_counts) | set(cur_counts))
    ref_vec  = np.array([ref_counts.get(c, 0) for c in all_cats], dtype=float)
    cur_vec  = np.array([cur_counts.get(c, 0) for c in all_cats], dtype=float)

    ref_total = ref_vec.sum()
    cur_total = cur_vec.sum()

    if ref_total == 0 or cur_total == 0:
        return None, None, None

    # Check expected frequencies (chi2 unreliable below min_expected)
    expected = np.outer(
        np.array([ref_total, cur_total]),
        (ref_vec + cur_vec) / (ref_total + cur_total),
    )
    if (expected < min_expected).any():
        return None, None, None

    contingency = np.array([ref_vec, cur_vec])
    chi2, pval, _, _ = _scipy_stats.chi2_contingency(contingency)
    return float(chi2), float(pval), pval < alpha


def _compute_cramers_v(chi2_stat: float, n_total: int, n_categories: int) -> float:
    """Bias-corrected Cramér's V for a 2×k contingency table.

    Subtracts the expected chi-square under the null so that V stays near 0
    for large-n datasets with negligible real effects.
    """
    if n_total <= 1 or n_categories <= 1:
        return 0.0
    return float(np.sqrt(max(0.0, chi2_stat / n_total - (n_categories - 1) / (n_total - 1))))


def _aggregate_severity(results_lists: List) -> str:
    max_rank = 0
    for item in results_lists:
        if isinstance(item, list):
            for r in item:
                max_rank = max(max_rank, _SEVERITY_RANK.get(r.severity, 0))
        else:
            max_rank = max(max_rank, _SEVERITY_RANK.get(item.severity, 0))
    return _RANK_SEVERITY[max_rank]


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-FEATURE DRIFT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _monitor_single_feature(
    feature: str,
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    cfg: Dict[str, Any],
    skipped: List[str],
) -> Optional[FeatureDriftResult]:
    if feature not in ref_df.columns or feature not in cur_df.columns:
        skipped.append(feature)
        return None

    ref_arr = pd.to_numeric(ref_df[feature], errors="coerce").values
    cur_arr = pd.to_numeric(cur_df[feature], errors="coerce").values

    psi_cfg  = cfg["psi"]
    ks_cfg   = cfg["ks"]
    pct_cfg  = cfg["percentile_shift"]

    ref_clean  = ref_arr[~np.isnan(ref_arr)]
    n_unique   = len(np.unique(ref_clean)) if len(ref_clean) > 0 else 0
    if n_unique < _LOW_CARDINALITY_THRESHOLD:
        logger.debug(
            "_monitor_single_feature: %s has %d unique values — using categorical PSI",
            feature, n_unique,
        )
        psi = _compute_psi_categorical(ref_arr, cur_arr, psi_cfg["epsilon"])
    else:
        psi = _compute_psi(ref_arr, cur_arr, psi_cfg["n_bins"], psi_cfg["epsilon"])
    psi_sev  = _psi_severity(psi, cfg)

    ks_stat, ks_pval, ks_sig = _compute_ks(ref_arr, cur_arr, ks_cfg["alpha"])
    ks_sev   = SEVERITY_ALERT if ks_sig else SEVERITY_STABLE

    pct_shift, pct_sev = _compute_percentile_shift(
        ref_arr, cur_arr,
        pct_cfg["percentiles"],
        pct_cfg["monitor_pct_change"],
        pct_cfg["alert_pct_change"],
    )

    severity = _RANK_SEVERITY[max(
        _SEVERITY_RANK[psi_sev],
        _SEVERITY_RANK[ks_sev],
        _SEVERITY_RANK[pct_sev],
    )]

    return FeatureDriftResult(
        feature=feature,
        psi=psi,
        psi_severity=psi_sev,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pval,
        ks_significant=ks_sig,
        percentile_shift=pct_shift if pct_shift else None,
        severity=severity,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 1 — INPUT DISTRIBUTION DRIFT
# ─────────────────────────────────────────────────────────────────────────────

def monitor_input_drift(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[FeatureDriftResult], List[str]]:
    """Returns (results, skipped_feature_names)."""
    cfg      = _get_drift_config(config)
    skipped: List[str] = []
    results: List[FeatureDriftResult] = []

    all_features: List[str] = []
    for group in cfg["input_features"].values():
        all_features.extend(group)

    for feat in all_features:
        r = _monitor_single_feature(feat, ref_df, cur_df, cfg, skipped)
        if r is not None:
            results.append(r)

    n_alert   = sum(1 for r in results if r.severity == SEVERITY_ALERT)
    n_monitor = sum(1 for r in results if r.severity == SEVERITY_MONITOR)
    logger.info(
        "monitor_input_drift: features=%d alert=%d monitor=%d skipped=%d",
        len(results), n_alert, n_monitor, len(skipped),
    )
    if skipped:
        logger.warning("monitor_input_drift: skipped absent features: %s", skipped)

    return results, skipped


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 2 — OUTPUT DISTRIBUTION DRIFT
# ─────────────────────────────────────────────────────────────────────────────

def _mean_limit_by_tier(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    limit_col: str = "assigned_limit",
    tier_col:  str = "risk_tier",
) -> Dict[str, Dict[str, float]]:
    if limit_col not in ref_df.columns or tier_col not in ref_df.columns:
        return {}
    if limit_col not in cur_df.columns or tier_col not in cur_df.columns:
        return {}

    result: Dict[str, Dict[str, float]] = {}
    ref_means = ref_df.groupby(tier_col)[limit_col].mean()

    for tier, ref_mean in ref_means.items():
        cur_mean = cur_df[cur_df[tier_col] == tier][limit_col].mean()
        if pd.isna(cur_mean):
            continue
        denom    = abs(ref_mean) if abs(ref_mean) > 1e-9 else 1e-9
        pct_chg  = (cur_mean - ref_mean) / denom
        result[str(tier)] = {
            "ref_mean":   float(ref_mean),
            "cur_mean":   float(cur_mean),
            "pct_change": float(pct_chg),
        }
    return result


def monitor_output_drift(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[FeatureDriftResult], List[str]]:
    """Returns (results, skipped_feature_names)."""
    cfg      = _get_drift_config(config)
    skipped: List[str] = []
    results: List[FeatureDriftResult] = []

    all_features: List[str] = []
    for group in cfg["output_features"].values():
        all_features.extend(group)

    for feat in all_features:
        r = _monitor_single_feature(feat, ref_df, cur_df, cfg, skipped)
        if r is not None:
            results.append(r)

    tier_stats = _mean_limit_by_tier(ref_df, cur_df)
    if tier_stats:
        logger.info("monitor_output_drift: mean assigned_limit by tier: %s", tier_stats)

    n_alert   = sum(1 for r in results if r.severity == SEVERITY_ALERT)
    n_monitor = sum(1 for r in results if r.severity == SEVERITY_MONITOR)
    logger.info(
        "monitor_output_drift: features=%d alert=%d monitor=%d skipped=%d",
        len(results), n_alert, n_monitor, len(skipped),
    )
    return results, skipped


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 3 — POPULATION COMPOSITION DRIFT
# ─────────────────────────────────────────────────────────────────────────────

def _discretize_agent_tier_multiplier(series: pd.Series) -> pd.Series:
    _MULTIPLIER_TO_TIER = {
        1.00: "diamond",
        0.75: "titanium",
        0.50: "platinum",
        0.35: "gold",
        0.25: "silver",
        0.10: "bronze",
        0.05: "new_bronze",
    }
    def _map(v):
        if pd.isna(v):
            return "unknown"
        rounded = round(float(v), 2)
        return _MULTIPLIER_TO_TIER.get(rounded, "other")
    return series.map(_map)


def _monitor_categorical(
    feature: str,
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    comp_cfg: Dict[str, Any],
    skipped: Optional[List[str]] = None,
) -> Optional[CompositionDriftResult]:
    if feature == "agent_tier_ceiling_multiplier":
        ref_col = _discretize_agent_tier_multiplier(ref_df[feature]) if feature in ref_df.columns else None
        cur_col = _discretize_agent_tier_multiplier(cur_df[feature]) if feature in cur_df.columns else None
    else:
        ref_col = ref_df[feature] if feature in ref_df.columns else None
        cur_col = cur_df[feature] if feature in cur_df.columns else None

    if ref_col is None or cur_col is None:
        if skipped is not None:
            skipped.append(feature)
        return None

    ref_counts = ref_col.value_counts().to_dict()
    cur_counts = cur_col.value_counts().to_dict()

    all_cats   = sorted(set(ref_counts) | set(cur_counts))
    ref_total  = sum(ref_counts.values()) or 1
    cur_total  = sum(cur_counts.values()) or 1

    ref_dist = {c: ref_counts.get(c, 0) / ref_total for c in all_cats}
    cur_dist = {c: cur_counts.get(c, 0) / cur_total for c in all_cats}

    max_abs_shift = max(abs(cur_dist[c] - ref_dist[c]) for c in all_cats)

    chi2, pval, sig = _compute_chi2(ref_counts, cur_counts, comp_cfg["chi2_alpha"],
                                     comp_cfg["min_expected_freq"])

    cramers_v: Optional[float] = None
    if sig:
        n_total = ref_total + cur_total
        cramers_v = _compute_cramers_v(chi2, n_total, len(all_cats))
        # Effect-size tiering: large-n datasets can be statistically significant
        # with negligible real impact — use V to distinguish alert from monitor.
        if cramers_v >= comp_cfg["cramers_v_alert_threshold"]:
            severity = SEVERITY_ALERT
        else:
            severity = SEVERITY_MONITOR
    elif sig is None:
        # scipy absent or expected-freq too sparse — fall back to fraction-delta
        severity = SEVERITY_ALERT if max_abs_shift >= comp_cfg["alert_fraction_delta"] * 2 else (
            SEVERITY_MONITOR if max_abs_shift >= comp_cfg["alert_fraction_delta"] else SEVERITY_STABLE
        )
    else:
        severity = SEVERITY_STABLE

    return CompositionDriftResult(
        feature=feature,
        ref_distribution=ref_dist,
        cur_distribution=cur_dist,
        chi2_statistic=chi2,
        chi2_pvalue=pval,
        chi2_significant=sig,
        cramers_v=cramers_v,
        max_absolute_shift=max_abs_shift,
        severity=severity,
    )


def monitor_composition_drift(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[CompositionDriftResult], List[str]]:
    """Returns (results, skipped_feature_names)."""
    cfg      = _get_drift_config(config)
    comp_cfg = cfg["composition"]
    skipped: List[str] = []
    results: List[CompositionDriftResult] = []

    for feat in cfg["composition_features"]:
        r = _monitor_categorical(feat, ref_df, cur_df, comp_cfg, skipped)
        if r is not None:
            results.append(r)

    n_alert = sum(1 for r in results if r.severity == SEVERITY_ALERT)
    logger.info(
        "monitor_composition_drift: features=%d alert=%d skipped=%d",
        len(results), n_alert, len(skipped),
    )
    if skipped:
        logger.warning("monitor_composition_drift: skipped absent features: %s", skipped)
    return results, skipped


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 4 — POLICY CALIBRATION HEALTH
# ─────────────────────────────────────────────────────────────────────────────

def _flag_rate(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    s = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return float((s > 0).mean())


def _health_result(
    metric: str,
    ref_rate: float,
    cur_rate: float,
    *,
    absolute_alert: Optional[float] = None,
    cur_rate_ceiling: Optional[float] = None,
    relative_monitor: Optional[float] = None,
    min_relative_baseline: float = 0.02,
    threshold_label: float = 0.0,
) -> PolicyHealthResult:
    eps = 1e-9
    abs_change = cur_rate - ref_rate
    rel_change = abs_change / (abs(ref_rate) + eps)

    severity = SEVERITY_STABLE

    if cur_rate_ceiling is not None and cur_rate >= cur_rate_ceiling:
        severity = SEVERITY_ALERT
    elif absolute_alert is not None and abs(abs_change) >= absolute_alert:
        severity = SEVERITY_ALERT
    elif (
        relative_monitor is not None
        and ref_rate >= min_relative_baseline
        and abs(rel_change) >= relative_monitor
    ):
        # Relative comparison is only meaningful when the baseline is large
        # enough that a proportional shift is operationally significant.
        severity = SEVERITY_MONITOR

    return PolicyHealthResult(
        metric=metric,
        ref_rate=ref_rate,
        cur_rate=cur_rate,
        absolute_change=float(abs_change),
        relative_change=float(rel_change),
        threshold=threshold_label,
        severity=severity,
    )


def monitor_policy_health(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> List[PolicyHealthResult]:
    cfg      = _get_drift_config(config)
    ph       = cfg["policy_health"]
    min_base = ph.get("min_relative_baseline", 0.02)
    results: List[PolicyHealthResult] = []

    # 1. Regulatory cap utilisation (absolute ceiling regardless of ref)
    results.append(_health_result(
        "regulatory_cap_applied_rate",
        _flag_rate(ref_df, "regulatory_cap_applied"),
        _flag_rate(cur_df, "regulatory_cap_applied"),
        cur_rate_ceiling=ph["regulatory_cap_rate_alert"],
        threshold_label=ph["regulatory_cap_rate_alert"],
    ))

    # 2. Thin-file rate (absolute pp change)
    results.append(_health_result(
        "thin_file_rate",
        _flag_rate(ref_df, "is_thin_file"),
        _flag_rate(cur_df, "is_thin_file"),
        absolute_alert=ph["thin_file_abs_change_alert"],
        threshold_label=ph["thin_file_abs_change_alert"],
    ))

    # 3. Fallback capacity inputs (absolute ceiling)
    results.append(_health_result(
        "capacity_fallback_inputs_rate",
        _flag_rate(ref_df, "capacity_fallback_inputs"),
        _flag_rate(cur_df, "capacity_fallback_inputs"),
        cur_rate_ceiling=ph["fallback_inputs_rate_alert"],
        threshold_label=ph["fallback_inputs_rate_alert"],
    ))

    # 4. Missing capacity inputs (absolute ceiling)
    results.append(_health_result(
        "capacity_missing_inputs_rate",
        _flag_rate(ref_df, "capacity_missing_inputs"),
        _flag_rate(cur_df, "capacity_missing_inputs"),
        cur_rate_ceiling=ph["missing_inputs_rate_alert"],
        threshold_label=ph["missing_inputs_rate_alert"],
    ))

    # 5. Proven-good override rate (relative monitor)
    results.append(_health_result(
        "proven_good_rate",
        _flag_rate(ref_df, "is_proven_good_borrower"),
        _flag_rate(cur_df, "is_proven_good_borrower"),
        relative_monitor=ph["proven_good_relative_monitor"],
        min_relative_baseline=min_base,
        threshold_label=ph["proven_good_relative_monitor"],
    ))

    # 6. Active floor triggered rate (relative monitor)
    results.append(_health_result(
        "active_floor_applied_rate",
        _flag_rate(ref_df, "active_floor_applied"),
        _flag_rate(cur_df, "active_floor_applied"),
        relative_monitor=ph["active_floor_relative_monitor"],
        min_relative_baseline=min_base,
        threshold_label=ph["active_floor_relative_monitor"],
    ))

    # 7. KYC block rate (relative monitor)
    results.append(_health_result(
        "kyc_block_rate",
        _flag_rate(ref_df, "is_kyc_blocked"),
        _flag_rate(cur_df, "is_kyc_blocked"),
        relative_monitor=ph["kyc_block_relative_monitor"],
        min_relative_baseline=min_base,
        threshold_label=ph["kyc_block_relative_monitor"],
    ))

    # 8. Usage gate inactive rate (relative monitor)
    def _inactive_rate(df: pd.DataFrame) -> float:
        if "recent_usage_active_flag" not in df.columns:
            return 0.0
        s = pd.to_numeric(df["recent_usage_active_flag"], errors="coerce").fillna(1)
        return float((s == 0).mean())

    results.append(_health_result(
        "usage_inactive_rate",
        _inactive_rate(ref_df),
        _inactive_rate(cur_df),
        relative_monitor=ph["usage_inactive_relative_monitor"],
        min_relative_baseline=min_base,
        threshold_label=ph["usage_inactive_relative_monitor"],
    ))

    n_alert   = sum(1 for r in results if r.severity == SEVERITY_ALERT)
    n_monitor = sum(1 for r in results if r.severity == SEVERITY_MONITOR)
    logger.info(
        "monitor_policy_health: metrics=%d alert=%d monitor=%d",
        len(results), n_alert, n_monitor,
    )
    for r in results:
        if r.severity == SEVERITY_ALERT:
            logger.warning(
                "monitor_policy_health: ALERT on %s ref=%.4f cur=%.4f threshold=%.4f",
                r.metric, r.ref_rate, r.cur_rate, r.threshold,
            )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 5 — CAP DRIVER COMPOSITION DRIFT ("silent recalibration")
# ─────────────────────────────────────────────────────────────────────────────

def _detect_silent_recalibration(
    result: CompositionDriftResult,
    alert_threshold: float,
) -> Optional[str]:
    if not result.ref_distribution or not result.cur_distribution:
        return None

    ref_top = max(result.ref_distribution, key=result.ref_distribution.get)
    cur_top = max(result.cur_distribution, key=result.cur_distribution.get)

    if ref_top == cur_top:
        return None

    shift = abs(
        result.ref_distribution.get(ref_top, 0.0) -
        result.cur_distribution.get(ref_top, 0.0)
    )
    if shift >= alert_threshold:
        return (
            f"Silent recalibration signal: dominant binding cap shifted from "
            f"{ref_top} (ref={result.ref_distribution.get(ref_top, 0):.1%}) to "
            f"{cur_top} (cur={result.cur_distribution.get(cur_top, 0):.1%}). "
            f"Formula unchanged but effective regime shifted."
        )
    return None


def monitor_cap_driver_drift(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[CompositionDriftResult], List[str]]:
    """Returns (results, skipped_feature_names)."""
    cfg     = _get_drift_config(config)
    cd_cfg  = cfg["cap_driver"]
    comp_cfg = {
        "chi2_alpha":               cd_cfg["chi2_alpha"],
        "min_expected_freq":        5,
        "alert_fraction_delta":     0.05,
        "cramers_v_alert_threshold":   cfg["composition"]["cramers_v_alert_threshold"],
        "cramers_v_monitor_threshold": cfg["composition"]["cramers_v_monitor_threshold"],
    }
    skipped: List[str] = []
    results: List[CompositionDriftResult] = []

    for feat in ["combined_top_driver", "policy_reason", "capacity_top_driver"]:
        r = _monitor_categorical(feat, ref_df, cur_df, comp_cfg, skipped)
        if r is not None:
            results.append(r)
            if feat == "combined_top_driver":
                msg = _detect_silent_recalibration(r, cd_cfg["dominant_shift_alert"])
                if msg:
                    logger.warning(msg)

    if skipped:
        logger.warning("monitor_cap_driver_drift: skipped absent features: %s", skipped)
    return results, skipped


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_drift_monitor(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    monitor_inputs: bool = True,
    monitor_outputs: bool = True,
) -> DriftReport:
    cfg = _get_drift_config(config)

    logger.info(
        "run_drift_monitor: ref_rows=%d cur_rows=%d scipy=%s",
        len(ref_df), len(cur_df), _SCIPY_AVAILABLE,
    )

    if monitor_inputs:
        input_drift, input_skipped = monitor_input_drift(ref_df, cur_df, cfg)
    else:
        input_drift, input_skipped = [], []

    if monitor_outputs:
        output_drift, output_skipped = monitor_output_drift(ref_df, cur_df, cfg)
    else:
        output_drift, output_skipped = [], []

    composition_drift, comp_skipped = monitor_composition_drift(ref_df, cur_df, cfg)
    policy_health                   = monitor_policy_health(ref_df, cur_df, cfg)
    cap_driver_drift,  cap_skipped  = monitor_cap_driver_drift(ref_df, cur_df, cfg)

    skipped = input_skipped + output_skipped + comp_skipped + cap_skipped

    overall = _aggregate_severity([
        input_drift, output_drift, composition_drift,
        policy_health, cap_driver_drift,
    ])

    report = DriftReport(
        input_drift=input_drift,
        output_drift=output_drift,
        composition_drift=composition_drift,
        policy_health=policy_health,
        cap_driver_drift=cap_driver_drift,
        overall_severity=overall,
        scipy_available=_SCIPY_AVAILABLE,
        ref_row_count=len(ref_df),
        cur_row_count=len(cur_df),
        skipped_features=skipped,
    )

    summary = report.summary_dict()
    logger.info("run_drift_monitor: complete — %s", summary)

    if overall == SEVERITY_ALERT:
        logger.warning(
            "run_drift_monitor: ALERT — overall drift severity is alert. "
            "Review top_input_alerts=%s top_output_alerts=%s top_policy_alerts=%s",
            summary["top_input_alerts"],
            summary["top_output_alerts"],
            summary["top_policy_alerts"],
        )

    return report
