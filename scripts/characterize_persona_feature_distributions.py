"""
Characterizes the distribution of the ~16-17 features that feed borrower
persona clustering (segmentation/borrower_persona_clustering.py), to check
whether heavy right-skew in the raw monetary columns (commission,
account_balance, avg_loan_size_lifetime, ...) explains why UMAP+HDBSCAN puts
~90-94% of active borrowers into one dominant cluster regardless of how the
clustering parameters are tuned.

Reuses load_and_join() and build_features() from borrower_persona_clustering.py
directly -- not a reimplementation -- so this profiles exactly the same
population and columns that feed clustering, not an approximation.

For each numeric feature, prints raw-scale vs. log1p-transformed-scale
skewness and a "tail ratio" (99th percentile / median, a simple heavy-tail
indicator), side by side -- so it's visible at a glance which columns are
skewed enough to matter for clustering, and whether log1p actually fixes it
for that specific column (some columns may already be well-behaved and not
need it; others may still be skewed even after the transform).

Usage:
    python scripts\\characterize_persona_feature_distributions.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from segmentation.borrower_persona_clustering import build_features, load_and_join  # noqa: E402
from segmentation.extrafloat_segmentation_features import _get_features_config  # noqa: E402


def _tail_ratio(s: pd.Series) -> float:
    """99th percentile / median -- how many times larger the top 1% is than
    a typical value. A ratio near 1 means the column is tightly clumped
    (exactly the shape that would make UMAP/HDBSCAN see one dense blob); a
    large ratio means a long right tail is stretching the scale."""
    median = s.median()
    if median == 0:
        return np.nan
    return s.quantile(0.99) / median


def main() -> None:
    print("Loading + joining real data (same population as borrower_persona_clustering.py)...")
    df = load_and_join()
    feat, all_cols = build_features(df)
    numeric_cols = [c for c in all_cols if not c.startswith("has_")]

    feat_cfg = _get_features_config(None)
    skew_threshold = feat_cfg["skew_threshold"]

    rows = []
    for col in numeric_cols:
        raw = feat[col].dropna().astype(float)
        if len(raw) == 0:
            rows.append({"column": col, "note": "all-NaN, skipped"})
            continue
        raw_skew = raw.skew()
        log_ok = bool((raw >= 0).all())
        log_vals = np.log1p(raw) if log_ok else None

        rows.append({
            "column": col,
            "raw_skew": raw_skew,
            "log1p_skew": log_vals.skew() if log_vals is not None else np.nan,
            "raw_tail_ratio_p99_median": _tail_ratio(raw),
            "log1p_tail_ratio_p99_median": _tail_ratio(log_vals) if log_vals is not None else np.nan,
            "raw_cv": (raw.std() / raw.mean()) if raw.mean() not in (0, np.nan) else np.nan,
            "would_auto_log_transform": bool(log_ok and raw_skew > skew_threshold),
        })

    profile = pd.DataFrame(rows)
    if "raw_skew" in profile.columns:
        profile = profile.sort_values("raw_skew", ascending=False, na_position="last")

    print(f"\nskew_threshold (from feature config): {skew_threshold} "
          f"-- columns above this auto-qualify for log1p+winsorize in Step 3")
    with pd.option_context("display.float_format", "{:.3f}".format,
                            "display.max_rows", None, "display.width", 220):
        print(profile.to_string(index=False))

    print("\nPercentile spread per column (raw scale) -- shows how 'clumped' "
          "the bulk of active agents are on each feature:")
    pct_rows = []
    for col in numeric_cols:
        raw = feat[col].dropna().astype(float)
        if len(raw) == 0:
            continue
        pct_rows.append({
            "column": col,
            "p1": raw.quantile(0.01), "p25": raw.quantile(0.25),
            "median": raw.median(), "p75": raw.quantile(0.75),
            "p95": raw.quantile(0.95), "p99": raw.quantile(0.99),
            "max": raw.max(),
        })
    pct_profile = pd.DataFrame(pct_rows)
    with pd.option_context("display.float_format", "{:,.1f}".format,
                            "display.max_rows", None, "display.width", 220):
        print(pct_profile.to_string(index=False))


if __name__ == "__main__":
    main()
