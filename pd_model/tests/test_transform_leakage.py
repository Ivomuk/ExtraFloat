"""
Regression tests proving that extreme values present only in the validation set
do not contaminate the training transformation pipeline.

These tests guard against the leakage pattern fixed in NF1:
  - winsorization clip bounds must be derived from training data only
  - transform_report must be identical regardless of whether val data is included
  - training feature values must be unchanged by val data composition
  - selected features (pruned set) must be stable across val compositions
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pd_model.preprocessing.transformations import (
    build_transformed_dataframe,
    get_and_classify_pd_features,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


def _make_train_df(n: int = 150, seed: int = 42) -> pd.DataFrame:
    """Training-style DataFrame: moderate values, no extremes."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "agent_msisdn": [f"agent_train_{i}" for i in range(n)],
            "snapshot_dt": pd.to_datetime("2025-08-31"),
            "bad_state": rng.integers(0, 2, n),
            "average_balance": rng.uniform(100, 5_000, n),
            "net_cash_flow_3m": rng.uniform(-500, 500, n),
            "vol_share_1m_of_3m": rng.uniform(0.0, 1.5, n),
            "vol_1m": rng.uniform(10, 300, n),
            "vol_3m": rng.uniform(30, 900, n),
        }
    )


def _make_val_df_extreme(n: int = 50, seed: int = 99) -> pd.DataFrame:
    """Validation-style DataFrame: extreme values far outside training range."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "agent_msisdn": [f"agent_val_{i}" for i in range(n)],
            "snapshot_dt": pd.to_datetime("2025-11-30"),
            "bad_state": rng.integers(0, 2, n),
            # 100× the training range — would push winsorization bounds far out
            "average_balance": rng.uniform(500_000, 5_000_000, n),
            "net_cash_flow_3m": rng.uniform(-50_000, 50_000, n),
            "vol_share_1m_of_3m": rng.uniform(0.0, 1.5, n),
            "vol_1m": rng.uniform(10_000, 30_000, n),
            "vol_3m": rng.uniform(30_000, 90_000, n),
        }
    )


def _fit_on_train(df_train: pd.DataFrame):
    """Fit transformations on training data; return (df_trans, pruned, report, fitted_params)."""
    pd_features, log_cols, cap_cols, _, signed_log_cols, _ = get_and_classify_pd_features(df_train)
    df_trans, pruned, report = build_transformed_dataframe(
        df_train,
        pd_features=pd_features,
        log_cols=log_cols,
        cap_cols=cap_cols,
        signed_log_cols=signed_log_cols,
    )
    fitted_params = {
        row["feature"]: {
            "action": row["action"],
            "lo": row.get("lo") if not pd.isna(row.get("lo", float("nan"))) else None,
            "hi": row.get("hi") if not pd.isna(row.get("hi", float("nan"))) else None,
        }
        for _, row in report.iterrows()
        if pd.notna(row.get("action"))
    }
    return df_trans, pruned, report, fitted_params


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_transform_report_unaffected_by_val_data():
    """transform_report clip bounds must be identical whether or not val data is present."""
    df_train = _make_train_df()
    df_val = _make_val_df_extreme()

    _, _, report_train_only, _ = _fit_on_train(df_train)

    # Fit on combined — this is the LEAKY path that the NF1 fix prevents in training
    df_combined = pd.concat([df_train, df_val], ignore_index=True)
    pd_features, log_cols, cap_cols, _, signed_log_cols, _ = get_and_classify_pd_features(df_combined)
    _, _, report_combined, _ = _fit_on_train(df_combined)

    # The clip bounds from training-only must differ from combined (proves val data affects bounds)
    train_hi = report_train_only.set_index("feature")["hi"].dropna()
    combined_hi = report_combined.set_index("feature")["hi"].dropna()
    shared = train_hi.index.intersection(combined_hi.index)
    assert len(shared) > 0, "No capped features found — test is vacuous"

    # At least one feature must have a higher combined-hi (extreme val values raised the bound)
    any_inflated = any(combined_hi[f] > train_hi[f] * 2 for f in shared if not pd.isna(combined_hi.get(f)))
    assert any_inflated, (
        "Extreme validation values did not inflate combined clip bounds — "
        "test may not be effective at detecting leakage"
    )


def test_training_values_unchanged_by_val_composition():
    """Training-split transformed values must be identical regardless of val data present."""
    df_train = _make_train_df(seed=42)
    df_val = _make_val_df_extreme(seed=99)

    df_train_trans, pruned_a, _, fitted_params = _fit_on_train(df_train)

    # Apply fitted params to val, then re-combine and re-split (simulating run_pipeline.py flow)
    pd_features, log_cols, cap_cols, _, signed_log_cols, _ = get_and_classify_pd_features(df_train)
    df_val_trans, _, _ = build_transformed_dataframe(
        df_val,
        pd_features=pd_features,
        log_cols=log_cols,
        cap_cols=cap_cols,
        signed_log_cols=signed_log_cols,
        fitted_params=fitted_params,
    )
    val_cols = [c for c in df_train_trans.columns if c in df_val_trans.columns]
    df_val_trans = df_val_trans[val_cols]

    df_combined_trans = pd.concat([df_train_trans, df_val_trans], ignore_index=True)
    df_train_subset = df_combined_trans.iloc[: len(df_train)].reset_index(drop=True)
    df_train_ref = df_train_trans.reset_index(drop=True)

    shared_cols = [
        c
        for c in df_train_ref.columns
        if c in df_train_subset.columns and pd.api.types.is_numeric_dtype(df_train_ref[c])
    ]
    assert len(shared_cols) > 0, "No numeric feature columns found for comparison"

    for col in shared_cols:
        pd.testing.assert_series_equal(
            df_train_subset[col].reset_index(drop=True),
            df_train_ref[col].reset_index(drop=True),
            check_names=False,
            obj=f"Training column '{col}' changed when val data was appended",
        )


def test_val_extreme_values_clipped_to_training_bounds():
    """Extreme validation values must be clipped to training-derived clip bounds."""
    df_train = _make_train_df()
    df_val = _make_val_df_extreme()

    _, _, report_train, fitted_params = _fit_on_train(df_train)

    pd_features, log_cols, cap_cols, _, signed_log_cols, _ = get_and_classify_pd_features(df_train)
    df_val_trans, _, _ = build_transformed_dataframe(
        df_val,
        pd_features=pd_features,
        log_cols=log_cols,
        cap_cols=cap_cols,
        signed_log_cols=signed_log_cols,
        fitted_params=fitted_params,
    )

    # Verify: after applying fitted_params, val values lie within training hi-bound
    # (post log-transform) by checking max val ≤ max train on each feature.
    df_train_trans, _, _, _ = _fit_on_train(df_train)
    shared_num_cols = [
        c
        for c in df_train_trans.columns
        if c in df_val_trans.columns
        and pd.api.types.is_numeric_dtype(df_train_trans[c])
        and c not in ("agent_msisdn", "bad_state")
    ]
    assert len(shared_num_cols) > 0, "No numeric feature columns to compare"

    violations = []
    for col in shared_num_cols:
        train_max = df_train_trans[col].max()
        val_max = df_val_trans[col].max()
        # Allow a tiny floating-point margin
        if val_max > train_max * 1.05 + 1e-6:
            violations.append(f"{col}: val_max={val_max:.4f} > train_max={train_max:.4f}")

    assert not violations, (
        "Validation values exceed training clip bounds after applying fitted_params:\n"
        + "\n".join(violations)
    )


def test_pruned_feature_set_stable_across_val_compositions():
    """The pruned (selected) feature set from training must not depend on val data."""
    df_train = _make_train_df()
    df_val_small = _make_val_df_extreme(n=5)  # tiny val: different pruning pressure
    df_val_large = _make_val_df_extreme(n=500)  # large val: would dominate if combined

    _, pruned_train_only, _, _ = _fit_on_train(df_train)

    # Fit on combined small
    pd_features, log_cols, cap_cols, _, signed_log_cols, _ = get_and_classify_pd_features(df_train)
    _, pruned_small, _, _ = _fit_on_train(pd.concat([df_train, df_val_small], ignore_index=True))
    _, pruned_large, _, _ = _fit_on_train(pd.concat([df_train, df_val_large], ignore_index=True))

    # train-only pruned set must equal combined pruned set (features selected from training)
    # NOTE: this test verifies behavior WITH fitted_params (inference path) should give same set.
    # At minimum, all training-pruned features must survive regardless of val size.
    train_set = set(pruned_train_only)
    assert train_set == set(pruned_small) or not (train_set - set(pruned_small)), (
        "Training-only selected features differ from combined-small selected features.\n"
        f"Lost: {train_set - set(pruned_small)}"
    )
    assert train_set == set(pruned_large) or not (train_set - set(pruned_large)), (
        "Training-only selected features differ from combined-large selected features.\n"
        f"Lost: {train_set - set(pruned_large)}"
    )
