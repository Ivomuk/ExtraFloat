"""
Tests proving that inference is deterministic and batch-independent after the
fitted_params fix in apply_pd_transformations / build_transformed_dataframe.

These tests do NOT require trained model artifacts.  They exercise the
transformation layer directly, building fitted_params from a "training" run
and then verifying that scoring individual agents or reordering rows produces
identical transformed feature values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pd_model.preprocessing.transformations import (
    apply_pd_transformations,
    get_and_classify_pd_features,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_training_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "agent_msisdn": [f"256700{i:06d}" for i in range(n)],
            "snapshot_dt": pd.to_datetime("2025-09-30"),
            "bad_state": rng.integers(0, 2, n),
            # LOG column
            "average_balance": rng.exponential(1_000, n),
            # SIGNED LOG column (has negatives)
            "net_cash_flow_3m": rng.normal(0, 500, n),
            # CAP column
            "vol_share_1m_of_3m": rng.uniform(0, 2, n),
            # Count / CAP column
            "vol_1m": rng.exponential(200, n),
            "vol_3m": rng.exponential(600, n),
            # PROTECTED column (untransformed)
            "is_fully_inactive_6m": rng.integers(0, 2, n).astype(float),
        }
    )


def _build_fitted_params(train_df: pd.DataFrame) -> dict:
    """Run a training-style transformation pass and extract fitted_params."""
    pd_features, log_cols, cap_cols, _prot, signed_log_cols, _ = get_and_classify_pd_features(train_df)
    feature_cols = [c for c in pd_features if c in train_df.columns]
    _, _, transform_report = apply_pd_transformations(
        train_df[feature_cols].copy(),
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
        for _, row in transform_report.iterrows()
        if pd.notna(row.get("action"))
    }
    return fitted_params, pd_features, log_cols, cap_cols, signed_log_cols


def _transform_with_fitted(
    df: pd.DataFrame, fitted_params, pd_features, log_cols, cap_cols, signed_log_cols
) -> pd.DataFrame:
    feature_cols = [c for c in pd_features if c in df.columns]
    result, _, _ = apply_pd_transformations(
        df[feature_cols].copy(),
        pd_features=pd_features,
        log_cols=log_cols,
        cap_cols=cap_cols,
        signed_log_cols=signed_log_cols,
        fitted_params=fitted_params,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 1. Saved clip bounds are present after training-style run
# ─────────────────────────────────────────────────────────────────────────────


def test_transform_report_contains_lo_hi_bounds():
    """Training pass saves lo/hi for winsorized features."""
    train_df = _make_training_df()
    fitted_params, *_ = _build_fitted_params(train_df)

    winsorized_actions = {
        "forced_signed_log_ok",
        "signed_log_ok",
        "log_cap_ok",
        "cap_ok",
        "cap_only_due_to_negatives",
        "forced_signed_log_reverted_raw_cap",
        "signed_log_reverted_raw_cap",
        "log_cap_reverted_raw_cap",
    }
    for feat, fp in fitted_params.items():
        if fp.get("action") in winsorized_actions:
            assert fp.get("lo") is not None, f"{feat}: lo missing"
            assert fp.get("hi") is not None, f"{feat}: hi missing"
            assert float(fp["hi"]) >= float(fp["lo"]), f"{feat}: hi < lo"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Single-row score equals that row scored in a large batch
# ─────────────────────────────────────────────────────────────────────────────


def test_single_row_score_equals_batch_row():
    """Scoring one agent alone must produce the same transformed values as scoring
    that agent inside a 200-row batch."""
    train_df = _make_training_df(n=500, seed=0)
    fitted_params, pd_features, log_cols, cap_cols, signed_log_cols = _build_fitted_params(train_df)

    batch_df = _make_training_df(n=200, seed=1)

    # Pick agent at index 7 from the batch
    target_idx = 7
    single_df = batch_df.iloc[[target_idx]].copy().reset_index(drop=True)

    batch_result = _transform_with_fitted(
        batch_df, fitted_params, pd_features, log_cols, cap_cols, signed_log_cols
    )
    single_result = _transform_with_fitted(
        single_df, fitted_params, pd_features, log_cols, cap_cols, signed_log_cols
    )

    feature_cols = [c for c in pd_features if c in batch_df.columns]
    for col in feature_cols:
        if col not in batch_result.columns or col not in single_result.columns:
            continue
        batch_val = float(batch_result.iloc[target_idx][col])
        single_val = float(single_result.iloc[0][col])
        assert abs(batch_val - single_val) < 1e-5, (
            f"Column '{col}': single={single_val:.6f} vs batch={batch_val:.6f} — "
            "scores differ by batch composition (batch-dependency bug)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Row order independence
# ─────────────────────────────────────────────────────────────────────────────


def test_row_order_independence():
    """Shuffling input rows must not change per-agent transformed values."""
    train_df = _make_training_df(n=500, seed=2)
    fitted_params, pd_features, log_cols, cap_cols, signed_log_cols = _build_fitted_params(train_df)

    scoring_df = _make_training_df(n=100, seed=3)
    shuffled_df = scoring_df.sample(frac=1, random_state=99).reset_index(drop=True)

    result_orig = _transform_with_fitted(
        scoring_df, fitted_params, pd_features, log_cols, cap_cols, signed_log_cols
    )
    result_shuf = _transform_with_fitted(
        shuffled_df, fitted_params, pd_features, log_cols, cap_cols, signed_log_cols
    )

    feature_cols = [c for c in pd_features if c in scoring_df.columns]
    for col in feature_cols:
        if col not in result_orig.columns or col not in result_shuf.columns:
            continue
        orig_vals = result_orig[col].values
        shuf_vals = result_shuf[col].values
        # Re-align by msisdn
        orig_series = pd.Series(orig_vals, index=scoring_df["agent_msisdn"].values)
        shuf_series = pd.Series(shuf_vals, index=shuffled_df["agent_msisdn"].values)
        common = orig_series.index.intersection(shuf_series.index)
        np.testing.assert_allclose(
            orig_series[common].values.astype(float),
            shuf_series[common].values.astype(float),
            atol=1e-5,
            err_msg=f"Column '{col}' values differ after row shuffle",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Duplicate extreme-value rows do not pollute other agents
# ─────────────────────────────────────────────────────────────────────────────


def test_duplicate_extreme_rows_do_not_pollute():
    """Adding many copies of an extreme-value agent must not change other agents'
    transformed values (this would happen if winsorization were recomputed from
    the inference batch)."""
    train_df = _make_training_df(n=500, seed=4)
    fitted_params, pd_features, log_cols, cap_cols, signed_log_cols = _build_fitted_params(train_df)

    normal_df = _make_training_df(n=50, seed=5)

    # Extreme agent: all numeric features set to 10× the column maximum from training
    extreme_row = normal_df.iloc[[0]].copy()
    for col in [c for c in pd_features if c in extreme_row.columns]:
        extreme_row[col] = float(train_df[col].max()) * 10

    # Batch: 50 normal agents + 500 copies of the extreme agent
    polluted_df = pd.concat([normal_df] + [extreme_row] * 500, ignore_index=True)

    result_clean = _transform_with_fitted(
        normal_df, fitted_params, pd_features, log_cols, cap_cols, signed_log_cols
    )
    result_polluted = _transform_with_fitted(
        polluted_df, fitted_params, pd_features, log_cols, cap_cols, signed_log_cols
    )

    feature_cols = [c for c in pd_features if c in normal_df.columns]
    n_normal = len(normal_df)
    for col in feature_cols:
        if col not in result_clean.columns or col not in result_polluted.columns:
            continue
        clean_vals = result_clean[col].values.astype(float)
        polluted_normal_vals = result_polluted[col].values[:n_normal].astype(float)
        np.testing.assert_allclose(
            clean_vals,
            polluted_normal_vals,
            atol=1e-5,
            err_msg=f"Column '{col}': normal agents' values changed when extreme rows were added",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Determinism across repeated executions
# ─────────────────────────────────────────────────────────────────────────────


def test_deterministic_across_repeated_executions():
    """Running inference twice on the same input produces identical output."""
    train_df = _make_training_df(n=500, seed=6)
    fitted_params, pd_features, log_cols, cap_cols, signed_log_cols = _build_fitted_params(train_df)

    scoring_df = _make_training_df(n=80, seed=7)

    result_1 = _transform_with_fitted(
        scoring_df, fitted_params, pd_features, log_cols, cap_cols, signed_log_cols
    )
    result_2 = _transform_with_fitted(
        scoring_df, fitted_params, pd_features, log_cols, cap_cols, signed_log_cols
    )

    feature_cols = [c for c in pd_features if c in scoring_df.columns]
    for col in feature_cols:
        if col not in result_1.columns or col not in result_2.columns:
            continue
        np.testing.assert_array_equal(
            result_1[col].values,
            result_2[col].values,
            err_msg=f"Column '{col}' is non-deterministic across identical runs",
        )
