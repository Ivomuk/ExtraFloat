"""Tests for pd_model.modeling.xgb_model."""
import numpy as np
import pandas as pd
import pytest

from pd_model.modeling.xgb_model import build_monotone_constraints, evaluate_xgb, train_xgb


def _make_data(n: int = 400, n_features: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(0, 1, (n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series((X["feat_0"] + rng.normal(0, 0.5, n) > 0).astype(int))
    split = int(n * 0.7)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


class TestBuildMonotoneConstraints:
    def test_returns_list_of_correct_length(self):
        Xtr, ytr, _, _ = _make_data()
        constraints = build_monotone_constraints(Xtr.columns.tolist(), Xtr, ytr)
        assert len(constraints) == Xtr.shape[1]

    def test_values_in_valid_set(self):
        Xtr, ytr, _, _ = _make_data()
        constraints = build_monotone_constraints(Xtr.columns.tolist(), Xtr, ytr)
        assert all(c in (-1, 0, 1) for c in constraints)

    def test_strongly_correlated_feature_gets_nonzero(self):
        rng = np.random.default_rng(7)
        n = 500
        x = rng.normal(0, 1, n)
        y = pd.Series((x + rng.normal(0, 0.1, n) > 0).astype(int))
        X = pd.DataFrame({"strong": x})
        constraints = build_monotone_constraints(["strong"], X, y)
        assert constraints[0] != 0

    def test_missing_feature_gets_zero(self):
        Xtr, ytr, _, _ = _make_data()
        constraints = build_monotone_constraints(["nonexistent"], Xtr, ytr)
        assert constraints == [0]


class TestTrainXgb:
    def test_returns_model_and_two_scored_dfs(self):
        Xtr, ytr, Xva, yva = _make_data(400)
        model, train_sc, val_sc = train_xgb(Xtr, ytr, Xva, yva)
        assert model is not None
        assert isinstance(train_sc, pd.DataFrame)
        assert isinstance(val_sc, pd.DataFrame)

    def test_scored_dfs_have_required_cols(self):
        Xtr, ytr, Xva, yva = _make_data(400)
        _, train_sc, val_sc = train_xgb(Xtr, ytr, Xva, yva)
        for df in (train_sc, val_sc):
            assert "bad_state" in df.columns
            assert "raw_score" in df.columns

    def test_raw_scores_in_01(self):
        Xtr, ytr, Xva, yva = _make_data(400)
        _, _, val_sc = train_xgb(Xtr, ytr, Xva, yva)
        assert val_sc["raw_score"].between(0, 1).all()

    def test_val_auc_above_05_on_separable_data(self):
        from sklearn.metrics import roc_auc_score
        Xtr, ytr, Xva, yva = _make_data(600)
        _, _, val_sc = train_xgb(Xtr, ytr, Xva, yva)
        auc = roc_auc_score(val_sc["bad_state"], val_sc["raw_score"])
        assert auc > 0.5

    def test_thin_file_flag_not_in_feature_matrix(self):
        Xtr, ytr, Xva, yva = _make_data(400)
        Xtr["thin_file_flag"] = 0
        Xva["thin_file_flag"] = 0
        with pytest.raises(AssertionError, match="thin_file_flag"):
            train_xgb(Xtr, ytr, Xva, yva)

    def test_column_mismatch_raises(self):
        Xtr, ytr, Xva, yva = _make_data(400)
        Xva2 = Xva.rename(columns={"feat_0": "feat_WRONG"})
        with pytest.raises(AssertionError, match="column order"):
            train_xgb(Xtr, ytr, Xva2, yva)


class TestEvaluateXgb:
    def test_returns_dict_with_auc_keys(self):
        Xtr, ytr, Xva, yva = _make_data(400)
        model, train_sc, val_sc = train_xgb(Xtr, ytr, Xva, yva)
        result = evaluate_xgb(model, train_sc, val_sc)
        assert "train_auc" in result
        assert "val_auc" in result
        assert "feature_importance" in result

    def test_feature_importance_is_series(self):
        Xtr, ytr, Xva, yva = _make_data(400)
        model, train_sc, val_sc = train_xgb(Xtr, ytr, Xva, yva)
        result = evaluate_xgb(model, train_sc, val_sc)
        assert isinstance(result["feature_importance"], pd.Series)
