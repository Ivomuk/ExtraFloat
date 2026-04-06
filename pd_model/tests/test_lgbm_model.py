"""Tests for pd_model.modeling.lgbm_model."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pd_model.modeling.lgbm_model import evaluate_lgbm, train_lgbm


def _make_data(n: int = 400, n_features: int = 5, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(0, 1, (n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series((X["feat_0"] + rng.normal(0, 0.5, n) > 0).astype(int))
    split = int(n * 0.7)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


class TestTrainLgbm:
    def test_returns_model_and_two_scored_dfs(self):
        Xtr, ytr, Xva, yva = _make_data()
        model, train_sc, val_sc = train_lgbm(Xtr, ytr, Xva, yva)
        assert model is not None
        assert isinstance(train_sc, pd.DataFrame)
        assert isinstance(val_sc, pd.DataFrame)

    def test_scored_dfs_have_required_cols(self):
        Xtr, ytr, Xva, yva = _make_data()
        _, train_sc, val_sc = train_lgbm(Xtr, ytr, Xva, yva)
        for df in (train_sc, val_sc):
            assert "bad_state" in df.columns
            assert "raw_score" in df.columns

    def test_raw_scores_in_01(self):
        Xtr, ytr, Xva, yva = _make_data()
        _, _, val_sc = train_lgbm(Xtr, ytr, Xva, yva)
        assert val_sc["raw_score"].between(0, 1).all()

    def test_val_auc_above_05_on_separable_data(self):
        Xtr, ytr, Xva, yva = _make_data(600)
        _, _, val_sc = train_lgbm(Xtr, ytr, Xva, yva)
        auc = roc_auc_score(val_sc["bad_state"], val_sc["raw_score"])
        assert auc > 0.5

    def test_column_mismatch_raises(self):
        Xtr, ytr, Xva, yva = _make_data()
        Xva2 = Xva.rename(columns={"feat_0": "feat_WRONG"})
        with pytest.raises(AssertionError, match="column order"):
            train_lgbm(Xtr, ytr, Xva2, yva)


class TestEvaluateLgbm:
    def test_returns_dict_with_auc_keys(self):
        Xtr, ytr, Xva, yva = _make_data()
        model, train_sc, val_sc = train_lgbm(Xtr, ytr, Xva, yva)
        result = evaluate_lgbm(model, train_sc, val_sc)
        assert "train_auc" in result
        assert "val_auc" in result
        assert "feature_importance" in result

    def test_val_deciles_returned(self):
        Xtr, ytr, Xva, yva = _make_data(600)
        model, train_sc, val_sc = train_lgbm(Xtr, ytr, Xva, yva)
        result = evaluate_lgbm(model, train_sc, val_sc)
        assert "val_deciles" in result
        assert "overall" in result["val_deciles"]
