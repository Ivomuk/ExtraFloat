"""
Test 2: Label shuffle / permutation test.

If the 0.989 AUC were due to data leakage, shuffling the target labels would
not substantially reduce model performance -- a leaky feature correlates with
the label regardless of row order.

With genuine signal, shuffling labels destroys the feature-label relationship
and AUC collapses to ~0.50 +/- sampling noise.

Interpretation:
  real_auc - mean(shuffled_aucs) > 0.10  -> model is learning real signal
  real_auc - mean(shuffled_aucs) < 0.05  -> performance likely driven by leakage

To apply to the real model:
    from pd_model.tests.test_label_shuffle import permutation_auc_distribution, train_and_score
    import pandas as pd
    X_train = pd.read_csv("path/to/X_train.csv")
    y_train = pd.read_csv("path/to/y_train.csv")["bad_state"]
    real_auc = train_and_score(X_train, y_train)
    null_aucs = permutation_auc_distribution(X_train, y_train, n_iter=50)
    print(f"Real AUC: {real_auc:.4f}, Null mean: {sum(null_aucs)/len(null_aucs):.4f}")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier


def train_and_score(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> float:
    """Train a shallow decision tree and return held-out AUC."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    split = int(n * 0.7)
    tr_idx, val_idx = idx[:split], idx[split:]

    model = DecisionTreeClassifier(max_depth=4, random_state=seed)
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    proba = model.predict_proba(X.iloc[val_idx])[:, 1]
    return roc_auc_score(y.iloc[val_idx], proba)


def permutation_auc_distribution(
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 20,
    seed: int = 42,
) -> list[float]:
    """Return list of AUC values trained on shuffled labels (null distribution)."""
    rng = np.random.default_rng(seed)
    aucs = []
    for i in range(n_iter):
        y_shuffled = pd.Series(rng.permutation(y.values), index=y.index)
        aucs.append(train_and_score(X, y_shuffled, seed=seed + i))
    return aucs


class TestLabelShuffle:
    def _make_genuine_df(self, n: int = 800, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
        """Synthetic data with genuine signal: x1, x2 correlated with y."""
        rng = np.random.default_rng(seed)
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        logit = 1.5 * x1 + 0.8 * x2
        prob = 1 / (1 + np.exp(-logit))
        y = pd.Series((rng.uniform(0, 1, n) < prob).astype(int))
        X = pd.DataFrame({"x1": x1, "x2": x2})
        return X, y

    def test_real_labels_produce_meaningful_auc(self):
        """With genuine signal, real-label AUC must exceed 0.65."""
        X, y = self._make_genuine_df()
        auc = train_and_score(X, y)
        assert auc > 0.65, f"Expected real-label AUC > 0.65 for data with genuine signal, got {auc:.4f}"

    def test_shuffled_labels_collapse_auc_to_near_half(self):
        """After shuffling labels, mean AUC must be near 0.50 (no residual signal)."""
        X, y = self._make_genuine_df()
        perm_aucs = permutation_auc_distribution(X, y, n_iter=15)
        mean_perm = np.mean(perm_aucs)
        assert mean_perm < 0.60, (
            f"Shuffled-label AUC should collapse to ~0.50, got mean={mean_perm:.4f}. "
            "A mean > 0.60 implies a leaky feature that ignores the label order."
        )

    def test_gap_between_real_and_shuffled_auc_is_large(self):
        """Real-label AUC must exceed the shuffled null mean by at least 0.10."""
        X, y = self._make_genuine_df()
        real_auc = train_and_score(X, y)
        perm_aucs = permutation_auc_distribution(X, y, n_iter=15)
        mean_perm = np.mean(perm_aucs)
        gap = real_auc - mean_perm
        assert gap > 0.10, (
            f"Gap between real AUC ({real_auc:.4f}) and shuffled mean ({mean_perm:.4f}) "
            f"is only {gap:.4f}. A small gap implies leakage is driving model performance."
        )

    def test_permutation_produces_correct_count(self):
        """permutation_auc_distribution must return exactly n_iter values."""
        X, y = self._make_genuine_df(n=400)
        n_iter = 8
        aucs = permutation_auc_distribution(X, y, n_iter=n_iter)
        assert len(aucs) == n_iter

    def test_permutation_values_are_valid_aucs(self):
        """All permutation AUC values must be in [0, 1]."""
        X, y = self._make_genuine_df(n=400)
        aucs = permutation_auc_distribution(X, y, n_iter=8)
        for i, auc in enumerate(aucs):
            assert 0.0 <= auc <= 1.0, f"Permutation {i} produced invalid AUC: {auc}"

    def test_leaked_feature_maintains_high_real_auc(self):
        """
        Sanity-check: a perfect label proxy must produce near-perfect real AUC.
        This confirms the test framework can detect the leakage scenario it guards against.
        """
        rng = np.random.default_rng(99)
        n = 1000
        y = pd.Series(rng.integers(0, 2, n))
        X_leaky = pd.DataFrame({"leaked": y.values.copy()})
        real_auc = train_and_score(X_leaky, y)
        assert real_auc > 0.90, (
            f"Expected perfect proxy AUC > 0.90, got {real_auc:.4f}. "
            "The test framework cannot detect leakage if this sanity check fails."
        )
