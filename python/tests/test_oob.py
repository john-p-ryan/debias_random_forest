import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from debiasforest import (BaggedEnsemble, brute_force_oob_predictions, default_forest,
                          oob_predictions, simulate)


@pytest.fixture(scope="module")
def data():
    X, y, _ = simulate("f1", 80, rng=3)
    return X, y


def test_oob_equals_sklearn_oob_prediction(data):
    X, y = data
    rf = RandomForestRegressor(n_estimators=150, oob_score=True, random_state=0,
                               min_samples_split=6).fit(X, y)
    ours = oob_predictions(rf, X)
    np.testing.assert_allclose(ours, rf.oob_prediction_, atol=1e-10)


@pytest.mark.parametrize("make", [
    lambda: RandomForestRegressor(60, random_state=1),
    lambda: ExtraTreesRegressor(60, bootstrap=True, random_state=1),
    lambda: BaggingRegressor(DecisionTreeRegressor(), n_estimators=60, random_state=1),
    lambda: BaggingRegressor(DecisionTreeRegressor(), n_estimators=60, max_features=0.6,
                             random_state=1),
    lambda: default_forest(random_state=1, n_estimators=60),
])
def test_oob_matches_brute_force_for_all_ensembles(data, make):
    X, y = data
    forest = make().fit(X, y)
    ours = oob_predictions(forest, X)
    brute = brute_force_oob_predictions(forest, X)
    np.testing.assert_allclose(ours, brute, atol=1e-10)
    ens = BaggedEnsemble(forest, n_samples=len(y))
    np.testing.assert_allclose(ens.predict(X), forest.predict(X), atol=1e-10)


def test_inbag_counts_are_bootstrap_draws(data):
    X, y = data
    rf = RandomForestRegressor(30, random_state=2).fit(X, y)
    ens = BaggedEnsemble(rf, n_samples=len(y))
    assert ens.inbag.shape == (30, len(y))
    assert np.all(ens.inbag.sum(axis=1) == len(y))  # n draws with replacement
    assert ens.inbag.max() > 1  # repeated draws are kept
    np.testing.assert_array_equal(ens.oob_mask, ens.inbag == 0)
    # about 37% of rows are OOB for each tree
    assert 0.3 < ens.oob_mask.mean() < 0.45


def test_fast_path_equals_fallback(data):
    X, y = data
    rf = RandomForestRegressor(20, random_state=2).fit(X, y)
    fast = BaggedEnsemble(rf, n_samples=len(y))
    slow = BaggedEnsemble(rf, n_samples=len(y))
    slow._fast = False
    np.testing.assert_allclose(fast.tree_predictions(X), slow.tree_predictions(X))
    np.testing.assert_array_equal(fast.apply(X), slow.apply(X))
    np.testing.assert_array_equal(fast.apply(X), rf.apply(X))


def test_never_oob_rows_fall_back_with_warning(data):
    X, y = data
    rf = RandomForestRegressor(2, random_state=0).fit(X, y)
    with pytest.warns(RuntimeWarning, match="never out-of-bag"):
        pred = oob_predictions(rf, X)
    assert np.all(np.isfinite(pred))


def test_default_forest_mtry_and_params():
    rng = np.random.RandomState(0)
    for p, mtry in [(4, 1), (5, 1), (10, 3), (1, 1)]:
        X = rng.rand(40, p)
        f = default_forest(random_state=0, n_estimators=3).fit(X, X.sum(1))
        assert f.estimators_[0].max_features == mtry
        assert f.max_features == "third"  # constructor parameter untouched
        assert f.min_samples_split == 6 and f.min_samples_leaf == 1 and f.bootstrap
    assert default_forest().n_estimators == 500
    c = clone(default_forest(random_state=5))
    assert c.get_params()["random_state"] == 5
