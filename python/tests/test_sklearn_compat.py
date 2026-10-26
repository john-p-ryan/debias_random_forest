"""scikit-learn API compliance, clone/set_params round trips and no-mutation guarantees."""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.utils.estimator_checks import parametrize_with_checks

from debiasforest import (BoostedForestRegressor, ConditionalBiasForestRegressor,
                          SmoothedForestRegressor, default_forest, predict_all_methods)
from debiasforest._forest import DefaultForestRegressor


def _base():
    return default_forest(n_estimators=10)


# Checks that cannot apply: stock RandomForestRegressor fails the sample-weight
# equivalence checks too (bootstrap draws differ between weighted/repeated data); the
# array-API check is skipped by sklearn itself unless SCIPY_ARRAY_API is set.
_RF_INAPPLICABLE = {
    "check_sample_weight_equivalence_on_dense_data":
        "bootstrap resampling is not weight-equivariant (also true for stock sklearn RF)",
    "check_sample_weight_equivalence_on_sparse_data":
        "bootstrap resampling is not weight-equivariant (also true for stock sklearn RF)",
}


def _expected_failed(estimator):
    return _RF_INAPPLICABLE if isinstance(estimator, DefaultForestRegressor) else {}


_ESTIMATORS = [
    DefaultForestRegressor(n_estimators=10),
    BoostedForestRegressor(base_estimator=_base()),
    SmoothedForestRegressor(base_estimator=_base()),
    ConditionalBiasForestRegressor(base_estimator=_base()),
]


@parametrize_with_checks(_ESTIMATORS, expected_failed_checks=_expected_failed)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def test_default_forest_params_round_trip():
    f = default_forest(n_estimators=7, max_samples=0.5, criterion="absolute_error",
                       max_features="third", random_state=3)
    c = clone(f)
    assert c.get_params() == f.get_params()
    assert c.max_samples == 0.5 and c.criterion == "absolute_error"
    assert c.max_features == "third"
    assert f.set_params(max_samples=0.3) is f and f.max_samples == 0.3
    with pytest.raises(ValueError):
        f.set_params(not_a_param=1)


def test_default_forest_fit_validates_before_p():
    with pytest.raises(ValueError):
        default_forest(n_estimators=5).fit(np.arange(10.0), np.arange(10.0))
    f = default_forest(n_estimators=5, random_state=0).fit(
        np.random.RandomState(0).rand(30, 6).tolist(), np.arange(30.0))
    assert f.estimators_[0].max_features == 2
    assert f.max_features == "third"


def test_boosted_stage_forests_inherit_base_params():
    rng = np.random.RandomState(0)
    X = rng.rand(80, 3)
    y = X[:, 0] + rng.randn(80) * 0.1
    m = BoostedForestRegressor(n_iterations=2, random_state=0,
                               base_estimator=default_forest(n_estimators=10, max_samples=0.5))
    m.fit(X, y)
    assert len(m.estimators_) == 3
    for f in m.estimators_:
        assert f.max_samples == 0.5
        assert len(f.estimators_samples_[0]) == 40  # bootstrap of 0.5 * n


def test_predict_all_methods_does_not_mutate_estimator():
    rng = np.random.RandomState(1)
    X = rng.rand(60, 3)
    y = X[:, 0] + rng.randn(60) * 0.1
    est = default_forest(n_estimators=10, random_state=99)
    before = est.get_params()
    out = predict_all_methods(X, y, X[:5], random_state=0, max_extra_iterations=1,
                              base_estimator=est)
    assert "Un" in out
    assert est.get_params() == before
    assert not hasattr(est, "estimators_")
