import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from debiasforest import (BoostedForestRegressor, brute_force_oob_predictions, default_forest,
                          paper_label, simulate)


def small(**kw):
    return BoostedForestRegressor(base_estimator=default_forest(n_estimators=40), **kw)


def test_paper_label():
    assert [paper_label(k) for k in range(5)] == ["Un", "Bf", "Bf(1)", "Bf(2)", "Bf(3)"]
    with pytest.raises(ValueError):
        paper_label(-1)


def test_zero_iterations_equals_plain_forest(f1_data):
    X, y, Xt, _ = f1_data
    base = default_forest(random_state=4, n_estimators=40).fit(X, y)
    m = BoostedForestRegressor(n_iterations=0, base_estimator=base, prefit=True).fit(X, y)
    np.testing.assert_allclose(m.predict(Xt), base.predict(Xt), atol=1e-10)
    # un-prefit: equals its own first stage, and matches an identically seeded forest
    m2 = small(n_iterations=0, random_state=0).fit(X, y)
    np.testing.assert_allclose(m2.predict(Xt), m2.estimators_[0].predict(Xt), atol=1e-10)
    assert m2.n_iterations_ == 0 and len(m2.estimators_) == 1


def test_stage_structure_and_oob_prediction(f1_data):
    X, y, Xt, _ = f1_data
    m = small(n_iterations=2, random_state=1).fit(X, y)
    f0, f1, f2 = m.estimators_
    o0 = brute_force_oob_predictions(f0, X)
    o1 = brute_force_oob_predictions(f1, X)
    o2 = brute_force_oob_predictions(f2, X)
    np.testing.assert_allclose(m.oob_prediction_, o0 + o1 + o2, atol=1e-9)
    # stage forests are fitted to successive OOB residuals
    r1 = y - o0
    np.testing.assert_allclose(f1.estimators_[0].tree_.value[0, 0, 0],
                               r1[f1.estimators_samples_[0]].mean(), atol=1e-9)
    np.testing.assert_allclose(f2.estimators_[0].tree_.value[0, 0, 0],
                               (r1 - o1)[f2.estimators_samples_[0]].mean(), atol=1e-9)
    np.testing.assert_allclose(m.predict(Xt), f0.predict(Xt) + f1.predict(Xt) + f2.predict(Xt),
                               atol=1e-9)
    # independent seeds per stage -> different bootstraps
    assert not np.array_equal(f0.estimators_samples_[0], f1.estimators_samples_[0])


def test_staged_predict_matches_final(f1_data):
    X, y, Xt, _ = f1_data
    m = small(n_iterations=3, random_state=2).fit(X, y)
    staged = list(m.staged_predict(Xt))
    assert len(staged) == 4
    np.testing.assert_allclose(staged[-1], m.predict(Xt), atol=1e-10)
    np.testing.assert_allclose(staged[0], m.estimators_[0].predict(Xt), atol=1e-10)
    for k in range(1, 4):
        m_k = small(n_iterations=k, random_state=2).fit(X, y)
        np.testing.assert_allclose(m_k.predict(Xt), staged[k], atol=1e-9)


def test_determinism(f1_data):
    X, y, Xt, _ = f1_data
    a = small(n_iterations=2, random_state=7).fit(X, y).predict(Xt)
    b = small(n_iterations=2, random_state=7).fit(X, y).predict(Xt)
    c = small(n_iterations=2, random_state=8).fit(X, y).predict(Xt)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_oob_stopping_picks_argmin():
    # pure noise: boosting residual noise only adds variance, the OOB MSE path rises
    rng = np.random.RandomState(0)
    X = rng.rand(100, 3)
    y = rng.randn(100)
    fixed = small(n_iterations=6, random_state=3).fit(X, y)
    path = fixed.oob_mse_path_
    assert len(path) == 7
    m = small(n_iterations=6, stopping="oob", random_state=3).fit(X, y)
    # same seeds -> the OOB run reproduces a prefix of the fixed path
    np.testing.assert_allclose(m.oob_mse_path_, path[:len(m.oob_mse_path_)])
    assert m.n_iterations_ == int(np.argmin(m.oob_mse_path_))
    assert len(m.estimators_) == m.n_iterations_ + 1
    # stopped as soon as the MSE exceeded 1.1 x the running minimum (or ran to the end)
    p = m.oob_mse_path_
    if len(p) < 7:
        assert p[-1] > 1.1 * p[:-1].min()
    assert all(p[j] <= 1.1 * p[:j].min() for j in range(1, len(p) - 1))
    assert m.n_iterations_ < len(p)
    staged = list(m.staged_predict(X))
    assert len(staged) == m.n_iterations_ + 1


def test_oob_stopping_keeps_signal_stages(f1_data):
    X, y, _, _ = f1_data
    m = small(n_iterations=4, stopping="oob", random_state=1).fit(X, y)
    assert m.n_iterations_ >= 1  # Friedman 1: one residual forest reduces the OOB MSE
    assert m.oob_mse_path_[m.n_iterations_] == m.oob_mse_path_.min()


def test_prefit_validation(f1_data):
    X, y, _, _ = f1_data
    with pytest.raises(ValueError):
        BoostedForestRegressor(prefit=True).fit(X, y)
    with pytest.raises(NotFittedError):
        BoostedForestRegressor(prefit=True, base_estimator=default_forest(n_estimators=5)).fit(X, y)
    base = default_forest(random_state=0, n_estimators=10).fit(X, y)
    with pytest.raises(ValueError):
        BoostedForestRegressor(prefit=True, base_estimator=base).fit(X[:50], y[:50])
    with pytest.raises(ValueError):
        BoostedForestRegressor(stopping="bogus").fit(X, y)
    with pytest.raises(NotFittedError):
        BoostedForestRegressor().predict(X)


def test_prefit_does_not_refit_base(f1_data):
    X, y, Xt, _ = f1_data
    base = default_forest(random_state=0, n_estimators=30).fit(X, y)
    before = base.predict(Xt)
    m = BoostedForestRegressor(n_iterations=1, base_estimator=base, prefit=True,
                               random_state=0).fit(X, y)
    assert m.estimators_[0] is base
    np.testing.assert_array_equal(base.predict(Xt), before)


def test_sklearn_compatibility(f1_data):
    X, y, _, _ = f1_data
    est = small(n_iterations=1, random_state=0)
    c = clone(est)
    assert c.get_params()["n_iterations"] == 1
    scores = cross_val_score(est, X, y, cv=3)
    assert scores.shape == (3,) and np.all(np.isfinite(scores))
    pipe = make_pipeline(StandardScaler(), small(n_iterations=1, random_state=0))
    assert pipe.fit(X, y).predict(X[:5]).shape == (5,)
    est.set_params(n_iterations=2)
    assert est.n_iterations == 2


def test_alternative_ensembles(f1_data):
    from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor
    from sklearn.tree import DecisionTreeRegressor
    X, y, Xt, _ = f1_data
    for base in (ExtraTreesRegressor(30, bootstrap=True, min_samples_split=6),
                 BaggingRegressor(DecisionTreeRegressor(), n_estimators=30, max_features=0.8)):
        m = BoostedForestRegressor(n_iterations=1, base_estimator=base, random_state=0).fit(X, y)
        assert m.predict(Xt).shape == (len(Xt),)
        assert np.all(np.isfinite(m.oob_prediction_))
        assert base.get_params().get("random_state") is None  # the template is not mutated


def test_bias_reduction_friedman1():
    X, y, _ = simulate("f1", 200, sigma=1.0, rng=21)
    Xt, _, ft = simulate("f1", 1000, sigma=0.0, rng=22)
    un = default_forest(random_state=0, n_estimators=150).fit(X, y)
    bf = BoostedForestRegressor(n_iterations=1, base_estimator=un, prefit=True,
                                random_state=0).fit(X, y)
    mse_un = np.mean((un.predict(Xt) - ft) ** 2)
    mse_bf = np.mean((bf.predict(Xt) - ft) ** 2)
    assert mse_bf < mse_un
