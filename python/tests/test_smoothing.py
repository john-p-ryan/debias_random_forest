from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from debiasforest import (SmoothedForestRegressor, SmoothingSpline, default_forest,
                          smooth_spline)
from debiasforest.smoothing import nknots_smspl

DATA = Path(__file__).parent / "data"
_data = pd.read_csv(DATA / "smooth_spline_data.csv")
_pred = pd.read_csv(DATA / "smooth_spline_pred.csv")
_meta = pd.read_csv(DATA / "smooth_spline_meta.csv")
CASES = list(_meta["case"])


@pytest.mark.parametrize("case", CASES)
def test_spline_matches_r_fixtures(case):
    d = _data[_data.case == case]
    p = _pred[_pred.case == case]
    m = _meta[_meta.case == case].iloc[0]
    fit = smooth_spline(d.x.values, d.y.values)
    assert len(fit.knots) - 6 == m.nknots
    assert len(fit.x) == m.nx
    assert fit.spar == pytest.approx(m.spar, abs=1e-4)
    assert fit.lambda_ == pytest.approx(m["lambda"], rel=1e-3)
    assert fit.df == pytest.approx(m.df, rel=1e-4)
    assert fit.crit == pytest.approx(m.crit, rel=1e-5)
    pred = fit.predict(p.x.values)  # includes points outside the data range
    scale = np.abs(p.pred.values).max()
    np.testing.assert_allclose(pred, p.pred.values, atol=1e-5 * scale, rtol=1e-5)


def test_linear_extrapolation():
    d = _data[_data.case == "moderate_cubic"]
    fit = smooth_spline(d.x.values, d.y.values)
    hi = d.x.max()
    xs = hi + np.array([0.5, 1.0, 1.5])
    v = fit.predict(xs)
    assert v[2] - v[1] == pytest.approx(v[1] - v[0], rel=1e-10)
    # continuous at the boundary, with the boundary slope
    eps = 1e-7
    assert fit.predict(hi + eps) == pytest.approx(fit.predict(hi), abs=1e-5)
    slope = fit.predict(hi, deriv=1)
    assert (v[1] - v[0]) / 0.5 == pytest.approx(slope, rel=1e-8)
    lo = d.x.min()
    w = fit.predict(lo - np.array([1.0, 2.0, 3.0]))
    assert w[2] - w[1] == pytest.approx(w[1] - w[0], rel=1e-10)


def test_ties_are_aggregated_and_order_does_not_matter():
    rng = np.random.RandomState(0)
    x = np.round(rng.rand(80) * 5, 1)
    y = np.sin(x) + 0.2 * rng.randn(80)
    f1 = smooth_spline(x, y)
    perm = rng.permutation(80)
    f2 = smooth_spline(x[perm], y[perm])
    grid = np.linspace(-1, 6, 30)
    np.testing.assert_allclose(f1.predict(grid), f2.predict(grid), atol=1e-8)
    assert len(f1.x) == len(np.unique(x))
    assert f1.w.sum() == 80  # tie counts are the aggregated weights


def test_equivariance_and_fixed_parameters():
    rng = np.random.RandomState(1)
    x = rng.rand(60)
    y = np.cos(3 * x) + 0.1 * rng.randn(60)
    f = smooth_spline(x, y)
    g = smooth_spline(2 * x + 1, 5 * y - 3)  # GCV is invariant to affine changes of y and x
    assert g.spar == pytest.approx(f.spar, abs=1e-4)
    np.testing.assert_allclose(g.predict(2 * x + 1), 5 * f.predict(x) - 3, atol=1e-3)
    fixed = smooth_spline(x, y, spar=f.spar)
    assert fixed.df == pytest.approx(f.df, rel=2e-3)
    assert smooth_spline(x, y, spar=1.5).df < smooth_spline(x, y, spar=0.1).df
    with pytest.raises(ValueError):
        smooth_spline(x, y, spar=1.0, lambda_=1.0)
    with pytest.raises(ValueError):
        smooth_spline([1, 2, 3], [1, 2, 3])


def test_nknots_rule():
    # values of R's stats:::.nknots.smspl
    got = [nknots_smspl(n) for n in (49, 50, 51, 100, 200, 300, 800, 900, 3200, 5000)]
    assert got == [49, 49, 50, 62, 99, 105, 140, 142, 200, 204]


@pytest.mark.parametrize("method,deg", [("linear", 1), ("quadratic", 2), ("cubic", 3)])
def test_polynomial_methods_match_closed_form(f1_data, method, deg):
    X, y, Xt, _ = f1_data
    base = default_forest(random_state=0, n_estimators=50).fit(X, y)
    m = SmoothedForestRegressor(method=method, base_estimator=base, prefit=True).fit(X, y)
    oob = m.base_oob_prediction_
    coef = np.polyfit(oob, y, deg)
    np.testing.assert_allclose(m.oob_prediction_, np.polyval(coef, oob), atol=1e-8)
    np.testing.assert_allclose(m.predict(Xt), np.polyval(coef, base.predict(Xt)), atol=1e-8)
    if deg == 1:  # normal equations
        A = np.c_[np.ones_like(oob), oob]
        b = np.linalg.solve(A.T @ A, A.T @ y)
        np.testing.assert_allclose(m.predict(Xt), b[0] + b[1] * base.predict(Xt), atol=1e-8)


def test_spline_method(f1_data):
    X, y, Xt, _ = f1_data
    base = default_forest(random_state=0, n_estimators=50).fit(X, y)
    m = SmoothedForestRegressor(method="spline", base_estimator=base, prefit=True).fit(X, y)
    ref = smooth_spline(m.base_oob_prediction_, y)
    np.testing.assert_allclose(m.predict(Xt), ref.predict(base.predict(Xt)), atol=1e-9)
    np.testing.assert_allclose(m.oob_prediction_, ref.predict(m.base_oob_prediction_),
                               atol=1e-9)
    with pytest.raises(ValueError):
        SmoothedForestRegressor(method="bogus").fit(X, y)


def test_smoothed_without_prefit_is_deterministic(f1_data):
    X, y, Xt, _ = f1_data
    mk = lambda: SmoothedForestRegressor(base_estimator=default_forest(n_estimators=30),
                                         random_state=3)
    np.testing.assert_array_equal(mk().fit(X, y).predict(Xt), mk().fit(X, y).predict(Xt))
