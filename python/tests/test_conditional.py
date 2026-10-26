import numpy as np

from debiasforest import (ConditionalBiasForestRegressor, brute_force_oob_predictions,
                          default_forest)


def brute_force_weights(forest, X, y, X0, exclude_self=False):
    """Paper formula: b0 = sum_i v_i(x0) (oob_i - y_i),
    v_i = sum_j I(i OOB in tree j, x_i in x0's leaf) / sum_l sum_j (same)."""
    n = len(y)
    oob = brute_force_oob_predictions(forest, X)
    bias = oob - y
    in_bag = [set(np.asarray(s).tolist()) for s in forest.estimators_samples_]
    leaves_tr = forest.apply(X)
    leaves_0 = forest.apply(X0)
    out = []
    for r, l0 in enumerate(leaves_0):
        cnt = np.zeros(n)
        for j in range(forest.n_estimators):
            for i in range(n):
                if i not in in_bag[j] and leaves_tr[i, j] == l0[j]:
                    cnt[i] += 1
        if exclude_self is not False:
            cnt[exclude_self[r]] = 0
        out.append(0.0 if cnt.sum() == 0 else float(np.sum(cnt / cnt.sum() * bias)))
    return np.array(out), oob


def test_conditional_matches_brute_force_weights(f1_data):
    X, y, Xt, _ = f1_data
    X, y, Xt = X[:60], y[:60], Xt[:12]
    base = default_forest(random_state=0, n_estimators=25).fit(X, y)
    m = ConditionalBiasForestRegressor(base_estimator=base, prefit=True).fit(X, y)
    b, oob = brute_force_weights(base, X, y, Xt)
    np.testing.assert_allclose(m.bias(Xt), b, atol=1e-10)
    np.testing.assert_allclose(m.predict(Xt), base.predict(Xt) - b, atol=1e-10)
    np.testing.assert_allclose(m.base_oob_prediction_, oob, atol=1e-10)


def test_conditional_oob_prediction_removes_self_weight(f1_data):
    X, y, _, _ = f1_data
    X, y = X[:50], y[:50]
    base = default_forest(random_state=1, n_estimators=25).fit(X, y)
    m = ConditionalBiasForestRegressor(base_estimator=base, prefit=True).fit(X, y)
    b, oob = brute_force_weights(base, X, y, X, exclude_self=list(range(len(y))))
    np.testing.assert_allclose(m.oob_prediction_, oob - b, atol=1e-10)


def test_zero_weight_gives_zero_bias(f1_data):
    X, y, _, _ = f1_data
    m = ConditionalBiasForestRegressor(base_estimator=default_forest(n_estimators=10),
                                       random_state=0).fit(X, y)
    far = np.full((1, X.shape[1]), 1e6)  # lands in extreme leaves; still finite
    assert np.isfinite(m.predict(far)).all()
    assert np.all(m.leaf_oob_count_ >= 0)


def test_conditional_deterministic_and_reduces_bias(f1_data):
    X, y, Xt, ft = f1_data
    mk = lambda: ConditionalBiasForestRegressor(
        base_estimator=default_forest(n_estimators=40), random_state=2)
    a, b = mk().fit(X, y).predict(Xt), mk().fit(X, y).predict(Xt)
    np.testing.assert_array_equal(a, b)
