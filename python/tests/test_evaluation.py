import numpy as np
import pytest

from debiasforest import (BoostedForestRegressor, DatasetUnavailableError, default_forest,
                          load_yacht, paper_label, predict_all_methods, repeated_splits,
                          select_n_iterations, simulate, simulation_study)
from debiasforest import datasets
from debiasforest.evaluation import method_labels

LABELS = ["Un", "De", "Qu", "Cu", "Sp", "Bf", "Bf(1)", "Bf(2)", "Bf(3)", "Cond"]


def test_predict_all_methods_order_and_shapes(f1_data):
    X, y, Xt, _ = f1_data
    out = predict_all_methods(X, y, Xt, random_state=0, n_estimators=40)
    assert list(out) == LABELS
    assert all(v.shape == (len(Xt),) for v in out.values())
    assert method_labels() == LABELS
    assert paper_label(2) == "Bf(1)"
    # Bf(k) = base + k+1 residual forests
    again = predict_all_methods(X, y, Xt, random_state=0, n_estimators=40)
    for k in out:
        np.testing.assert_array_equal(out[k], again[k])
    short = predict_all_methods(X, y, Xt, random_state=0, n_estimators=20,
                                max_extra_iterations=1)
    assert list(short) == ["Un", "De", "Qu", "Cu", "Sp", "Bf", "Bf(1)", "Cond"]


def test_predict_all_methods_shares_base_forest(f1_data):
    X, y, Xt, _ = f1_data
    out = predict_all_methods(X, y, Xt, random_state=5, n_estimators=30)
    # Bf's first stage is the Un forest: Bf - Un = residual forest, so Bf(1) - Bf = next one
    assert not np.allclose(out["Un"], out["Bf"])
    diffs = out["Bf"] - out["Un"]
    assert np.std(diffs) > 0


def test_simulate_functions():
    X, y, f = simulate("fig1", 5, rng=0)
    assert X.shape == (5, 4)
    np.testing.assert_allclose(f, 3 * X[:, 0] + X[:, 1] ** 2 + np.sin(X[:, 2]) - np.sqrt(X[:, 3]))
    for name, p in [("friedman1", 5), ("f1", 5), ("f2", 10), ("f3", 10), ("f4", 5), ("f5", 5)]:
        X, y, f = simulate(name, 50, rng=1)
        assert X.shape == (50, p) and y.shape == f.shape == (50,)
        assert X.min() >= 0 and X.max() <= 1
    X, _, f = simulate("f1", 4, rng=3)
    np.testing.assert_allclose(f, 10 * np.sin(np.pi * X[:, 0] * X[:, 1])
                               + 20 * (X[:, 2] - .5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4])
    X, _, f = simulate("f5", 4, rng=3)
    np.testing.assert_allclose(f, 10 * np.cos(np.pi * X[:, 0] * X[:, 2]) + 10 * X[:, 2] ** 2
                               + 10 * (X[:, 0] + X[:, 3]))
    # noise level: default sigma 0.5 for fig1 / 1 else; explicit sigma
    X, y, f = simulate("fig1", 20000, rng=2)
    assert np.std(y - f) == pytest.approx(0.5, rel=0.03)
    X, y, f = simulate("f1", 20000, sigma=4, rng=2)
    assert np.std(y - f) == pytest.approx(4, rel=0.03)
    with pytest.raises(ValueError):
        simulate("nope", 3)
    a = simulate("f2", 10, rng=7)
    b = simulate("f2", 10, rng=7)
    np.testing.assert_array_equal(a[1], b[1])


def test_simulation_study_small():
    kw = dict(n=60, n_test=100, n_reps=4, random_state=0, n_estimators=15)
    r = simulation_study("f1", 1.0, **kw)
    assert r.labels == LABELS
    assert r.mspe.shape == (4, 10)
    assert r.mean.shape == r.sd.shape == r.bias2.shape == (10,)
    assert np.all(r.bias2 >= 0) and np.all(r.mean > 0)
    s = r.summary()
    assert set(s) == set(LABELS) and set(s["Un"]) == {"mspe", "sd", "bias2"}
    r2 = simulation_study("f1", 1.0, n_jobs=2, **kw)
    np.testing.assert_allclose(r.mspe, r2.mspe)
    np.testing.assert_allclose(r.bias2, r2.bias2)


def test_repeated_splits_small(f1_data):
    X, y, _, _ = f1_data
    r = repeated_splits(X, y, n_splits=3, random_state=0, n_estimators=15)
    assert r.mspe.shape == (3, 10) and r.labels == LABELS
    r2 = repeated_splits(X, y, n_splits=3, random_state=0, n_estimators=15, n_jobs=2)
    np.testing.assert_allclose(r.mspe, r2.mspe)


def test_select_n_iterations_sanity():
    X, y, _ = simulate("f1", 150, sigma=1.0, rng=5)
    res = select_n_iterations(X, y, max_iterations=3, n_splits=4, random_state=0,
                              n_estimators=40)
    assert res.mean_mspe.shape == res.sd_mspe.shape == (4,)
    assert res.mspe.shape == (4, 4)
    assert 0 <= res.n_iterations <= 3
    assert res.mean_mspe[1] < res.mean_mspe[0]  # boosting helps on Friedman 1
    assert res.n_iterations >= 1
    m = res.mean_mspe
    k = res.n_iterations
    assert all(m[j] >= m[j + 1] for j in range(k))  # no increase before the selected k
    assert k == 3 or m[k] < m[k + 1]


def test_select_n_iterations_rule_on_noise():
    rng = np.random.RandomState(0)
    X, y = rng.rand(60, 3), rng.randn(60)
    res = select_n_iterations(X, y, max_iterations=3, n_splits=3, random_state=1,
                              n_estimators=20)
    m = res.mean_mspe
    expected = next((j for j in range(3) if m[j] < m[j + 1]), 3)
    assert res.n_iterations == expected


def test_load_yacht_offline_and_cache(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(datasets.urllib.request, "urlopen", boom)
    with pytest.raises(DatasetUnavailableError):
        load_yacht(cache_dir=tmp_path)
    rng = np.random.RandomState(0)
    arr = rng.rand(308, 7)
    (tmp_path / "yacht_hydrodynamics.data").write_text(
        "\n".join(" ".join(f"{v:.4f}" for v in row) for row in arr))
    X, y = load_yacht(cache_dir=tmp_path)
    assert X.shape == (308, 6) and y.shape == (308,)
    np.testing.assert_allclose(y, arr[:, -1], atol=1e-4)


def test_bias_plot_runs(f1_data):
    import matplotlib
    matplotlib.use("Agg")
    from debiasforest import bias_plot
    X, y, Xt, ft = f1_data
    ax = bias_plot(ft, {"Un": ft + 0.1, "Bf": ft - 0.1})
    assert len(ax.collections) == 2
    ax2 = bias_plot(ft, ft * 0.9)
    assert ax2.get_xlabel() == "observed"
