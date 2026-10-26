"""Method comparison utilities: all methods from one base forest, simulations, splits."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import check_random_state
from threadpoolctl import threadpool_limits

from ._forest import default_forest
from .boosted import BoostedForestRegressor, paper_label
from .conditional import ConditionalBiasForestRegressor
from .datasets import default_sigma, simulate
from .smoothing import SmoothedForestRegressor

__all__ = ["paper_label", "method_labels", "predict_all_methods", "select_n_iterations",
           "simulation_study", "repeated_splits", "SimulationResult", "SplitsResult",
           "SelectionResult"]

_MAX_SEED = 2**31 - 1


def method_labels(max_extra_iterations=3):
    """``["Un","De","Qu","Cu","Sp","Bf","Bf(1)",...,"Cond"]``."""
    return (["Un", "De", "Qu", "Cu", "Sp"]
            + [paper_label(k) for k in range(1, max_extra_iterations + 2)] + ["Cond"])


def predict_all_methods(X_train, y_train, X_test, random_state=None, max_extra_iterations=3,
                        n_estimators=500, base_estimator=None):
    """Test predictions of every method, all built on one shared base forest.

    Returns a dict ``label -> predictions`` in the order Un, De, Qu, Cu, Sp, Bf, Bf(1),
    ..., Bf(max_extra_iterations), Cond.  ``Bf(k)`` has ``k + 1`` residual forests and the
    boosted stages share the base forest (stage 0).
    """
    rng = check_random_state(random_state)
    seed_base, seed_boost = rng.randint(0, _MAX_SEED, size=2)
    base = (default_forest(random_state=seed_base, n_estimators=n_estimators)
            if base_estimator is None else clone(base_estimator).set_params(random_state=seed_base))
    base.fit(X_train, y_train)
    out = {"Un": base.predict(X_test)}
    for lab, method in (("De", "linear"), ("Qu", "quadratic"), ("Cu", "cubic"),
                        ("Sp", "spline")):
        m = SmoothedForestRegressor(method=method, base_estimator=base, prefit=True)
        out[lab] = m.fit(X_train, y_train).predict(X_test)
    bf = BoostedForestRegressor(n_iterations=max_extra_iterations + 1, base_estimator=base,
                                prefit=True, random_state=seed_boost)
    bf.fit(X_train, y_train)
    for k, pred in enumerate(bf.staged_predict(X_test)):
        if k >= 1:
            out[paper_label(k)] = pred
    cond = ConditionalBiasForestRegressor(base_estimator=base, prefit=True)
    out["Cond"] = cond.fit(X_train, y_train).predict(X_test)
    return out


# --------------------------------------------------------------------------- results
@dataclass
class SimulationResult:
    """Output of :func:`simulation_study`.

    ``mspe`` has shape (n_reps, n_methods); ``mean``, ``sd`` and ``bias2`` are per method.
    ``bias2`` is the mean over test points of (average prediction over reps - f(x))^2.
    """
    labels: list
    mspe: np.ndarray
    bias2: np.ndarray
    func_name: str = ""
    sigma: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def mean(self):
        return self.mspe.mean(axis=0)

    @property
    def sd(self):
        return self.mspe.std(axis=0, ddof=1) if self.mspe.shape[0] > 1 else np.zeros(len(self.labels))

    def summary(self):
        return {lab: {"mspe": float(m), "sd": float(s), "bias2": float(b)}
                for lab, m, s, b in zip(self.labels, self.mean, self.sd, self.bias2)}

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame({"MSPE": self.mean, "SD": self.sd, "bias2": self.bias2},
                            index=self.labels).T


@dataclass
class SplitsResult:
    """Output of :func:`repeated_splits`: ``mspe`` has shape (n_splits, n_methods)."""
    labels: list
    mspe: np.ndarray

    @property
    def mean(self):
        return self.mspe.mean(axis=0)

    @property
    def sd(self):
        return self.mspe.std(axis=0, ddof=1) if self.mspe.shape[0] > 1 else np.zeros(len(self.labels))

    def summary(self):
        return {lab: {"mspe": float(m), "sd": float(s)}
                for lab, m, s in zip(self.labels, self.mean, self.sd)}

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame({"MSPE": self.mean, "SD": self.sd}, index=self.labels).T


@dataclass
class SelectionResult:
    """Output of :func:`select_n_iterations`."""
    n_iterations: int
    mean_mspe: np.ndarray  # length max_iterations + 1 (k = 0..max)
    sd_mspe: np.ndarray
    mspe: np.ndarray  # (n_splits, max_iterations + 1)


# ------------------------------------------------------------------------ simulations
def _sim_rep(func_name, sigma, n, X_test, f_test, y_noise_seed, train_seed, fit_seed,
             max_extra_iterations, n_estimators):
    Xtr, ytr, _ = simulate(func_name, n, sigma=sigma, rng=train_seed)
    rng = np.random.RandomState(y_noise_seed)
    y_test = f_test + sigma * rng.standard_normal(f_test.shape[0])
    with threadpool_limits(1):  # no BLAS/OpenMP oversubscription inside workers
        preds = predict_all_methods(Xtr, ytr, X_test, random_state=fit_seed,
                                    max_extra_iterations=max_extra_iterations,
                                    n_estimators=n_estimators)
    P = np.array(list(preds.values()))  # (methods, n_test)
    return P, ((P - y_test[None, :]) ** 2).mean(axis=1)


def simulation_study(func_name, sigma=None, n=200, n_test=2000, n_reps=200,
                     random_state=None, n_jobs=None, n_estimators=500,
                     max_extra_iterations=3):
    """Monte-Carlo comparison of all methods on one of the paper's functions.

    The test predictors (and ``f(X_test)``) are generated once and held fixed; each
    replication draws a fresh training set, fresh test noise, fits all methods from one
    base forest and records the test MSPE.  Squared bias uses the average prediction over
    replications, so MSPE and bias^2 come from the same fits.  Replications run in parallel
    through joblib (``n_jobs``); forests are single-threaded inside workers.
    """
    rng = check_random_state(random_state)
    X_test, _, f_test = simulate(func_name, n_test, sigma=0.0, rng=rng)
    if sigma is None:
        sigma = default_sigma(func_name)
    seeds = rng.randint(0, _MAX_SEED, size=(n_reps, 3))
    res = Parallel(n_jobs=n_jobs)(
        delayed(_sim_rep)(func_name, sigma, n, X_test, f_test, s[0], s[1], s[2],
                          max_extra_iterations, n_estimators) for s in seeds)
    labels = method_labels(max_extra_iterations)
    preds = np.stack([r[0] for r in res])  # (reps, methods, n_test)
    mspe = np.stack([r[1] for r in res])
    avg = preds.mean(axis=0)
    bias2 = ((avg - f_test[None, :]) ** 2).mean(axis=1)
    return SimulationResult(labels=labels, mspe=mspe, bias2=bias2, func_name=str(func_name),
                            sigma=float(sigma))


# --------------------------------------------------------------------- real-data splits
def _split_rep(X, y, tr, te, fit_seed, max_extra_iterations, n_estimators):
    preds = predict_all_methods(X[tr], y[tr], X[te], random_state=fit_seed,
                                max_extra_iterations=max_extra_iterations,
                                n_estimators=n_estimators)
    return np.array([np.mean((y[te] - p) ** 2) for p in preds.values()])


def repeated_splits(X, y, n_splits=100, test_size=1 / 3, random_state=None, n_jobs=None,
                    n_estimators=500, max_extra_iterations=3):
    """Test MSPE of every method over repeated random train/test splits."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = check_random_state(random_state)
    splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size,
                            random_state=rng.randint(0, _MAX_SEED))
    seeds = rng.randint(0, _MAX_SEED, size=n_splits)
    res = Parallel(n_jobs=n_jobs)(
        delayed(_split_rep)(X, y, tr, te, s, max_extra_iterations, n_estimators)
        for (tr, te), s in zip(splitter.split(X), seeds))
    return SplitsResult(labels=method_labels(max_extra_iterations), mspe=np.stack(res))


# ------------------------------------------------------------- number of iterations
def _select_rep(X, y, tr, te, seed, max_iterations, n_estimators, base_estimator):
    m = BoostedForestRegressor(n_iterations=max_iterations, random_state=seed,
                               base_estimator=base_estimator
                               if base_estimator is not None
                               else default_forest(n_estimators=n_estimators))
    m.fit(X[tr], y[tr])
    return np.array([np.mean((y[te] - p) ** 2) for p in m.staged_predict(X[te])])


def select_n_iterations(X, y, max_iterations=6, n_splits=20, test_size=1 / 3,
                        random_state=None, n_jobs=None, n_estimators=500,
                        base_estimator=None):
    """The paper's stopping rule: repeated splits, stop when the mean test MSPE rises.

    On each split one boosted forest with ``max_iterations`` residual forests is fitted
    and ``staged_predict`` gives the test MSPE for k = 0..max_iterations.  The mean MSPE
    curve is averaged over splits; the selected k is the first k whose mean MSPE is lower
    than that at k + 1 (``max_iterations`` if the curve never increases).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = check_random_state(random_state)
    splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size,
                            random_state=rng.randint(0, _MAX_SEED))
    seeds = rng.randint(0, _MAX_SEED, size=n_splits)
    res = Parallel(n_jobs=n_jobs)(
        delayed(_select_rep)(X, y, tr, te, s, max_iterations, n_estimators, base_estimator)
        for (tr, te), s in zip(splitter.split(X), seeds))
    mspe = np.stack(res)
    mean = mspe.mean(axis=0)
    sd = mspe.std(axis=0, ddof=1) if n_splits > 1 else np.zeros_like(mean)
    k = max_iterations
    for j in range(max_iterations):
        if mean[j] < mean[j + 1]:
            k = j
            break
    return SelectionResult(n_iterations=k, mean_mspe=mean, sd_mspe=sd, mspe=mspe)
