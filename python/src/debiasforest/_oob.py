"""Shared machinery for the estimators: base-forest handling and OOB predictions."""
from __future__ import annotations

import weakref

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from ._forest import BaggedEnsemble, default_forest

__all__ = ["oob_predictions", "brute_force_oob_predictions"]

_MAX_SEED = 2**31 - 1


def oob_predictions(forest, X, n_samples=None):
    """Out-of-bag predictions of a fitted bagging ensemble on its training matrix ``X``."""
    X = np.asarray(X, dtype=np.float64)
    ens = BaggedEnsemble(forest, n_samples=X.shape[0] if n_samples is None else n_samples)
    return ens.oob_prediction(X)


def brute_force_oob_predictions(forest, X):
    """Slow reference implementation (per-estimator ``predict`` and membership sets)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    feats = getattr(forest, "estimators_features_", None)
    sums, cnts = np.zeros(n), np.zeros(n)
    for t, (est, idx) in enumerate(zip(forest.estimators_, forest.estimators_samples_)):
        oob = np.setdiff1d(np.arange(n), np.asarray(idx))
        if oob.size == 0:
            continue
        Xt = X[oob] if feats is None else X[oob][:, feats[t]]
        sums[oob] += est.predict(Xt)
        cnts[oob] += 1
    out = np.full(n, np.nan)
    ok = cnts > 0
    out[ok] = sums[ok] / cnts[ok]
    return out


_SAMPLES_CACHE = weakref.WeakKeyDictionary()
_CACHE_LIMIT = 50_000_000  # n_trees * n_samples


def _estimators_samples(forest):
    """``forest.estimators_samples_`` with a weak cache (sklearn rebuilds it on each access,
    which dominates run time when several methods share one forest)."""
    est = forest.estimators_
    hit = _SAMPLES_CACHE.get(forest)
    if hit is not None and hit[0] is est and hit[1] == len(est):
        return hit[2]
    samples = forest.estimators_samples_
    try:
        if len(samples) * len(samples[0]) <= _CACHE_LIMIT:
            _SAMPLES_CACHE[forest] = (est, len(est), samples)
    except TypeError:  # not weak-referenceable
        pass
    return samples


try:  # scikit-learn >= 1.6
    from sklearn.utils.validation import validate_data as _sk_validate_data
except ImportError:  # pragma: no cover - older scikit-learn
    _sk_validate_data = None


def _validate(est, X, y="no_validation", **kwargs):
    """``validate_data`` (sklearn >= 1.6) or ``est._validate_data`` (sklearn < 1.6)."""
    if _sk_validate_data is not None:
        return _sk_validate_data(est, X, y, **kwargs)
    return est._validate_data(X, y, **kwargs)


def _set_if_present(est, **params):
    valid = est.get_params()
    est.set_params(**{k: v for k, v in params.items() if k in valid and v is not None})
    return est


class _DebiasBase(RegressorMixin, BaseEstimator):
    """Common fitting logic: (re)use or fit the base forest and compute its OOB output."""

    def _check_Xy(self, X, y):
        """Validate training data (sets ``n_features_in_`` / ``feature_names_in_``)."""
        # check_X_y: standard error for y=None, column_or_1d(y, warn=True) for (n, 1) y
        return _validate(self, X, y, reset=True, dtype=np.float64, y_numeric=True)

    def _check_X(self, X):
        """Validate prediction data against the fitted number of features."""
        return _validate(self, X, reset=False, dtype=np.float64)

    def _new_forest(self, seed):
        """An unfitted forest from ``base_estimator`` (or the default) with this seed."""
        if self.base_estimator is None:
            return default_forest(random_state=seed, n_jobs=self.n_jobs)
        est = clone(self.base_estimator)
        return _set_if_present(est, random_state=seed, n_jobs=self.n_jobs)

    def _base_forest(self, X, y, seed):
        """Return (fitted forest, BaggedEnsemble, training tree predictions, leaves)."""
        n = X.shape[0]
        if self.prefit:
            if self.base_estimator is None:
                raise ValueError("prefit=True requires `base_estimator` to be a fitted forest")
            forest = self.base_estimator
            check_is_fitted(forest, "estimators_")
            nfi = getattr(forest, "n_features_in_", X.shape[1])
            if nfi != X.shape[1]:
                raise ValueError("prefit base_estimator was fitted with a different number "
                                 "of features")
            samples = _estimators_samples(forest)
            for idx in samples:
                if len(idx) and np.max(idx) >= n:
                    raise ValueError("prefit base_estimator was not fitted on these "
                                     "training rows (bootstrap index out of range); "
                                     "X, y must be the forest's training data")
        else:
            forest = self._new_forest(seed).fit(X, y)
            samples = _estimators_samples(forest)
        ens = BaggedEnsemble(forest, n_samples=n, samples=samples)
        leaves = ens.apply(X) if ens._fast else None
        P = ens.tree_predictions(X, leaves)
        return forest, ens, P, leaves

    def _seeds(self, k):
        rng = check_random_state(self.random_state)
        return rng.randint(0, _MAX_SEED, size=k)
