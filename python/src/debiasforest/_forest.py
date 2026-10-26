"""Forest construction and the bagging-ensemble adapter.

``default_forest`` builds a scikit-learn random forest that mirrors the defaults of
the R package ``randomForest`` used in the paper (500 trees, ``mtry = max(floor(p/3), 1)``,
``nodesize = 5``).  ``BaggedEnsemble`` wraps any *fitted* scikit-learn bagging ensemble
(``RandomForestRegressor``, ``ExtraTreesRegressor(bootstrap=True)``, ``BaggingRegressor``)
and exposes the quantities every bias-correction method needs: in-bag counts, out-of-bag
(OOB) predictions, per-tree predictions and leaf indices.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import check_array

import inspect as _inspect

_ALL_FINITE = "ensure_all_finite" in _inspect.signature(check_array).parameters

__all__ = ["DefaultForestRegressor", "default_forest", "BaggedEnsemble"]


class DefaultForestRegressor(RandomForestRegressor):
    """``RandomForestRegressor`` whose ``max_features`` may be ``"third"``.

    ``"third"`` means ``mtry = max(floor(p / 3), 1)`` (R ``randomForest`` default for
    regression).  scikit-learn only accepts fractions, which are wrong for small ``p``
    (e.g. ``1/3`` of 4 features would give 2 instead of 1), so the integer is resolved at
    fit time, when ``p`` is known.  The constructor parameter itself is left untouched
    (so ``clone`` / ``get_params`` behave normally); the trees carry the resolved integer
    in ``estimators_[i].max_features``.
    """

    def __init__(self, n_estimators=500, *, criterion="squared_error", max_depth=None,
                 min_samples_split=6, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features="third", max_leaf_nodes=None, min_impurity_decrease=0.0,
                 bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
                 verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None,
                 monotonic_cst=None):
        super().__init__(
            n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
            max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs,
            random_state=random_state, verbose=verbose, warm_start=warm_start,
            ccp_alpha=ccp_alpha, max_samples=max_samples, monotonic_cst=monotonic_cst)

    def fit(self, X, y, sample_weight=None):
        original = self.max_features
        if isinstance(original, str) and original == "third":
            Xv = check_array(X, dtype=np.float32, accept_sparse="csc",
                            ensure_all_finite=False) if _ALL_FINITE else \
                check_array(X, dtype=np.float32, accept_sparse="csc",
                            force_all_finite=False)
            self.max_features = max(Xv.shape[1] // 3, 1)
        try:
            return super().fit(X, y, sample_weight=sample_weight)
        finally:
            self.max_features = original


def default_forest(random_state=None, n_estimators=500, n_jobs=None, **kwargs):
    """Random forest with the defaults of R ``randomForest`` (regression).

    500 trees, ``mtry = max(floor(p/3), 1)`` (resolved at fit time), ``nodesize = 5``
    (``min_samples_split=6``, ``min_samples_leaf=1``), bootstrap sample of size ``n``
    drawn with replacement.  Extra keyword arguments override any of these.
    """
    return DefaultForestRegressor(n_estimators=n_estimators, random_state=random_state,
                                  n_jobs=n_jobs, **kwargs)


class BaggedEnsemble:
    """Adapter around a fitted scikit-learn bagging ensemble.

    Fast path (every member is a fitted sklearn tree and no feature subsetting, i.e.
    random forests, extra trees, bagged trees): leaves and predictions come straight from
    ``tree_.apply`` / ``tree_.value`` with no per-tree validation or joblib overhead.
    Otherwise (``estimators_features_`` present, non-tree members) it falls back to the
    members' own ``apply`` / ``predict``.

    Parameters
    ----------
    forest : fitted estimator with ``estimators_`` and ``estimators_samples_``.
    n_samples : number of training rows (inferred from ``estimators_samples_`` if omitted).
    samples : optional pre-computed ``forest.estimators_samples_`` (sklearn rebuilds that
        list on every access, which is slow for large forests).

    The in-bag count matrix ``inbag`` (n_trees x n, number of times each row was drawn in
    each tree's bootstrap) and the OOB mask are computed lazily; call :meth:`release_oob`
    to drop them (e.g. before storing the adapter in a fitted estimator).
    """

    def __init__(self, forest, n_samples=None, samples=None):
        try:
            if samples is None:
                samples = forest.estimators_samples_  # rebuilt by sklearn on every access
            estimators = forest.estimators_
        except AttributeError:
            raise TypeError("`forest` must be a fitted sklearn bagging ensemble exposing "
                            "`estimators_` and `estimators_samples_`") from None
        self.forest = forest
        self.estimators = list(estimators)
        self.n_trees = len(self.estimators)
        self._samples = samples
        if n_samples is None:
            n_samples = int(max(int(np.max(s)) for s in self._samples)) + 1
        self.n_samples = int(n_samples)
        self._inbag = None
        feats = getattr(forest, "estimators_features_", None)
        self._features = None if feats is None else [np.asarray(f) for f in feats]
        self._fast = (self._features is None
                      and all(hasattr(e, "tree_") for e in self.estimators))
        self._values = ([e.tree_.value[:, 0, 0] for e in self.estimators]
                        if self._fast else None)
        self._offsets = None

    # -- in-bag information ------------------------------------------------------
    @property
    def inbag(self):
        """In-bag counts, shape (n_trees, n)."""
        if self._inbag is None:
            if self._samples is None:
                self._samples = self.forest.estimators_samples_
            n = self.n_samples
            inbag = np.zeros((self.n_trees, n), dtype=np.int32)
            for t, idx in enumerate(self._samples):
                idx = np.asarray(idx)
                if idx.dtype == bool:
                    idx = np.flatnonzero(idx)
                inbag[t] = np.bincount(idx, minlength=n)
            self._inbag = inbag
        return self._inbag

    @property
    def oob_mask(self):
        """Boolean (n_trees, n): True where the row is out-of-bag for the tree."""
        return self.inbag == 0

    def release_oob(self):
        """Drop the cached in-bag matrix and sample indices (prediction still works)."""
        self._inbag = None
        self._samples = None
        return self

    # -- per tree quantities ------------------------------------------------------
    @staticmethod
    def _as_float32(X):
        return np.ascontiguousarray(X, dtype=np.float32)

    def apply(self, X):
        """Leaf index of every row in every tree, shape (n, n_trees)."""
        X = np.asarray(X, dtype=np.float64)
        out = np.empty((X.shape[0], self.n_trees), dtype=np.intp)
        if self._fast:
            X32 = self._as_float32(X)
            for t, e in enumerate(self.estimators):
                out[:, t] = e.tree_.apply(X32)
            return out
        for t, e in enumerate(self.estimators):
            Xt = X if self._features is None else X[:, self._features[t]]
            out[:, t] = e.apply(Xt)
        return out

    def tree_predictions(self, X, leaves=None):
        """Prediction of each tree for each row, shape (n, n_trees)."""
        X = np.asarray(X, dtype=np.float64)
        if self._fast:
            if leaves is None:
                leaves = self.apply(X)
            out = np.empty(leaves.shape, dtype=np.float64)
            for t in range(self.n_trees):
                out[:, t] = self._values[t][leaves[:, t]]
            return out
        out = np.empty((X.shape[0], self.n_trees), dtype=np.float64)
        for t, e in enumerate(self.estimators):
            Xt = X if self._features is None else X[:, self._features[t]]
            out[:, t] = e.predict(Xt)
        return out

    def predict(self, X):
        """Ensemble prediction (average of tree predictions; single-threaded)."""
        X = np.asarray(X, dtype=np.float64)
        if self._fast:
            X32 = self._as_float32(X)
            acc = np.zeros(X.shape[0])
            for t, e in enumerate(self.estimators):
                acc += self._values[t][e.tree_.apply(X32)]
            return acc / self.n_trees
        return self.tree_predictions(X).mean(axis=1)

    @property
    def node_offsets(self):
        """Cumulative node counts, used to give leaves a global id across trees."""
        if self._offsets is None:
            counts = np.array([e.tree_.node_count for e in self.estimators])
            self._offsets = np.concatenate([[0], np.cumsum(counts)])
        return self._offsets

    # -- OOB ------------------------------------------------------------------
    def oob_from_tree_predictions(self, P, X_train_pred=None):
        """OOB predictions from training-row tree predictions ``P`` (n, n_trees).

        Rows that are never out-of-bag fall back to the full ensemble prediction (with a
        warning).
        """
        oob = self.oob_mask.T  # (n, n_trees)
        cnt = oob.sum(axis=1)
        num = np.where(oob, P, 0.0).sum(axis=1)
        pred = np.empty(P.shape[0])
        ok = cnt > 0
        pred[ok] = num[ok] / cnt[ok]
        if not ok.all():
            warnings.warn(f"{int((~ok).sum())} training rows were never out-of-bag; "
                          "their OOB prediction falls back to the full-ensemble prediction. "
                          "Use more trees.", RuntimeWarning, stacklevel=3)
            pred[~ok] = P[~ok].mean(axis=1)
        return pred, cnt

    def oob_prediction(self, X):
        """OOB prediction of the training rows ``X`` (must be the training matrix)."""
        pred, _ = self.oob_from_tree_predictions(self.tree_predictions(X))
        return pred
