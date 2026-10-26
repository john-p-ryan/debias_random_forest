"""Boosted random forest (Breiman 2001a; Chen, Withana Gamage & Ryan 2022, section 2.2.2)."""
from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._forest import BaggedEnsemble
from ._oob import _DebiasBase

__all__ = ["BoostedForestRegressor", "paper_label"]


def paper_label(n_iterations):
    """Paper notation: 0 -> "Un", 1 -> "Bf", k -> "Bf(k-1)" for k >= 2."""
    k = int(n_iterations)
    if k < 0:
        raise ValueError("n_iterations must be >= 0")
    return {0: "Un", 1: "Bf"}.get(k, f"Bf({k - 1})")


class BoostedForestRegressor(_DebiasBase):
    """Iteratively boosted random forest.

    Stage 0 is the base forest fitted to ``y``.  With OOB prediction ``o_j`` of stage j,
    the residual ``r_{j+1} = r_j - o_j`` (``r_0 = y``) is fitted by a fresh forest (own
    seed) at stage j+1.  The prediction is the sum of the stage predictions.

    Parameters
    ----------
    n_iterations : number of residual forests after the base forest.  0 -> uncorrected,
        1 -> the paper's ``Bf``, k+1 -> ``Bf(k)``.  With ``stopping="oob"`` it is the
        maximum number of residual forests.
    stopping : ``"fixed"`` fits exactly ``n_iterations`` residual forests.  ``"oob"``
        tracks the OOB MSE of ``y`` against the cumulative OOB prediction after each
        stage, stops growing once it exceeds 1.1 x the minimum so far, and keeps the
        stages up to the argmin.
    base_estimator : forest used at every stage (cloned; seeds are overridden).  Default
        :func:`debiasforest.default_forest`.  Any sklearn bagging ensemble works.
    prefit : if True ``base_estimator`` is an already fitted forest used as stage 0
        (``fit(X, y)`` must receive its training data); later stages use unfitted clones.
    random_state, n_jobs : seeding and parallelism of the forests.

    Attributes
    ----------
    estimators_ : list of fitted stage forests (length ``n_iterations_ + 1``)
    oob_prediction_ : OOB prediction of the (bias-corrected) training rows
    n_iterations_ : number of residual forests kept
    oob_mse_path_ : OOB MSE for stages 0..K as computed (also beyond the argmin)
    """

    def __init__(self, n_iterations=1, stopping="fixed", base_estimator=None, prefit=False,
                 random_state=None, n_jobs=None):
        self.n_iterations = n_iterations
        self.stopping = stopping
        self.base_estimator = base_estimator
        self.prefit = prefit
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        if self.stopping not in ("fixed", "oob"):
            raise ValueError("stopping must be 'fixed' or 'oob'")
        K = int(self.n_iterations)
        if K < 0:
            raise ValueError("n_iterations must be >= 0")
        X, y = self._check_Xy(X, y)
        seeds = self._seeds(K + 1)
        forest, ens, P, _ = self._base_forest(X, y, seeds[0])
        oob0, _ = ens.oob_from_tree_predictions(P)
        forests = [forest]
        ensembles = [ens]
        stage_oob = [oob0]
        cum = oob0.copy()
        mse = [float(np.mean((y - cum) ** 2))]
        resid = y - oob0
        for j in range(1, K + 1):
            f = self._new_forest(seeds[j]).fit(X, resid)
            e = BaggedEnsemble(f, n_samples=X.shape[0])
            o, _ = e.oob_from_tree_predictions(e.tree_predictions(X))
            forests.append(f)
            ensembles.append(e)
            stage_oob.append(o)
            cum = cum + o
            resid = resid - o
            mse.append(float(np.mean((y - cum) ** 2)))
            if self.stopping == "oob" and mse[-1] > 1.1 * min(mse[:-1]):
                break
        keep = len(forests)
        if self.stopping == "oob":
            keep = int(np.argmin(mse)) + 1
        self.estimators_ = forests[:keep]
        self._ensembles = [e.release_oob() for e in ensembles[:keep]]
        self.n_iterations_ = keep - 1
        self.oob_mse_path_ = np.array(mse)
        self.oob_prediction_ = np.sum(stage_oob[:keep], axis=0)
        self.stage_oob_predictions_ = np.array(stage_oob[:keep])
        return self

    def _check_predict(self, X):
        check_is_fitted(self, "estimators_")
        X = self._check_X(X)
        return X

    def predict(self, X):
        X = self._check_predict(X)
        return np.sum([e.predict(X) for e in self._ensembles], axis=0)

    def staged_predict(self, X):
        """Yield cumulative predictions for ``n_iterations = 0, 1, ..., n_iterations_``."""
        X = self._check_predict(X)
        cum = 0.0
        for e in self._ensembles:
            cum = cum + e.predict(X)
            yield np.array(cum, copy=True)
