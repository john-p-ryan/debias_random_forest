"""Conditional bias correction (Lu & Hardin 2021; paper section 2.2.3)."""
from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._forest import BaggedEnsemble
from ._oob import _DebiasBase

__all__ = ["ConditionalBiasForestRegressor"]


class ConditionalBiasForestRegressor(_DebiasBase):
    """Forest prediction minus a leaf-weighted average of the training OOB biases.

    For tree t and leaf L let ``S_t[L]`` be the sum of ``(oob_pred_i - y_i)`` over the rows
    i that are out-of-bag for tree t and fall in leaf L, and ``C_t[L]`` their count.  The
    bias estimate at x0 is ``b(x0) = sum_t S_t[leaf_t(x0)] / sum_t C_t[leaf_t(x0)]`` (0 if
    the denominator is 0) and the prediction is ``forest(x0) - b(x0)``.  This is the weight
    formula ``v_i(x0)`` of the paper written per leaf.

    ``oob_prediction_`` for training row j is ``oob_pred_j - b_{-j}(x_j)`` where the bias
    estimate uses all trees but removes row j's own contribution (``bias_j`` from the sums,
    one count per tree in which j is OOB).  Rows' OOB biases themselves are computed from
    the full OOB predictions (as in the paper), so this is an approximate OOB estimate.

    Parameters: ``base_estimator``, ``prefit``, ``random_state``, ``n_jobs`` as in
    :class:`~debiasforest.BoostedForestRegressor`.  Requires tree-based ensemble members.
    """

    def __init__(self, base_estimator=None, prefit=False, random_state=None, n_jobs=None):
        self.base_estimator = base_estimator
        self.prefit = prefit
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X, y = self._check_Xy(X, y)
        forest, ens, P, leaves = self._base_forest(X, y, self._seeds(1)[0])
        if leaves is None:
            leaves = ens.apply(X)
        oob, k = ens.oob_from_tree_predictions(P)
        bias = oob - y
        offs = ens.node_offsets
        ids = leaves + offs[:-1][None, :]  # global leaf ids (n, T)
        mask = ens.oob_mask.T  # (n, T)
        total = int(offs[-1])
        rows = np.broadcast_to(np.arange(X.shape[0])[:, None], ids.shape)
        S = np.bincount(ids[mask], weights=bias[rows[mask]], minlength=total)
        C = np.bincount(ids[mask], minlength=total).astype(float)
        num = S[ids].sum(axis=1) - k * bias
        den = C[ids].sum(axis=1) - k
        b = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        self.estimator_ = forest
        self._ensemble = ens.release_oob()
        self.leaf_bias_sum_ = S
        self.leaf_oob_count_ = C
        self.base_oob_prediction_ = oob
        self.oob_prediction_ = oob - b
        return self

    def bias(self, X):
        """Estimated bias ``b(x0)`` for each row of ``X``."""
        check_is_fitted(self, "leaf_bias_sum_")
        X = self._check_X(X)
        ens = self._ensemble
        ids = ens.apply(X) + ens.node_offsets[:-1][None, :]
        num = self.leaf_bias_sum_[ids].sum(axis=1)
        den = self.leaf_oob_count_[ids].sum(axis=1)
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    def predict(self, X):
        check_is_fitted(self, "leaf_bias_sum_")
        X = self._check_X(X)
        ens = self._ensemble
        leaves = ens.apply(X)
        ids = leaves + ens.node_offsets[:-1][None, :]
        pred = ens.tree_predictions(X, leaves).mean(axis=1)
        num = self.leaf_bias_sum_[ids].sum(axis=1)
        den = self.leaf_oob_count_[ids].sum(axis=1)
        return pred - np.divide(num, den, out=np.zeros_like(num), where=den > 0)
