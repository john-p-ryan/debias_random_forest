"""R-compatible ``smooth.spline`` port and the univariate smoothing estimator.

The smoothing spline follows ``stats::smooth.spline`` (R 4.x, GCV, ``cv = FALSE``):

* tied ``x`` values (after rounding to ``tol = 1e-6 * IQR(x)``) are aggregated into
  weighted means, ``x`` is scaled to [0, 1];
* cubic B-spline basis on the knot subset ``xbar[seq(1, nx, length.out = nknots)]`` with
  ``nknots = .nknots.smspl(nx)`` (all knots if ``nx < 50``);
* penalty ``Omega_ij = int B_i''(t) B_j''(t) dt``; ``lambda = r * 256**(3*spar - 1)`` with
  ``r = tr(X'WX) / tr(Omega)`` over the same index range as R's Fortran ``sbart``;
* GCV criterion, minimised over ``spar`` in [-1.5, 1.5] by R's Brent / golden-section
  routine (``tol = 1e-4``, ``eps = 2e-8``);
* linear extrapolation beyond the data range.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import solveh_banded
from sklearn.utils.validation import check_is_fitted

from ._oob import _DebiasBase

__all__ = ["SmoothingSpline", "smooth_spline", "nknots_smspl", "SmoothedForestRegressor"]

_SGRAM_C = 0.3330  # constant in R's sgram.f (a rounded 1/3)
_GOLD = 0.381966011250105151795413165634
_GL_X = np.array([-1.0, 1.0]) / np.sqrt(3.0)  # 2-point Gauss-Legendre nodes (exact to cubics)


def nknots_smspl(n):
    """R's ``.nknots.smspl``: number of inner knots for ``n`` unique x values."""
    if n < 50:
        return int(n)
    a1, a2, a3, a4 = np.log2(50), np.log2(100), np.log2(140), np.log2(200)
    if n < 200:
        v = 2 ** (a1 + (a2 - a1) * (n - 50) / 150)
    elif n < 800:
        v = 2 ** (a2 + (a3 - a2) * (n - 200) / 600)
    elif n < 3200:
        v = 2 ** (a3 + (a4 - a3) * (n - 800) / 2400)
    else:
        v = 200 + (n - 3200) ** 0.2
    return int(np.trunc(v))


def _r_seq_index(nx, nknots):
    """0-based indices of ``seq.int(1, nx, length.out = nknots)`` truncated like R."""
    if nknots == 1:
        return np.array([0])
    by = (nx - 1) / (nknots - 1)
    seq = 1.0 + np.arange(nknots) * by
    seq[-1] = nx
    return np.trunc(seq).astype(int) - 1


class SmoothingSpline:
    """Cubic smoothing spline fitted at construction, compatible with R's ``smooth.spline``.

    Parameters
    ----------
    x, y : data; ``w`` optional non-negative weights.
    spar : fixed smoothing parameter (skips the search).
    lambda_ : fixed penalty (on the scaled-x problem, as R's ``lambda``).
    all_knots : use every unique x as a knot.
    nknots : number of inner knots (int or callable of nx); default ``nknots_smspl``.
    control_spar : dict overriding ``low, high, tol, eps, maxit`` (R's ``control.spar``).

    Attributes: ``spar``, ``lambda_``, ``df``, ``crit`` (R's ``cv.crit``), ``fit_crit``
    (Fortran GCV criterion at the optimum), ``ratio``, ``knots``, ``coef``, ``x`` (unique
    x), ``y`` (fitted values at unique x), ``lev``, ``w`` (aggregated weights).
    """

    def __init__(self, x, y, w=None, *, spar=None, lambda_=None, all_knots=False,
                 nknots=None, control_spar=None):
        self._fit(x, y, w, spar, lambda_, all_knots, nknots, control_spar)

    # ------------------------------------------------------------------ fitting
    def _fit(self, x, y, w, fixed_spar, fixed_lambda, all_knots, nknots_arg, control_spar):
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        if x.shape != y.shape:
            raise ValueError("x and y must have the same length")
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("missing or infinite values in inputs are not allowed")
        n = x.size
        if fixed_spar is not None and fixed_lambda is not None:
            raise ValueError("must not specify both 'spar' and 'lambda_'")
        ctrl = dict(low=-1.5, high=1.5, tol=1e-4, eps=2e-8, maxit=500)
        ctrl.update(control_spar or {})

        no_wgts = w is None
        if no_wgts:
            w = np.ones(n)
        else:
            w = np.asarray(w, dtype=np.float64).ravel()
            if w.shape != x.shape or np.any(w < 0) or np.all(w == 0):
                raise ValueError("invalid weights")
            w = w * np.sum(w > 0) / np.sum(w)

        iqr = np.subtract(*np.percentile(x, [75, 25]))
        tol = 1e-6 * iqr
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("'tol' must be strictly positive and finite (IQR(x) > 0 needed)")
        xx = np.round((x - x.mean()) / tol)
        order = np.argsort(x, kind="stable")
        xxs = xx[order]
        nd = np.concatenate([[True], xxs[:-1] < xxs[1:]])
        ux = x[order][nd]
        nx = ux.size
        if nx <= 3:
            raise ValueError("need at least four unique 'x' values")
        gid_sorted = np.cumsum(nd) - 1
        ox = np.empty(n, dtype=int)
        ox[order] = gid_sorted
        wbar = np.bincount(ox, weights=w, minlength=nx)
        sy = np.bincount(ox, weights=w * y, minlength=nx)
        syy = np.bincount(ox, weights=w * y * y, minlength=nx)
        ybar = sy / np.where(wbar > 0, wbar, 1.0)
        yssw = float(np.sum(syy - wbar * ybar ** 2))

        r_ux = ux[-1] - ux[0]
        xbar = (ux - ux[0]) / r_ux
        if all_knots:
            nknots = nx
        elif nknots_arg is None:
            nknots = nknots_smspl(nx)
        elif callable(nknots_arg):
            nknots = int(nknots_arg(nx))
        else:
            nknots = int(nknots_arg)
        if nknots < 1 or nknots > nx:
            raise ValueError("invalid number of knots")
        inner = xbar if all_knots else xbar[_r_seq_index(nx, nknots)]
        knots = np.concatenate([np.repeat(xbar[0], 3), inner, np.repeat(xbar[-1], 3)])
        nk = nknots + 2

        # design matrix, weighted normal matrix and penalty (dense; nk is at most ~200)
        eye = np.eye(nk)
        B = BSpline(knots, eye, 3, extrapolate=False)
        Xd = self._design(B, xbar, nk)
        XtWX = (Xd * wbar[:, None]).T @ Xd
        Xty = Xd.T @ (wbar * ybar)
        Omega = self._penalty(knots, nk)
        # R (sbart.c): ratio over 1-based i = 3 .. nk-3
        ratio = np.sum(np.diag(XtWX)[2:nk - 3]) / np.sum(np.diag(Omega)[2:nk - 3])
        sw = np.sqrt(wbar)
        sumw = float(np.sum(sw ** 2))

        # each row of the design matrix has at most 4 (contiguous) non-zeros
        start = np.minimum(np.argmax(Xd != 0, axis=1), nk - 4)
        cols = start[:, None] + np.arange(4)[None, :]  # (nx, 4)
        vals = np.take_along_axis(Xd, cols, axis=1)  # (nx, 4)
        rows_idx = cols[:, :, None]
        cols_idx = cols[:, None, :]
        vv = vals[:, :, None] * vals[:, None, :]

        def evaluate(lam):
            A = XtWX + lam * Omega
            ab = np.zeros((4, nk))
            for k in range(4):
                ab[k, :nk - k] = np.diag(A, -k)
            try:
                coef = solveh_banded(ab, Xty, lower=True, check_finite=False)
                Ainv = solveh_banded(ab, eye, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                return np.inf, None
            sz = Xd @ coef
            # leverage_i = w_i * b_i' A^{-1} b_i, using only the 4x4 block of A^{-1}
            lev = wbar * np.sum(vv * Ainv[rows_idx, cols_idx], axis=(1, 2))
            rss = yssw + np.sum(((ybar - sz) * sw) ** 2)
            df = np.sum(lev)
            crit = (rss / sumw) / (1.0 - (0.0 + 1.0 * df) / sumw) ** 2
            return crit, (coef, sz, lev, lam)

        if fixed_lambda is not None:
            crit, st = evaluate(float(fixed_lambda))
            spar = np.nan
            if st is None:
                raise ValueError("smoothing parameter value too extreme")
        elif fixed_spar is not None:
            spar = float(fixed_spar)
            crit, st = evaluate(ratio * 256.0 ** (3.0 * spar - 1.0))
            if st is None:
                raise ValueError("smoothing parameter value too extreme")
        else:
            spar, crit, st = self._brent(evaluate, ratio, ctrl)

        coef, sz, lev, lam = st
        df = float(np.sum(lev))
        resid = y - sz[ox]
        if no_wgts:
            cv_crit = np.mean(resid ** 2) / (1.0 - df / n) ** 2
        else:
            cv_crit = np.average(resid ** 2, weights=w) / (1.0 - df / n) ** 2

        self.spar = spar
        self.lambda_ = float(lam)
        self.df = df
        self.crit = float(cv_crit)
        self.fit_crit = float(crit)
        self.ratio = float(ratio)
        self.knots = knots
        self.coef = coef
        self.x = ux
        self.y = sz
        self.lev = lev
        self.w = wbar
        self.min_ = ux[0]
        self.range_ = r_ux
        self._bspline = BSpline(knots, coef, 3, extrapolate=False)
        e = np.array([0.0, 1.0])
        self._end_val = np.nan_to_num(self._bspline(e))
        self._end_slope = np.nan_to_num(self._bspline.derivative(1)(e))
        return self

    @staticmethod
    def _design(B, xbar, nk):
        """Dense design matrix; scipy gives nan at the right end point, where only the last
        basis function is non-zero (equal to 1)."""
        Xd = B(xbar)
        bad = ~np.isfinite(Xd).all(axis=1)
        if bad.any():
            Xd[bad] = 0.0
            Xd[bad, nk - 1] = 1.0
        return Xd

    @staticmethod
    def _penalty(knots, nk):
        """Penalty matrix ``Omega_ij ~ integral of B_i'' B_j''``.

        On each knot interval B'' is linear, ``B''(t) = y1 + y2 (t - a) / h``, and the
        integral of a product is ``h (y1 y1' + (y2 y1' + y2' y1)/2 + c y2 y2')`` with
        c = 1/3.  R's Fortran ``sgram`` uses the rounded constant 0.3330 instead of 1/3,
        which is reproduced here (``_SGRAM_C``) so that lambda has the same meaning.
        """
        d2 = BSpline(knots, np.eye(nk), 3).derivative(2)  # extrapolate=True is fine inside
        Om = np.zeros((nk, nk))
        for a, b in zip(knots[:-1], knots[1:]):
            h = b - a
            if h <= 0:
                continue
            pts = 0.5 * (a + b) + 0.5 * h * _GL_X
            D = d2(pts)  # (2, nk): B'' at two interior points of the interval
            slope = (D[1] - D[0]) / (pts[1] - pts[0])
            ya = D[0] - slope * (pts[0] - a)  # B'' at left end of the interval
            y2 = slope * h  # B''(b) - B''(a)
            Om += h * (np.outer(ya, ya) + 0.5 * (np.outer(y2, ya) + np.outer(ya, y2))
                       + _SGRAM_C * np.outer(y2, y2))
        return Om

    @staticmethod
    def _brent(evaluate, ratio, ctrl):
        """R's golden-section / parabolic search of ``sbart.c`` over spar."""
        a, b = ctrl["low"], ctrl["high"]
        tol, eps, maxit = ctrl["tol"], ctrl["eps"], ctrl["maxit"]
        BIG = 1e100
        cache = {}

        def f(spar):
            crit, st = evaluate(ratio * 256.0 ** (3.0 * spar - 1.0))
            if not np.isfinite(crit):
                crit = 2.0 * BIG
            cache["last"] = (crit, st)
            return crit

        v = a + _GOLD * (b - a)
        w = x = v
        d = e = 0.0
        fx = f(x)
        fv = fw = fx
        it = 0
        while True:
            xm = 0.5 * (a + b)
            tol1 = eps * abs(x) + tol / 3.0
            t2 = tol1 * 2.0
            it += 1
            if abs(x - xm) <= t2 - 0.5 * (b - a) or it > maxit:
                break
            golden = True
            if abs(e) > tol1 and np.isfinite(fx) and np.isfinite(fv) and np.isfinite(fw):
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p = (x - v) * q - (x - w) * r
                q = (q - r) * 2.0
                if q > 0.0:
                    p = -p
                q = abs(q)
                r = e
                e = d
                if not (abs(p) >= abs(0.5 * q * r) or q == 0.0):
                    if not (p <= q * (a - x) or p >= q * (b - x)):
                        d = p / q
                        u = x + d
                        if u - a < t2 or b - u < t2:
                            d = tol1 if xm - x >= 0.0 else -tol1  # fsign(tol1, xm - x)
                        golden = False
            if golden:
                e = (a - x) if x >= xm else (b - x)
                d = _GOLD * e
            u = x + (d if abs(d) >= tol1 else np.copysign(tol1, d))
            fu = f(u)
            if fu <= fx:
                if u >= x:
                    a = x
                else:
                    b = x
                v, fv = w, fw
                w, fw = x, fx
                x, fx = u, fu
            else:
                if u < x:
                    a = u
                else:
                    b = u
                if fu <= fw or w == x:
                    v, fv = w, fw
                    w, fw = u, fu
                elif fu <= fv or v == x or v == w:
                    v, fv = u, fu
        # R keeps the fit of the last evaluated point but reports spar = x, crit = fx
        crit_last, st = cache["last"]
        return x, fx, st

    # --------------------------------------------------------------- prediction
    def predict(self, x, deriv=0):
        """Evaluate the fitted spline (or its derivative) at ``x``; linear extrapolation."""
        x = np.asarray(x, dtype=np.float64)
        shape = x.shape
        xs = (x.ravel() - self.min_) / self.range_
        out = np.empty_like(xs)
        left, right = xs < 0, xs > 1
        inner = ~(left | right)
        if deriv == 0:
            sp = self._bspline
            vals = sp(np.clip(xs[inner], 0.0, 1.0))
            vals = np.where(np.isfinite(vals), vals, self._end_val[1])
            out[inner] = vals
            out[left] = self._end_val[0] + self._end_slope[0] * xs[left]
            out[right] = self._end_val[1] + self._end_slope[1] * (xs[right] - 1.0)
        elif deriv == 1:
            sp = self._bspline.derivative(1)
            vals = sp(np.clip(xs[inner], 0.0, 1.0))
            vals = np.where(np.isfinite(vals), vals, self._end_slope[1] / self.range_)
            out[inner] = vals / self.range_
            out[left] = self._end_slope[0] / self.range_
            out[right] = self._end_slope[1] / self.range_
        else:
            raise ValueError("deriv must be 0 or 1")
        return out.reshape(shape)

    __call__ = predict


def smooth_spline(x, y, w=None, **kwargs):
    """Fit an R-compatible smoothing spline; see :class:`SmoothingSpline`."""
    return SmoothingSpline(x, y, w, **kwargs)


_DEGREES = {"linear": 1, "quadratic": 2, "cubic": 3}


class SmoothedForestRegressor(_DebiasBase):
    """Univariate smoothing bias correction (De / Qu / Cu / Sp in the paper).

    Fits ``y ~ g(yhat_oob)`` on the training data, where ``yhat_oob`` is the base forest's
    OOB prediction, and predicts ``g(forest.predict(X))``.

    Parameters
    ----------
    method : ``"linear"`` (R ``corr.bias``), ``"quadratic"``, ``"cubic"`` (least-squares
        polynomials) or ``"spline"`` (R-compatible ``smooth.spline`` with GCV).
    base_estimator, prefit, random_state, n_jobs : as in
        :class:`~debiasforest.BoostedForestRegressor`.

    Attributes
    ----------
    estimator_ : fitted base forest;  smoother_ : the fitted correction ``g`` (callable);
    oob_prediction_ : ``g(yhat_oob)``;  base_oob_prediction_ : ``yhat_oob``.
    """

    def __init__(self, method="linear", base_estimator=None, prefit=False,
                 random_state=None, n_jobs=None):
        self.method = method
        self.base_estimator = base_estimator
        self.prefit = prefit
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        if self.method not in (*_DEGREES, "spline"):
            raise ValueError("method must be 'linear', 'quadratic', 'cubic' or 'spline'")
        X, y = self._check_Xy(X, y)
        forest, ens, P, _ = self._base_forest(X, y, self._seeds(1)[0])
        oob, _ = ens.oob_from_tree_predictions(P)
        if self.method == "spline":
            g = SmoothingSpline(oob, y)
        else:
            g = np.polynomial.Polynomial.fit(oob, y, _DEGREES[self.method])
        self.estimator_ = forest
        self._ensemble = ens.release_oob()
        self.smoother_ = g
        self.base_oob_prediction_ = oob
        self.oob_prediction_ = np.asarray(g(oob), dtype=float)
        return self

    def predict(self, X):
        check_is_fitted(self, "smoother_")
        X = self._check_X(X)
        return np.asarray(self.smoother_(self._ensemble.predict(X)), dtype=float)
