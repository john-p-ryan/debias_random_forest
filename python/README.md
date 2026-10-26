# debiasforest (Python)

Bias correction for random forest regression, after Chen, Withana Gamage & Ryan (2022),
*Debias Random Forest Regression Predictors*, J. Stat. Research 56(2):115-131. All methods
are thin wrappers around scikit-learn forests that track bootstrap / out-of-bag (OOB)
membership; the flagship is the **boosted forest** (Breiman 2001a): fit a forest to the OOB
residuals of the previous forest and add the predictions.

## Install

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate   # or any Python >= 3.10
uv pip install -e "python[test]"        # scikit-learn >= 1.4, numpy, scipy (+ pytest)
uv pip install -e "python[test,demo]"   # + matplotlib, pandas, jupyter, nbconvert, ...
pytest python/tests                      # about 10 s
```

## Quick start

```python
from debiasforest import (BoostedForestRegressor, SmoothedForestRegressor,
                          ConditionalBiasForestRegressor, default_forest, simulate)

X, y, _ = simulate("friedman1", 200, sigma=1.0, rng=0)
X_test, _, f_test = simulate("friedman1", 2000, sigma=0.0, rng=1)

bf = BoostedForestRegressor(n_iterations=1, random_state=0).fit(X, y)   # the paper's "Bf"
pred = bf.predict(X_test)
bf.oob_prediction_            # bias-corrected OOB predictions of the training rows
for k, p in enumerate(bf.staged_predict(X_test)):   # k = 0 (Un), 1 (Bf), ...
    print(k, ((p - f_test) ** 2).mean())

# One base forest for every method (as in the paper's comparison):
base = default_forest(random_state=0).fit(X, y)
sp = SmoothedForestRegressor(method="spline", base_estimator=base, prefit=True).fit(X, y)
cond = ConditionalBiasForestRegressor(base_estimator=base, prefit=True).fit(X, y)
```

All estimators follow the scikit-learn API (`clone`, `Pipeline`, `cross_val_score`, ...).

## API overview

| Paper | Python |
|---|---|
| Un | `default_forest(random_state, n_estimators=500)` (or any sklearn forest) |
| De / Qu / Cu / Sp | `SmoothedForestRegressor(method="linear"/"quadratic"/"cubic"/"spline")` |
| Bf, Bf(k) | `BoostedForestRegressor(n_iterations=1, stopping="fixed"/"oob")`, `staged_predict` |
| Cond | `ConditionalBiasForestRegressor()` |
| labels | `paper_label(k)`: 0 -> `"Un"`, 1 -> `"Bf"`, k -> `"Bf(k-1)"` |
| stopping rule | `select_n_iterations(X, y, max_iterations=6, n_splits=20)` |
| comparisons | `predict_all_methods`, `simulation_study`, `repeated_splits` |
| data | `simulate(name, n, sigma, rng)` (`fig1`, `friedman1`/`f1`, `f2`..`f5`), `load_yacht()` |
| plot | `bias_plot(reference, {label: predictions})` (matplotlib imported lazily) |
| building blocks | `BaggedEnsemble` (in-bag counts, OOB predictions, leaves), `smooth_spline(x, y)` |

Common estimator parameters: `base_estimator` (template forest, default `default_forest()`),
`prefit` (use an already fitted forest; `fit(X, y)` must get its training data), `random_state`
and `n_jobs`.  `n_iterations` is the number of residual forests after the base forest
(0 -> Un, 1 -> Bf, k+1 -> Bf(k)).  With `stopping="oob"` the boosted forest tracks the OOB MSE
of the cumulative OOB prediction, stops once it exceeds 1.1 x its minimum and keeps the stages up
to the argmin (`n_iterations` is then the maximum; `n_iterations_` and `oob_mse_path_` report
what happened).  Fitted estimators expose `oob_prediction_`.

Any fitted scikit-learn bagging ensemble exposing `estimators_` and `estimators_samples_` can be
the base: `RandomForestRegressor`, `ExtraTreesRegressor(bootstrap=True)`, `BaggingRegressor`
(including `max_features < 1`, handled through `estimators_features_`).

## Notes

* **R-compatible defaults.** `default_forest` mirrors R `randomForest`: 500 trees,
  `mtry = max(floor(p/3), 1)`, `nodesize = 5` (`min_samples_split=6`, `min_samples_leaf=1`),
  bootstrap of size n with replacement.  Because a fraction `max_features=1/3` is wrong for small p,
  the forest class `DefaultForestRegressor` accepts `max_features="third"` and resolves the integer at
  fit time (the constructor parameter stays `"third"`, so `clone`/`get_params` behave normally).
* **Distinct vs drawn rows.** scikit-learn applies `min_samples_split` / `min_samples_leaf` to the number of
  *distinct* in-bag rows in a node, whereas R (and the Julia package) count bootstrap draws.  Trees are
  therefore slightly different from R's, a minor effect.
* **OOB predictions.** Computed from per-tree predictions and the in-bag count matrix rebuilt from
  `estimators_samples_` (indices drawn with replacement).  Rows that are never out-of-bag fall back to
  the full-ensemble prediction with a `RuntimeWarning` (use more trees).
* **Conditional bias.** `b(x0) = sum_t S_t[leaf_t(x0)] / sum_t C_t[leaf_t(x0)]`, with `S_t`/`C_t` the sum / count
  of the OOB biases `oob_i - y_i` per leaf; this is the paper's weights `v_i(x0)` written per leaf.
  `oob_prediction_` removes row j's own contribution from its bias estimate (the OOB biases themselves
  are the usual full-OOB ones, so it is an approximate OOB estimate).
* **Smoothing spline.** `smooth_spline` is a native port of R's `smooth.spline` (GCV): tie aggregation with
  `tol = 1e-6 * IQR(x)`, x scaled to [0,1], cubic B-splines on R's knot subset, `lambda = r * 256^(3 spar - 1)`,
  R's Brent search on `spar` in [-1.5, 1.5] and linear extrapolation.  Following R it reports the
  `lambda`/`df` of the last evaluated point of the search, and `crit` is R's `cv.crit`.  The penalty matrix uses
  the constant 0.3330 as R's Fortran `sgram` does.  Fixtures in `tests/data` come from `dev/spline_reference.R`.
* **Performance.** Forest predictions inside the package use `tree_.apply` directly (single-threaded,
  no joblib overhead).  `simulation_study` / `repeated_splits` / `select_n_iterations` parallelise over
  replications with joblib (`n_jobs`), keeping each forest single-threaded.  One simulation replication
  (n = 200, 500 trees, all ten methods, 2000 test points) takes about 3 s on one core.
* `load_yacht()` downloads the UCI Yacht data (falling back to a GitHub mirror), caches it under
  `$DEBIASFOREST_DATA` or `~/.cache/debiasforest`, and raises `DatasetUnavailableError` when offline.
