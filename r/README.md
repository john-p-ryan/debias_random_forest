# debiasforest (R)

Bias correction for random forest regression, after Chen, Withana Gamage & Ryan (2022),
*Debias Random Forest Regression Predictors*, J. Stat. Research 56(2):115-131.  Forests are
[`randomForest`](https://CRAN.R-project.org/package=randomForest) objects fitted with
`keep.inbag = TRUE`, the package the paper used, so in-bag counts, out-of-bag (OOB)
predictions and terminal nodes come straight from `randomForest`.

| Paper | R |
|---|---|
| Un | `fit_forest(x, y)` (or any regression `randomForest` with `keep.inbag = TRUE`) |
| De / Qu / Cu / Sp | `smoothed_forest(x, y, method = "linear" / "quadratic" / "cubic" / "spline")` |
| Bf, Bf(k) | `boosted_forest(x, y, n_iterations = 1, stopping = "fixed" / "oob")`, `staged_predict()` |
| Cond | `conditional_bias_forest(x, y)` |

## Install

```sh
Rscript -e 'install.packages(c("randomForest", "testthat"))'   # testthat only for the tests
R CMD INSTALL r                                                # from the repository root
Rscript -e 'testthat::test_local("r")'                         # about 5 s
```

R >= 4.0.  `randomForest` is the only dependency outside base R.

## Quick start

```r
library(debiasforest)

d <- simulate_data("f1", 200, seed = 1)                  # Friedman 1, sigma = 1
test <- simulate_data("f1", 2000, sigma = 0, seed = 2)

un <- fit_forest(d$x, d$y, seed = 1)                     # uncorrected forest (randomForest defaults)
bf <- boosted_forest(d$x, forest = un, seed = 1)         # paper's Bf, re-using the base forest
mean((predict(un, test$x) - test$f)^2); mean((predict(bf, test$x) - test$f)^2)

staged_predict(bf, test$x)       # columns Un, Bf (more with n_iterations > 1)
predict(bf)                      # bias-corrected OOB predictions of the training rows

smoothed_forest(d$x, method = "spline", forest = un)    # smoothing spline of y on the OOB prediction
conditional_bias_forest(d$x, forest = un)

P <- predict_all_methods(d$x, d$y, test$x, seed = 1)    # matrix, one column per method
res <- simulation_study("f1", n_reps = 20, seed = 1, n_cores = 4)   # MSPE mean/SD, bias^2
```

## API

* `fit_forest(x, y, ntree = 500, mtry = max(floor(p/3), 1), nodesize = 5, seed, ...)` returns a
  `randomForest` object with `inbag`; `...` goes to `randomForest()`.  `oob_predictions(forest, x)`,
  `default_mtry(x)`.
* `boosted_forest(x, y, n_iterations = 1, stopping = "fixed", forest = NULL, ntree, mtry, nodesize, seed, ...)`:
  `predict`, `staged_predict`; fields `forests`, `n_iterations` (used), `oob_mse_path`,
  `oob_prediction`, `stage_oob_predictions`.  `n_iterations` counts residual forests after the base
  forest: 0 -> Un, 1 -> Bf, k + 1 -> Bf(k) (`paper_label(k)`).  With `stopping = "oob"`,
  `n_iterations` is the maximum: stages are added until the OOB MSE exceeds 1.1 x its minimum so far,
  and the model is truncated to the argmin.
* `smoothed_forest(x, y, method = "linear", forest = NULL, ...)`: fields `smoother`,
  `base_oob_prediction`, `oob_prediction`.
* `conditional_bias_forest(x, y, forest = NULL, ...)`, `bias_estimate(object, newdata)`.
* `predict(object)` without `newdata` returns the bias-corrected OOB predictions (as
  `predict.randomForest` does).  With `forest =` supplied, `y` defaults to the forest's response and
  `x` must be its training predictors (checked against `forest$inbag` and `forest$y`).
* `predict_all_methods`, `simulation_study`, `repeated_splits`, `select_n_iterations` (with
  `print` and `as.data.frame` methods), `method_labels()`.
* `simulate_data(name, n, sigma, seed)` (names `"fig1" "f1"/"friedman1" "f2" ... "f5"`),
  `true_mean`, `simulation_names`, `simulation_sigma`, `load_yacht(cache_dir)` (signals an error of
  class `debiasforest_dataset_unavailable` offline), `bias_plot(reference, predictions)`.

## Notes

* **Forest.** `randomForest` defaults for regression: 500 trees, `mtry = max(floor(p/3), 1)`,
  `nodesize = 5` (a node with at most 5 bootstrap draws is not split; duplicate draws count
  separately).  `x` may be a matrix or a data frame (factors are handled by `randomForest`).  A
  supplied `forest` must be a regression forest fitted with `keep.inbag = TRUE` and without
  `corr.bias`, and `x` must be its training data in the original row order (spot-checked against
  the stored OOB predictions).  Residual stages reuse its `ntree` and `mtry`, and its `nodesize`
  if it comes from `fit_forest()` (`randomForest` does not store it); other `randomForest()`
  arguments must be repeated in `...`.
* **OOB predictions** are `forest$predicted`, recomputed from `inbag` and the tree predictions for
  forests from `randomForest::combine()` (whose stored OOB predictions are out of date).  Rows that
  are never out-of-bag (only with few trees) fall back to the full-forest prediction, with a
  warning.
* **Spline.** `method = "spline"` calls `stats::smooth.spline` with its default GCV choice and
  predicts with linear extrapolation; the Python and Julia packages port this function.
* **Seeds.** Every fitting function takes `seed`: with an integer the result is reproducible and
  the caller's random number stream is restored afterwards; with `NULL` the current stream is used,
  so `set.seed()` also works.  `simulation_study`, `repeated_splits` and `select_n_iterations` draw
  one seed per replication up front, so results do not depend on `n_cores`.
* **Parallelism.** `n_cores > 1` forks workers with `parallel::mclapply` (sequential on Windows).
  Each replication of a simulation study (n = 200, 500 trees, all ten methods, 2000 test points)
  takes about 0.7 s on one core.
* **Conditional bias.** `b(x0) = sum_t S_t[leaf_t(x0)] / sum_t C_t[leaf_t(x0)]`, with `S_t` / `C_t`
  the sum / count of the OOB biases `oob_i - y_i` per terminal node, stored as `nrnodes x ntree`
  matrices.  `oob_prediction` removes each row's own contribution from its bias estimate.
* `load_yacht()` downloads the UCI Yacht data (falling back to a GitHub mirror) and caches it under
  `$DEBIASFOREST_DATA` or `tools::R_user_dir("debiasforest", "cache")`.
