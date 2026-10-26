# DebiasForest.jl

Bias correction for random-forest regression, after Chen, Withana Gamage & Ryan (2022),
*Debias Random Forest Regression Predictors*, J. Stat. Research 56(2):115-131.
Forests are built from [DecisionTree.jl](https://github.com/JuliaAI/DecisionTree.jl) trees
inside our own bagging loop, which keeps in-bag counts, out-of-bag (OOB) predictions and
leaf ids for every tree.

| Paper | Julia |
|---|---|
| Un | `fit_forest(X, y)` -> `BaggedForest` |
| De / Qu / Cu / Sp | `fit(SmoothedForest, X, y; method = :linear / :quadratic / :cubic / :spline)` |
| Bf, Bf(k) | `fit(BoostedForest, X, y; n_iterations = 1, stopping = :fixed / :oob)`, `staged_predict` |
| Cond | `fit(ConditionalBiasForest, X, y)` |

## Install

```julia
using Pkg
Pkg.develop(path = "path/to/debias_random_forest/julia")   # or Pkg.activate("julia"); Pkg.instantiate()
```

Julia >= 1.10.  If the Julia package server is unreachable: `export JULIA_PKG_SERVER=""`.
Start Julia with `-t auto` (or `-t 4`) to build trees and run replications in parallel.

## Quick start

```julia
using DebiasForest, Random, Statistics

X, y, f = simulate("f1", 200; rng = Xoshiro(1))          # Friedman 1, sigma = 1
Xt, _, ft = simulate("f1", 2000; sigma = 0, rng = Xoshiro(2))

un = fit_forest(X, y; seed = 1)                          # uncorrected forest (R randomForest defaults)
bf = fit(BoostedForest, un; n_iterations = 1)            # paper's Bf, re-using the base forest
mean(abs2, predict(un, Xt) .- ft), mean(abs2, predict(bf, Xt) .- ft)

staged_predict(bf, Xt)            # columns: Un, Bf (more with n_iterations > 1)
oob_prediction(bf)                # bias-corrected OOB predictions of the training rows

fit(SmoothedForest, un; method = :spline)                # smoothing spline of y on OOB prediction
fit(ConditionalBiasForest, un)

P = predict_all_methods(X, y, Xt; seed = 1)              # Dict label => test predictions
res = simulation_study("f1"; n = 200, n_reps = 20, seed = 1)   # MSPE mean/SD, bias^2
```

## API

* `fit_forest(X, y; n_trees=500, mtry=max(p÷3,1), nodesize=5, seed|rng, threaded=true)` -> `BaggedForest`;
  `predict`, `oob_prediction`, `leaf_matrix(forest, X)`; fields `trees`, `inbag`, `leaves`, `X`, `y`.
* `fit(BoostedForest, X, y; n_iterations=1, stopping=:fixed|:oob, n_trees, mtry, nodesize, seed)` and
  `fit(BoostedForest, forest::BaggedForest; ...)`; `predict`, `staged_predict`, `oob_prediction`,
  `oob_staged_prediction`; fields `n_iterations` (used), `oob_mse` (path), `stages`.
  `n_iterations` counts residual forests after the base forest: 0 -> Un, 1 -> Bf, k+1 -> Bf(k)
  (`paper_label(k)`).  With `stopping = :oob`, `n_iterations` is the maximum: stages are added until
  the OOB MSE exceeds 1.1 x its minimum so far, and the model is truncated to the argmin.
* `fit(SmoothedForest, X, y; method=:linear)` / `fit(SmoothedForest, forest; method)`.
* `fit(ConditionalBiasForest, X, y)` / `fit(ConditionalBiasForest, forest)`; `bias_estimate`.
  `oob_prediction` removes a row's own contribution to the bias weights.
* `smooth_spline(x, y; w, spar, lambda, ...)` -> `SmoothSpline`, `predict(fit, xnew; deriv=0)` -
  native port of R's `stats::smooth.spline` (GCV).
* `predict_all_methods`, `repeated_splits`, `simulation_study`, `select_n_iterations`,
  `simulate(name, n; sigma, rng)` (names `"fig1" "f1"/"friedman1" "f2" ... "f5"`),
  `load_yacht(; cache_dir)` (throws `DatasetUnavailableError` offline).
* MLJModelInterface models `BoostedForestRegressor`, `SmoothedForestRegressor`,
  `ConditionalBiasForestRegressor` (Deterministic, `Table(Continuous)` -> `Continuous`).
  Hyper-parameters `n_trees`, `mtry` (0 = default), `nodesize`, `seed` (-1 = random) plus
  `n_iterations`/`stopping` or `method`.

## Notes

* **Nodesize.** R's `nodesize = 5` means nodes with <= 5 bootstrap draws are not split.  In
  DecisionTree.jl terms that is `min_samples_split = 6`, `min_samples_leaf = 1`; bootstrap duplicates
  count as separate rows (as in R).  `mtry` is re-drawn at every split.
* **Reproducibility.** Every tree (and every simulation replication) gets its own RNG derived
  from `seed`, so results do not depend on the number of threads.  Boosting stage `j` uses a seed
  derived from the base seed; `fit(BoostedForest, X, y; n_iterations=0, seed=s)` equals `fit_forest(X, y; seed=s)`.
* **Rows never out-of-bag** (only with few trees) fall back to the full-forest prediction, with a warning.
* **Spline.** Reproduces R 4.3: tie aggregation (tolerance 1e-6 IQR), x scaled to [0,1], knot subset
  `.nknots.smspl`, penalty matrix (including `sgram.f`'s constant 0.3330), `lambda = r*256^(3 spar-1)`,
  GCV, Brent search on spar in [-1.5, 1.5] with R's tolerances; like R, the reported `spar` is the best
  abscissa while coefficients, `lambda` and `df` come from the last evaluation.  It uses dense rather
  than banded linear algebra (fine for the few hundred knots R allows).
* Test fixtures `test/data/smooth_spline_*.csv` come from `dev/spline_reference.R` (R 4.3.3).

## Tests

```
julia -t 4 --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```
