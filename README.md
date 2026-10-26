# Debias Random Forest Regression Predictors

Planned Python (`debiasforest`) and Julia (`DebiasForest.jl`) implementations of
random forest bias correction, with reproducible simulations and a UCI Yacht
regression demo.

**Status:** planning stage. Packages, tests, and demos are not yet implemented.

## Planned methods

- **Boosted forests (Bf):** fit additional forests to out-of-bag (OOB) residuals.
  The default, `n_iterations=1`, is the paper's one-step correction;
  `n_iterations=0` gives the uncorrected forest.
- **Univariate smoothing (De/Qu/Cu/Sp):** linear, quadratic, cubic, or
  R-compatible smoothing-spline correction.
- **Conditional bias correction (Cond):** estimate bias using OOB observations
  sharing terminal leaves.

Both packages will track bootstrap membership, expose corrected OOB predictions,
and support fixed iterations, OOB stopping, and repeated-split iteration selection.
Defaults follow the paper's R forests: 500 trees, `mtry=max(floor(p/3), 1)`,
and node size 5. Python will use scikit-learn ≥1.4; Julia will use
DecisionTree.jl with a custom bagging loop and an MLJ interface.

## Planned Python quick start

These commands and APIs will become available after implementation
(target environment: Python 3.12):

```sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e "python[test,demo]"
```

```python
from debiasforest import BoostedForestRegressor

model = BoostedForestRegressor(n_iterations=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Planned layout and validation

- `python/` — scikit-learn-compatible estimators and pytest tests.
- `julia/` — Julia package, MLJ models, and Julia tests.
- `demo/` — Python notebook, Julia script, simulation tables, and figures.
- `dev/` — R script generating reference smoothing-spline fixtures.

Demos will compare prediction error and squared bias across methods, reproduce
the paper's main comparisons, and cache Yacht data when available. Validation
will cover OOB calculations, correction formulas, reproducibility, R spline
agreement, and end-to-end demo execution. An MIT license is planned.

## Reference

Chen, L., Withana Gamage, P., & Ryan, J. (2022). *Debias Random Forest Regression
Predictors*. Journal of Statistical Research, 56(2), 115–131.
[doi:10.3329/jsr.v56i2.67466](https://doi.org/10.3329/jsr.v56i2.67466).
