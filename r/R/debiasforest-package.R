#' debiasforest: bias correction for random forest regression
#'
#' Implements the methods compared in Chen, Withana Gamage & Ryan (2022),
#' "Debias Random Forest Regression Predictors", *J. Stat. Research* 56(2):115-131,
#' on top of [randomForest::randomForest()] forests fitted with `keep.inbag = TRUE`.
#'
#' | Paper | R |
#' |---|---|
#' | Un | [fit_forest()] (any regression `randomForest` with `keep.inbag = TRUE`) |
#' | De / Qu / Cu / Sp | [smoothed_forest()] with `method = "linear"`, `"quadratic"`, `"cubic"`, `"spline"` |
#' | Bf, Bf(k) | [boosted_forest()] with `n_iterations = 1`, `stopping = "fixed"` or `"oob"`; [staged_predict()] |
#' | Cond | [conditional_bias_forest()] |
#' | stopping rule | [select_n_iterations()] |
#' | comparisons | [predict_all_methods()], [simulation_study()], [repeated_splits()] |
#'
#' `n_iterations` is the number of residual forests fitted after the base forest:
#' 0 is Un, 1 is Bf (the default) and k + 1 is Bf(k); [paper_label()] gives the
#' paper's notation.
#'
#' @references Chen, L., Withana Gamage, P. & Ryan, J. (2022). Debias random forest
#'   regression predictors. *Journal of Statistical Research*, 56(2), 115-131.
#'   \doi{10.3329/jsr.v56i2.67466}
#' @keywords internal
"_PACKAGE"
