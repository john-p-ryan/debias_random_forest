# Univariate smoothing bias correction (De / Qu / Cu / Sp in the paper).

#' Univariate smoothing bias correction
#'
#' Fits `y ~ g(yhat_oob)` on the training data, where `yhat_oob` is the base
#' forest's OOB prediction, and predicts `g(forest prediction)`.  `"linear"`,
#' `"quadratic"` and `"cubic"` are least-squares polynomials (the paper's De, Qu, Cu;
#' `"linear"` is the regression `randomForest`'s `corr.bias` fits to the OOB
#' predictions, although `predict.randomForest` applies it differently), `"spline"` is
#' [stats::smooth.spline()] with its default GCV choice of smoothing (Sp), which
#' extrapolates linearly beyond the range of the OOB predictions.
#'
#' @inheritParams boosted_forest
#' @param method `"linear"`, `"quadratic"`, `"cubic"` or `"spline"`.
#' @param ntree,mtry,nodesize settings of the base forest (ignored with `forest`).
#' @param ... further arguments to [randomForest::randomForest()].
#' @return An object of class `smoothed_forest`: a list with `forest`, `method`,
#'   `smoother` (the fitted `smooth.spline`, or a list with the polynomial's
#'   `center`, `scale` and `coef` on the standardised OOB prediction),
#'   `base_oob_prediction` (`yhat_oob`) and `oob_prediction` (`g(yhat_oob)`).
#' @examples
#' d <- simulate_data("f1", 200, seed = 1)
#' rf <- fit_forest(d$x, d$y, ntree = 100, seed = 1)
#' sp <- smoothed_forest(d$x, method = "spline", forest = rf)
#' head(predict(sp, d$x))
#' @export
smoothed_forest <- function(x, y = NULL, method = c("linear", "quadratic", "cubic", "spline"),
                            forest = NULL, ntree = 500, mtry = default_mtry(x),
                            nodesize = 5, seed = NULL, ...) {
  method <- match.arg(method)
  x <- check_x(x)
  base <- with_seed(seed, base_forest(x, y, forest, ntree, mtry, nodesize, ...))
  u <- oob_predictions(base$forest, x)
  g <- if (method == "spline") {
    stats::smooth.spline(u, base$y)
  } else {
    fit_polynomial(u, base$y, c(linear = 1L, quadratic = 2L, cubic = 3L)[[method]])
  }
  structure(list(forest = base$forest, method = method, smoother = g,
                 base_oob_prediction = u, oob_prediction = apply_smoother(g, u)),
            class = "smoothed_forest")
}

# Least-squares polynomial of `y` on the standardised `u` (stable for degree <= 3).
fit_polynomial <- function(u, y, degree) {
  center <- mean(u)
  scale <- stats::sd(u)
  if (!is.finite(scale) || scale == 0) scale <- 1
  coef <- stats::lm.fit(outer((u - center) / scale, 0:degree, `^`), y)$coefficients
  coef[is.na(coef)] <- 0  # aliased terms (e.g. fewer distinct values than degree + 1)
  list(center = center, scale = scale, coef = unname(coef))
}

apply_smoother <- function(g, u) {
  if (inherits(g, "smooth.spline")) return(stats::predict(g, u)$y)
  z <- (u - g$center) / g$scale
  drop(outer(z, seq_along(g$coef) - 1L, `^`) %*% g$coef)
}

#' @rdname predict.debiasforest
#' @export
predict.smoothed_forest <- function(object, newdata, ...) {
  if (missing(newdata)) return(object$oob_prediction)
  apply_smoother(object$smoother, unname(stats::predict(object$forest, newdata)))
}

#' @export
print.smoothed_forest <- function(x, ...) {
  label <- c(linear = "De", quadratic = "Qu", cubic = "Cu", spline = "Sp")[[x$method]]
  cat("Smoothed random forest (", label, "): ", x$method, " map of y on the OOB prediction\n",
      sep = "")
  if (x$method == "spline")
    cat("  smooth.spline: df = ", format(x$smoother$df, digits = 4), ", lambda = ",
        format(x$smoother$lambda, digits = 4), "\n", sep = "")
  cat("  base forest: ", x$forest$ntree, " trees, mtry = ", x$forest$mtry, "\n", sep = "")
  invisible(x)
}
