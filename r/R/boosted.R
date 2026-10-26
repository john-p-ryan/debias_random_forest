# Boosted random forest (Breiman 2001a; Chen, Withana Gamage & Ryan 2022, section 2.2.2).

#' Paper label for a number of residual iterations
#'
#' @param k number(s) of residual forests after the base forest.
#' @return `"Un"` for 0, `"Bf"` for 1 and `"Bf(k-1)"` for k >= 2.
#' @examples
#' paper_label(0:4)
#' @export
paper_label <- function(k) {
  if (!is.numeric(k) || anyNA(k) || any(k < 0) || any(k != round(k)))
    stop("`k` must be non-negative integers", call. = FALSE)
  ifelse(k == 0, "Un", ifelse(k == 1, "Bf", paste0("Bf(", k - 1, ")")))
}

#' Boosted random forest
#'
#' Stage 0 is the base forest fitted to `y`.  With `o_j` the OOB prediction of stage
#' j, stage j + 1 is a fresh forest fitted to the OOB residual `r_{j+1} = r_j - o_j`
#' (`r_0 = y`).  The prediction is the sum of the stage predictions and the
#' bias-corrected OOB prediction is the sum of the stage OOB predictions.
#'
#' With `stopping = "oob"`, `n_iterations` is a maximum: after each stage the OOB MSE
#' of `y` against the cumulative OOB prediction is recorded, growing stops once it
#' exceeds 1.1 times the minimum so far, and the stages up to the minimum are kept.
#'
#' @param x predictor matrix or data frame.
#' @param y numeric response (defaults to the response of `forest`).
#' @param n_iterations number of residual forests after the base forest: 0 is the
#'   uncorrected forest (Un), 1 the paper's Bf, k + 1 the paper's Bf(k).
#' @param stopping `"fixed"` (exactly `n_iterations` residual forests) or `"oob"`.
#' @param forest optional base forest already fitted on `x`, `y` with
#'   `keep.inbag = TRUE` (for example from [fit_forest()]); it is used as stage 0.
#' @param ntree,mtry,nodesize forest settings for every stage.  With a supplied
#'   `forest`, they default to its own values (`nodesize` only for forests from
#'   [fit_forest()], since `randomForest` does not store it); other
#'   `randomForest()` arguments it was fitted with must be repeated in `...`.
#' @param seed optional integer seed (see [fit_forest()]).
#' @param ... further arguments to [randomForest::randomForest()] for every stage.
#' @return An object of class `boosted_forest`: a list with `forests` (the stage
#'   forests, length `n_iterations + 1`), `n_iterations` (number of residual forests
#'   kept), `stopping`, `oob_mse_path` (OOB MSE after stages 0, 1, ... as computed,
#'   including a stage that triggered the stop), `oob_prediction` and
#'   `stage_oob_predictions` (n x stages matrix).
#' @seealso [staged_predict()], [select_n_iterations()]
#' @examples
#' d <- simulate_data("f1", 200, seed = 1)
#' test <- simulate_data("f1", 500, sigma = 0, seed = 2)
#' bf <- boosted_forest(d$x, d$y, n_iterations = 1, ntree = 100, seed = 1)
#' colMeans((staged_predict(bf, test$x) - test$f)^2)   # Un, Bf
#' @export
boosted_forest <- function(x, y = NULL, n_iterations = 1, stopping = c("fixed", "oob"),
                           forest = NULL, ntree = 500, mtry = default_mtry(x),
                           nodesize = 5, seed = NULL, ...) {
  stopping <- match.arg(stopping)
  K <- check_count(n_iterations, "n_iterations")
  x <- check_x(x)
  if (!is.null(forest)) {
    check_forest(forest)
    if (missing(ntree)) ntree <- forest$ntree
    if (missing(mtry)) mtry <- forest$mtry
    if (missing(nodesize) && !is.null(forest$nodesize)) nodesize <- forest$nodesize
  }
  with_seed(seed, {
    base <- base_forest(x, y, forest, ntree, mtry, nodesize, ...)
    y <- base$y
    oob <- oob_predictions(base$forest, x)
    forests <- list(base$forest)
    stage_oob <- list(oob)
    cum <- oob
    mse <- mean((y - cum)^2)
    resid <- y - oob
    for (j in seq_len(K)) {
      f <- fit_forest(x, resid, ntree = ntree, mtry = mtry, nodesize = nodesize, ...)
      o <- oob_predictions(f, x)
      forests[[j + 1L]] <- f
      stage_oob[[j + 1L]] <- o
      cum <- cum + o
      resid <- resid - o
      mse <- c(mse, mean((y - cum)^2))
      if (stopping == "oob" && mse[j + 1L] > 1.1 * min(mse[seq_len(j)])) break
    }
  })
  keep <- if (stopping == "oob") which.min(mse) else length(forests)
  O <- do.call(cbind, stage_oob[seq_len(keep)])
  colnames(O) <- paper_label(seq_len(keep) - 1L)
  structure(list(forests = forests[seq_len(keep)], n_iterations = keep - 1L,
                 stopping = stopping, oob_mse_path = mse,
                 oob_prediction = rowSums(O), stage_oob_predictions = O),
            class = "boosted_forest")
}

#' Cumulative predictions of a boosted forest
#'
#' @param object a fitted model.
#' @param newdata predictors.
#' @param ... unused.
#' @return Matrix with one column per number of residual forests k = 0, 1, ...,
#'   `n_iterations`, named with [paper_label()] (`"Un"`, `"Bf"`, `"Bf(1)"`, ...).
#' @export
staged_predict <- function(object, newdata, ...) UseMethod("staged_predict")

#' @rdname staged_predict
#' @export
staged_predict.boosted_forest <- function(object, newdata, ...) {
  P <- vapply(object$forests, function(f) unname(stats::predict(f, newdata)),
              numeric(NROW(newdata)))
  S <- matrix(P, nrow = NROW(newdata))
  for (k in seq_len(ncol(S))[-1L]) S[, k] <- S[, k - 1L] + S[, k]
  colnames(S) <- paper_label(seq_len(ncol(S)) - 1L)
  S
}

#' Predict from a bias-corrected forest
#'
#' @param object a fitted `boosted_forest`, `smoothed_forest` or
#'   `conditional_bias_forest`.
#' @param newdata predictors.  If missing, the bias-corrected OOB predictions of
#'   the training rows are returned (like `predict.randomForest`).
#' @param ... unused.
#' @return Numeric vector of predictions.
#' @name predict.debiasforest
NULL

#' @rdname predict.debiasforest
#' @export
predict.boosted_forest <- function(object, newdata, ...) {
  if (missing(newdata)) return(object$oob_prediction)
  S <- staged_predict(object, newdata)
  unname(S[, ncol(S)])
}

#' @export
print.boosted_forest <- function(x, ...) {
  f <- x$forests[[1L]]
  cat("Boosted random forest (", paper_label(x$n_iterations), "): ", x$n_iterations,
      " residual forest", if (x$n_iterations != 1L) "s", " after the base forest\n",
      sep = "")
  cat("  stopping: ", x$stopping, "; trees per stage: ", f$ntree, "; mtry: ", f$mtry,
      "\n", sep = "")
  cat("  OOB MSE by stage:", format(signif(x$oob_mse_path, 4)), "\n")
  invisible(x)
}
