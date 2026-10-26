# Base forests: fitting with R randomForest defaults, validation of prefit forests and
# out-of-bag (OOB) predictions.

#' Default `mtry` for regression
#'
#' `max(floor(p / 3), 1)`, the [randomForest::randomForest()] default for regression.
#'
#' @param x predictor matrix or data frame (or the number of predictors).
#' @return An integer.
#' @export
default_mtry <- function(x) {
  p <- if (is.numeric(x) && length(x) == 1L && is.null(dim(x))) x else NCOL(x)
  max(as.integer(p %/% 3), 1L)
}

#' Fit a random forest that keeps its bootstrap samples
#'
#' Calls [randomForest::randomForest()] with the defaults used in the paper (500
#' trees, `mtry = max(floor(p / 3), 1)`, `nodesize = 5`, bootstrap of size n drawn
#' with replacement) and `keep.inbag = TRUE`, so that every bias correction in this
#' package can use the forest.
#'
#' `nodesize` follows `randomForest`: a node with at most `nodesize` bootstrap draws
#' is not split (duplicated draws count separately).
#'
#' @param x predictor matrix or data frame.
#' @param y numeric response.
#' @param ntree number of trees.
#' @param mtry number of predictors sampled at each split.
#' @param nodesize minimum size of terminal nodes.
#' @param seed optional integer seed.  If given, the fit is reproducible and the
#'   global random number stream is restored afterwards; if `NULL` the current
#'   stream is used (so `set.seed()` beforehand also works).
#' @param ... further arguments to [randomForest::randomForest()] (for example
#'   `replace`, `sampsize` or `maxnodes`).
#' @return A `randomForest` object with `inbag`, plus `nodesize`, which
#'   [boosted_forest()] reuses for its residual forests.  Its `call` refers to
#'   `fit_forest()`'s arguments, so use `randomForest(..., keep.inbag = TRUE)`
#'   directly if you need `update()` or `randomForest::grow()`.
#' @seealso [oob_predictions()]
#' @examples
#' d <- simulate_data("f1", 100, seed = 1)
#' rf <- fit_forest(d$x, d$y, ntree = 100, seed = 1)
#' head(oob_predictions(rf, d$x))
#' @export
fit_forest <- function(x, y, ntree = 500, mtry = default_mtry(x), nodesize = 5,
                       seed = NULL, ...) {
  x <- check_x(x)
  y <- check_y(y, x)
  rf <- with_seed(seed, randomForest::randomForest(
    x = x, y = y, ntree = ntree, mtry = mtry, nodesize = nodesize,
    keep.inbag = TRUE, keep.forest = TRUE, ...))
  rf$nodesize <- nodesize
  rf
}

#' Out-of-bag predictions of a random forest
#'
#' The OOB prediction of training row i is the average prediction of the trees whose
#' bootstrap sample does not contain row i (`forest$predicted`).  For forests whose
#' stored OOB predictions are out of date (after [randomForest::combine()]) they are
#' recomputed from `forest$inbag` and the tree predictions at `x`.  Rows that are in
#' the bootstrap sample of every tree (only likely with few trees) fall back to the
#' full-forest prediction, with a warning.
#'
#' @param forest a regression `randomForest` object.
#' @param x the training predictors (needed only for combined forests and for the
#'   fallback).
#' @return Numeric vector of length n.
#' @export
oob_predictions <- function(forest, x = NULL) {
  if (!inherits(forest, "randomForest") || !identical(forest$type, "regression"))
    stop("`forest` must be a regression randomForest object", call. = FALSE)
  if (!is.null(forest$coefs))
    stop("forests fitted with `corr.bias = TRUE` are not supported", call. = FALSE)
  pred <- unname(as.numeric(forest$predicted))
  n_oob <- forest$oob.times
  if (!is.null(forest$inbag) && !oob_counts_current(forest)) {
    if (is.null(x))
      stop("the OOB predictions stored in `forest` are out of date (e.g. after ",
           "combine()); pass the training predictors `x`", call. = FALSE)
    pred <- tree_oob_mean(forest, x)
    n_oob <- rowSums(forest$inbag == 0)
  }
  never <- n_oob == 0
  if (any(never)) {
    if (is.null(x))
      stop("some rows were never out-of-bag; pass the training predictors `x`",
           call. = FALSE)
    warning(sum(never), " training rows were never out-of-bag; their OOB prediction ",
            "falls back to the full-forest prediction. Use more trees.", call. = FALSE)
    pred[never] <- unname(stats::predict(forest, x[never, , drop = FALSE]))
  }
  pred
}

# FALSE when `predicted` / `oob.times` no longer describe `inbag` (randomForest::combine()).
oob_counts_current <- function(forest) {
  identical(as.integer(forest$oob.times), as.integer(rowSums(forest$inbag == 0)))
}

# Mean prediction of the trees for which each of `rows` of `x` is OOB (NaN if none).
tree_oob_mean <- function(forest, x, rows = seq_len(nrow(x))) {
  ind <- stats::predict(forest, x[rows, , drop = FALSE], predict.all = TRUE)$individual
  oob <- forest$inbag[rows, , drop = FALSE] == 0
  unname(rowSums(ind * oob) / rowSums(oob))
}

# Return the base forest of a debiasing method: either fit one on (x, y) or check that
# the supplied `forest` was fitted on these data.  `y` defaults to the forest's response.
base_forest <- function(x, y, forest, ntree, mtry, nodesize, ...) {
  if (is.null(forest)) {
    if (is.null(y)) stop("`y` is required unless `forest` is supplied", call. = FALSE)
    return(list(forest = fit_forest(x, y, ntree = ntree, mtry = mtry,
                                    nodesize = nodesize, ...),
                y = check_y(y, x)))
  }
  check_forest(forest)
  if (nrow(forest$inbag) != nrow(x))
    stop("`forest` was fitted on ", nrow(forest$inbag), " rows but `x` has ",
         nrow(x), "; x must be the forest's training data", call. = FALSE)
  if (length(forest$forest$ncat) != ncol(x))
    stop("`forest` was fitted with a different number of predictors", call. = FALSE)
  y <- if (is.null(y)) unname(as.numeric(forest$y)) else check_y(y, x)
  if (!isTRUE(all.equal(y, unname(as.numeric(forest$y)))))
    stop("`y` is not the response `forest` was fitted on", call. = FALSE)
  if (oob_counts_current(forest)) {
    # spot check that `x` holds the training rows in their original order
    rows <- unique(round(seq(1, nrow(x), length.out = min(nrow(x), 10L))))
    rows <- rows[forest$oob.times[rows] > 0]
    stored <- unname(forest$predicted[rows])
    if (length(rows) && max(abs(tree_oob_mean(forest, x, rows) - stored)) >
        1e-8 * (1 + max(abs(stored))))
      stop("`x` is not the training data of `forest` (its OOB predictions differ); ",
           "pass the predictors it was fitted on, in the same row order", call. = FALSE)
  }
  list(forest = forest, y = y)
}

check_forest <- function(forest) {
  if (!inherits(forest, "randomForest") || !identical(forest$type, "regression"))
    stop("`forest` must be a regression randomForest object", call. = FALSE)
  if (is.null(forest$inbag))
    stop("`forest` must be fitted with keep.inbag = TRUE (see fit_forest())",
         call. = FALSE)
  if (is.null(forest$forest))
    stop("`forest` must be fitted with keep.forest = TRUE", call. = FALSE)
  invisible(forest)
}

check_x <- function(x) {
  if (is.data.frame(x)) return(x)
  if (is.null(dim(x))) stop("`x` must be a matrix or data frame", call. = FALSE)
  x <- as.matrix(x)
  if (!is.numeric(x)) stop("`x` must be numeric (or a data frame)", call. = FALSE)
  x
}

check_y <- function(y, x) {
  if (!is.numeric(y) || (!is.null(dim(y)) && NCOL(y) != 1L))
    stop("`y` must be a numeric vector", call. = FALSE)
  y <- as.numeric(y)
  if (length(y) != nrow(x))
    stop("`x` has ", nrow(x), " rows but `y` has ", length(y), " values", call. = FALSE)
  y
}

check_count <- function(k, name, min = 0L) {
  if (!is.numeric(k) || length(k) != 1L || is.na(k) || k < min || k != round(k))
    stop("`", name, "` must be a ", if (min > 0) "positive" else "non-negative",
         " integer", call. = FALSE)
  as.integer(k)
}

# Evaluate `expr` after set.seed(seed) and restore the global RNG state afterwards;
# with seed = NULL `expr` simply uses the current stream.
with_seed <- function(seed, expr) {
  if (is.null(seed)) return(expr)
  env <- globalenv()
  had_seed <- exists(".Random.seed", envir = env, inherits = FALSE)
  if (had_seed) old <- get(".Random.seed", envir = env, inherits = FALSE)
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old, envir = env)
    } else if (exists(".Random.seed", envir = env, inherits = FALSE)) {
      rm(".Random.seed", envir = env)
    }
  })
  set.seed(seed)
  expr
}
