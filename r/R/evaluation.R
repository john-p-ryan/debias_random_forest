# Method comparison: all methods from one base forest, simulation studies, repeated
# train/test splits and the paper's stopping rule for the number of iterations.

#' Method labels in paper order
#'
#' @param max_extra_iterations largest k of the boosted forests Bf(k) reported.
#' @return `c("Un", "De", "Qu", "Cu", "Sp", "Bf", "Bf(1)", ..., "Cond")`.
#' @examples
#' method_labels()
#' @export
method_labels <- function(max_extra_iterations = 3) {
  c("Un", "De", "Qu", "Cu", "Sp", paper_label(seq_len(max_extra_iterations + 1)), "Cond")
}

#' Test predictions of every method from one base forest
#'
#' Fits the base forest once and builds every correction on it: Un, the univariate
#' corrections De, Qu, Cu, Sp, the boosted forests Bf, Bf(1), ...,
#' Bf(`max_extra_iterations`) (Bf(k) has k + 1 residual forests, and the base forest
#' is their stage 0) and Cond.
#'
#' @param x_train,y_train training data.
#' @param x_test test predictors.
#' @param max_extra_iterations largest k of Bf(k).
#' @param ntree,mtry,nodesize forest settings (every forest).
#' @param seed optional integer seed (see [fit_forest()]).
#' @param ... further arguments to [randomForest::randomForest()].
#' @return Matrix of test predictions, one column per method, named as in
#'   [method_labels()].
#' @examples
#' d <- simulate_data("f1", 200, seed = 1)
#' test <- simulate_data("f1", 500, sigma = 0, seed = 2)
#' P <- predict_all_methods(d$x, d$y, test$x, ntree = 100, seed = 1)
#' round(colMeans((P - test$f)^2), 2)
#' @export
predict_all_methods <- function(x_train, y_train, x_test, max_extra_iterations = 3,
                                ntree = 500, mtry = default_mtry(x_train), nodesize = 5,
                                seed = NULL, ...) {
  x_train <- check_x(x_train)
  with_seed(seed, {
    base <- fit_forest(x_train, y_train, ntree = ntree, mtry = mtry, nodesize = nodesize,
                       ...)
    smooth <- vapply(c("linear", "quadratic", "cubic", "spline"), function(m) {
      stats::predict(smoothed_forest(x_train, method = m, forest = base), x_test)
    }, numeric(NROW(x_test)))
    bf <- boosted_forest(x_train, n_iterations = max_extra_iterations + 1, forest = base,
                         nodesize = nodesize, ...)
    cond <- conditional_bias_forest(x_train, forest = base)
  })
  P <- cbind(Un = unname(stats::predict(base, x_test)),
             matrix(smooth, nrow = NROW(x_test),
                    dimnames = list(NULL, c("De", "Qu", "Cu", "Sp"))),
             staged_predict(bf, x_test)[, -1L, drop = FALSE],
             Cond = stats::predict(cond, x_test))
  stopifnot(identical(colnames(P), method_labels(max_extra_iterations)))
  P
}

# lapply over `X`, in forked processes when n_cores > 1 (sequential on Windows).  Every
# task seeds itself, so results do not depend on n_cores.
par_lapply <- function(X, FUN, n_cores = 1L) {
  if (n_cores <= 1L || length(X) <= 1L || .Platform$OS.type == "windows")
    return(lapply(X, FUN))
  res <- parallel::mclapply(X, FUN, mc.cores = n_cores, mc.preschedule = FALSE)
  failed <- vapply(res, function(r) inherits(r, "try-error") || is.null(r), logical(1))
  if (any(failed)) {
    err <- res[[which(failed)[1L]]]
    stop("a parallel task failed: ",
         if (inherits(err, "try-error")) conditionMessage(attr(err, "condition"))
         else "no result (worker killed?)", call. = FALSE)
  }
  res
}

draw_seeds <- function(k) sample.int(.Machine$integer.max, k)

mspe_summary <- function(mspe) {
  list(mean = colMeans(mspe),
       sd = if (nrow(mspe) > 1L) apply(mspe, 2L, stats::sd) else rep(0, ncol(mspe)))
}

#' Monte-Carlo comparison of all methods on a simulation design
#'
#' The test predictors (and `f(x_test)`) are drawn once and held fixed; each
#' replication draws a fresh training set and fresh test noise, fits every method
#' with [predict_all_methods()] and records the test MSPE against the noisy test
#' response.  The squared bias uses the average prediction over replications, so the
#' MSPE and the squared bias come from the same fits.
#'
#' @param name design name (see [simulate_data()]).
#' @param sigma noise standard deviation (default: the design's).
#' @param n,n_test training and test sizes.
#' @param n_reps number of replications.
#' @param seed optional integer seed (see [fit_forest()]).
#' @param n_cores number of forked worker processes (see [parallel::mclapply()];
#'   ignored on Windows).  Every replication has its own seed, so results do not
#'   depend on `n_cores`.
#' @param ntree,max_extra_iterations,... passed to [predict_all_methods()].
#' @return An object of class `debias_simulation`: a list with `labels`, `mspe`
#'   (`n_reps x methods` matrix), `mean` and `sd` of the MSPE, `bias2` (mean over test
#'   points of the squared difference between the average prediction and `f`), and
#'   the inputs.  `as.data.frame()` gives one row per method.
#' @examples
#' \donttest{
#' res <- simulation_study("f1", n = 100, n_test = 200, n_reps = 3, ntree = 50, seed = 1)
#' res
#' }
#' @export
simulation_study <- function(name, sigma = NULL, n = 200, n_test = 2000, n_reps = 200,
                             seed = NULL, n_cores = 1L, ntree = 500,
                             max_extra_iterations = 3, ...) {
  if (is.null(sigma)) sigma <- simulation_sigma(name)
  n_reps <- check_count(n_reps, "n_reps", min = 1L)
  setup <- with_seed(seed, list(test = simulate_data(name, n_test, sigma = 0),
                                seeds = draw_seeds(n_reps)))
  test <- setup$test
  res <- par_lapply(setup$seeds, function(s) with_seed(s, {
    train <- simulate_data(name, n, sigma = sigma)
    y_test <- test$f + sigma * stats::rnorm(n_test)
    P <- predict_all_methods(train$x, train$y, test$x, ntree = ntree,
                             max_extra_iterations = max_extra_iterations, ...)
    list(P = P, mspe = colMeans((P - y_test)^2))
  }), n_cores)
  mspe <- do.call(rbind, lapply(res, `[[`, "mspe"))
  avg <- Reduce(`+`, lapply(res, `[[`, "P")) / n_reps
  s <- mspe_summary(mspe)
  structure(list(labels = colnames(mspe), mspe = mspe, mean = s$mean, sd = s$sd,
                 bias2 = colMeans((avg - test$f)^2), name = as.character(name),
                 sigma = sigma, n = n, n_test = n_test, n_reps = n_reps),
            class = "debias_simulation")
}

split_indices <- function(n, test_size) {
  # ceiling matches scikit-learn's ShuffleSplit (and the Julia package)
  n_test <- min(max(ceiling(n * test_size), 1), n - 1)
  sample.int(n, n_test)
}

check_test_size <- function(test_size) {
  if (!is.numeric(test_size) || length(test_size) != 1L || is.na(test_size) ||
      test_size <= 0 || test_size >= 1)
    stop("`test_size` must be a fraction strictly between 0 and 1", call. = FALSE)
}

#' Comparison of all methods over repeated random train/test splits
#'
#' @param x,y data.
#' @param n_splits number of random splits.
#' @param test_size fraction of rows in each test set (rounded up).
#' @inheritParams simulation_study
#' @return An object of class `debias_splits`: a list with `labels`, `mspe`
#'   (`n_splits x methods`), `mean` and `sd`.
#' @export
repeated_splits <- function(x, y, n_splits = 100, test_size = 1 / 3, seed = NULL,
                            n_cores = 1L, ntree = 500, max_extra_iterations = 3, ...) {
  x <- check_x(x)
  y <- check_y(y, x)
  check_test_size(test_size)
  seeds <- with_seed(seed, draw_seeds(check_count(n_splits, "n_splits", min = 1L)))
  res <- par_lapply(seeds, function(s) with_seed(s, {
    te <- split_indices(nrow(x), test_size)
    P <- predict_all_methods(x[-te, , drop = FALSE], y[-te], x[te, , drop = FALSE],
                             ntree = ntree, max_extra_iterations = max_extra_iterations,
                             ...)
    colMeans((P - y[te])^2)
  }), n_cores)
  mspe <- do.call(rbind, res)
  s <- mspe_summary(mspe)
  structure(list(labels = colnames(mspe), mspe = mspe, mean = s$mean, sd = s$sd),
            class = "debias_splits")
}

#' The paper's stopping rule for the number of boosting iterations
#'
#' On each of `n_splits` random splits one [boosted_forest()] with `max_iterations`
#' residual forests is fitted and [staged_predict()] gives the test MSPE for
#' k = 0, ..., `max_iterations`.  The selected k is the first one whose mean MSPE
#' (over splits) is lower than that at k + 1, or `max_iterations` if the mean curve
#' never increases.
#'
#' @inheritParams repeated_splits
#' @param max_iterations largest number of residual forests tried.
#' @param ntree number of trees of every forest.
#' @param ... further arguments to [boosted_forest()] (for example `mtry`,
#'   `nodesize`).
#' @return An object of class `debias_selection`: a list with `n_iterations` (the
#'   selected k), `mean_mspe` and `sd_mspe` (named by [paper_label()], k = 0, ...,
#'   `max_iterations`) and `mspe` (`n_splits x (max_iterations + 1)`).
#' @export
select_n_iterations <- function(x, y, max_iterations = 6, n_splits = 20, test_size = 1 / 3,
                                seed = NULL, n_cores = 1L, ntree = 500, ...) {
  x <- check_x(x)
  y <- check_y(y, x)
  max_iterations <- check_count(max_iterations, "max_iterations")
  check_test_size(test_size)
  seeds <- with_seed(seed, draw_seeds(check_count(n_splits, "n_splits", min = 1L)))
  res <- par_lapply(seeds, function(s) with_seed(s, {
    te <- split_indices(nrow(x), test_size)
    bf <- boosted_forest(x[-te, , drop = FALSE], y[-te], n_iterations = max_iterations,
                         ntree = ntree, ...)
    colMeans((staged_predict(bf, x[te, , drop = FALSE]) - y[te])^2)
  }), n_cores)
  mspe <- do.call(rbind, res)
  s <- mspe_summary(mspe)
  rises <- which(s$mean[-length(s$mean)] < s$mean[-1L])
  k <- if (length(rises)) rises[1L] - 1L else max_iterations
  structure(list(n_iterations = as.integer(k), mean_mspe = s$mean, sd_mspe = s$sd,
                 mspe = mspe),
            class = "debias_selection")
}

#' @export
as.data.frame.debias_simulation <- function(x, row.names = NULL, optional = FALSE, ...) {
  data.frame(method = x$labels, mspe = unname(x$mean), sd = unname(x$sd),
             bias2 = unname(x$bias2), row.names = row.names, ...)
}

#' @export
as.data.frame.debias_splits <- function(x, row.names = NULL, optional = FALSE, ...) {
  data.frame(method = x$labels, mspe = unname(x$mean), sd = unname(x$sd),
             row.names = row.names, ...)
}

#' @export
print.debias_simulation <- function(x, digits = 3, ...) {
  cat("Simulation study '", x$name, "' (sigma = ", x$sigma, ", n = ", x$n,
      ", n_test = ", x$n_test, ", ", x$n_reps, " replications)\n", sep = "")
  print(as.data.frame(x), digits = digits, row.names = FALSE)
  invisible(x)
}

#' @export
print.debias_splits <- function(x, digits = 3, ...) {
  cat("Repeated splits (", nrow(x$mspe), " splits): test MSPE\n", sep = "")
  print(as.data.frame(x), digits = digits, row.names = FALSE)
  invisible(x)
}

#' @export
print.debias_selection <- function(x, digits = 3, ...) {
  cat("Selected n_iterations = ", x$n_iterations, " (", paper_label(x$n_iterations),
      ") from ", nrow(x$mspe), " splits\n", sep = "")
  print(rbind(mean = x$mean_mspe, sd = x$sd_mspe), digits = digits)
  invisible(x)
}
