# Shared data and brute-force references.

f1_train <- simulate_data("f1", 120, seed = 11)
f1_test <- simulate_data("f1", 60, seed = 12)

noise_data <- function(n = 100, p = 3, seed = 0) {
  set.seed(seed)
  list(x = matrix(stats::runif(n * p), n, p), y = stats::rnorm(n))
}

# OOB prediction from the individual tree predictions and the inbag counts.
brute_force_oob <- function(rf, x) {
  ind <- stats::predict(rf, x, predict.all = TRUE)$individual
  oob <- rf$inbag == 0
  unname(rowSums(ind * oob) / rowSums(oob))
}

# Paper formula: b(x0) = sum_i v_i(x0) (oob_i - y_i), v_i(x0) = (number of trees in which
# i is OOB and in x0's terminal node) / (sum over i of the same).  `exclude_self[r]` is a
# training row whose weight is set to 0 for row r of x0.
brute_force_bias <- function(rf, x, y, x0, exclude_self = NULL) {
  oob <- brute_force_oob(rf, x)
  bias <- oob - y
  oob_mask <- rf$inbag == 0
  leaves <- attr(stats::predict(rf, x, nodes = TRUE), "nodes")
  leaves0 <- attr(stats::predict(rf, x0, nodes = TRUE), "nodes")
  b <- numeric(nrow(x0))
  for (r in seq_len(nrow(x0))) {
    cnt <- numeric(length(y))
    for (t in seq_len(rf$ntree)) {
      for (i in seq_along(y)) {
        if (oob_mask[i, t] && leaves[i, t] == leaves0[r, t]) cnt[i] <- cnt[i] + 1
      }
    }
    if (!is.null(exclude_self)) cnt[exclude_self[r]] <- 0
    b[r] <- if (sum(cnt) == 0) 0 else sum((cnt / sum(cnt) * bias)[cnt > 0])
  }
  list(bias = b, oob = oob)
}

# `expr` (evaluated lazily) must leave the global random number stream untouched.
expect_rng_preserved <- function(expr) {
  set.seed(2024)
  before <- get(".Random.seed", envir = globalenv())
  force(expr)
  expect_identical(get(".Random.seed", envir = globalenv()), before)
}
