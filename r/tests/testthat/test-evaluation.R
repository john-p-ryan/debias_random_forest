labels <- c("Un", "De", "Qu", "Cu", "Sp", "Bf", "Bf(1)", "Bf(2)", "Bf(3)", "Cond")
x <- f1_train$x
y <- f1_train$y
xt <- f1_test$x

test_that("method_labels lists the methods in paper order", {
  expect_identical(method_labels(), labels)
  expect_identical(method_labels(1), c("Un", "De", "Qu", "Cu", "Sp", "Bf", "Bf(1)", "Cond"))
  expect_identical(method_labels(0), c("Un", "De", "Qu", "Cu", "Sp", "Bf", "Cond"))
})

test_that("predict_all_methods returns one column per method, reproducibly", {
  P <- predict_all_methods(x, y, xt, ntree = 40, seed = 0)
  expect_true(is.matrix(P))
  expect_identical(colnames(P), labels)
  expect_identical(dim(P), c(60L, 10L))
  expect_true(all(is.finite(P)))
  expect_identical(predict_all_methods(x, y, xt, ntree = 40, seed = 0), P)
  Q <- predict_all_methods(x, y, xt, ntree = 40, seed = 1)
  expect_false(isTRUE(all.equal(P, Q)))
  short <- predict_all_methods(x, y, xt, ntree = 20, seed = 0, max_extra_iterations = 1)
  expect_identical(colnames(short), method_labels(1))
  expect_rng_preserved(predict_all_methods(x, y, xt, ntree = 20, seed = 0))
})

test_that("predict_all_methods works for a single test row", {
  P <- predict_all_methods(x, y, xt, ntree = 40, seed = 0)
  P1 <- predict_all_methods(x, y, xt[1, , drop = FALSE], ntree = 40, seed = 0)
  expect_identical(dim(P1), c(1L, 10L))
  expect_equal(P1[1, ], P[1, ])
})

test_that("every method is built on the same base forest", {
  P <- predict_all_methods(x, y, xt, ntree = 40, seed = 5)
  set.seed(5)
  base <- fit_forest(x, y, ntree = 40)
  bf <- boosted_forest(x, n_iterations = 4, forest = base)
  expect_identical(P[, "Un"], unname(predict(base, xt)))
  smooth <- c(De = "linear", Qu = "quadratic", Cu = "cubic", Sp = "spline")
  for (m in names(smooth)) {
    expect_identical(P[, m], predict(smoothed_forest(x, method = smooth[[m]], forest = base), xt))
  }
  expect_identical(unname(P[, c("Bf", "Bf(1)", "Bf(2)", "Bf(3)")]),
                   unname(staged_predict(bf, xt)[, -1]))
  expect_identical(P[, "Cond"], predict(conditional_bias_forest(x, forest = base), xt))
  expect_false(isTRUE(all.equal(P[, "Un"], P[, "Bf"])))
})

test_that("simulation_study summarises the replications", {
  r <- simulation_study("f1", sigma = 1, n = 60, n_test = 100, n_reps = 4, ntree = 25,
                        seed = 0)
  expect_s3_class(r, "debias_simulation")
  expect_identical(r$labels, labels)
  expect_identical(dim(r$mspe), c(4L, 10L))
  expect_identical(colnames(r$mspe), labels)
  for (v in c("mean", "sd", "bias2")) {
    expect_length(r[[v]], 10)
    expect_identical(names(r[[v]]), labels)
  }
  expect_true(all(r$bias2 >= 0))
  expect_true(all(r$mean > 0))
  expect_equal(r$mean, colMeans(r$mspe))
  expect_equal(r$sd, apply(r$mspe, 2, sd))
  expect_identical(r$sigma, 1)
  expect_identical(r$name, "f1")
  r2 <- simulation_study("f1", sigma = 1, n = 60, n_test = 100, n_reps = 4, ntree = 25,
                         seed = 0, n_cores = 2)
  expect_identical(r2$mspe, r$mspe)
  expect_identical(r2$bias2, r$bias2)
  r3 <- simulation_study("f1", sigma = 1, n = 60, n_test = 100, n_reps = 4, ntree = 25,
                         seed = 0)
  expect_identical(r3$mspe, r$mspe)
  df <- as.data.frame(r)
  expect_identical(names(df), c("method", "mspe", "sd", "bias2"))
  expect_identical(df$method, labels)
  expect_equal(df$bias2, unname(r$bias2))
  expect_output(print(r), "Simulation study 'f1'")
  expect_rng_preserved(simulation_study("f1", n = 40, n_test = 20, n_reps = 1, ntree = 25,
                                        seed = 0))
})

test_that("simulation_study's squared bias is bounded by the noiseless MSPE", {
  r1 <- simulation_study("f1", sigma = 0, n = 60, n_test = 80, n_reps = 1, ntree = 25,
                         seed = 1)
  expect_equal(r1$mspe[1, ], r1$bias2)
  expect_identical(unname(r1$sd), rep(0, 10))
  r3 <- simulation_study("f1", sigma = 0, n = 60, n_test = 80, n_reps = 3, ntree = 25,
                         seed = 1)
  expect_true(all(r3$bias2 <= r3$mean + 1e-12))
})

test_that("simulation_study uses the design's default sigma", {
  r <- simulation_study("fig1", n = 50, n_test = 40, n_reps = 2, ntree = 25, seed = 1)
  expect_identical(r$sigma, 0.5)
  expect_identical(dim(r$mspe), c(2L, 10L))
})

test_that("repeated_splits summarises the splits", {
  r <- repeated_splits(x, y, n_splits = 3, ntree = 25, seed = 0)
  expect_s3_class(r, "debias_splits")
  expect_identical(r$labels, labels)
  expect_identical(dim(r$mspe), c(3L, 10L))
  expect_equal(r$mean, colMeans(r$mspe))
  expect_true(all(r$sd >= 0))
  r2 <- repeated_splits(x, y, n_splits = 3, ntree = 25, seed = 0, n_cores = 2)
  expect_identical(r2$mspe, r$mspe)
  expect_identical(repeated_splits(x, y, n_splits = 3, ntree = 25, seed = 0)$mspe, r$mspe)
  short <- repeated_splits(x, y, n_splits = 2, ntree = 25, seed = 0, max_extra_iterations = 1)
  expect_identical(short$labels, method_labels(1))
  df <- as.data.frame(r)
  expect_identical(names(df), c("method", "mspe", "sd"))
  expect_identical(df$method, labels)
  expect_output(print(r), "Repeated splits \\(3 splits\\)")
  expect_error(repeated_splits(x, y[-1], n_splits = 2), "rows but `y` has")
})

test_that("test sets are a rounded-up fraction of the rows", {
  set.seed(1)
  expect_length(split_indices(10, 1 / 3), 4)
  expect_length(split_indices(9, 1 / 3), 3)
  expect_length(split_indices(10, 0), 1)
  expect_length(split_indices(10, 1), 9)
  te <- split_indices(30, 0.5)
  expect_false(anyDuplicated(te) > 0)
  expect_true(all(te %in% 1:30))
})

check_selection_rule <- function(res, max_iterations) {
  m <- res$mean_mspe
  rises <- which(m[-length(m)] < m[-1])
  expected <- if (length(rises)) rises[1] - 1L else max_iterations
  expect_identical(res$n_iterations, as.integer(expected))
}

test_that("select_n_iterations picks the first k before the mean MSPE rises (noise)", {
  d <- noise_data(60, 3, seed = 0)
  res <- select_n_iterations(d$x, d$y, max_iterations = 3, n_splits = 3, ntree = 25,
                             seed = 1)
  expect_s3_class(res, "debias_selection")
  expect_identical(names(res$mean_mspe), paper_label(0:3))
  expect_length(res$sd_mspe, 4)
  expect_identical(dim(res$mspe), c(3L, 4L))
  expect_equal(res$mean_mspe, colMeans(res$mspe))
  check_selection_rule(res, 3)
  expect_identical(res$n_iterations, 0L)
})

test_that("select_n_iterations keeps boosting on Friedman 1", {
  d <- simulate_data("f1", 150, seed = 5)
  res <- select_n_iterations(d$x, d$y, max_iterations = 3, n_splits = 4, ntree = 40,
                             seed = 0)
  check_selection_rule(res, 3)
  expect_lt(res$mean_mspe[["Bf"]], res$mean_mspe[["Un"]])
  expect_gte(res$n_iterations, 1)
  k <- res$n_iterations
  expect_true(all(diff(res$mean_mspe[seq_len(k + 1)]) <= 0))
  res2 <- select_n_iterations(d$x, d$y, max_iterations = 3, n_splits = 4, ntree = 40,
                              seed = 0, n_cores = 2)
  expect_identical(res2$mspe, res$mspe)
  expect_output(print(res), "Selected n_iterations")
})

test_that("select_n_iterations returns max_iterations when the curve never rises", {
  d <- simulate_data("f1", 150, seed = 5)
  one <- select_n_iterations(d$x, d$y, max_iterations = 1, n_splits = 3, ntree = 40,
                             seed = 0)
  expect_lt(one$mean_mspe[["Bf"]], one$mean_mspe[["Un"]])
  expect_identical(one$n_iterations, 1L)
  zero <- select_n_iterations(d$x, d$y, max_iterations = 0, n_splits = 2, ntree = 40,
                              seed = 0)
  expect_identical(zero$n_iterations, 0L)
  expect_identical(dim(zero$mspe), c(2L, 1L))
  expect_error(select_n_iterations(d$x, d$y, max_iterations = -1), "non-negative integer")
})

test_that("methods beat the uncorrected forest on Friedman 1", {
  mse <- 0
  for (k in 1:3) {
    train <- simulate_data("f1", 200, seed = 100 + k)
    test <- simulate_data("f1", 1000, sigma = 0, seed = 200 + k)
    P <- predict_all_methods(train$x, train$y, test$x, ntree = 150, seed = k)
    mse <- mse + colMeans((P - test$f)^2) / 3
  }
  expect_lt(mse[["Bf"]], mse[["Un"]])
  expect_lt(mse[["De"]], mse[["Un"]])
  expect_lt(mse[["Cond"]], mse[["Un"]])
})

test_that("replication counts and test sizes are validated", {
  expect_error(simulation_study("f1", n_reps = 0), "positive integer")
  expect_error(repeated_splits(x, y, n_splits = 0), "positive integer")
  expect_error(select_n_iterations(x, y, n_splits = 0), "positive integer")
  expect_error(repeated_splits(x, y, test_size = 5), "strictly between 0 and 1")
  expect_error(select_n_iterations(x, y, test_size = 0), "strictly between 0 and 1")
})

test_that("as.data.frame methods follow the generic's arguments", {
  r <- repeated_splits(x, y, n_splits = 2, ntree = 30, seed = 1)
  expect_identical(names(data.frame(r)), c("method", "mspe", "sd"))
  expect_identical(nrow(as.data.frame(r)), 10L)
})
