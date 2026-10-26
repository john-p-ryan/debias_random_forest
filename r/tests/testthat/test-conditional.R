x <- f1_train$x[1:60, ]
y <- f1_train$y[1:60]
xt <- f1_test$x[1:12, ]
base <- fit_forest(x, y, ntree = 25, seed = 1)
cond <- conditional_bias_forest(x, forest = base)

test_that("bias_estimate matches the paper's weight formula", {
  ref <- brute_force_bias(base, x, y, xt)
  expect_equal(bias_estimate(cond, xt), ref$bias, tolerance = 1e-10)
  expect_equal(predict(cond, xt), unname(predict(base, xt)) - ref$bias, tolerance = 1e-10)
  expect_equal(cond$base_oob_prediction, ref$oob, tolerance = 1e-10)
  expect_identical(cond$forest, base)
})

test_that("the training OOB prediction excludes each row's own weight", {
  ref <- brute_force_bias(base, x, y, x, exclude_self = seq_len(60))
  expect_equal(cond$oob_prediction, ref$oob - ref$bias, tolerance = 1e-10)
  expect_identical(predict(cond), cond$oob_prediction)
})

test_that("leaf sums and counts aggregate the OOB biases per tree", {
  C <- cond$leaf_oob_count
  S <- cond$leaf_bias_sum
  expect_identical(dim(C), c(as.integer(base$forest$nrnodes), 25L))
  expect_identical(dim(S), dim(C))
  expect_true(all(C >= 0))
  expect_equal(colSums(C), colSums(base$inbag == 0))
  bias <- cond$base_oob_prediction - y
  expect_equal(colSums(S), colSums((base$inbag == 0) * bias), tolerance = 1e-10)
})

test_that("far-away points get finite predictions", {
  far <- matrix(1e6, 1, ncol(x))
  expect_true(all(is.finite(predict(cond, far))))
  expect_true(all(is.finite(bias_estimate(cond, -far))))
})

test_that("the bias is 0 where no OOB training row shares a leaf", {
  empty <- cond
  empty$leaf_bias_sum[] <- 0
  empty$leaf_oob_count[] <- 0
  expect_identical(bias_estimate(empty, xt), rep(0, 12))
  expect_equal(predict(empty, xt), unname(predict(base, xt)))
})

test_that("fitting the base forest internally matches a supplied forest", {
  a <- conditional_bias_forest(x, y, ntree = 20, seed = 2)
  b <- conditional_bias_forest(x, y, ntree = 20, seed = 2)
  c <- conditional_bias_forest(x, forest = fit_forest(x, y, ntree = 20, seed = 2))
  expect_identical(predict(a, xt), predict(b, xt))
  expect_identical(predict(a, xt), predict(c, xt))
  expect_rng_preserved(conditional_bias_forest(x, y, ntree = 20, seed = 1))
})

test_that("conditional_bias_forest validates its inputs and accepts data frames", {
  expect_error(conditional_bias_forest(x[1:30, ], y[1:30], forest = base), "fitted on 60 rows")
  expect_error(conditional_bias_forest(x), "`y` is required")
  expect_error(bias_estimate(list(), xt))
  df <- conditional_bias_forest(as.data.frame(x), y, ntree = 20, seed = 2)
  m <- conditional_bias_forest(x, y, ntree = 20, seed = 2)
  expect_equal(predict(df, as.data.frame(xt)), predict(m, xt))
  expect_output(print(cond), "Cond")
})

test_that("two trees (a two-column id matrix) still index the leaf sums", {
  two <- fit_forest(x, y, ntree = 2, seed = 3)
  m <- suppressWarnings(conditional_bias_forest(x, forest = two))
  ref <- brute_force_bias(two, x, y, xt)
  expect_equal(bias_estimate(m, xt), ref$bias, tolerance = 1e-10)
  expect_equal(predict(m, xt), unname(predict(two, xt)) - ref$bias, tolerance = 1e-10)
})
