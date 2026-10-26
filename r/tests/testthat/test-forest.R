test_that("default_mtry is max(floor(p / 3), 1)", {
  for (p in c(1, 2, 4, 5, 10)) {
    expected <- max(floor(p / 3), 1)
    expect_identical(default_mtry(p), as.integer(expected))
    expect_identical(default_mtry(matrix(0, 3, p)), as.integer(expected))
    expect_identical(default_mtry(as.data.frame(matrix(0, 3, p))), as.integer(expected))
  }
})

test_that("fit_forest uses the paper's defaults and keeps the bootstrap samples", {
  expect_identical(formals(fit_forest)$ntree, 500)
  expect_identical(formals(fit_forest)$nodesize, 5)
  set.seed(0)
  for (p in c(1, 4, 5, 10)) {
    x <- matrix(runif(40 * p), 40, p)
    rf <- fit_forest(x, rowSums(x), ntree = 3, seed = 1)
    expect_identical(rf$mtry, max(floor(p / 3), 1))
    expect_identical(dim(rf$inbag), c(40L, 3L))
    expect_false(is.null(rf$forest))
    expect_identical(rf$type, "regression")
  }
  rf <- fit_forest(f1_train$x, f1_train$y, seed = 1)
  expect_identical(rf$ntree, 500)
  expect_identical(dim(rf$inbag), c(120L, 500L))
})

test_that("inbag counts are bootstrap draws of size n", {
  rf <- fit_forest(f1_train$x, f1_train$y, ntree = 30, seed = 2)
  expect_true(all(colSums(rf$inbag) == 120))
  expect_gt(max(rf$inbag), 1)
  expect_gt(mean(rf$inbag == 0), 0.3)
  expect_lt(mean(rf$inbag == 0), 0.45)
  expect_identical(unname(rf$oob.times), as.integer(rowSums(rf$inbag == 0)))
})

test_that("fit_forest forwards further arguments to randomForest", {
  rf <- fit_forest(f1_train$x, f1_train$y, ntree = 10, replace = FALSE, sampsize = 50,
                   seed = 1)
  expect_true(all(colSums(rf$inbag) == 50))
  expect_equal(max(rf$inbag), 1)
  rf <- fit_forest(f1_train$x, f1_train$y, ntree = 5, mtry = 4, seed = 1)
  expect_identical(rf$mtry, 4)
})

test_that("oob_predictions equals the brute-force OOB average", {
  rf <- fit_forest(f1_train$x, f1_train$y, ntree = 50, seed = 1)
  oob <- oob_predictions(rf, f1_train$x)
  expect_length(oob, 120)
  expect_null(names(oob))
  expect_equal(oob, brute_force_oob(rf, f1_train$x), tolerance = 1e-12)
  expect_equal(oob_predictions(rf), oob)
})

test_that("rows never out-of-bag fall back to the full-forest prediction", {
  rf <- fit_forest(f1_train$x, f1_train$y, ntree = 2, seed = 1)
  never <- rf$oob.times == 0
  expect_true(any(never))
  expect_true(any(!never))
  expect_warning(oob <- oob_predictions(rf, f1_train$x), "never out-of-bag")
  expect_true(all(is.finite(oob)))
  expect_equal(oob[never], unname(predict(rf, f1_train$x[never, , drop = FALSE])))
  expect_equal(oob[!never], brute_force_oob(rf, f1_train$x)[!never])
  expect_error(oob_predictions(rf), "never out-of-bag")
})

test_that("oob_predictions rejects unsupported forests", {
  cls <- randomForest::randomForest(f1_train$x, factor(f1_train$y > 15), ntree = 10)
  expect_error(oob_predictions(cls), "regression randomForest")
  expect_error(oob_predictions(list()), "regression randomForest")
  cb <- randomForest::randomForest(f1_train$x, f1_train$y, ntree = 10, corr.bias = TRUE)
  expect_error(oob_predictions(cb), "corr.bias")
})

test_that("fit_forest is reproducible with a seed and restores the RNG state", {
  a <- fit_forest(f1_train$x, f1_train$y, ntree = 20, seed = 42)
  b <- fit_forest(f1_train$x, f1_train$y, ntree = 20, seed = 42)
  c <- fit_forest(f1_train$x, f1_train$y, ntree = 20, seed = 43)
  expect_identical(predict(a, f1_test$x), predict(b, f1_test$x))
  expect_identical(a$inbag, b$inbag)
  expect_false(isTRUE(all.equal(predict(a, f1_test$x), predict(c, f1_test$x))))
  set.seed(42)
  d <- fit_forest(f1_train$x, f1_train$y, ntree = 20)
  expect_identical(predict(d, f1_test$x), predict(a, f1_test$x))
  expect_rng_preserved(fit_forest(f1_train$x, f1_train$y, ntree = 5, seed = 1))
})

test_that("a seed does not leave a .Random.seed behind when there was none", {
  env <- globalenv()
  old <- get0(".Random.seed", envir = env, inherits = FALSE)
  on.exit(if (!is.null(old)) assign(".Random.seed", old, envir = env))
  if (!is.null(old)) rm(".Random.seed", envir = env)
  fit_forest(f1_train$x, f1_train$y, ntree = 5, seed = 1)
  expect_false(exists(".Random.seed", envir = env, inherits = FALSE))
})

test_that("fit_forest validates its inputs and accepts data frames", {
  expect_error(fit_forest(f1_train$x[, 1], f1_train$y, ntree = 5), "matrix or data frame")
  expect_error(fit_forest(f1_train$x, f1_train$y[-1], ntree = 5), "120 rows but `y` has 119")
  expect_error(fit_forest(f1_train$x, as.character(f1_train$y), ntree = 5), "numeric vector")
  expect_error(fit_forest(matrix("a", 4, 2), 1:4, ntree = 5), "numeric")
  df <- as.data.frame(f1_train$x)
  a <- fit_forest(df, f1_train$y, ntree = 20, seed = 3)
  b <- fit_forest(f1_train$x, f1_train$y, ntree = 20, seed = 3)
  expect_equal(oob_predictions(a, df), oob_predictions(b, f1_train$x))
})

test_that("combined forests get their OOB predictions recomputed from inbag", {
  d <- simulate_data("f1", 80, seed = 5)
  a <- fit_forest(d$x, d$y, ntree = 30, seed = 1)
  b <- fit_forest(d$x, d$y, ntree = 30, seed = 2)
  ab <- randomForest::combine(a, b)
  expect_false(identical(as.integer(ab$oob.times), as.integer(rowSums(ab$inbag == 0))))
  expect_equal(oob_predictions(ab, d$x), brute_force_oob(ab, d$x), tolerance = 1e-12)
  expect_error(oob_predictions(ab), "out of date")
  sm <- smoothed_forest(d$x, method = "linear", forest = ab)
  expect_equal(sm$base_oob_prediction, brute_force_oob(ab, d$x), tolerance = 1e-12)
})

test_that("a supplied forest must come with its own training rows, in order", {
  d <- simulate_data("f1", 80, seed = 5)
  rf <- fit_forest(d$x, d$y, ntree = 30, seed = 1)
  perm <- rev(seq_len(80))
  expect_error(boosted_forest(d$x[perm, ], forest = rf), "not the training data")
  expect_error(conditional_bias_forest(d$x[perm, ], forest = rf), "not the training data")
  expect_s3_class(boosted_forest(d$x, forest = rf), "boosted_forest")
})
