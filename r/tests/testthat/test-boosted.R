x <- f1_train$x
y <- f1_train$y
xt <- f1_test$x

test_that("paper_label maps iteration counts to the paper's notation", {
  expect_identical(paper_label(0:4), c("Un", "Bf", "Bf(1)", "Bf(2)", "Bf(3)"))
  expect_error(paper_label(-1), "non-negative")
  expect_error(paper_label(1.5), "non-negative")
})

test_that("zero iterations gives the base forest", {
  base <- fit_forest(x, y, ntree = 40, seed = 4)
  b0 <- boosted_forest(x, forest = base, n_iterations = 0)
  expect_identical(b0$n_iterations, 0L)
  expect_length(b0$forests, 1)
  expect_identical(predict(b0, xt), unname(predict(base, xt)))
  expect_identical(b0$oob_prediction, oob_predictions(base, x))
  fresh <- boosted_forest(x, y, n_iterations = 0, ntree = 40, seed = 4)
  expect_identical(predict(fresh, xt), unname(predict(base, xt)))
})

test_that("a supplied forest is used as stage 0 without refitting", {
  base <- fit_forest(x, y, ntree = 30, mtry = 2, seed = 1)
  copy <- base
  bf <- boosted_forest(x, y, forest = base, n_iterations = 1, seed = 1)
  expect_identical(bf$forests[[1]], base)
  expect_identical(base, copy)
  expect_equal(bf$forests[[2]]$ntree, 30)
  expect_equal(bf$forests[[2]]$mtry, 2)
  bf <- boosted_forest(x, forest = base, n_iterations = 1, ntree = 25, mtry = 1, seed = 1)
  expect_equal(bf$forests[[2]]$ntree, 25)
  expect_equal(bf$forests[[2]]$mtry, 1)
})

test_that("each stage is fitted to the OOB residual of the previous stages", {
  b3 <- boosted_forest(x, y, n_iterations = 3, ntree = 40, seed = 2)
  expect_identical(b3$n_iterations, 3L)
  expect_length(b3$forests, 4)
  o <- lapply(b3$forests, oob_predictions, x = x)
  expect_equal(unname(b3$forests[[2]]$y), y - o[[1]])
  expect_equal(unname(b3$forests[[3]]$y), unname(b3$forests[[2]]$y) - o[[2]])
  expect_equal(unname(b3$forests[[4]]$y), unname(b3$forests[[3]]$y) - o[[3]])
  expect_equal(unname(b3$stage_oob_predictions), do.call(cbind, o))
  expect_identical(colnames(b3$stage_oob_predictions), paper_label(0:3))
  expect_equal(b3$oob_prediction, Reduce(`+`, o))
  expect_equal(b3$oob_mse_path, sapply(1:4, function(k) mean((y - Reduce(`+`, o[1:k]))^2)))
  expect_false(identical(b3$forests[[1]]$inbag, b3$forests[[2]]$inbag))
})

test_that("staged_predict gives cumulative sums of the stage predictions", {
  b3 <- boosted_forest(x, y, n_iterations = 3, ntree = 40, seed = 2)
  S <- staged_predict(b3, xt)
  expect_identical(dim(S), c(60L, 4L))
  expect_identical(colnames(S), c("Un", "Bf", "Bf(1)", "Bf(2)"))
  stage <- sapply(b3$forests, function(f) unname(predict(f, xt)))
  expect_equal(unname(S), t(apply(stage, 1, cumsum)))
  expect_identical(unname(S[, 1]), unname(predict(b3$forests[[1]], xt)))
  expect_identical(unname(S[, 4]), predict(b3, xt))
  one <- staged_predict(b3, xt[1, , drop = FALSE])
  expect_identical(dim(one), c(1L, 4L))
  expect_equal(one[1, ], S[1, ])
})

test_that("fewer iterations with the same seed reproduce the staged predictions", {
  S <- staged_predict(boosted_forest(x, y, n_iterations = 3, ntree = 40, seed = 2), xt)
  for (k in 0:2) {
    bk <- boosted_forest(x, y, n_iterations = k, ntree = 40, seed = 2)
    expect_identical(predict(bk, xt), unname(S[, k + 1]))
  }
})

test_that("boosted_forest is deterministic with a seed and restores the RNG state", {
  a <- predict(boosted_forest(x, y, n_iterations = 2, ntree = 20, seed = 7), xt)
  b <- predict(boosted_forest(x, y, n_iterations = 2, ntree = 20, seed = 7), xt)
  c <- predict(boosted_forest(x, y, n_iterations = 2, ntree = 20, seed = 8), xt)
  expect_identical(a, b)
  expect_false(isTRUE(all.equal(a, c)))
  set.seed(7)
  d <- predict(boosted_forest(x, y, n_iterations = 2, ntree = 20), xt)
  expect_identical(d, a)
  expect_rng_preserved(boosted_forest(x, y, n_iterations = 1, ntree = 20, seed = 5))
})

test_that("OOB stopping on pure noise keeps the argmin of a prefix of the fixed path", {
  d <- noise_data(100, 3, seed = 0)
  for (s in 1:3) {
    fixed <- boosted_forest(d$x, d$y, n_iterations = 6, ntree = 40, seed = s)
    path <- fixed$oob_mse_path
    expect_length(path, 7)
    m <- boosted_forest(d$x, d$y, n_iterations = 6, stopping = "oob", ntree = 40, seed = s)
    p <- m$oob_mse_path
    expect_identical(m$stopping, "oob")
    expect_lte(length(p), 7)
    expect_identical(p, path[seq_along(p)])
    expect_identical(m$n_iterations, which.min(p) - 1L)
    expect_length(m$forests, m$n_iterations + 1)
    expect_identical(ncol(m$stage_oob_predictions), m$n_iterations + 1L)
    expect_identical(ncol(staged_predict(m, d$x)), m$n_iterations + 1L)
    if (length(p) < 7) expect_gt(p[length(p)], 1.1 * min(p[-length(p)]))
    if (length(p) > 2) {
      for (j in 2:(length(p) - 1)) expect_lte(p[j], 1.1 * min(p[seq_len(j - 1)]))
    }
    expect_equal(mean((d$y - m$oob_prediction)^2), p[m$n_iterations + 1])
    expect_lte(m$n_iterations, 1)
  }
})

test_that("OOB stopping on Friedman 1 keeps at least one residual forest", {
  for (s in 1:3) {
    m <- boosted_forest(x, y, n_iterations = 4, stopping = "oob", ntree = 40, seed = s)
    expect_gte(m$n_iterations, 1)
    expect_identical(m$oob_mse_path[m$n_iterations + 1], min(m$oob_mse_path))
  }
})

test_that("boosted_forest validates its inputs", {
  base <- fit_forest(x, y, ntree = 20, seed = 1)
  expect_error(boosted_forest(x, y, stopping = "bogus", ntree = 5))
  expect_error(boosted_forest(x, y, n_iterations = -1, ntree = 5), "non-negative integer")
  expect_error(boosted_forest(x, y, n_iterations = 1.5, ntree = 5), "non-negative integer")
  no_inbag <- randomForest::randomForest(x, y, ntree = 10)
  expect_error(boosted_forest(x, y, forest = no_inbag), "keep.inbag")
  no_forest <- randomForest::randomForest(x, y, ntree = 10, keep.inbag = TRUE,
                                          keep.forest = FALSE)
  expect_error(boosted_forest(x, y, forest = no_forest), "keep.forest")
  expect_error(boosted_forest(x[1:50, ], y[1:50], forest = base), "fitted on 120 rows")
  expect_error(boosted_forest(x[, 1:4], y, forest = base), "number of predictors")
  expect_error(boosted_forest(x, y + 1, forest = base), "not the response")
  expect_error(boosted_forest(x), "`y` is required")
  cls <- randomForest::randomForest(x, factor(y > 15), ntree = 10, keep.inbag = TRUE)
  expect_error(boosted_forest(x, forest = cls), "regression randomForest")
  expect_error(boosted_forest(x, forest = list()), "regression randomForest")
})

test_that("a forest that is not a list gets the same validation error", {
  expect_error(smoothed_forest(x, forest = "not a forest"), "regression randomForest")
  expect_error(conditional_bias_forest(x, forest = "not a forest"), "regression randomForest")
  expect_error(boosted_forest(x, forest = "not a forest"), "regression randomForest")
})

test_that("predict without newdata returns the bias-corrected OOB prediction", {
  base <- fit_forest(x, y, ntree = 30, seed = 1)
  bf <- boosted_forest(x, forest = base, n_iterations = 2, seed = 1)
  expect_identical(predict(bf), bf$oob_prediction)
  expect_equal(bf$oob_prediction, rowSums(sapply(bf$forests, oob_predictions, x = x)))
})

test_that("print summarises the stages", {
  bf <- boosted_forest(x, y, n_iterations = 2, ntree = 20, seed = 1)
  expect_output(print(bf), "Bf\\(1\\)")
  expect_output(print(bf), "2 residual forests")
  expect_output(print(boosted_forest(x, y, n_iterations = 1, ntree = 20, seed = 1)),
                "1 residual forest after")
  expect_output(expect_invisible(print(bf)), "OOB MSE by stage")
})

test_that("the boosted forest reduces the bias of the forest on Friedman 1", {
  train <- simulate_data("f1", 200, seed = 21)
  test <- simulate_data("f1", 1000, sigma = 0, seed = 22)
  un <- fit_forest(train$x, train$y, ntree = 150, seed = 0)
  bf <- boosted_forest(train$x, forest = un, n_iterations = 1, seed = 0)
  mse_un <- mean((predict(un, test$x) - test$f)^2)
  mse_bf <- mean((predict(bf, test$x) - test$f)^2)
  expect_lt(mse_bf, mse_un)
})

test_that("residual stages reuse the nodesize of a forest from fit_forest", {
  base <- fit_forest(x, y, ntree = 30, nodesize = 20, seed = 1)
  expect_identical(base$nodesize, 20)
  bf <- boosted_forest(x, forest = base, seed = 2)
  ref <- {
    set.seed(2)
    fit_forest(x, y - oob_predictions(base, x), ntree = 30, mtry = base$mtry, nodesize = 20)
  }
  expect_identical(bf$forests[[2]]$forest$ndbigtree, ref$forest$ndbigtree)
  small <- boosted_forest(x, forest = base, nodesize = 5, seed = 2)
  expect_gt(sum(small$forests[[2]]$forest$ndbigtree), sum(bf$forests[[2]]$forest$ndbigtree))
})
