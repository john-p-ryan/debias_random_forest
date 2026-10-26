x <- f1_train$x
y <- f1_train$y
xt <- f1_test$x
base <- fit_forest(x, y, ntree = 50, seed = 1)
u <- oob_predictions(base, x)
ut <- unname(predict(base, xt))

test_that("polynomial methods equal least squares of y on powers of the OOB prediction", {
  for (method in c("linear", "quadratic", "cubic")) {
    degree <- c(linear = 1, quadratic = 2, cubic = 3)[[method]]
    s <- smoothed_forest(x, method = method, forest = base)
    expect_identical(s$method, method)
    expect_identical(s$forest, base)
    expect_identical(s$base_oob_prediction, u)
    fit <- lm(y ~ poly(u, degree, raw = TRUE), data = data.frame(y = y, u = u))
    expect_equal(s$oob_prediction, unname(fitted(fit)), tolerance = 1e-10)
    expect_equal(predict(s, xt), unname(predict(fit, data.frame(u = ut))), tolerance = 1e-10)
  }
  s <- smoothed_forest(x, y, method = "linear", forest = base)
  slope <- cov(u, y) / var(u)
  expect_equal(predict(s, xt), mean(y) + slope * (ut - mean(u)), tolerance = 1e-10)
})

test_that("the spline method is smooth.spline of y on the OOB prediction", {
  s <- smoothed_forest(x, method = "spline", forest = base)
  expect_s3_class(s$smoother, "smooth.spline")
  ref <- smooth.spline(u, y)
  expect_equal(s$smoother$lambda, ref$lambda)
  expect_equal(s$oob_prediction, predict(ref, u)$y)
  expect_equal(predict(s, xt), predict(ref, ut)$y)
})

test_that("the spline extrapolates linearly beyond the range of the OOB predictions", {
  s <- smoothed_forest(x, method = "spline", forest = base)
  p <- unname(predict(base, x))  # in-bag predictions reach beyond the OOB range
  lo <- min(u)
  hi <- max(u)
  expect_true(any(p > hi))
  expect_true(any(p < lo))
  g <- function(v, deriv = 0) predict(s$smoother, v, deriv = deriv)$y
  above <- p > hi
  below <- p < lo
  expect_equal(predict(s, x)[above], g(hi) + g(hi, 1) * (p[above] - hi), tolerance = 1e-8)
  expect_equal(predict(s, x)[below], g(lo) + g(lo, 1) * (p[below] - lo), tolerance = 1e-8)
})

test_that("predict without newdata returns g(OOB prediction)", {
  for (method in c("linear", "quadratic", "cubic", "spline")) {
    s <- smoothed_forest(x, method = method, forest = base)
    expect_identical(predict(s), s$oob_prediction)
    expect_length(s$oob_prediction, 120)
  }
})

test_that("fitting the base forest internally matches a supplied forest", {
  a <- smoothed_forest(x, y, method = "quadratic", ntree = 30, seed = 3)
  b <- smoothed_forest(x, y, method = "quadratic", ntree = 30, seed = 3)
  c <- smoothed_forest(x, method = "quadratic",
                       forest = fit_forest(x, y, ntree = 30, seed = 3))
  expect_identical(predict(a, xt), predict(b, xt))
  expect_identical(predict(a, xt), predict(c, xt))
  expect_identical(a$forest$ntree, 30)
  expect_rng_preserved(smoothed_forest(x, y, ntree = 20, seed = 1))
})

test_that("smoothed_forest validates its inputs", {
  expect_error(smoothed_forest(x, method = "bogus", forest = base))
  expect_error(smoothed_forest(x[1:50, ], y[1:50], forest = base), "fitted on 120 rows")
  expect_error(smoothed_forest(x, y + 1, forest = base), "not the response")
  expect_error(smoothed_forest(x), "`y` is required")
})

test_that("degenerate OOB predictions give a finite polynomial fit", {
  x2 <- matrix(rep(c(0, 1), 25), 50, 1)
  y2 <- rep(c(0, 1), 25)
  s <- suppressWarnings(smoothed_forest(x2, y2, method = "cubic", ntree = 30, seed = 1))
  expect_true(all(is.finite(s$smoother$coef)))
  expect_equal(predict(s, matrix(c(0, 1), 2, 1)), c(0, 1))
  g <- fit_polynomial(rep(2, 10), 1:10, 3)
  expect_identical(g$scale, 1)
  expect_equal(apply_smoother(g, c(2, 5)), c(5.5, 5.5))
})

test_that("data frame predictors are accepted", {
  df <- as.data.frame(x)
  s <- smoothed_forest(df, y, method = "spline", ntree = 30, seed = 2)
  m <- smoothed_forest(x, y, method = "spline", ntree = 30, seed = 2)
  expect_equal(predict(s, as.data.frame(xt)), predict(m, xt))
})

test_that("print names the paper's method", {
  labels <- c(linear = "De", quadratic = "Qu", cubic = "Cu", spline = "Sp")
  for (method in names(labels)) {
    s <- smoothed_forest(x, method = method, forest = base)
    expect_output(print(s), paste0("(", labels[[method]], ")"), fixed = TRUE)
  }
  expect_output(print(smoothed_forest(x, method = "spline", forest = base)), "lambda")
})
