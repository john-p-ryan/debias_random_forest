formulas <- list(
  fig1 = function(x) 3 * x[, 1] + x[, 2]^2 + sin(x[, 3]) - sqrt(x[, 4]),
  f1 = function(x) 10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 0.5)^2 + 10 * x[, 4] +
    5 * x[, 5],
  f2 = function(x) 0.1 * exp(4 * x[, 1]) + 4 / (1 + exp(-20 * x[, 2] + 10)) + 3 * x[, 2] +
    2 * x[, 4] + x[, 5],
  f3 = function(x) 10 * (0.878 * x[, 1] + 0.642 * x[, 2] + 0.766 * x[, 3] + 0.666 * x[, 4] +
                           0.933 * x[, 5]),
  f4 = function(x) 10 * x[, 1] + 0.5 * x[, 2]^2 + 15 * x[, 2] - 25 * sqrt(x[, 3]) +
    0.05 * x[, 4],
  f5 = function(x) 10 * cos(pi * x[, 1] * x[, 3]) + 10 * x[, 3]^2 + 10 * (x[, 1] + x[, 4])
)
n_predictors <- c(fig1 = 4L, f1 = 5L, f2 = 10L, f3 = 10L, f4 = 5L, f5 = 5L)

test_that("every design has the right shape, ranges and regression function", {
  expect_identical(simulation_names(), names(n_predictors))
  for (name in simulation_names()) {
    p <- n_predictors[[name]]
    d <- simulate_data(name, 50, seed = 1)
    expect_identical(dim(d$x), c(50L, p))
    expect_identical(colnames(d$x), paste0("x", seq_len(p)))
    expect_length(d$y, 50)
    expect_length(d$f, 50)
    expect_true(all(d$x >= 0 & d$x <= 1))
    expect_equal(d$f, formulas[[name]](d$x))
    expect_equal(true_mean(name, d$x), d$f)
  }
})

test_that("design names are case- and space-insensitive aliases", {
  aliases <- c(friedman1 = "f1", `Friedman 1` = "f1", F1 = "f1", function1 = "f1",
               function2 = "f2", function3 = "f3", function4 = "f4", function5 = "f5",
               figure1 = "fig1", FIG1 = "fig1")
  for (alias in names(aliases)) {
    expect_identical(simulate_data(alias, 10, seed = 3),
                     simulate_data(aliases[[alias]], 10, seed = 3))
  }
  expect_equal(true_mean("friedman1", f1_train$x), f1_train$f)
})

test_that("regression functions take the documented values", {
  half <- matrix(0.5, 1, 5)
  expect_equal(true_mean("f1", half), 10 * sin(pi / 4) + 5 + 2.5)
  expect_equal(true_mean("f5", half), 10 * cos(pi / 4) + 2.5 + 10)
  expect_equal(true_mean("fig1", matrix(0.5, 1, 4)), 1.5 + 0.25 + sin(0.5) - sqrt(0.5))
  expect_equal(true_mean("f3", matrix(1, 1, 10)), 10 * (0.878 + 0.642 + 0.766 + 0.666 + 0.933))
})

test_that("noise has the design's standard deviation", {
  expect_identical(vapply(simulation_names(), simulation_sigma, numeric(1)),
                   c(fig1 = 0.5, f1 = 1, f2 = 1, f3 = 1, f4 = 1, f5 = 1))
  d <- simulate_data("fig1", 20000, seed = 2)
  expect_equal(sd(d$y - d$f), 0.5, tolerance = 0.03)
  d <- simulate_data("f1", 20000, seed = 2)
  expect_equal(sd(d$y - d$f), 1, tolerance = 0.03)
  d <- simulate_data("f1", 20000, sigma = 4, seed = 2)
  expect_equal(sd(d$y - d$f), 4, tolerance = 0.03)
  d <- simulate_data("friedman1", 10, sigma = 0, seed = 1)
  expect_identical(d$y, d$f)
})

test_that("simulate_data is reproducible with a seed", {
  expect_identical(simulate_data("f2", 10, seed = 7), simulate_data("f2", 10, seed = 7))
  expect_false(identical(simulate_data("f2", 10, seed = 7)$y,
                         simulate_data("f2", 10, seed = 8)$y))
  set.seed(7)
  expect_identical(simulate_data("f2", 10), simulate_data("f2", 10, seed = 7))
  expect_rng_preserved(simulate_data("f1", 10, seed = 1))
})

test_that("unknown designs and wrong inputs are errors", {
  expect_error(simulate_data("nope", 3), "unknown simulation design 'nope'")
  expect_error(simulation_sigma("nope"), "unknown simulation design")
  expect_error(simulate_data(c("f1", "f2"), 3), "single design name")
  expect_error(simulate_data("f1", -1), "non-negative integer")
  expect_error(true_mean("f1", matrix(0, 2, 4)), "5 predictors")
})

write_yacht <- function(path, values) {
  lines <- apply(values, 1, function(r) paste(formatC(r, format = "f", digits = 4),
                                              collapse = "  "))
  writeLines(lines, path)
}

yacht_values <- function() {
  set.seed(0)
  matrix(round(runif(308 * 7), 4), 308, 7)
}

no_urls <- c("file:///nonexistent/a", "file:///nonexistent/b")

test_that("load_yacht signals a classed error when the data are unavailable", {
  dir <- tempfile("yacht")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE))
  expect_error(load_yacht(cache_dir = dir, urls = no_urls),
               class = "debiasforest_dataset_unavailable")
  expect_error(load_yacht(cache_dir = dir, urls = no_urls), "nonexistent/b")
  expect_length(list.files(dir), 0)
  writeLines(c("1 2 3", "4 5 6"), file.path(dir, "yacht_hydrodynamics.data"))
  expect_error(load_yacht(cache_dir = dir, urls = no_urls),
               class = "debiasforest_dataset_unavailable")
})

test_that("load_yacht reads a cached file", {
  dir <- tempfile("yacht")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE))
  values <- yacht_values()
  write_yacht(file.path(dir, "yacht_hydrodynamics.data"), values)
  d <- load_yacht(cache_dir = dir, urls = no_urls)
  expect_true(is.matrix(d$x))
  expect_identical(dim(d$x), c(308L, 6L))
  expect_length(d$y, 308)
  expect_equal(unname(d$x), values[, 1:6])
  expect_equal(d$y, values[, 7])
  expect_identical(colnames(d$x)[6], "froude_number")
})

test_that("load_yacht falls back through the URLs and caches the download", {
  src <- tempfile(fileext = ".data")
  dir <- file.path(tempfile("yacht"), "nested")
  on.exit(unlink(c(src, dirname(dir)), recursive = TRUE))
  values <- yacht_values()
  write_yacht(src, values)
  d <- load_yacht(cache_dir = dir, urls = c(no_urls[1], paste0("file://", src)))
  expect_equal(d$y, values[, 7])
  expect_true(file.exists(file.path(dir, "yacht_hydrodynamics.data")))
  expect_equal(load_yacht(cache_dir = dir, urls = no_urls)$y, values[, 7])
})

test_that("load_yacht uses $DEBIASFOREST_DATA as the default cache", {
  dir <- tempfile("yacht")
  dir.create(dir)
  old <- Sys.getenv("DEBIASFOREST_DATA", unset = NA)
  on.exit({
    if (is.na(old)) Sys.unsetenv("DEBIASFOREST_DATA") else Sys.setenv(DEBIASFOREST_DATA = old)
    unlink(dir, recursive = TRUE)
  })
  Sys.setenv(DEBIASFOREST_DATA = dir)
  values <- yacht_values()
  write_yacht(file.path(dir, "yacht_hydrodynamics.data"), values)
  expect_equal(load_yacht(urls = no_urls)$y, values[, 7])
})
