# Generate reference fixtures for the native R-compatible smooth.spline ports.
#
# Usage (from the repository root):
#   Rscript dev/spline_reference.R
#
# Writes three CSV files into both python/tests/data and julia/test/data:
#   smooth_spline_data.csv  case, x, y            (training data)
#   smooth_spline_pred.csv  case, x, pred         (predictions incl. extrapolation)
#   smooth_spline_meta.csv  case, n, nx, nknots, spar, lambda, df, crit
#
# Only base R (stats::smooth.spline with default GCV) is used.  The fixtures
# are committed, so R is not needed when running either test suite.

set.seed(20221019)

make_case <- function(name, x, y) list(name = name, x = x, y = y)

cases <- list()

# small n (< 50): every unique x is a knot
x <- sort(runif(30))
cases[[length(cases) + 1]] <- make_case("small_sine", x, sin(2 * pi * x) + rnorm(30, sd = 0.2))

# moderate n (knot subset), unsorted input
x <- runif(120, -2, 3)
cases[[length(cases) + 1]] <- make_case("moderate_cubic", x, 0.5 * x^3 - x + rnorm(120, sd = 0.8))

# large n, noisy, nearly linear (typical OOB prediction vs response)
yhat <- runif(300, 5, 25)
cases[[length(cases) + 1]] <- make_case("oob_like", yhat, 1.3 * (yhat - 15) + 15 + 0.02 * (yhat - 15)^2 + rnorm(300, sd = 2))

# tied x values (aggregated by R before fitting)
x <- round(runif(150, 0, 10), 1)
cases[[length(cases) + 1]] <- make_case("ties", x, log1p(x) + rnorm(150, sd = 0.3))

# n >= 800 knot rule branch
x <- rnorm(900)
cases[[length(cases) + 1]] <- make_case("large_n", x, tanh(2 * x) + rnorm(900, sd = 0.25))

# exactly linear signal (GCV should choose a very smooth fit)
x <- runif(80, 0, 1)
cases[[length(cases) + 1]] <- make_case("linear", x, 2 + 3 * x + rnorm(80, sd = 0.1))

data_rows <- list()
pred_rows <- list()
meta_rows <- list()

for (cs in cases) {
  fit <- smooth.spline(cs$x, cs$y)
  rng <- range(cs$x)
  span <- diff(rng)
  # evaluation grid inside and outside (linear extrapolation) the data range
  xnew <- c(seq(rng[1], rng[2], length.out = 41),
            rng[1] - c(0.05, 0.25) * span,
            rng[2] + c(0.05, 0.25) * span,
            cs$x[1:5])
  pr <- predict(fit, xnew)$y
  data_rows[[cs$name]] <- data.frame(case = cs$name, x = cs$x, y = cs$y)
  pred_rows[[cs$name]] <- data.frame(case = cs$name, x = xnew, pred = pr)
  meta_rows[[cs$name]] <- data.frame(
    case = cs$name, n = length(cs$x), nx = length(fit$x),
    nknots = fit$fit$nk - 2L, spar = fit$spar, lambda = fit$lambda,
    df = fit$df, crit = fit$cv.crit
  )
}

data_df <- do.call(rbind, data_rows)
pred_df <- do.call(rbind, pred_rows)
meta_df <- do.call(rbind, meta_rows)

for (dir in c("python/tests/data", "julia/test/data")) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(data_df, file.path(dir, "smooth_spline_data.csv"), row.names = FALSE)
  write.csv(pred_df, file.path(dir, "smooth_spline_pred.csv"), row.names = FALSE)
  write.csv(meta_df, file.path(dir, "smooth_spline_meta.csv"), row.names = FALSE)
}

cat(R.version.string, "\n")
print(meta_df, digits = 10, row.names = FALSE)
