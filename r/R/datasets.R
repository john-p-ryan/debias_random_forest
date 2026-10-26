# Simulation designs of Chen, Withana Gamage & Ryan (2022), pp. 122-123, and the UCI
# Yacht Hydrodynamics data.

# name -> number of predictors, default sigma, regression function of the n x p matrix
sim_designs <- list(
  fig1 = list(p = 4L, sigma = 0.5, f = function(x)
    3 * x[, 1] + x[, 2]^2 + sin(x[, 3]) - sqrt(x[, 4])),
  # Function 1 (Friedman 1)
  f1 = list(p = 5L, sigma = 1, f = function(x)
    10 * sin(pi * x[, 1] * x[, 2]) + 20 * (x[, 3] - 0.5)^2 + 10 * x[, 4] + 5 * x[, 5]),
  # Function 2 as printed in the paper (p. 123): 3 x2 (Friedman's original has 3 x3);
  # x6..x10 are noise variables
  f2 = list(p = 10L, sigma = 1, f = function(x)
    0.1 * exp(4 * x[, 1]) + 4 / (1 + exp(-20 * x[, 2] + 10)) + 3 * x[, 2] + 2 * x[, 4] +
      x[, 5]),
  # Function 3: linear, coefficients as printed (drawn once from U(0.5, 1));
  # x6..x10 are noise variables
  f3 = list(p = 10L, sigma = 1, f = function(x)
    10 * drop(x[, 1:5] %*% c(0.878, 0.642, 0.766, 0.666, 0.933))),
  # Function 4: polynomial; x5 is a noise variable
  f4 = list(p = 5L, sigma = 1, f = function(x)
    10 * x[, 1] + 0.5 * x[, 2]^2 + 15 * x[, 2] - 25 * sqrt(x[, 3]) + 0.05 * x[, 4]),
  # Function 5: non-linear
  f5 = list(p = 5L, sigma = 1, f = function(x)
    10 * cos(pi * x[, 1] * x[, 3]) + 10 * x[, 3]^2 + 10 * (x[, 1] + x[, 4]))
)

sim_aliases <- c(friedman1 = "f1", function1 = "f1", function2 = "f2", function3 = "f3",
                 function4 = "f4", function5 = "f5", figure1 = "fig1")

sim_design <- function(name) {
  if (length(name) != 1L) stop("`name` must be a single design name", call. = FALSE)
  key <- gsub(" ", "", tolower(as.character(name)), fixed = TRUE)
  if (key %in% names(sim_aliases)) key <- sim_aliases[[key]]
  if (!key %in% names(sim_designs))
    stop("unknown simulation design '", name, "'; choose from ",
         paste(simulation_names(), collapse = ", "), " (or 'friedman1')", call. = FALSE)
  sim_designs[[key]]
}

#' Simulation designs of the paper
#'
#' `simulate_data()` draws `n` rows with i.i.d. U(0, 1) predictors and
#' `y = f(x) + N(0, sigma^2)` noise from one of the paper's designs: `"fig1"`
#' (4 predictors, sigma 0.5), `"f1"` / `"friedman1"` (Friedman 1, 5 predictors),
#' `"f2"` (10), `"f3"` (linear, 10), `"f4"` (5) or `"f5"` (5); sigma defaults to 1
#' except for `"fig1"`.  Function 2 uses the `3 x2` term as printed in the paper.
#'
#' @param name design name (see [simulation_names()]).
#' @param n number of rows.
#' @param sigma noise standard deviation (default: the design's).
#' @param seed optional integer seed (see [fit_forest()]).
#' @return `simulate_data()`: a list with `x` (n x p matrix, columns `x1`, ...),
#'   `y` and `f` (the true regression function at `x`).
#' @examples
#' d <- simulate_data("friedman1", 5, seed = 1)
#' all.equal(d$f, true_mean("f1", d$x))
#' simulation_names()
#' @export
simulate_data <- function(name, n, sigma = NULL, seed = NULL) {
  design <- sim_design(name)
  n <- check_count(n, "n")
  if (is.null(sigma)) sigma <- design$sigma
  with_seed(seed, {
    x <- matrix(stats::runif(n * design$p), n, design$p,
                dimnames = list(NULL, paste0("x", seq_len(design$p))))
    f <- unname(design$f(x))
    list(x = x, y = f + sigma * stats::rnorm(n), f = f)
  })
}

#' @rdname simulate_data
#' @param x predictor matrix with the design's number of columns.
#' @return `true_mean()`: `f(x)` for every row of `x`.
#' @export
true_mean <- function(name, x) {
  design <- sim_design(name)
  x <- as.matrix(x)
  if (ncol(x) != design$p)
    stop("design '", name, "' has ", design$p, " predictors", call. = FALSE)
  unname(design$f(x))
}

#' @rdname simulate_data
#' @return `simulation_names()`: the design names.
#' @export
simulation_names <- function() names(sim_designs)

#' @rdname simulate_data
#' @return `simulation_sigma()`: the default noise standard deviation of a design.
#' @export
simulation_sigma <- function(name) sim_design(name)$sigma

yacht_urls <- c(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
  "https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/master/UCI_Datasets/yacht/data/data.txt"
)

yacht_names <- c("longitudinal_position", "prismatic_coefficient",
                 "length_displacement_ratio", "beam_draught_ratio", "length_beam_ratio",
                 "froude_number", "residuary_resistance")

read_yacht <- function(path) {
  d <- utils::read.table(path, header = FALSE, col.names = yacht_names,
                         colClasses = "numeric", strip.white = TRUE, blank.lines.skip = TRUE)
  if (!identical(dim(d), c(308L, 7L)))
    stop("unexpected yacht data shape ", paste(dim(d), collapse = " x "), call. = FALSE)
  d
}

#' UCI Yacht Hydrodynamics data
#'
#' Downloads the data (308 rows, 6 hull predictors and the residuary resistance)
#' from the UCI repository, falling back to a GitHub mirror, and caches the file in
#' `cache_dir`.  When the file is neither cached nor downloadable (for example
#' offline) an error of class `debiasforest_dataset_unavailable` is signalled, so
#' callers can skip gracefully with `tryCatch()`.
#'
#' @param cache_dir cache directory; default `$DEBIASFOREST_DATA` or
#'   `tools::R_user_dir("debiasforest", "cache")`.
#' @param urls download locations, tried in order.
#' @param timeout download timeout in seconds.
#' @return A list with `x` (308 x 6 matrix) and `y` (residuary resistance).
#' @examples
#' \dontrun{
#' yacht <- tryCatch(load_yacht(), debiasforest_dataset_unavailable = function(e) NULL)
#' }
#' @export
load_yacht <- function(cache_dir = NULL, urls = yacht_urls, timeout = 20) {
  if (is.null(cache_dir)) {
    cache_dir <- Sys.getenv("DEBIASFOREST_DATA", unset = "")
    if (!nzchar(cache_dir)) cache_dir <- tools::R_user_dir("debiasforest", "cache")
  }
  path <- file.path(cache_dir, "yacht_hydrodynamics.data")
  d <- if (file.exists(path)) tryCatch(read_yacht(path), error = function(e) NULL)
  errors <- character()
  if (is.null(d)) {
    op <- options(timeout = timeout)
    on.exit(options(op), add = TRUE)
    tmp <- tempfile(fileext = ".data")
    on.exit(unlink(tmp), add = TRUE)
    for (url in urls) {
      d <- tryCatch({
        utils::download.file(url, tmp, quiet = TRUE, mode = "wb")
        read_yacht(tmp)
      }, error = function(e) {
        errors <<- c(errors, paste0(url, ": ", conditionMessage(e)))
        NULL
      }, warning = function(w) {
        errors <<- c(errors, paste0(url, ": ", conditionMessage(w)))
        NULL
      })
      if (!is.null(d)) {
        dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
        file.copy(tmp, path, overwrite = TRUE)
        break
      }
    }
  }
  if (is.null(d))
    stop(errorCondition(paste("could not obtain the yacht data set:",
                              paste(errors, collapse = "; ")),
                        class = "debiasforest_dataset_unavailable"))
  list(x = as.matrix(d[, 1:6]), y = d[, 7])
}
