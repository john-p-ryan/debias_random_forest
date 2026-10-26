# =============================================================================
#  Demonstration of debiasforest (R)
#  Chen, Withana Gamage & Ryan (2022), "Debias Random Forest Regression
#  Predictors", J. Stat. Research 56(2):115-131.
#
#  Run (from the repo root or from demo/):
#      Rscript demo/debias_demo.R
#  Uses the installed package (R CMD INSTALL r) if there is one, otherwise
#  loads r/ with pkgload::load_all().
#
#  Full ("thorough") mode: K = 200 replications, 100 yacht splits
#                          (about 5.5 min with 8 cores, 10 min with 4).
#  Quick mode:  DEBIAS_DEMO_QUICK=1  ->  K = 10, 10 splits (smoke test, ~40 s).
#  Cores:       DEBIAS_DEMO_CORES (default 4) forked worker processes;
#               results do not depend on the number of cores.
#
#  Outputs: tables  -> demo/results/r/*.csv, *.md
#           figures -> demo/figures/r/*.png
#           data cache -> demo/data/   (git-ignored, shared with the Julia and
#                                       Python demos)
# =============================================================================

script_dir <- function() {
  arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(arg)) return(dirname(normalizePath(sub("^--file=", "", arg[1]))))
  for (i in rev(seq_len(sys.nframe()))) {            # source("demo/debias_demo.R")
    f <- sys.frame(i)$ofile
    if (!is.null(f)) return(dirname(normalizePath(f)))
  }
  if (file.exists("debias_demo.R")) getwd() else file.path(getwd(), "demo")
}
DEMO_DIR <- script_dir()
REPO_DIR <- dirname(DEMO_DIR)

# --- make sure debiasforest can be loaded -------------------------------------
if (requireNamespace("debiasforest", quietly = TRUE)) {
  library(debiasforest)
} else {
  pkgload::load_all(file.path(REPO_DIR, "r"), export_all = FALSE, helpers = FALSE,
                    quiet = TRUE)
}

# --- settings ----------------------------------------------------------------
QUICK    <- Sys.getenv("DEBIAS_DEMO_QUICK") == "1"
K_REPS   <- if (QUICK) 10 else 200      # Monte-Carlo replications (paper: 1000)
N_SPLITS <- if (QUICK) 10 else 100      # random yacht splits (paper: 1000)
N_TRAIN  <- 200
N_TEST   <- 2000
SEED     <- 20220101
MAX_ITER <- 6                           # iteration curve: k = 0..6
N_CORES  <- suppressWarnings(as.integer(Sys.getenv("DEBIAS_DEMO_CORES", "4")))
N_CORES  <- if (is.na(N_CORES)) 4L else max(N_CORES, 1L)
if (.Platform$OS.type == "windows") N_CORES <- 1L   # mclapply cannot fork there
RES_DIR  <- file.path(DEMO_DIR, "results", "r")
FIG_DIR  <- file.path(DEMO_DIR, "figures", "r")
DATA_DIR <- file.path(DEMO_DIR, "data")
for (d in c(RES_DIR, FIG_DIR, DATA_DIR)) dir.create(d, recursive = TRUE, showWarnings = FALSE)

LABELS <- method_labels()               # Un De Qu Cu Sp Bf Bf(1) Bf(2) Bf(3) Cond
UNIVARIATE <- c("De", "Qu", "Cu", "Sp")
COL <- "#0072B2"

cat("debiasforest demo   mode = ", if (QUICK) "QUICK" else "full",
    "   K = ", K_REPS, "   yacht splits = ", N_SPLITS, "   cores = ", N_CORES, "\n",
    sep = "")
now <- function() proc.time()[["elapsed"]]
T0 <- now()
timer <- function(label, t) cat(sprintf("  [%s] %.1f s\n", label, now() - t))

# --- small helpers -------------------------------------------------------------
f2 <- function(x) sprintf("%.2f", x)
as_rows <- function(rows) if (is.list(rows)) do.call(rbind, rows) else rows

# Write rows (a list of character vectors or a character matrix) as CSV (header +
# rows) and as a markdown table.
write_csv <- function(path, header, rows) {
  writeLines(c(paste(header, collapse = ","),
               apply(as_rows(rows), 1, paste, collapse = ",")), path)
}
write_md <- function(path, header, rows, title = NULL, note = NULL) {
  writeLines(c(if (!is.null(title)) c(paste("###", title), ""),
               paste("|", paste(header, collapse = " | "), "|"),
               paste0("|", paste(rep("---", length(header)), collapse = "|"), "|"),
               apply(as_rows(rows), 1, function(r) paste("|", paste(r, collapse = " | "), "|")),
               if (!is.null(note)) c("", note)), path)
}

# lapply in forked workers; every task seeds itself, so results do not depend on N_CORES.
par_map <- function(X, FUN) {
  res <- parallel::mclapply(X, FUN, mc.cores = N_CORES, mc.preschedule = FALSE)
  bad <- vapply(res, function(r) inherits(r, "try-error") || is.null(r), logical(1))
  if (any(bad)) stop("task ", which(bad)[1], " failed: ", format(res[[which(bad)[1]]]),
                     call. = FALSE)
  res
}

# --- paper reference values (Tables 1 and 2, n = 200 rows) ---------------------
# Order of columns: Un De Qu Cu Sp Bf Bf(1) Bf(2) Bf(3) Cond
PAPER <- list(
  "f1/1" = list(mspe = c(6.44, 3.92, 3.79, 3.81, 3.83, 3.39, 2.95, 3.13, 3.13, 3.92),
                sd   = c(0.42, 0.41, 0.34, 0.34, 0.46, 0.29, 0.23, 0.26, 0.33, 0.34),
                bias2 = c(4.76, 1.44, 1.43, 1.42, 1.43, 1.46, 1.46, 0.74, 0.53, 2.01)),
  "f1/2" = list(mspe = c(9.42, 7.11, 7.15, 7.19, 7.20, 6.78, 6.78, 6.87, 7.71, 7.20),
                sd   = c(0.71, 0.5, 0.51, 0.51, 0.57, 0.46, 0.46, 0.44, 0.54, 0.52),
                bias2 = c(4.36, 1.53, 1.52, 1.53, 1.52, 1.38, 1.39, 0.77, 0.57, 1.96)),
  "f1/4" = list(mspe = c(21.90, 20.58, 20.67, 20.82, 20.81, 20.85, 20.85, 22.94, 26.45, 20.58),
                sd   = c(1.03, 0.91, 0.93, 0.98, 1.09, 0.90, 0.90, 1.14, 1.63, 0.90),
                bias2 = c(4.39, 1.75, 1.73, 1.79, 1.75, 1.22, 1.22, 0.64, 0.45, 1.90)),
  "f2/1" = list(mspe = c(2.01, 1.50, 1.49, 1.50, 1.52, 1.41, 1.41, 1.47, 1.65, 1.49),
                sd   = c(0.21, 0.12, 0.12, 0.15, 0.12, 0.12, 0.12, 0.11, 0.14, 0.13),
                bias2 = c(0.79, 0.15, 0.14, 0.13, 0.14, 0.11, 0.11, 0.02, 0.01, 0.21)),
  "f3/1" = list(mspe = c(8.68, 4.12, 4.15, 4.18, 4.19, 3.49, 3.49, 2.40, 2.46, 5.04),
                sd   = c(0.78, 0.58, 0.59, 0.59, 0.61, 0.40, 0.41, 0.21, 0.17, 0.61),
                bias2 = c(6.50, 0.78, 0.76, 0.74, 0.75, 1.34, 1.34, 0.18, 0.06, 2.56)),
  "f4/1" = list(mspe = c(8.83, 4.15, 4.12, 4.14, 4.17, 3.06, 3.07, 2.59, 2.96, 3.76),
                sd   = c(1.18, 0.67, 0.67, 0.68, 0.73, 0.46, 0.46, 0.27, 0.29, 0.65),
                bias2 = c(7.51, 1.30, 1.25, 1.22, 1.23, 1.01, 1.00, 0.22, 0.21, 1.80)),
  "f5/1" = list(mspe = c(2.79, 2.20, 2.20, 2.15, 2.12, 1.68, 1.68, 1.61, 1.80, 1.67),
                sd   = c(0.18, 0.19, 0.19, 0.18, 0.19, 0.10, 0.10, 0.07, 0.08, 0.11),
                bias2 = c(1.45, 0.77, 0.76, 0.67, 0.67, 0.29, 0.30, 0.08, 0.04, 0.37))
)
# Paper Table 4, Yacht row: (mean, sd); Bf(3) not reported.
PAPER_YACHT <- list(Un = c(12.89, 4.65), De = c(5.75, 1.58), Qu = c(5.36, 1.45),
                    Cu = c(4.52, 1.63), Sp = c(4.08, 1.62), Bf = c(1.69, 0.99),
                    "Bf(1)" = c(1.55, 0.69), "Bf(2)" = c(2.19, 0.81), Cond = c(3.46, 2.24))

# =============================================================================
# 1. Fig. 1 replica: OOB prediction vs true f(x), Un vs Bf   (n = 500, sigma = 0.5)
# =============================================================================
cat("\n[1] Fig. 1 replica\n")
t <- now()
d1 <- simulate_data("fig1", 500, sigma = 0.5, seed = SEED)
un1 <- fit_forest(d1$x, d1$y, seed = SEED)                          # uncorrected forest
bf1 <- boosted_forest(d1$x, forest = un1, n_iterations = 1, seed = SEED + 1)  # Breiman's correction
oob_un <- oob_predictions(un1)
oob_bf <- predict(bf1)                  # no newdata: bias-corrected OOB predictions

# Calibration slope of f(x) on the OOB prediction would be 1 for an unbiased predictor;
# regressing the OOB prediction on f(x) shows the shrinkage (<1) of the uncorrected forest.
slope <- function(a, b) cov(a, b) / var(a)
f1v <- d1$f
write_csv(file.path(RES_DIR, "fig1.csv"), c("i", "f_true", "oob_Un", "oob_Bf"),
          cbind(seq_along(f1v), sprintf("%.6f", f1v), sprintf("%.6f", oob_un),
                sprintf("%.6f", oob_bf)))
write_md(file.path(RES_DIR, "fig1_summary.md"),
         c("Predictor", "OOB MSE vs f(x)", "OOB MSE vs y", "slope of OOB pred on f(x)"),
         list(c("Un", f2(mean((oob_un - f1v)^2)), f2(mean((oob_un - d1$y)^2)), f2(slope(f1v, oob_un))),
              c("Bf", f2(mean((oob_bf - f1v)^2)), f2(mean((oob_bf - d1$y)^2)), f2(slope(f1v, oob_bf)))),
         title = "Fig. 1 replica (n = 500, sigma = 0.5)",
         note = "A slope below 1 means the OOB predictions are pulled towards the mean (bias at the extremes).")

png(file.path(FIG_DIR, "fig1_oob_vs_true.png"), width = 900, height = 450, pointsize = 14)
par(mfrow = c(1, 2), pty = "s", mar = c(4.5, 4.5, 3, 1), las = 1)
lim1 <- range(f1v, oob_un, oob_bf)
bias_plot(f1v, oob_un, xlab = "true f(x)", ylab = "OOB prediction",
          main = "Un: uncorrected forest", col = COL, cex = 0.8, lim = lim1)
bias_plot(f1v, oob_bf, xlab = "true f(x)", ylab = "OOB prediction",
          main = "Bf: boosted (1 iteration)", col = COL, cex = 0.8, lim = lim1)
invisible(dev.off())
cat(sprintf("  OOB MSE vs f:  Un %.3f   Bf %.3f\n", mean((oob_un - f1v)^2), mean((oob_bf - f1v)^2)))
timer("fig1", t)

# =============================================================================
# 2. Simulation tables (Tables 1-2 style), n = 200, n_test = 2000
#    Note: test X is fixed across replications and the test noise is fresh in each
#    replication, so MSPE (vs noisy y) and squared bias come from the same fits.
# =============================================================================
cat(sprintf("\n[2] Simulation tables (K = %d)\n", K_REPS))
t <- now()
designs <- list(c("f1", 1), c("f1", 2), c("f1", 4), c("f2", 1), c("f3", 1), c("f4", 1), c("f5", 1))
sim_csv <- list()
sim_md <- list()
for (di in seq_along(designs)) {
  td <- now()
  name <- designs[[di]][1]
  sigma <- as.numeric(designs[[di]][2])
  res <- simulation_study(name, sigma = sigma, n = N_TRAIN, n_test = N_TEST, n_reps = K_REPS,
                          seed = SEED + di, n_cores = N_CORES)
  ref <- PAPER[[sprintf("%s/%g", name, sigma)]]
  dlabel <- sprintf("%s (sigma=%g)", toupper(name), sigma)
  sim_md <- c(sim_md, list(c(dlabel, "MSPE (SD), this run", sprintf("%.2f (%.2f)", res$mean, res$sd)),
                           c(dlabel, "MSPE (SD), paper", sprintf("%.2f (%.2f)", ref$mspe, ref$sd)),
                           c(dlabel, "bias^2, this run", f2(res$bias2)),
                           c(dlabel, "bias^2, paper", f2(ref$bias2))))
  ours <- list(mspe = res$mean, sd = res$sd, bias2 = res$bias2)
  for (stat in names(ours)) {
    sim_csv <- c(sim_csv, list(c(name, sprintf("%.1f", sigma), stat, "r", sprintf("%.4f", ours[[stat]])),
                               c(name, sprintf("%.1f", sigma), stat, "paper", f2(ref[[stat]]))))
  }
  timer(dlabel, td)
}
write_csv(file.path(RES_DIR, "simulation_table.csv"),
          c("function", "sigma", "statistic", "source", LABELS), sim_csv)
write_md(file.path(RES_DIR, "simulation_table.md"), c("Design", "Row", LABELS), sim_md,
         title = sprintf("Simulation study, n = %d, n_test = %d, K = %d (paper: K = 1000)",
                         N_TRAIN, N_TEST, K_REPS),
         note = paste("Paper values: Chen, Withana Gamage & Ryan (2022), Tables 1-2 (n = 200 rows).",
                      "MSPE is measured against the noisy test response; bias^2 uses the average prediction over replications."))
timer("simulation tables", t)

# =============================================================================
# 3. Iteration curve (Fig. 5 style): mean test MSPE vs number of residual forests k = 0..6
#    (paper labels Un, Bf, Bf(1), ..., Bf(5)).  Yacht is added in section 4.
# =============================================================================
cat(sprintf("\n[3] Iteration curves (K = %d)\n", K_REPS))
t <- now()
curve_labels <- paper_label(0:MAX_ITER)

# Mean (over K replications) test MSPE of the staged boosted predictions.
iteration_curve <- function(name, sigma, seed, n = N_TRAIN, n_test = N_TEST, reps = K_REPS) {
  test <- simulate_data(name, n_test, sigma = 0, seed = seed)
  out <- par_map(seq_len(reps), function(r) {
    set.seed(seed + 1000 * r)
    train <- simulate_data(name, n, sigma = sigma)
    y_test <- test$f + sigma * rnorm(n_test)
    bf <- boosted_forest(train$x, train$y, n_iterations = MAX_ITER, seed = seed + 1000 * r + 1)
    colMeans((staged_predict(bf, test$x) - y_test)^2)
  })
  colMeans(do.call(rbind, out))
}

curve_designs <- list(c("f1", 1), c("f3", 1))
curves <- list()
for (i in seq_along(curve_designs)) {
  name <- curve_designs[[i]][1]
  sigma <- as.numeric(curve_designs[[i]][2])
  curves[[sprintf("%s (sigma=%g)", toupper(name), sigma)]] <-
    iteration_curve(name, sigma, seed = SEED + 100 + i)
}
timer("iteration curves", t)

# =============================================================================
# 4. Yacht hydrodynamics (UCI): 100 random 2/3-1/3 splits
# =============================================================================
cat(sprintf("\n[4] Yacht data (splits = %d)\n", N_SPLITS))
t <- now()
yacht <- tryCatch(load_yacht(cache_dir = DATA_DIR), debiasforest_dataset_unavailable = function(e) {
  warning("Yacht data unavailable; skipping section 4: ", conditionMessage(e),
          call. = FALSE, immediate. = TRUE)
  NULL
})

if (!is.null(yacht)) {
  Xy <- yacht$x
  yy <- yacht$y
  cat("  loaded yacht data: ", nrow(Xy), " rows, ", ncol(Xy), " features\n", sep = "")

  # 4a. MSPE table for all methods
  ty <- now()
  rs <- repeated_splits(Xy, yy, n_splits = N_SPLITS, seed = SEED + 200, n_cores = N_CORES)
  rows_csv <- list()
  rows_md <- list()
  for (j in seq_along(rs$labels)) {
    l <- rs$labels[j]
    pv <- PAPER_YACHT[[l]]
    rows_csv[[j]] <- c(l, sprintf("%.4f", rs$mean[j]), sprintf("%.4f", rs$sd[j]),
                       if (is.null(pv)) c("", "") else as.character(pv))
    rows_md[[j]] <- c(l, sprintf("%.2f (%.2f)", rs$mean[j], rs$sd[j]),
                      if (is.null(pv)) "-" else sprintf("%.2f (%.2f)", pv[1], pv[2]))
  }
  write_csv(file.path(RES_DIR, "yacht_mspe.csv"),
            c("method", "mspe_mean", "mspe_sd", "paper_mean", "paper_sd"), rows_csv)
  write_md(file.path(RES_DIR, "yacht_mspe.md"),
           c("Method", "MSPE (SD), this run", "MSPE (SD), paper Table 4"), rows_md,
           title = sprintf("Yacht hydrodynamics: test MSPE over %d random 2/3-1/3 splits (paper: 1000)",
                           N_SPLITS))
  timer("yacht splits", ty)

  # 4b. selected number of iterations (paper's stopping rule)
  ts <- now()
  sel <- select_n_iterations(Xy, yy, max_iterations = MAX_ITER, n_splits = N_SPLITS,
                             seed = SEED + 300, n_cores = N_CORES)
  write_md(file.path(RES_DIR, "yacht_select_n_iterations.md"), c("k", "label", "mean MSPE", "SD"),
           cbind(0:MAX_ITER, curve_labels, f2(sel$mean_mspe), f2(sel$sd_mspe)),
           title = sprintf("Yacht: select_n_iterations (max_iterations = %d, %d splits)",
                           MAX_ITER, N_SPLITS),
           note = sprintf("Selected number of residual forests: **k = %d** (%s).",
                          sel$n_iterations, paper_label(sel$n_iterations)))
  cat(sprintf("  select_n_iterations -> k = %d (%s)\n", sel$n_iterations,
              paper_label(sel$n_iterations)))
  curves[["Yacht"]] <- unname(sel$mean_mspe)
  timer("select_n_iterations", ts)

  # 4c. predicted vs observed for one split (Fig. 3 style): Un, best univariate, Bf
  best_uni <- UNIVARIATE[which.min(rs$mean[match(UNIVARIATE, rs$labels)])]
  set.seed(SEED + 400)
  perm <- sample.int(nrow(Xy))
  nte <- round(nrow(Xy) / 3)
  te <- perm[seq_len(nte)]
  tr <- perm[-seq_len(nte)]
  base <- fit_forest(Xy[tr, ], yy[tr], seed = SEED + 401)
  smoother <- c(De = "linear", Qu = "quadratic", Cu = "cubic", Sp = "spline")[[best_uni]]
  preds <- list(Un = unname(predict(base, Xy[te, ])),
                predict(smoothed_forest(Xy[tr, ], yy[tr], method = smoother, forest = base), Xy[te, ]),
                Bf = predict(boosted_forest(Xy[tr, ], yy[tr], forest = base, n_iterations = 1,
                                            seed = SEED + 402), Xy[te, ]))
  names(preds)[2] <- best_uni
  png(file.path(FIG_DIR, "yacht_pred_vs_obs.png"), width = 1250, height = 430, pointsize = 14)
  par(mfrow = c(1, 3))
  par(cex = 1, pty = "s", mar = c(4.5, 4.5, 3, 1), las = 1)   # mfrow alone shrinks the text
  for (l in names(preds)) {
    bias_plot(yy[te], preds[[l]], xlab = "observed", ylab = "predicted", col = COL, alpha = 0.7,
              cex = 1, main = sprintf("%s  (MSPE %.2f)", l, mean((yy[te] - preds[[l]])^2)))
  }
  invisible(dev.off())
  cat("  predicted-vs-observed plot: Un vs ", best_uni, " vs Bf\n", sep = "")
} else {
  write_md(file.path(RES_DIR, "yacht_mspe.md"), "Note",
           list("Yacht data unavailable (offline); section skipped."))
}
timer("yacht", t)

# --- iteration-curve output (needs Yacht curve from section 4) ------------------
cnames <- c(sort(setdiff(names(curves), "Yacht")), intersect("Yacht", names(curves)))
ks <- 0:MAX_ITER
write_csv(file.path(RES_DIR, "iteration_curve.csv"), c("k", "label", cnames),
          cbind(ks, curve_labels, sapply(cnames, function(cn) sprintf("%.4f", curves[[cn]]))))
write_md(file.path(RES_DIR, "iteration_curve.md"), c("k", "label", cnames),
         cbind(ks, curve_labels, sapply(cnames, function(cn) f2(curves[[cn]]))),
         title = "Mean test MSPE vs number of residual forests k",
         note = sprintf("Simulations: K = %d replications, n = %d. Yacht: %d random splits.",
                        K_REPS, N_TRAIN, N_SPLITS))
png(file.path(FIG_DIR, "iteration_curve.png"), width = 400 * length(cnames), height = 400,
    pointsize = 14)
par(mfrow = c(1, length(cnames)))
par(cex = 1, mar = c(5.5, 4.5, 3, 1), las = 1)
for (cn in cnames) {
  plot(ks, curves[[cn]], type = "o", pch = 16, lwd = 2, col = COL, xaxt = "n",
       xlab = "", ylab = "mean MSPE", main = cn)
  axis(1, at = ks, labels = FALSE)
  text(ks, par("usr")[3] - 0.05 * diff(par("usr")[3:4]), curve_labels, srt = 30,
       adj = c(1, 1), xpd = TRUE)
  title(xlab = "iterations k", line = 4)
}
invisible(dev.off())

cat(sprintf("\nDone in %.1f s.  Tables: %s\n      Figures: %s\n", now() - T0, RES_DIR, FIG_DIR))
