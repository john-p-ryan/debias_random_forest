# Conditional bias correction (Lu & Hardin 2021; paper section 2.2.3).

#' Conditional bias correction
#'
#' Forest prediction minus a leaf-weighted average of the training OOB biases.  For
#' tree t and terminal node L let `S_t[L]` be the sum of `oob_i - y_i` over the
#' training rows i that are out-of-bag for tree t and fall in L, and `C_t[L]` their
#' number.  The bias estimate at x0 is
#' `b(x0) = sum_t S_t[leaf_t(x0)] / sum_t C_t[leaf_t(x0)]` (0 if the denominator is
#' 0) and the prediction is `forest(x0) - b(x0)`; this is the paper's weight formula
#' `v_i(x0)` written per leaf.
#'
#' `oob_prediction` for training row j is `oob_j - b_{-j}(x_j)`, where row j's own
#' contribution (its bias in the numerator and one count per tree in which it is
#' OOB) is removed from the sums.  The OOB biases themselves use the full OOB
#' predictions (as in the paper), so this is an approximate OOB estimate.
#'
#' @inheritParams smoothed_forest
#' @return An object of class `conditional_bias_forest`: a list with `forest`,
#'   `leaf_bias_sum` and `leaf_oob_count` (`nrnodes x ntree` matrices `S` and `C`),
#'   `base_oob_prediction` and `oob_prediction`.
#' @seealso [bias_estimate()]
#' @examples
#' d <- simulate_data("f1", 200, seed = 1)
#' cond <- conditional_bias_forest(d$x, d$y, ntree = 100, seed = 1)
#' head(bias_estimate(cond, d$x))
#' @export
conditional_bias_forest <- function(x, y = NULL, forest = NULL, ntree = 500,
                                    mtry = default_mtry(x), nodesize = 5, seed = NULL,
                                    ...) {
  x <- check_x(x)
  base <- with_seed(seed, base_forest(x, y, forest, ntree, mtry, nodesize, ...))
  rf <- base$forest
  oob <- oob_predictions(rf, x)
  bias <- oob - base$y
  n <- nrow(x)
  nr <- rf$forest$nrnodes
  ids <- leaf_ids(rf, attr(stats::predict(rf, x, nodes = TRUE), "nodes"))
  oob_mask <- rf$inbag == 0
  g <- ids[oob_mask]
  S <- matrix(0, nr, rf$ntree)
  S[sort(unique(g))] <- rowsum(rep(bias, rf$ntree)[oob_mask], g)  # rows sorted by group
  C <- matrix(tabulate(g, nbins = nr * rf$ntree), nr, rf$ntree)
  k <- rowSums(oob_mask)
  num <- rowSums(matrix(S[c(ids)], n)) - k * bias
  den <- rowSums(matrix(C[c(ids)], n)) - k
  structure(list(forest = rf, leaf_bias_sum = S, leaf_oob_count = C,
                 base_oob_prediction = oob,
                 oob_prediction = oob - ifelse(den > 0, num / den, 0)),
            class = "conditional_bias_forest")
}

# Terminal nodes (n x ntree, from predict(rf, nodes = TRUE)) as linear indices into
# nrnodes x ntree matrices.  Index with c(ids): with ntree = 2 the matrix itself would be
# read as (row, column) pairs.
leaf_ids <- function(rf, nodes) {
  dimnames(nodes) <- NULL
  nodes + rep((seq_len(rf$ntree) - 1L) * rf$forest$nrnodes, each = nrow(nodes))
}

leaf_bias <- function(object, ids) {
  num <- rowSums(matrix(object$leaf_bias_sum[c(ids)], nrow(ids)))
  den <- rowSums(matrix(object$leaf_oob_count[c(ids)], nrow(ids)))
  ifelse(den > 0, num / den, 0)
}

#' Estimated conditional bias
#'
#' @param object a fitted `conditional_bias_forest`.
#' @param newdata predictors.
#' @return The bias estimate `b(x0)` for each row of `newdata`.
#' @export
bias_estimate <- function(object, newdata) {
  stopifnot(inherits(object, "conditional_bias_forest"))
  rf <- object$forest
  leaf_bias(object, leaf_ids(rf, attr(stats::predict(rf, newdata, nodes = TRUE), "nodes")))
}

#' @rdname predict.debiasforest
#' @export
predict.conditional_bias_forest <- function(object, newdata, ...) {
  if (missing(newdata)) return(object$oob_prediction)
  rf <- object$forest
  p <- stats::predict(rf, newdata, nodes = TRUE)
  as.vector(p) - leaf_bias(object, leaf_ids(rf, attr(p, "nodes")))
}

#' @export
print.conditional_bias_forest <- function(x, ...) {
  cat("Conditional bias corrected random forest (Cond)\n")
  cat("  base forest: ", x$forest$ntree, " trees, mtry = ", x$forest$mtry, "\n", sep = "")
  invisible(x)
}
