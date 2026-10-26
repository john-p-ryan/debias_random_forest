#' Predicted versus observed values with a 45 degree line
#'
#' Scatter plot (base graphics) of one or several methods' predictions against the
#' observed responses or the true regression function.  Points above (below) the red
#' 45 degree line are over- (under-) predicted; systematic curvature around the line
#' is bias that a correction can remove.
#'
#' @param reference observed responses (or true function values), on the x axis.
#' @param predictions a numeric vector, a named list of vectors or a matrix with one
#'   column per method (such as the output of [predict_all_methods()]).
#' @param xlab,ylab,main axis labels and title.
#' @param col colours, one per method (default: the Okabe-Ito palette, or
#'   `hcl.colors(k, "Dark 3")` for more than 8 methods).
#' @param alpha opacity of the points.
#' @param pch,cex plotting symbol and size.
#' @param legend_pos position of the legend (used when the methods have names).
#' @param lim common limits of both axes (default: the range of all values), for
#'   example to give several panels the same scale.
#' @param ... further arguments to [graphics::plot()].
#' @return The predictions as a list, invisibly.
#' @examples
#' d <- simulate_data("fig1", 300, seed = 1)
#' rf <- fit_forest(d$x, d$y, ntree = 100, seed = 1)
#' bf <- boosted_forest(d$x, forest = rf)
#' bias_plot(d$f, list(Un = oob_predictions(rf), Bf = predict(bf)), xlab = "f(x)",
#'           ylab = "OOB prediction")
#' @export
bias_plot <- function(reference, predictions, xlab = "observed", ylab = "predicted",
                      main = NULL, col = NULL, alpha = 0.6, pch = 16, cex = 0.6,
                      legend_pos = "topleft", lim = NULL, ...) {
  if (is.matrix(predictions)) {
    predictions <- stats::setNames(lapply(seq_len(ncol(predictions)),
                                          function(j) predictions[, j]),
                                   colnames(predictions))
  } else if (!is.list(predictions)) {
    predictions <- list(predictions)
  }
  if (is.null(col)) {
    k <- max(length(predictions), 1L)
    col <- if (k <= 8L) grDevices::palette.colors(k + 1L)[-1L] else grDevices::hcl.colors(k, "Dark 3")
  }
  col <- rep_len(grDevices::adjustcolor(col, alpha.f = alpha), length(predictions))
  if (is.null(lim)) lim <- range(reference, unlist(predictions), finite = TRUE)
  graphics::plot(NA, xlim = lim, ylim = lim, xlab = xlab, ylab = ylab, main = main, ...)
  for (j in seq_along(predictions))
    graphics::points(reference, predictions[[j]], col = col[j], pch = pch, cex = cex)
  graphics::abline(0, 1, col = "red", lwd = 1)
  if (!is.null(names(predictions)))
    graphics::legend(legend_pos, legend = names(predictions), col = col, pch = pch,
                     bty = "n")
  invisible(predictions)
}
