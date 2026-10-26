test_that("bias_plot draws a vector, a named list and a matrix", {
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off())
  f <- f1_test$f
  out <- expect_invisible(bias_plot(f, f * 0.9))
  expect_identical(out, list(f * 0.9))
  preds <- list(Un = f + 0.1, Bf = f - 0.1)
  expect_identical(bias_plot(f, preds, xlab = "f(x)", main = "title"), preds)
  M <- cbind(Un = f + 0.1, Bf = f - 0.1, Cond = f)
  out <- bias_plot(f, M, col = c("black", "blue", "green"), legend_pos = "bottomright")
  expect_identical(names(out), colnames(M))
  expect_equal(out$Cond, f)
  expect_silent(bias_plot(f, list(a = c(f[-1], NA)), alpha = 1))
})

test_that("bias_plot gives distinct default colours to many methods and takes limits", {
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off())
  f <- f1_test$f
  M <- sapply(1:10, function(k) f + k / 10)
  colnames(M) <- method_labels()
  expect_silent(bias_plot(f, M))
  expect_silent(bias_plot(f, f, lim = c(-5, 40)))
  expect_equal(graphics::par("usr")[1:2], c(-5, 40) + c(-1, 1) * 0.04 * 45)
})
