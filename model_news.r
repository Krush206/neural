#
# Linear Model in the Style of "The New S Language"
# --------------------------------------------------
# New S (Chambers & Hastie, 1992) introduced the idioms that R still uses:
#   * formulas        (response ~ predictors)
#   * classes & S3 methods (UseMethod dispatch)
#   * model.frame / model.matrix
#   * generic fit / predict / print / plot
#
# R is an implementation of the S language, so this code runs in R.
# It demonstrates *how* an AI/model would be written on top of New S.
#

# --------------------------------------------------
# 1. Data (same study-hours data as main.r)
# --------------------------------------------------

study <- data.frame(
  hours = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
  score = c(42, 45, 51, 53, 60, 64, 68, 72, 79, 85)
)

# --------------------------------------------------
# 2. New S OOP: constructor, class, S3 generics
# --------------------------------------------------

# Constructor: captures the formula and data in a New S way.
news_lm <- function(formula, data,
                    learning_rate = 0.05, iterations = 5000) {

  # New S idiom: turn the formula + data frame into a model frame.
  mf <- model.frame(formula, data)

  y <- mf[, 1]                                  # response
  X <- as.matrix(cbind(1, mf[, -1, drop = FALSE]))  # design matrix w/ intercept

  # Gradient descent (same math as main.r, packaged as a New S object).
  n <- nrow(X)
  beta <- numeric(ncol(X))
  history <- numeric(iterations)

  for (i in 1:iterations) {
    fitted <- X %*% beta
    resid <- fitted - y
    history[i] <- sum(resid^2) / (2 * n)
    grad <- t(X) %*% resid / n
    beta <- beta - learning_rate * as.vector(grad)
  }

  obj <- list(
    coefficients = beta,
    history      = history,
    terms        = terms(formula),
    call         = match.call(),
    xnames       = colnames(X)
  )
  class(obj) <- c("news_lm", "news_model")   # New S: class attribute -> dispatch
  obj
}

# --------------------------------------------------
# 3. S3 generics and methods (UseMethod dispatch)
# --------------------------------------------------

fit <- function(object, ...) UseMethod("fit")
predict.news_model <- function(object, newdata, ...) {
  if (missing(newdata)) {
    # Fitted values on the training data.
    X <- model.matrix(object$terms, model.frame(object$terms, eval(object$call$data)))
    drop(X %*% object$coefficients)
  } else {
    mf <- model.frame(object$terms, newdata)
    X  <- as.matrix(cbind(1, mf[, -1, drop = FALSE]))
    drop(X %*% object$coefficients)
  }
}
print.news_lm <- function(x, ...) {
  cat("New S linear model: score ~ hours\n")
  cat("Coefficients:\n")
  print(setNames(x$coefficients, x$xnames))
  cat("Final cost:", tail(x$history, 1), "\n")
  invisible(x)
}
plot.news_lm <- function(x, ...) {
  plot(x$history, type = "l", lwd = 2, col = "steelblue",
       main = "New S Model: Gradient-Descent Convergence",
       xlab = "Iteration", ylab = "Cost")
  grid()
}

# --------------------------------------------------
# 4. Train: formula interface, New S style
# --------------------------------------------------

model <- news_lm(score ~ hours, data = study)

print(model)

# --------------------------------------------------
# 5. Predict
# --------------------------------------------------

fitted <- predict(model)                      # training predictions
new_hours <- data.frame(hours = c(3.5, 7.5))
forecast  <- predict(model, newdata = new_hours)

cat("\nFitted values on training data:\n")
print(round(fitted, 2))
cat("\nForecasts:\n")
print(data.frame(hours = new_hours$hours, predicted_score = round(forecast, 2)))

cat("\nResidual MSE:", mean((study$score - fitted)^2), "\n")

# --------------------------------------------------
# 6. Visuals (headless-safe)
# --------------------------------------------------

png("news_fit.png", width = 700, height = 600)
par(mfrow = c(2, 1))
plot(study$hours, study$score, pch = 19, col = "steelblue",
     main = "New S Model: score ~ hours",
     xlab = "Hours of Study", ylab = "Examination Score")
grid()
abline(a = model$coefficients[1], b = model$coefficients[2], lwd = 2, col = "tomato")
points(new_hours$hours, forecast, pch = 17, col = "darkgreen", cex = 2)
legend("topleft", legend = c("data", "fitted line", "forecast"),
       pch = c(19, NA, 17), lty = c(NA, 1, NA), col = c("steelblue", "tomato", "darkgreen"))

plot(model)  # dispatches to plot.news_lm
par(mfrow = c(1, 1))
dev.off()

cat("\nWrote news_fit.png\n")
cat("\n---\n")
cat("New S notes:\n")
cat("- Formulas (score ~ hours), model.frame, model.matrix: core New S.\n")
cat("- class() + UseMethod: S3 method dispatch from New S; R inherited it.\n")
cat("- True New S (1992, S-PLUS) used static scoping; R uses lexical scoping.\n")
cat("- R later added S4 classes, reference classes, and packages on top of New S.\n")
