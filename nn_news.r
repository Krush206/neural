#
# Neural Network in the Style of "The New S Language"
# ----------------------------------------------------
# A multi-layer perceptron trained with backpropagation from scratch.
# No external packages -- only base R.
#
# New S (Chambers & Hastie, 1992) idioms used here:
#   * formulas        (response ~ predictors)
#   * model.frame / model.matrix / terms / delete.response
#   * classes & S3 methods (UseMethod dispatch)
#   * generics: fit / predict / print / plot
#
# R is an implementation of the S language, so this code runs in R.
#
# Task: XOR -- the classic non-linear problem no straight line can solve.
#

# --------------------------------------------------
# 0. Backpropagation engine (bare R, no packages)
# --------------------------------------------------

sigmoid <- function(z) 1 / (1 + exp(-z))
sigmoid_prime <- function(a) a * (1 - a)
tanh_prime <- function(a) 1 - a^2

# One training run: full-batch gradient descent on mean squared error.
# Returns the updated weights and the loss history.
backprop <- function(X, y,
                     W1, b1, W2, b2,
                     learning_rate, iterations) {

  n       <- nrow(X)
  n_hidden <- ncol(W1)

  loss_history <- numeric(iterations)

  for (i in 1:iterations) {

    # ---- Forward ----
    z1 <- X %*% W1 + matrix(b1, n, n_hidden, byrow = TRUE)
    a1 <- tanh(z1)
    z2 <- a1 %*% W2 + matrix(b2, n, 1, byrow = TRUE)
    a2 <- sigmoid(z2)

    # ---- Loss: mean squared error ----
    err <- a2 - matrix(y, ncol = 1)
    loss_history[i] <- mean(err^2)

    # ---- Backward ----
    d_a2 <- err / n
    d_z2 <- d_a2 * sigmoid_prime(a2)
    d_W2 <- t(a1) %*% d_z2
    d_b2 <- colSums(d_z2)

    d_a1 <- d_z2 %*% t(W2)
    d_z1 <- d_a1 * tanh_prime(a1)
    d_W1 <- t(X) %*% d_z1
    d_b1 <- colSums(d_z1)

    # ---- Update ----
    W2 <- W2 - learning_rate * d_W2
    b2 <- b2 - learning_rate * d_b2
    W1 <- W1 - learning_rate * d_W1
    b1 <- b1 - learning_rate * d_b1
  }

  list(W1 = W1, b1 = b1, W2 = W2, b2 = b2, history = loss_history)
}

# --------------------------------------------------
# 1. New S OOP: constructor, class, S3 generics
# --------------------------------------------------

# Constructor: formula interface in the New S style.
news_nn <- function(formula, data,
                    hidden = 4,
                    learning_rate = 0.5,
                    iterations = 10000,
                    seed = 42) {

  set.seed(seed)

  # New S idiom: turn the formula + data frame into a model frame,
  # then into a design matrix (with intercept column).
  mf    <- model.frame(formula, data)
  mt    <- terms(formula)
  y     <- mf[, 1]
  X     <- model.matrix(mt, mf)

  n_input    <- ncol(X)            # includes the intercept column
  n_predictors <- ncol(X) - 1
  n_hidden   <- hidden
  n_output   <- 1

  # Xavier-ish initialization.
  W1 <- matrix(rnorm(n_input * n_hidden, sd = sqrt(1 / n_input)),
               nrow = n_input, ncol = n_hidden)
  b1 <- numeric(n_hidden)
  W2 <- matrix(rnorm(n_hidden * n_output, sd = sqrt(1 / n_hidden)),
               nrow = n_hidden, ncol = n_output)
  b2 <- numeric(n_output)

  # Train: backpropagation from scratch.
  fitted <- backprop(X, y, W1, b1, W2, b2, learning_rate, iterations)

  obj <- list(
    weights     = fitted[c("W1", "b1", "W2", "b2")],
    history     = fitted$history,
    terms       = mt,
    model       = mf,                 # New S: model objects keep their data
    call        = match.call(),
    xnames      = colnames(X),
    ynames      = colnames(mf)[1],
    n_input     = n_input,
    n_predictors = n_predictors,
    n_hidden    = n_hidden,
    n_output    = n_output,
    learning_rate = learning_rate,
    iterations  = iterations
  )
  class(obj) <- c("news_nn", "news_model")   # New S: class attribute -> dispatch
  obj
}

# Generic for (re)fitting.
fit <- function(object, ...) UseMethod("fit")

# Refit from scratch using the data stored in the object.
fit.news_nn <- function(object, ...) {
  mf <- object$model
  y <- mf[, 1]
  X <- model.matrix(object$terms, mf)
  refit <- backprop(X, y,
                    object$weights$W1, object$weights$b1,
                    object$weights$W2, object$weights$b2,
                    object$learning_rate, object$iterations)
  object$weights <- refit[c("W1", "b1", "W2", "b2")]
  object$history <- refit$history
  object
}

# --------------------------------------------------
# 2. S3 methods
# --------------------------------------------------

predict.news_nn <- function(object, newdata,
                            type = c("probability", "class"), ...) {
  type <- match.arg(type)

  forward <- function(X, w) {
    z1 <- X %*% w$W1 + matrix(w$b1, nrow(X), ncol(w$W1), byrow = TRUE)
    a1 <- tanh(z1)
    z2 <- a1 %*% w$W2 + matrix(w$b2, nrow(X), 1, byrow = TRUE)
    sigmoid(z2)
  }

  w <- object$weights
  if (missing(newdata)) {
    # Fitted values on the training data.
    X <- model.matrix(object$terms, object$model)
  } else {
    # Design matrix for new data, built the New S way. Drop the response
    # term so newdata need not contain the response column.
    mt <- delete.response(object$terms)
    mf <- model.frame(mt, newdata)
    X  <- model.matrix(mt, mf)
  }

  probs <- drop(forward(X, w))
  if (type == "class") as.integer(probs > 0.5) else probs
}

print.news_nn <- function(x, ...) {
  cat("New S neural network:", paste(deparse(x$call$formula), collapse = " "), "\n")
  cat("  Architecture:", x$n_predictors, "inputs + intercept ->",
      x$n_hidden, "(tanh) ->", x$n_output, "(sigmoid)\n")
  cat("  Learning rate:", x$learning_rate, ", iterations:", x$iterations, "\n")
  cat("  Final loss:", tail(x$history, 1), "\n")
  invisible(x)
}

plot.news_nn <- function(x, ...) {
  plot(x$history, type = "l", lwd = 2, col = "steelblue",
       main = "New S MLP: Training Loss",
       xlab = "Iteration", ylab = "Mean Squared Error")
  grid()
  invisible(x)
}

# --------------------------------------------------
# 3. Demo: XOR (non-linear classification)
# --------------------------------------------------

if (sys.nframe() == 0) {

  # Replicated, jittered XOR so gradient descent has something to learn.
  xor_data <- data.frame(
    x1 = c(0, 0, 1, 1),
    x2 = c(0, 1, 0, 1),
    y  = c(0, 1, 1, 0)
  )
  set.seed(42)
  n_rep <- 200
  X <- matrix(0, nrow = 4 * n_rep, ncol = 2)
  y <- numeric(4 * n_rep)
  for (r in 1:4) {
    idx <- ((r - 1) * n_rep + 1):(r * n_rep)
    X[idx, 1] <- xor_data$x1[r] + rnorm(n_rep, sd = 0.05)
    X[idx, 2] <- xor_data$x2[r] + rnorm(n_rep, sd = 0.05)
    y[idx]    <- xor_data$y[r]
  }
  train_df <- data.frame(x1 = X[, 1], x2 = X[, 2], y = y)

  # Train: New S style -- formula interface, object returned.
  model <- news_nn(y ~ x1 + x2, data = train_df,
                   hidden = 4, learning_rate = 0.5, iterations = 10000)
  print(model)

  # In-sample predictions: probabilities and classes.
  probs  <- predict(model)                 # training predictions
  preds  <- predict(model, type = "class")
  cat("\nIn-sample accuracy:", round(mean(preds == train_df$y) * 100, 2), "%\n")

  # Refitting continues training from the current weights.
  model <- fit(model)
  cat("After refit, loss:", round(tail(model$history, 1), 6), "\n")

  # --------------------------------------------------
  # 4. Visuals (headless-safe, written to file)
  # --------------------------------------------------

  png("nn_news_loss.png", width = 600, height = 400)
  plot(model)                              # dispatches to plot.news_nn
  dev.off()

  png("nn_news_boundary.png", width = 700, height = 600)
  grid_size <- 200
  g <- seq(-0.2, 1.2, length.out = grid_size)
  G <- data.frame(expand.grid(x1 = g, x2 = g))
  P <- matrix(predict(model, newdata = G), nrow = grid_size, ncol = grid_size)

  image(g, g, P, col = terrain.colors(50),
        xlab = "x1", ylab = "x2",
        main = "New S MLP Decision Boundary (XOR)")
  contour(g, g, P, levels = 0.5, add = TRUE, lwd = 2, col = "black")
  points(X[, 1], X[, 2], pch = 21, col = "black",
         bg = ifelse(y == 1, "red", "blue"), cex = 0.6)
  dev.off()

  cat("Wrote nn_news_loss.png and nn_news_boundary.png\n")
}
