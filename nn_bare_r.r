#
# Neural Network in Bare R
# ------------------------
# A multi-layer perceptron trained with backpropagation from scratch.
# No external packages -- only base R.
#
# Task: XOR (classic non-linear problem a straight line cannot solve).
#

# --------------------------------------------------
# 1. Data: XOR
# --------------------------------------------------

xor_data <- data.frame(
  x1 = c(0, 0, 1, 1),
  x2 = c(0, 1, 0, 1),
  y  = c(0, 1, 1, 0)
)

# Replicate with jitter so gradient descent has something to learn.
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

# --------------------------------------------------
# 2. Network architecture: 2 -> 4 (tanh) -> 1 (sigmoid)
# --------------------------------------------------

n_input  <- 2
n_hidden <- 4
n_output <- 1

# Xavier-ish initialization.
W1 <- matrix(rnorm(n_input * n_hidden, sd = sqrt(1 / n_input)),
             nrow = n_input, ncol = n_hidden)
b1 <- numeric(n_hidden)
W2 <- matrix(rnorm(n_hidden * n_output, sd = sqrt(1 / n_hidden)),
             nrow = n_hidden, ncol = n_output)
b2 <- numeric(n_output)

sigmoid <- function(z) 1 / (1 + exp(-z))
sigmoid_prime <- function(a) a * (1 - a)
tanh_prime <- function(a) 1 - a^2

# --------------------------------------------------
# 3. Forward / backward pass (one mini-batch = full data)
# --------------------------------------------------

learning_rate <- 0.5
iterations <- 10000
loss_history <- numeric(iterations)

for (i in 1:iterations) {

  # ---- Forward ----
  z1 <- X %*% W1 + matrix(b1, nrow = nrow(X), ncol = n_hidden, byrow = TRUE)
  a1 <- tanh(z1)
  z2 <- a1 %*% W2 + matrix(b2, nrow = nrow(X), ncol = n_output, byrow = TRUE)
  a2 <- sigmoid(z2)

  # ---- Loss: mean squared error ----
  err <- a2 - matrix(y, ncol = 1)
  loss <- mean(err^2)
  loss_history[i] <- loss

  # ---- Backward ----
  d_a2 <- err / nrow(X)
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

# --------------------------------------------------
# 4. Prediction function
# --------------------------------------------------

predict_nn <- function(Xnew) {
  z1 <- Xnew %*% W1 + matrix(b1, nrow = nrow(Xnew), ncol = n_hidden, byrow = TRUE)
  a1 <- tanh(z1)
  z2 <- a1 %*% W2 + matrix(b2, nrow = nrow(Xnew), ncol = n_output, byrow = TRUE)
  sigmoid(z2)
}

prob  <- predict_nn(X)
pred  <- as.integer(prob > 0.5)
acc   <- mean(pred == y)

cat("Final loss:", round(tail(loss_history, 1), 6), "\n")
cat("Accuracy :", round(acc * 100, 2), "%\n")
cat("Predictions (probability -> class):\n")
print(cbind(prob, pred, y))

# --------------------------------------------------
# 5. Visuals (saved to file, headless-safe)
# --------------------------------------------------

png("nn_loss.png", width = 600, height = 400)
plot(loss_history, type = "l", lwd = 2, col = "steelblue",
     main = "Bare-R MLP: Training Loss (XOR)",
     xlab = "Iteration", ylab = "Mean Squared Error")
grid()
dev.off()

png("nn_boundary.png", width = 700, height = 600)
grid_size <- 200
g <- seq(-0.2, 1.2, length.out = grid_size)
G <- expand.grid(x1 = g, x2 = g)
P <- matrix(predict_nn(as.matrix(G)), nrow = grid_size, ncol = grid_size)

image(g, g, P, col = terrain.colors(50),
      xlab = "x1", ylab = "x2",
      main = "Bare-R MLP Decision Boundary (XOR)")
contour(g, g, P, levels = 0.5, add = TRUE, lwd = 2, col = "black")
points(X[, 1], X[, 2], pch = 21, col = "black",
       bg = ifelse(y == 1, "red", "blue"), cex = 0.6)
dev.off()

cat("Wrote nn_loss.png and nn_boundary.png\n")
