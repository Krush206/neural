
#
# Linear Regression in Bare R
#
# Implementation using gradient descent.
# No external packages required.
#

# --------------------------------------------------
# 1. Data
# --------------------------------------------------

data <- data.frame(
  hours = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
  score = c(42, 45, 51, 53, 60, 64, 68, 72, 79, 85)
)

# --------------------------------------------------
# 2. Exploratory data analysis
# --------------------------------------------------

print(data)
summary(data)

plot(
  data$hours,
  data$score,
  main = "Study Hours and Examination Scores",
  xlab = "Hours of Study",
  ylab = "Examination Score",
  pch = 19
)

# --------------------------------------------------
# 3. Implement gradient descent
# --------------------------------------------------

gradient_descent <- function(x, y,
                             learning_rate = 0.05,
                             iterations = 5000) {

  # Scale x to improve numerical stability.
  x <- x / 10

  n <- length(x)

  # Model parameters.
  intercept <- 0
  slope <- 0

  # Store the cost at each iteration.
  history <- numeric(iterations)

  for (i in 1:iterations) {

    predictions <- intercept + slope * x

    errors <- predictions - y

    # Mean squared error.
    cost <- sum(errors^2) / (2 * n)

    history[i] <- cost

    # Calculate gradients.
    d_intercept <- sum(errors) / n
    d_slope <- sum(errors * x) / n

    # Update parameters.
    intercept <- intercept -
      learning_rate * d_intercept

    slope <- slope -
      learning_rate * d_slope
  }

  list(
    intercept = intercept,
    slope = slope,
    history = history
  )
}

# --------------------------------------------------
# 4. Train the model
# --------------------------------------------------

model <- gradient_descent(
  data$hours,
  data$score
)

print(model$intercept)
print(model$slope)

# --------------------------------------------------
# 5. Prediction function
# --------------------------------------------------

predict <- function(model, x) {

  model$intercept +
    model$slope * (x / 10)

}

# Make predictions.
predictions <- predict(model, data$hours)

print(predictions)

# --------------------------------------------------
# 6. Evaluate the model
# --------------------------------------------------

errors <- data$score - predictions

mse <- mean(errors^2)
rmse <- sqrt(mse)

cat("Mean Squared Error:", mse, "\n")
cat("Root Mean Squared Error:", rmse, "\n")

# --------------------------------------------------
# 7. Visualize the fitted model
# --------------------------------------------------

plot(
  data$hours,
  data$score,
  main = "Linear Regression",
  xlab = "Hours of Study",
  ylab = "Examination Score",
  pch = 19
)

abline(
  a = model$intercept,
  b = model$slope / 10,
  lwd = 2
)

# --------------------------------------------------
# 8. Visualize the learning process
# --------------------------------------------------

plot(
  model$history,
  type = "l",
  main = "Gradient Descent Convergence",
  xlab = "Iteration",
  ylab = "Cost"
)

# --------------------------------------------------
# 9. Predict a new observation
# --------------------------------------------------

new_hours <- 7.5

predicted_score <- predict(model, new_hours)

cat(
  "Predicted score for",
  new_hours,
  "hours of study:",
  predicted_score,
  "\n"
)
