"""
The real thing gradient descent is for: fitting a model to data.

Problem: predict exam score from hours studied.
Model:   score = m * hours + b            <- this IS the "AI model"
Params:  m and b                          <- this IS "the weights"
Loss:    how wrong the line is on our real data
Training: use gradient descent to find the m, b that make loss smallest

This is the same update rule as phase1_gradient_descent.py. The only
difference is WHAT we're minimizing: instead of a fake (x-3)^2, the loss
is now computed FROM REAL DATA.
"""

import numpy as np

# --- Fake but realistic data: (hours studied, exam score) ---
# In a real project this would be loaded from a file. The relationship
# is roughly score = 5*hours + 50, plus some random noise, because real
# students never sit exactly on a line - that's true of all real data.
rng = np.random.default_rng(seed=0)
hours = np.linspace(0, 10, 20)
true_m, true_b = 5.0, 50.0
noise = rng.normal(0, 5, size=hours.shape)
scores = true_m * hours + true_b + noise
scores = np.clip(scores, 0, 100)  # scores can't exceed 100 or go below 0


# --- The model: a straight line ---
def predict(hours, m, b):
    return m * hours + b


# --- The loss function: Mean Squared Error ---
# For every student, (predicted - actual)^2, then average it.
# Squaring does two jobs: makes errors positive (so over- and
# under-predicting don't cancel out), and punishes big misses harder
# than small ones.
def mse_loss(m, b):
    predictions = predict(hours, m, b)
    errors = predictions - scores
    return np.mean(errors ** 2)


# --- The gradient of the loss, with respect to m and b ---
# This is calculus applied to real data instead of a toy function.
# loss = mean( (m*h + b - y)^2 )
# d(loss)/dm = mean( 2 * (m*h + b - y) * h )   <- chain rule, times d/dm of (m*h+b) = h
# d(loss)/db = mean( 2 * (m*h + b - y) * 1 )   <- chain rule, times d/db of (m*h+b) = 1
def gradients(m, b):
    predictions = predict(hours, m, b)
    errors = predictions - scores
    grad_m = np.mean(2 * errors * hours)
    grad_b = np.mean(2 * errors)
    return grad_m, grad_b


def train(learning_rate=0.01, steps=500):
    # Start with a deliberately bad guess: a flat line at 0.
    # Gradient descent has to find its way from "obviously wrong" to
    # "matches the data" using nothing but the slope of the loss.
    m, b = 0.0, 0.0
    history = []

    for step in range(steps):
        loss = mse_loss(m, b)
        history.append((step, m, b, loss))

        grad_m, grad_b = gradients(m, b)
        # Same update rule as before, just two numbers instead of one.
        m = m - learning_rate * grad_m
        b = b - learning_rate * grad_b

        if step % 50 == 0:
            print(f"step {step:>4} | m={m:6.3f}  b={b:6.3f}  | loss={loss:8.3f}")

    print(f"\nFinal:  m={m:.3f}, b={b:.3f}")
    print(f"Truth:  m={true_m:.3f}, b={true_b:.3f}  (what generated the data, minus noise)")
    return history, m, b


if __name__ == "__main__":
    train()
