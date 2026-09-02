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

Like that file, this is the single home of the algorithm - the terminal
trainer, the plots and the web UI all call train() here.
"""

import math

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


def train(learning_rate=0.01, steps=500, verbose=True, print_every=50):
    """Fit m and b by gradient descent.

    Returns (history, m, b), where history is a list of
    (step, m, b, loss) tuples - one per step, plus a final entry for the
    finished model so the last row really is the answer you're given.
    """
    # Start with a deliberately bad guess: a flat line at 0.
    # Gradient descent has to find its way from "obviously wrong" to
    # "matches the data" using nothing but the slope of the loss.
    m, b = 0.0, 0.0
    history = []

    # Deliberately trying a too-large learning rate is a useful experiment,
    # and it overflows on purpose. diagnose() explains what happened in
    # words, so numpy's RuntimeWarnings would just be noise on top.
    with np.errstate(over="ignore", invalid="ignore"):
        for step in range(steps):
            loss = mse_loss(m, b)
            history.append((step, m, b, loss))

            grad_m, grad_b = gradients(m, b)
            # Same update rule as before, just two numbers instead of one.
            m = m - learning_rate * grad_m
            b = b - learning_rate * grad_b

            if verbose and step % print_every == 0:
                print(f"step {step:>4} | m={m:6.3f}  b={b:6.3f}  | loss={loss:8.3f}")

            if not (math.isfinite(m) and math.isfinite(b)):
                # A learning rate too big for this data makes the weights
                # explode to inf/NaN. Stop rather than filling the history
                # with meaningless numbers.
                break

        # Record where training actually ended up. Without this the last row
        # is the state BEFORE the final update, and "final loss" is a lie by
        # one step.
        history.append((len(history), m, b, mse_loss(m, b)))

    if verbose:
        print(f"\nFinal:  m={m:.3f}, b={b:.3f}")
        print(f"Truth:  m={true_m:.3f}, b={true_b:.3f}  (what generated the data, minus noise)")
    return history, m, b


def closed_form_solution():
    """The exact best-fit line, solved algebraically instead of searched for.

    For a straight line there IS a formula for the optimal m and b, so we
    can check gradient descent's homework: it should creep toward exactly
    these numbers and never beat them.

    The reason this matters: for a neural network (or Qwen) no such
    formula exists, which is precisely why gradient descent - a method
    that only ever needs the local slope - is the thing that scales.
    """
    h_mean, s_mean = hours.mean(), scores.mean()
    m = np.sum((hours - h_mean) * (scores - s_mean)) / np.sum((hours - h_mean) ** 2)
    b = s_mean - m * h_mean
    return float(m), float(b)


def diagnose(history):
    """Turn a training run into a plain-language verdict, shared by the
    terminal trainer and the web UI."""
    first_loss, final_loss = history[0][3], history[-1][3]

    if not math.isfinite(final_loss):
        return (
            "The loss blew up to infinity. The learning rate was too high: "
            "each step overcorrected further than the last until the "
            "numbers ran off the end of what a float can hold. Try 0.01."
        )

    # Loss going UP is the signature of too-large steps, not too-small
    # ones - worth naming separately, because the fix is the opposite.
    if final_loss > first_loss:
        return (
            f"Diverging: loss went UP, {first_loss:.1f} -> {final_loss:.1f}. "
            "The learning rate is too big, so each correction overshoots the "
            "bottom of the loss curve and lands somewhere worse. Lower it "
            "(try 0.01) rather than training for longer - more steps of the "
            "wrong size only makes this worse."
        )

    best_loss = mse_loss(*closed_form_solution())

    if final_loss <= best_loss * 1.01:
        return (
            f"Converged. Final loss {final_loss:.3f} is within 1% of "
            f"{best_loss:.3f}, which is the best any straight line can do on "
            "this data - confirmed by solving for it exactly in "
            "closed_form_solution(). The leftover error isn't the model "
            "failing, it's the random noise in the data, which no line can fit."
        )

    if final_loss > first_loss * 0.5:
        return (
            f"Barely moved: loss went {first_loss:.1f} -> {final_loss:.1f}, "
            f"still far off the achievable {best_loss:.3f}. Usually that "
            "means the learning rate is too small to cover the distance in "
            "this many steps. Raise it, or train for longer."
        )

    return (
        f"Making progress but not finished: loss {first_loss:.1f} -> "
        f"{final_loss:.3f}, against {best_loss:.3f} for the best possible "
        "line. More steps (or a slightly larger learning rate) will close "
        "the gap."
    )


if __name__ == "__main__":
    train()
