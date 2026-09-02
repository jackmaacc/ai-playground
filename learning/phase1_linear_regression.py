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

IMPORTANT: the model is trained on one part of the data and scored on
another part it has never seen. Judging a model by how well it fits the
examples it already memorised tells you nothing about whether it works -
that split is the difference between "it learned" and "it memorised".

Like phase1_gradient_descent.py, this is the single home of the
algorithm - the terminal trainer, the plots and the web UI all call
train() here.
"""

import math
from typing import NamedTuple

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
# Scores can't exceed 100 or go below 0. Note this flattens the very top
# of the range (5*10 + 50 is already 100), so the true relationship isn't
# perfectly straight at the right-hand edge - that's real data for you,
# and a little of the leftover error comes from exactly this.
scores = np.clip(scores, 0, 100)

# Which points are held back for testing. Fixed so every run is
# reproducible; change it and you get a different (equally valid) split.
SPLIT_SEED = 0
TEST_FRACTION = 0.25


def split_data(test_fraction=TEST_FRACTION, seed=SPLIT_SEED):
    """Randomly hold back some students the model will never train on.

    It has to be RANDOM. `hours` is sorted, so slicing off the last few
    points would hold out only the longest-studying students - that
    measures whether the model can extrapolate beyond anything it has
    seen, which is a different (and much harder) question than whether
    it generalises to new students in the range it was trained on.
    """
    order = np.random.default_rng(seed).permutation(len(hours))
    n_test = max(1, round(len(hours) * test_fraction))
    test_idx = np.sort(order[:n_test])
    train_idx = np.sort(order[n_test:])
    return hours[train_idx], scores[train_idx], hours[test_idx], scores[test_idx]


train_hours, train_scores, test_hours, test_scores = split_data()


class TrainingStep(NamedTuple):
    """One row of training history.

    It's a NamedTuple, so it still behaves exactly like the plain tuple
    this used to be - row[3] works - while also allowing row.test_loss.
    """
    step: int
    m: float
    b: float
    loss: float       # on the training split: what training minimises
    test_loss: float  # on the held-out split: what we actually care about


# --- The model: a straight line ---
def predict(hours, m, b):
    return m * hours + b


# --- The loss function: Mean Squared Error ---
# For every student, (predicted - actual)^2, then average it.
# Squaring does two jobs: makes errors positive (so over- and
# under-predicting don't cancel out), and punishes big misses harder
# than small ones.
#
# Defaults to the TRAINING split, because that's the only data training
# is allowed to look at. Pass x/y explicitly to score any other set.
def mse_loss(m, b, x=None, y=None):
    x = train_hours if x is None else x
    y = train_scores if y is None else y
    errors = predict(x, m, b) - y
    return np.mean(errors ** 2)


# --- The gradient of the loss, with respect to m and b ---
# This is calculus applied to real data instead of a toy function.
# loss = mean( (m*h + b - y)^2 )
# d(loss)/dm = mean( 2 * (m*h + b - y) * h )   <- chain rule, times d/dm of (m*h+b) = h
# d(loss)/db = mean( 2 * (m*h + b - y) * 1 )   <- chain rule, times d/db of (m*h+b) = 1
def gradients(m, b, x=None, y=None):
    x = train_hours if x is None else x
    y = train_scores if y is None else y
    errors = predict(x, m, b) - y
    grad_m = np.mean(2 * errors * x)
    grad_b = np.mean(2 * errors)
    return grad_m, grad_b


def evaluate(m, b, x, y):
    """Score a line on a set of students, in three ways.

    mse  - what training minimises, but in squared points, so hard to feel
    rmse - the same thing back in exam points: "typically off by this much"
    r2   - fraction of the variation explained. 1.0 is perfect, 0.0 is no
           better than always guessing the average, and negative means
           worse than that (which held-out data really can be).
    """
    errors = predict(x, m, b) - y
    mse = float(np.mean(errors ** 2))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "mse": mse,
        "rmse": math.sqrt(mse) if math.isfinite(mse) and mse >= 0 else float("nan"),
        "r2": 1 - ss_res / ss_tot if ss_tot else float("nan"),
    }


def train(learning_rate=0.01, steps=500, verbose=True, print_every=50):
    """Fit m and b by gradient descent, on the training split only.

    Returns (history, m, b). history is a list of TrainingStep, one per
    step plus a final entry for the finished model, so the last row
    really is the answer you're given.
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
            history.append(TrainingStep(
                step, m, b,
                mse_loss(m, b),
                # Measured every step but never used to update the
                # weights - the moment test data influences training, it
                # stops being a fair test.
                mse_loss(m, b, test_hours, test_scores),
            ))

            grad_m, grad_b = gradients(m, b)
            # Same update rule as before, just two numbers instead of one.
            m = m - learning_rate * grad_m
            b = b - learning_rate * grad_b

            if verbose and step % print_every == 0:
                row = history[-1]
                print(f"step {step:>4} | m={m:6.3f}  b={b:6.3f}  | "
                      f"train loss={row.loss:8.3f}  test loss={row.test_loss:8.3f}")

            if not (math.isfinite(m) and math.isfinite(b)):
                # A learning rate too big for this data makes the weights
                # explode to inf/NaN. Stop rather than filling the history
                # with meaningless numbers.
                break

        # Record where training actually ended up. Without this the last row
        # is the state BEFORE the final update, and "final loss" is a lie by
        # one step.
        history.append(TrainingStep(
            len(history), m, b,
            mse_loss(m, b),
            mse_loss(m, b, test_hours, test_scores),
        ))

    if verbose:
        print(f"\nFinal:  m={m:.3f}, b={b:.3f}")
        print(f"Truth:  m={true_m:.3f}, b={true_b:.3f}  (what generated the data, minus noise)")
        print(f"Trained on {len(train_hours)} students, scored on {len(test_hours)} it never saw.")
    return history, m, b


def closed_form_solution(x=None, y=None):
    """The exact best-fit line, solved algebraically instead of searched for.

    For a straight line there IS a formula for the optimal m and b, so we
    can check gradient descent's homework: it should creep toward exactly
    these numbers and never beat them.

    The reason this matters: for a neural network (or Qwen) no such
    formula exists, which is precisely why gradient descent - a method
    that only ever needs the local slope - is the thing that scales.
    """
    x = train_hours if x is None else x
    y = train_scores if y is None else y
    x_mean, y_mean = np.mean(x), np.mean(y)
    m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b = y_mean - m * x_mean
    return float(m), float(b)


def diagnose(history):
    """Turn a training run into a plain-language verdict, shared by the
    terminal trainer and the web UI."""
    first_loss, final_loss = history[0].loss, history[-1].loss
    final_test = history[-1].test_loss

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
        verdict = (
            f"Converged. Training loss {final_loss:.3f} is within 1% of "
            f"{best_loss:.3f}, the best any straight line can do on the "
            "training students - confirmed by solving for it exactly in "
            "closed_form_solution()."
        )
    elif final_loss > first_loss * 0.5:
        verdict = (
            f"Barely moved: loss went {first_loss:.1f} -> {final_loss:.1f}, "
            f"still far off the achievable {best_loss:.3f}. Usually that "
            "means the learning rate is too small to cover the distance in "
            "this many steps. Raise it, or train for longer."
        )
    else:
        verdict = (
            f"Making progress but not finished: loss {first_loss:.1f} -> "
            f"{final_loss:.3f}, against {best_loss:.3f} for the best possible "
            "line. More steps (or a slightly larger learning rate) will close "
            "the gap."
        )

    return f"{verdict}\n\n{generalization_verdict(final_loss, final_test)}"


def generalization_verdict(train_loss, test_loss):
    """The question that actually matters: does it work on new students?"""
    if not math.isfinite(test_loss):
        return "Test loss isn't a finite number, so there's nothing to compare."

    gap = test_loss - train_loss
    summary = (f"Train loss {train_loss:.3f} vs test loss {test_loss:.3f} "
               f"(gap {gap:+.3f}). ")

    if test_loss <= train_loss * 1.5:
        return summary + (
            "Those are close, which means the model genuinely learned the "
            "pattern rather than memorising particular students. Note WHY "
            "it can't memorise: a straight line only has two numbers to "
            "store anything in. Give a model enough freedom and it will "
            "memorise instead - that's what the overfitting demo shows."
        )
    if test_loss <= train_loss * 3:
        return summary + (
            "The model does noticeably worse on students it never saw. Some "
            "of that is luck - only a handful are held out - but this is the "
            "gap to watch: it is the difference between fitting and working."
        )
    return summary + (
        "The model is far worse on new students than on its own training "
        "data. That is overfitting: it has learned the noise in these "
        "particular students rather than the trend underneath."
    )


if __name__ == "__main__":
    train()
