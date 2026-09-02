"""
Overfitting: the moment a model stops learning and starts memorising.

phase1_linear_regression.py fits a straight line, which has exactly two
numbers to store knowledge in - far too few to memorise anything. Here we
hand the model more and more freedom (a curve of increasing degree) and
watch the two errors come apart:

  training error  - always falls. More freedom always fits seen data better.
  test error      - falls, bottoms out, then CLIMBS.

That U-shape is the most important picture in classical machine learning.
The bottom of it is the model you want; everything to the right is a model
that has memorised the noise in its training data.

Run with: python phase1_overfitting.py
"""

import math
from pathlib import Path
from typing import NamedTuple

import numpy as np

import console
from phase1_linear_regression import (
    test_hours,
    test_scores,
    train_hours,
    train_scores,
)

OUT_DIR = Path(__file__).parent
MAX_DEGREE = 12

# Rescale hours before building powers of it. Without this, degree 12 on
# hours up to 10 means columns spanning 1 to 10^12, and the fit becomes
# numerically hopeless - you would be looking at floating-point noise
# rather than overfitting.
#
# Note it is fitted on the TRAINING data only, then applied to both
# splits. Anything learned from the test set - even something as small as
# a mean - is a leak that makes the test look better than it is.
X_MEAN = float(np.mean(train_hours))
X_STD = float(np.std(train_hours))


def scale(x):
    return (np.asarray(x, dtype=float) - X_MEAN) / X_STD


def polynomial_features(x, degree):
    """Turn one number into [1, z, z^2, ... z^degree].

    This is the whole trick behind "nonlinear" models built out of linear
    machinery: the model is still just a weighted sum, it's the FEATURES
    that curve.
    """
    return np.vander(scale(x), degree + 1, increasing=True)


def fit_polynomial(x, y, degree):
    """Solve for the best coefficients directly, by least squares.

    Deliberately NOT gradient descent. Powers of x produce a wildly
    ill-conditioned system, and no learning rate converges on it: too
    small and it crawls, big enough to move and it diverges. You would
    end up blaming overfitting for what is really a conditioning problem.

    Using the exact solver removes the optimiser from the experiment, so
    the only thing varying is model freedom - which is the point.
    """
    coeffs, *_ = np.linalg.lstsq(polynomial_features(x, degree), y, rcond=None)
    return coeffs


def predict_polynomial(x, coeffs):
    return polynomial_features(x, len(coeffs) - 1) @ coeffs


def mse(coeffs, x, y):
    return float(np.mean((predict_polynomial(x, coeffs) - y) ** 2))


class DegreeResult(NamedTuple):
    degree: int
    train_mse: float
    test_mse: float
    coeffs: np.ndarray


def sweep(max_degree=MAX_DEGREE):
    """Fit every degree from a straight line up to an absurd curve."""
    results = []
    with np.errstate(over="ignore", invalid="ignore"):
        for degree in range(1, max_degree + 1):
            coeffs = fit_polynomial(train_hours, train_scores, degree)
            results.append(DegreeResult(
                degree,
                mse(coeffs, train_hours, train_scores),
                mse(coeffs, test_hours, test_scores),
                coeffs,
            ))
    return results


def best_result(results):
    """The degree that actually generalises best - chosen on test error,
    which is the only column that reflects unseen students."""
    finite = [r for r in results if math.isfinite(r.test_mse)]
    return min(finite, key=lambda r: r.test_mse) if finite else results[0]


def diagnose(results):
    best = best_result(results)
    simplest, most_complex = results[0], results[-1]

    lines = [
        f"Best on held-out students: degree {best.degree} "
        f"(test MSE {best.test_mse:.2f}).",
        "",
        f"Degree 1 (a straight line):  train {simplest.train_mse:8.2f}   "
        f"test {simplest.test_mse:8.2f}",
        f"Degree {most_complex.degree} (very wiggly):     "
        f"train {most_complex.train_mse:8.2f}   test {most_complex.test_mse:8.2f}",
        "",
    ]

    if most_complex.train_mse < simplest.train_mse:
        lines.append(
            "Training error fell as the model got more flexible - it always "
            "does. More freedom can always hug the points it can see."
        )
    if most_complex.test_mse > best.test_mse:
        factor = most_complex.test_mse / best.test_mse if best.test_mse else float("inf")
        lines.append(
            f"But test error at degree {most_complex.degree} is {factor:.1f}x worse "
            f"than at degree {best.degree}. The extra freedom went into "
            "reproducing the random noise in the training students, which is "
            "worse than useless on anyone else."
        )
        lines.append("")
        lines.append(
            "This is why a model is never judged on its training data. Had we "
            "only looked at the training column, degree "
            f"{most_complex.degree} would look like the best model here."
        )
    else:
        lines.append(
            "Test error hasn't turned upward yet - try raising MAX_DEGREE, or "
            "re-run with a different SPLIT_SEED; with only a handful of "
            "held-out students the curve can be lumpy."
        )
    return "\n".join(lines)


def print_table(results):
    print(f"\n{'degree':>6} | {'train MSE':>12} | {'test MSE':>12} | verdict")
    print("-" * 58)
    best = best_result(results)
    for row in results:
        note = "  <-- best on unseen data" if row.degree == best.degree else ""
        print(f"{row.degree:>6} | {row.train_mse:>12.3f} | {row.test_mse:>12.3f} |{note}")


def save(fig, filename):
    import matplotlib.pyplot as plt

    path = OUT_DIR / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved {path}")


def plot_curve(results):
    import matplotlib.pyplot as plt

    degrees = [r.degree for r in results]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(degrees, [r.train_mse for r in results], "o-", color="crimson",
            label="training error (data the model saw)")
    ax.plot(degrees, [r.test_mse for r in results], "s--", color="darkorange",
            label="test error (students held back)")

    best = best_result(results)
    ax.axvline(best.degree, color="green", linestyle=":", linewidth=1.5,
               label=f"best generalisation (degree {best.degree})")

    ax.set_yscale("log")
    ax.set_xlabel("polynomial degree (how much freedom the model has)")
    ax.set_ylabel("mean squared error (log scale)")
    ax.set_title("Training error always falls. Test error tells the truth.")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "overfitting_curve.png")


def plot_fits(results):
    import matplotlib.pyplot as plt

    best = best_result(results)
    chosen = sorted({1, best.degree, results[-1].degree})
    colors = ["tab:blue", "green", "crimson"]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(train_hours, train_scores, color="black", zorder=3, label="trained on")
    ax.scatter(test_hours, test_scores, color="darkorange", marker="s", s=60,
               zorder=4, edgecolor="black", linewidth=0.5, label="held out")

    xs = np.linspace(float(np.min(train_hours)), float(np.max(train_hours)), 400)
    for color, degree in zip(colors, chosen):
        row = next(r for r in results if r.degree == degree)
        label = f"degree {degree}"
        if degree == best.degree:
            label += " (best)"
        elif degree == results[-1].degree:
            label += " (overfit)"
        ax.plot(xs, predict_polynomial(xs, row.coeffs), color=color, linewidth=2, label=label)

    # The wiggly fit shoots far outside the plausible score range; clamp
    # the view so the sensible curves stay readable.
    ax.set_ylim(min(0, float(np.min(train_scores)) - 10), float(np.max(train_scores)) + 15)
    ax.set_xlabel("hours studied")
    ax.set_ylabel("exam score")
    ax.set_title("The same data, fitted with increasing freedom")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "overfitting_fits.png")


def show_table():
    results = sweep()
    print_table(results)
    print(f"\n{diagnose(results)}")


def make_plots():
    results = sweep()
    plot_curve(results)
    plot_fits(results)


def run():
    console.run_menu(
        "Overfitting: when more model stops meaning more skill",
        [
            ("Sweep polynomial degrees and show the numbers", show_table),
            ("Save the plots (U-curve + the fits themselves)", make_plots),
        ],
        subtitle=f"trained on {len(train_hours)} students, tested on {len(test_hours)} held back",
    )


if __name__ == "__main__":
    show_table()
    make_plots()
