"""
Visualizations for phase1_gradient_descent.py.

Reuses f() and f_prime() from that file (the actual math), just adds a
version of the loop that records every step so we can plot it, instead of
only printing it.

Produces two PNGs in this folder:
  - gradient_descent_paths.png        the "ball rolling down the bowl" view
  - gradient_descent_convergence.png  x (and distance-to-minimum) vs step
"""

import matplotlib.pyplot as plt
import numpy as np

from phase1_gradient_descent import f, f_prime

TRUE_MINIMUM = 3.0

SCENARIOS = [
    ("lr = 0.1  (well-behaved)", 0.0, 0.1, 15),
    ("lr = 0.01 (too small - crawls)", 0.0, 0.01, 15),
    ("lr = 1.05 (too big - diverges)", 0.0, 1.05, 15),
]


def gradient_descent_history(start_x, learning_rate, steps):
    """Same algorithm as gradient_descent() in phase1_gradient_descent.py,
    but returns the list of x-values visited instead of printing them."""
    x = start_x
    history = [x]
    for _ in range(steps):
        x = x - learning_rate * f_prime(x)
        history.append(x)
    return history


def plot_paths(all_histories):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (label, history) in zip(axes, all_histories):
        # Draw the actual curve f(x) = (x-3)^2 across whatever range this
        # particular run visited, with a little padding.
        lo = min(min(history), TRUE_MINIMUM) - 1
        hi = max(max(history), TRUE_MINIMUM) + 1
        xs = np.linspace(lo, hi, 300)
        ax.plot(xs, f(xs), color="lightgray", linewidth=2, zorder=1)

        # Overlay the path gradient descent actually took, step by step.
        ys = [f(x) for x in history]
        ax.plot(history, ys, "o-", color="crimson", markersize=4, linewidth=1, zorder=2)
        ax.scatter([history[0]], [f(history[0])], color="black", zorder=3, label="start")
        ax.scatter([TRUE_MINIMUM], [f(TRUE_MINIMUM)], color="green", marker="*",
                   s=150, zorder=3, label="true minimum (x=3)")

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend(fontsize=8)

    fig.suptitle("Gradient descent: the path taken across the loss curve", fontsize=13)
    fig.tight_layout()
    out = "gradient_descent_paths.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


def plot_convergence(all_histories):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for label, history in all_histories:
        steps = list(range(len(history)))
        ax1.plot(steps, history, "o-", markersize=3, label=label)

        distance = [abs(x - TRUE_MINIMUM) for x in history]
        ax2.plot(steps, distance, "o-", markersize=3, label=label)

    ax1.axhline(TRUE_MINIMUM, color="green", linestyle="--", linewidth=1, label="true minimum")
    ax1.set_xlabel("step")
    ax1.set_ylabel("x")
    ax1.set_title("x over time")
    ax1.legend(fontsize=8)

    ax2.set_yscale("log")
    ax2.set_xlabel("step")
    ax2.set_ylabel("|x - 3|  (log scale)")
    ax2.set_title("distance from the true minimum")
    ax2.legend(fontsize=8)

    fig.suptitle("Convergence, crawling, and divergence - same algorithm, different learning rate", fontsize=12)
    fig.tight_layout()
    out = "gradient_descent_convergence.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


def run():
    all_histories = []
    for label, start_x, lr, steps in SCENARIOS:
        history = gradient_descent_history(start_x, lr, steps)
        all_histories.append((label, history))

    plot_paths(all_histories)
    plot_convergence(all_histories)


if __name__ == "__main__":
    run()
