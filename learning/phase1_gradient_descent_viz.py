"""
Visualizations for phase1_gradient_descent.py.

Reuses the real gradient_descent() from that file - it already returns the
list of x values it visited, so there's nothing to re-implement here. This
file only turns that trajectory into pictures.

Produces two PNGs next to this script:
  - gradient_descent_paths.png        the "ball rolling down the bowl" view
  - gradient_descent_convergence.png  x (and distance-to-minimum) vs step
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # writing files, not opening windows
import matplotlib.pyplot as plt
import numpy as np

from phase1_gradient_descent import MINIMUM, f, gradient_descent

# Save next to this file, not wherever the program happened to be started
# from - main.py can be launched from anywhere.
OUT_DIR = Path(__file__).parent

SCENARIOS = [
    ("lr = 0.1  (well-behaved)", 0.0, 0.1, 15),
    ("lr = 0.01 (too small - crawls)", 0.0, 0.01, 15),
    ("lr = 1.05 (too big - diverges)", 0.0, 1.05, 15),
]


def save(fig, filename):
    path = OUT_DIR / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)  # a menu can run this repeatedly; don't leak figures
    print(f"saved {path}")


def plot_paths(all_histories):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (label, history) in zip(axes, all_histories):
        # Draw the actual curve f(x) = (x-3)^2 across whatever range this
        # particular run visited, with a little padding.
        lo = min(min(history), MINIMUM) - 1
        hi = max(max(history), MINIMUM) + 1
        xs = np.linspace(lo, hi, 300)
        ax.plot(xs, f(xs), color="lightgray", linewidth=2, zorder=1)

        # Overlay the path gradient descent actually took, step by step.
        ys = [f(x) for x in history]
        ax.plot(history, ys, "o-", color="crimson", markersize=4, linewidth=1, zorder=2)
        ax.scatter([history[0]], [f(history[0])], color="black", zorder=3, label="start")
        ax.scatter([MINIMUM], [f(MINIMUM)], color="green", marker="*",
                   s=150, zorder=3, label=f"true minimum (x={MINIMUM:.0f})")

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend(fontsize=8)

    fig.suptitle("Gradient descent: the path taken across the loss curve", fontsize=13)
    fig.tight_layout()
    save(fig, "gradient_descent_paths.png")


def plot_convergence(all_histories):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for label, history in all_histories:
        steps = list(range(len(history)))
        ax1.plot(steps, history, "o-", markersize=3, label=label)

        distance = [abs(x - MINIMUM) for x in history]
        ax2.plot(steps, distance, "o-", markersize=3, label=label)

    ax1.axhline(MINIMUM, color="green", linestyle="--", linewidth=1, label="true minimum")
    ax1.set_xlabel("step")
    ax1.set_ylabel("x")
    ax1.set_title("x over time")
    ax1.legend(fontsize=8)

    ax2.set_yscale("log")
    ax2.set_xlabel("step")
    ax2.set_ylabel(f"|x - {MINIMUM:.0f}|  (log scale)")
    ax2.set_title("distance from the true minimum")
    ax2.legend(fontsize=8)

    fig.suptitle("Convergence, crawling, and divergence - same algorithm, different learning rate", fontsize=12)
    fig.tight_layout()
    save(fig, "gradient_descent_convergence.png")


def run():
    all_histories = [
        (label, gradient_descent(start_x, lr, steps, verbose=False))
        for label, start_x, lr, steps in SCENARIOS
    ]
    plot_paths(all_histories)
    plot_convergence(all_histories)


if __name__ == "__main__":
    run()
