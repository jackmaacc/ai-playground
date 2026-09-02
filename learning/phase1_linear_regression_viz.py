"""
Visualization for phase1_linear_regression.py.

Two PNGs, written next to this script:
  - linreg_fit.png    the data points + how the line evolves during training
  - linreg_loss.png   training loss AND held-out test loss over time
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # writing files, not opening windows
import matplotlib.pyplot as plt
import numpy as np

from phase1_linear_regression import (
    closed_form_solution,
    predict,
    test_hours,
    test_scores,
    train,
    train_hours,
    train_scores,
)

OUT_DIR = Path(__file__).parent


def save(fig, filename):
    path = OUT_DIR / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)  # a menu can run this repeatedly; don't leak figures
    print(f"saved {path}")


def plot_fit(history):
    fig, ax = plt.subplots(figsize=(7, 5))
    # Draw the two splits differently: the model only ever saw the black
    # ones. The orange ones are the exam it hasn't sat yet.
    ax.scatter(train_hours, train_scores, color="black", zorder=3, label="trained on")
    ax.scatter(test_hours, test_scores, color="darkorange", marker="s", s=60,
               zorder=4, edgecolor="black", linewidth=0.5, label="held out (never seen)")

    xs = np.linspace(0, 10, 100)
    # Show a few snapshots of the line as training progresses, fading in
    # color, so you can watch it rotate/shift from a bad guess to the fit.
    # Clamped so a short training run can't index past the end.
    snapshot_steps = sorted({min(step, len(history) - 1) for step in (0, 10, 50, 150, len(history) - 1)})
    colors = plt.cm.Blues(np.linspace(0.35, 1.0, len(snapshot_steps)))
    for color, step_idx in zip(colors, snapshot_steps):
        row = history[step_idx]
        ax.plot(xs, predict(xs, row.m, row.b), color=color, linewidth=2,
                label=f"step {step_idx}: m={row.m:.2f}, b={row.b:.2f}")

    # Where it was always heading: the exact least-squares line. Training
    # is converging on this, it can never beat it.
    best_m, best_b = closed_form_solution()
    ax.plot(xs, predict(xs, best_m, best_b), color="green", linestyle="--",
            linewidth=1.5, label=f"best possible: m={best_m:.2f}, b={best_b:.2f}")

    ax.set_xlabel("hours studied")
    ax.set_ylabel("exam score")
    ax.set_title("Gradient descent fitting a line to real data")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "linreg_fit.png")


def plot_loss(history):
    steps = [row.step for row in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, [row.loss for row in history], color="crimson",
            label="training loss (what training minimises)")
    ax.plot(steps, [row.test_loss for row in history], color="darkorange",
            linestyle="--", label="test loss (what we actually care about)")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (mean squared error)")
    ax.set_title("Loss going down = the line getting less wrong")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "linreg_loss.png")


def run():
    history, _, _ = train(learning_rate=0.01, steps=500)
    plot_fit(history)
    plot_loss(history)


if __name__ == "__main__":
    run()
