"""
Visualization for phase1_linear_regression.py.

Two PNGs:
  - linreg_fit.png    the data points + how the line evolves during training
  - linreg_loss.png   loss going down over training steps
"""

import matplotlib.pyplot as plt
import numpy as np

from phase1_linear_regression import hours, scores, predict, train


def plot_fit(history):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(hours, scores, color="black", zorder=3, label="real data (hours, score)")

    xs = np.linspace(0, 10, 100)
    # Show a few snapshots of the line as training progresses, fading in
    # color, so you can watch it rotate/shift from a bad guess to the fit.
    snapshot_steps = [0, 10, 50, 150, len(history) - 1]
    colors = plt.cm.Blues(np.linspace(0.35, 1.0, len(snapshot_steps)))
    for color, step_idx in zip(colors, snapshot_steps):
        _, m, b, _ = history[step_idx]
        ax.plot(xs, predict(xs, m, b), color=color, linewidth=2,
                 label=f"step {step_idx}: m={m:.2f}, b={b:.2f}")

    ax.set_xlabel("hours studied")
    ax.set_ylabel("exam score")
    ax.set_title("Gradient descent fitting a line to real data")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("linreg_fit.png", dpi=130)
    print("saved linreg_fit.png")


def plot_loss(history):
    steps = [h[0] for h in history]
    losses = [h[3] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, losses, color="crimson")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (mean squared error)")
    ax.set_title("Loss going down = the line getting less wrong")
    fig.tight_layout()
    fig.savefig("linreg_loss.png", dpi=130)
    print("saved linreg_loss.png")


def run():
    history, final_m, final_b = train(learning_rate=0.01, steps=500)
    plot_fit(history)
    plot_loss(history)


if __name__ == "__main__":
    run()
