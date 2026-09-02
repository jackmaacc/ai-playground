"""
Interactive wrapper around the linear regression model from
phase1_linear_regression.py: train it with your own settings, test
predictions, and get plain-language explanations of what's going on.

The maths all lives in phase1_linear_regression.py - this file only
handles asking you questions and explaining the answers.

Run it with: python phase1_interactive.py
"""

import console
from phase1_linear_regression import (
    closed_form_solution,
    diagnose,
    evaluate,
    generalization_verdict,
    mse_loss,
    predict,
    test_hours,
    test_scores,
    train,
    train_hours,
    train_scores,
    true_b,
    true_m,
)


def print_metrics(m, b):
    """Score the model on both splits, side by side.

    RMSE is the one to read: it's in exam points, so "3.8" means the
    model is typically about four points out.
    """
    train_metrics = evaluate(m, b, train_hours, train_scores)
    test_metrics = evaluate(m, b, test_hours, test_scores)
    print(f"\n{'':<18}{'trained on':>12}{'never seen':>12}")
    print(f"{'students':<18}{len(train_hours):>12}{len(test_hours):>12}")
    for key, label in (("rmse", "RMSE (points)"), ("mse", "MSE"), ("r2", "R2")):
        print(f"{label:<18}{train_metrics[key]:>12.3f}{test_metrics[key]:>12.3f}")

# Model state, lives for the duration of this interactive session.
state = {"m": 0.0, "b": 0.0, "trained": False, "history": []}


def train_model():
    print("\n--- Train ---")
    learning_rate = console.ask_float("learning rate (try 0.01 to start)", default=0.01)
    steps = console.ask_int("number of steps (try 500)", default=500, minimum=1)

    history, m, b = train(learning_rate, steps, verbose=False)

    state.update({"m": m, "b": b, "trained": True, "history": history})

    first_loss, final_loss = history[0].loss, history[-1].loss
    print(f"\nDone. Final training loss: {final_loss:.3f}  (started at {first_loss:.3f})")
    print(f"Learned:  score = {m:.3f} * hours + {b:.3f}")
    print_metrics(m, b)

    # Teach, don't just report: tell them what actually happened based on
    # the real numbers this run produced, not a canned message.
    print(f"\n{diagnose(history)}")


def compare_learning_rates():
    """Same data, same steps, two learning rates - the fastest way to feel
    what the learning rate actually controls."""
    print("\n--- Compare two learning rates ---")
    lr_a = console.ask_float("learning rate A", default=0.001)
    lr_b = console.ask_float("learning rate B", default=0.02)
    steps = console.ask_int("steps for both", default=500, minimum=1)

    print()
    for label, learning_rate in (("A", lr_a), ("B", lr_b)):
        history, m, b = train(learning_rate, steps, verbose=False)
        row = history[-1]
        print(f"=== {label}: lr={learning_rate} ===")
        print(f"  score = {m:.3f} * hours + {b:.3f}")
        print(f"  train loss {row.loss:.3f}   test loss {row.test_loss:.3f}")
        print(f"  {diagnose(history)}\n")

    print(
        "Identical data, identical starting point, identical number of steps.\n"
        "The only thing that differed was how far each step moved - and that\n"
        "alone decides whether training finishes, crawls, or explodes."
    )


def test_model():
    if not state["trained"]:
        print("\nTrain the model first (option 1) - it needs values for m and b.")
        return

    print("\n--- Test ---")
    h = console.ask_float("hours studied to predict for", default=5.0)
    m, b = state["m"], state["b"]
    predicted = predict(h, m, b)

    print(f"\npredicted score = m * hours + b")
    print(f"                = {m:.3f} * {h} + {b:.3f}")
    print(f"                = {predicted:.3f}")

    true_predicted = predict(h, true_m, true_b)
    print(f"\n(For reference: the relationship the fake data was generated from")
    print(f"would predict {true_predicted:.3f} here - your model's guess should be close but")
    print(f"not identical, since it only ever saw noisy data, never the true rule.)")


def explain_model():
    print("\n--- Explain ---")
    if not state["trained"]:
        print("Nothing trained yet, so there's nothing to explain. Run option 1 first.")
        return

    m, b = state["m"], state["b"]
    print(f"Your model learned: score = {m:.3f} * hours + {b:.3f}\n")
    print(f"Slope (m = {m:.3f}): every extra hour of studying is associated with")
    print(f"about {m:.2f} more points on the exam, according to the data it trained on.\n")
    print(f"Intercept (b = {b:.3f}): a student who studied 0 hours is predicted")
    print(f"to score about {b:.2f}.\n")

    print(f"How close is this to the truth? The data was actually generated from")
    print(f"score = {true_m} * hours + {true_b}, plus random noise (real students never")
    print(f"land exactly on a line). Your model recovered m={m:.2f} vs true m={true_m},")
    print(f"and b={b:.2f} vs true b={true_b} - it's estimating the REAL underlying pattern")
    print(f"despite only seeing noisy examples of it. That's the whole point of training:")
    print(f"find the signal underneath the noise.\n")

    # Was this the best it COULD have done? For a straight line we can
    # answer that exactly, which is a luxury you never get with a real
    # network - worth seeing at least once.
    best_m, best_b = closed_form_solution()
    print(f"Best possible line (solved algebraically, no training needed):")
    print(f"  score = {best_m:.3f} * hours + {best_b:.3f}   loss {mse_loss(best_m, best_b):.3f}")
    print(f"Your trained line:")
    print(f"  score = {m:.3f} * hours + {b:.3f}   loss {mse_loss(m, b):.3f}\n")
    print("Gradient descent is CREEPING toward that exact answer. It's slower")
    print("than the formula here - but the formula only exists for simple")
    print("models like this one. There's no closed form for Qwen's billions of")
    print("weights, so the slow, general method is the one that scales.\n")

    first, last = state["history"][0].loss, state["history"][-1].loss
    print(f"Loss went from {first:.1f} to {last:.1f} over training - each step, gradient")
    print(f"descent nudged m and b in whichever direction reduced that error.\n")

    # The part that actually decides whether a model is any good.
    print("--- Does it work on students it never saw? ---")
    print_metrics(m, b)
    print(f"\n{generalization_verdict(mse_loss(m, b), mse_loss(m, b, test_hours, test_scores))}\n")
    print(f"Every number in the 'trained on' column is measured on the "
          f"{len(train_hours)} students")
    print(f"this model already learned from, so it is flattering by construction.")
    print(f"The {len(test_hours)} students in the other column were held back before training")
    print("started and never influenced a single weight update - which is the")
    print("only reason that column means anything. Judging a model on its own")
    print("training data is the most common way to fool yourself in ML.")


def run():
    console.run_menu(
        "Linear regression trainer: predict exam score from hours studied",
        [
            ("Train the model", train_model),
            ("Test a prediction", test_model),
            ("Explain what the model learned", explain_model),
            ("Compare two learning rates", compare_learning_rates),
        ],
        back_label="Quit",
    )


if __name__ == "__main__":
    run()
