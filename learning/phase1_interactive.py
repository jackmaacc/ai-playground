"""
Interactive wrapper around the linear regression model from
phase1_linear_regression.py: train it with your own settings, test
predictions, and get plain-language explanations of what's going on.

Run it with: python phase1_interactive.py
"""

from phase1_linear_regression import hours, scores, predict, mse_loss, gradients, true_m, true_b

# Model state, lives for the duration of this interactive session.
state = {"m": 0.0, "b": 0.0, "trained": False, "loss_history": []}


def train_model():
    print("\n--- Train ---")
    lr = float(input("learning rate (try 0.01 to start): ") or 0.01)
    steps = int(input("number of steps (try 500): ") or 500)

    m, b = 0.0, 0.0
    loss_history = []
    for step in range(steps):
        loss = mse_loss(m, b)
        loss_history.append(loss)
        grad_m, grad_b = gradients(m, b)
        m = m - lr * grad_m
        b = b - lr * grad_b

    state["m"], state["b"] = m, b
    state["trained"] = True
    state["loss_history"] = loss_history

    print(f"\nDone. Final loss: {loss_history[-1]:.3f}  (started at {loss_history[0]:.3f})")
    print(f"Learned:  score = {m:.3f} * hours + {b:.3f}")

    # Teach, don't just report: tell them what actually happened based on
    # the real numbers this run produced, not a canned message.
    if loss_history[-1] > loss_history[0] * 0.5:
        print("Note: loss barely dropped. That usually means the learning rate")
        print("was too small to make real progress in this many steps, or too")
        print("large and it's bouncing around without settling. Try changing")
        print("either the learning rate or the step count and retrain.")
    elif loss_history[-1] != loss_history[-1]:  # NaN check
        print("Note: loss became NaN - the learning rate was too high and the")
        print("update blew up. Try a smaller learning rate.")
    else:
        print("This converged well - loss dropped steadily and leveled off.")


def test_model():
    if not state["trained"]:
        print("\nTrain the model first (option 1) - it needs values for m and b.")
        return

    print("\n--- Test ---")
    h = float(input("hours studied to predict for: "))
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

    first, last = state["loss_history"][0], state["loss_history"][-1]
    print(f"Loss went from {first:.1f} to {last:.1f} over training - each step, gradient")
    print(f"descent nudged m and b in whichever direction reduced that error.")


def menu():
    print("\n" + "=" * 50)
    print("Linear regression trainer: predict exam score from hours studied")
    print("=" * 50)
    print("1) Train the model")
    print("2) Test a prediction")
    print("3) Explain what the model learned")
    print("4) Quit")


if __name__ == "__main__":
    while True:
        menu()
        choice = input("choose (1-4): ").strip()
        if choice == "1":
            train_model()
        elif choice == "2":
            test_model()
        elif choice == "3":
            explain_model()
        elif choice == "4":
            break
        else:
            print("Not a valid option, try again.")
