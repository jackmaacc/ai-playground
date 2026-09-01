"""
Gradient descent, from scratch, on the simplest possible function.

The big idea: every "trained" model (your local Qwen model included) got its
weights by doing this exact loop, millions of times, on a much more
complicated function. If you understand this file, you understand the core
mechanic of training - everything else (backprop, Adam, LoRA...) is this
same idea plus bookkeeping.
"""

# f(x) = (x - 3)^2
# A parabola. Its minimum is obviously at x = 3 just by looking at it -
# that's the whole point of picking something this simple. We're not going
# to "solve" for the minimum algebraically. We're going to let the computer
# FIND it by repeatedly asking "which way is downhill from here?" and
# stepping that way. That's gradient descent.
def f(x):
    return (x - 3) ** 2


# The derivative of (x-3)^2 is 2(x-3), by the power/chain rule.
# f'(x) tells you the SLOPE of f at point x:
#   - positive slope -> f is increasing here -> "downhill" is to the LEFT (smaller x)
#   - negative slope -> f is increasing to the left -> "downhill" is to the RIGHT
#   - slope = 0 -> you're at a flat point (the minimum, for this function)
def f_prime(x):
    return 2 * (x - 3)


def gradient_descent(start_x, learning_rate, steps):
    x = start_x
    print(f"{'step':>4} | {'x':>10} | {'f(x)':>10} | {'slope f\'(x)':>12}")
    print("-" * 46)

    for step in range(steps):
        slope = f_prime(x)
        print(f"{step:>4} | {x:>10.5f} | {f(x):>10.5f} | {slope:>12.5f}")

        # THE ENTIRE ALGORITHM IS THIS ONE LINE:
        # move x a little bit in the direction that DECREASES f.
        # The slope points uphill, so we step in the OPPOSITE direction
        # (hence the minus sign), scaled by how big a step we're willing
        # to take (the learning rate).
        x = x - learning_rate * slope

    print(f"\nFinal x = {x:.5f}  (true minimum is x = 3)")
    return x


def demo():
    print("=== Run 1: a well-behaved learning rate ===")
    gradient_descent(start_x=0.0, learning_rate=0.1, steps=15)

    print("\n=== Run 2: learning rate too SMALL (watch it crawl) ===")
    gradient_descent(start_x=0.0, learning_rate=0.01, steps=15)

    print("\n=== Run 3: learning rate too BIG (watch it overshoot / blow up) ===")
    gradient_descent(start_x=0.0, learning_rate=1.05, steps=15)


def run():
    """Interactive menu: run the canned demo or try your own numbers."""
    print("\n" + "=" * 50)
    print("Gradient descent: minimize f(x) = (x-3)^2")
    print("=" * 50)
    while True:
        print("\n1) Run the canned demo (good / too small / too big learning rates)")
        print("2) Run with your own start_x, learning rate, steps")
        print("3) Back")
        choice = input("choose (1-3): ").strip()
        if choice == "1":
            demo()
        elif choice == "2":
            start_x = float(input("start_x (try 0): ") or 0)
            learning_rate = float(input("learning rate (try 0.1): ") or 0.1)
            steps = int(input("steps (try 15): ") or 15)
            final_x = gradient_descent(start_x, learning_rate, steps)

            distance = abs(final_x - 3)
            if distance != distance:  # NaN check
                print("\nThat diverged to NaN - the learning rate was too big and the")
                print("update blew up. Try something smaller.")
            elif distance > 0.5:
                print(f"\nStill {distance:.3f} away from the true minimum (x=3) after")
                print(f"{steps} steps. Either the learning rate is too small to get")
                print("there in time, or too big and it's oscillating around the")
                print("minimum instead of settling. Try adjusting it.")
            else:
                print(f"\nConverged: within {distance:.4f} of the true minimum (x=3).")
        elif choice == "3":
            return
        else:
            print("Not a valid option, try again.")


if __name__ == "__main__":
    demo()
