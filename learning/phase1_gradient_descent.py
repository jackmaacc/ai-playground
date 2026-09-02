"""
Gradient descent, from scratch, on the simplest possible function.

The big idea: every "trained" model (your local Qwen model included) got its
weights by doing this exact loop, millions of times, on a much more
complicated function. If you understand this file, you understand the core
mechanic of training - everything else (backprop, Adam, LoRA...) is this
same idea plus bookkeeping.

This is the one place the algorithm lives. The terminal menu, the plots in
phase1_gradient_descent_viz.py and the web UI all call gradient_descent()
here rather than each keeping their own copy of the loop.
"""

import math

import console

# Where f actually bottoms out. Used for the "how did we do?" reporting -
# the algorithm itself never gets told this.
MINIMUM = 3.0

# If |x| passes this, the run has diverged and is never coming back. Worth
# stopping there: f(x) would eventually overflow to a hard crash, and
# there's nothing to learn from watching inf scroll past.
DIVERGENCE_LIMIT = 1e12


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


def gradient_descent(start_x, learning_rate, steps, verbose=True, print_every=1):
    """Walk downhill from start_x and return every x visited along the way.

    Returning the whole trajectory (rather than just the final x) is what
    lets the plots draw the path and lets diagnose() work out WHY a run
    behaved the way it did.
    """
    x = float(start_x)
    history = [x]

    if verbose:
        print(f"{'step':>4} | {'x':>12} | {'f(x)':>14} | {'slope f\'(x)':>14}")
        print("-" * 52)

    for step in range(steps):
        slope = f_prime(x)

        # Printing every step is unreadable once you ask for thousands of
        # them, so long runs get sampled instead - but always show the last
        # one, since that's the answer.
        if verbose and (step % print_every == 0 or step == steps - 1):
            print(f"{step:>4} | {x:>12.5f} | {f(x):>14.5f} | {slope:>14.5f}")

        # THE ENTIRE ALGORITHM IS THIS ONE LINE:
        # move x a little bit in the direction that DECREASES f.
        # The slope points uphill, so we step in the OPPOSITE direction
        # (hence the minus sign), scaled by how big a step we're willing
        # to take (the learning rate).
        x = x - learning_rate * slope

        history.append(x)

        if not math.isfinite(x) or abs(x) > DIVERGENCE_LIMIT:
            if verbose:
                print(f"\nStopped early at step {step + 1}: x has run away to {x:.3e}.")
            break

    if verbose:
        print(f"\nFinal x = {history[-1]:.5f}  (true minimum is x = {MINIMUM})")
    return history


def diagnose(history):
    """Explain what the learning rate actually did, reading it off the
    trajectory rather than guessing.

    Both the terminal menu and the web UI call this, so the two teach
    exactly the same lesson instead of drifting apart.
    """
    final_x = history[-1]
    start_distance = abs(history[0] - MINIMUM)
    distance = abs(final_x - MINIMUM)

    # Did it repeatedly cross the minimum, or creep up on it from one side?
    # That distinction is the whole difference between "too big" and "too
    # small", and you can read it straight off the sign of (x - minimum).
    offsets = [x - MINIMUM for x in history[-6:]]
    crossings = sum(1 for a, b in zip(offsets, offsets[1:]) if a * b < 0)

    # The real test for divergence isn't "did the number get huge" - it's
    # "is it getting FURTHER away every step". A run can be visibly
    # diverging long before it overflows anything.
    if not math.isfinite(final_x) or distance > start_distance:
        return (
            f"Diverged - it ended up {distance:.3g} away from x={MINIMUM:.0f}, "
            f"having started {start_distance:.3g} away. The learning rate is "
            "big enough that every step overshoots by more than it was off by "
            "to begin with, so the error compounds. For "
            f"f(x)=(x-{MINIMUM:.0f})^2 anything above lr=1.0 does this, no "
            "matter how many steps you give it."
        )

    if distance < 0.01:
        return (
            f"Converged: within {distance:.5f} of the true minimum "
            f"(x={MINIMUM:.0f}). Notice the slope shrinking as it got closer - "
            "the steps get smaller automatically near the bottom, which is why "
            "it settles instead of orbiting forever."
        )

    # lr = 1.0 exactly is a genuine curiosity: each step lands the same
    # distance the other side of the minimum, so it cycles forever.
    if crossings >= 2 and abs(distance - start_distance) < start_distance * 0.01:
        return (
            f"Stuck in a perfect cycle, {distance:.3f} either side of "
            f"x={MINIMUM:.0f}. At exactly lr=1.0 each step lands the same "
            "distance past the minimum as it started, so it bounces between "
            "two points forever - never diverging, never arriving."
        )

    progress = 1 - (distance / start_distance) if start_distance else 1.0

    if crossings >= 2:
        return (
            f"Overshooting but still closing in - {distance:.3f} from "
            f"x={MINIMUM:.0f}. It leaps over the minimum and lands on the "
            "other side each time, because the steps are slightly too big. It "
            "gets there eventually; a smaller learning rate gets there "
            "straighter."
        )

    if progress > 0.8:
        return (
            f"On track - closed {progress:.0%} of the distance, {distance:.3f} "
            f"still to go. Nothing is wrong here: each step shrinks along with "
            "the slope, so the last stretch is always the slowest. Give it "
            "more steps and it lands on the minimum."
        )

    return (
        f"Crawling - only {progress:.0%} of the way there, still {distance:.3f} "
        f"from x={MINIMUM:.0f}, approaching from one side without ever "
        "overshooting. The direction is right, the steps are just too small. "
        "Raise the learning rate or the step count."
    )


def demo():
    print("=== Run 1: a well-behaved learning rate ===")
    gradient_descent(start_x=0.0, learning_rate=0.1, steps=15)

    print("\n=== Run 2: learning rate too SMALL (watch it crawl) ===")
    gradient_descent(start_x=0.0, learning_rate=0.01, steps=15)

    print("\n=== Run 3: learning rate too BIG (watch it overshoot / blow up) ===")
    gradient_descent(start_x=0.0, learning_rate=1.05, steps=15)


def custom_run():
    start_x = console.ask_float("start_x", default=0.0)
    learning_rate = console.ask_float("learning rate", default=0.1)
    steps = console.ask_int("steps", default=15, minimum=1)

    # Keep the table readable no matter how many steps were asked for.
    print_every = max(1, steps // 20)
    history = gradient_descent(start_x, learning_rate, steps, print_every=print_every)
    print(f"\n{diagnose(history)}")


def run():
    console.run_menu(
        "Gradient descent: minimize f(x) = (x-3)^2",
        [
            ("Run the canned demo (good / too small / too big learning rates)", demo),
            ("Run with your own start_x, learning rate, steps", custom_run),
        ],
    )


if __name__ == "__main__":
    demo()
