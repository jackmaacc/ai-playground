"""Gradient descent: the algorithm and every verdict diagnose() can give."""

import contextlib
import io
import math
import sys
import unittest
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parent.parent
if str(LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(LEARNING_DIR))

import phase1_gradient_descent as gd  # noqa: E402


def verdict(learning_rate, steps, start=0.0):
    history = gd.gradient_descent(start, learning_rate, steps, verbose=False)
    return gd.diagnose(history)


class TheFunction(unittest.TestCase):
    def test_minimum_is_where_we_say_it_is(self):
        self.assertEqual(gd.f(gd.MINIMUM), 0)
        self.assertEqual(gd.f_prime(gd.MINIMUM), 0)

    def test_slope_points_uphill(self):
        self.assertGreater(gd.f_prime(gd.MINIMUM + 1), 0)
        self.assertLess(gd.f_prime(gd.MINIMUM - 1), 0)


class Mechanics(unittest.TestCase):
    def test_history_begins_at_the_starting_point(self):
        history = gd.gradient_descent(-4.0, 0.1, 5, verbose=False)
        self.assertEqual(history[0], -4.0)

    def test_one_entry_per_step_plus_the_start(self):
        history = gd.gradient_descent(0.0, 0.1, 25, verbose=False)
        self.assertEqual(len(history), 26)

    def test_each_step_moves_downhill_when_stable(self):
        history = gd.gradient_descent(0.0, 0.1, 10, verbose=False)
        values = [gd.f(x) for x in history]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_divergence_stops_early_instead_of_overflowing(self):
        """lr=5 for thousands of steps used to reach OverflowError."""
        history = gd.gradient_descent(0.0, 5.0, 4000, verbose=False)
        self.assertLess(len(history), 4001)
        self.assertTrue(all(math.isfinite(x) for x in history))

    def test_verbose_false_is_silent(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            gd.gradient_descent(0.0, 0.1, 5, verbose=False)
        self.assertEqual(buffer.getvalue(), "")

    def test_print_every_samples_the_table(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            gd.gradient_descent(0.0, 0.1, 100, verbose=True, print_every=50)
        rows = [line for line in buffer.getvalue().splitlines()
                if line[:4].strip().isdigit()]
        # steps 0, 50 and the final step 99
        self.assertEqual(len(rows), 3)


class Diagnoses(unittest.TestCase):
    """The scenarios that exposed two real bugs earlier in this project."""

    def test_converged(self):
        self.assertTrue(verdict(0.1, 60).startswith("Converged"))

    def test_on_track_but_not_finished_is_not_called_crawling(self):
        self.assertTrue(verdict(0.1, 15).startswith("On track"))

    def test_crawling(self):
        self.assertTrue(verdict(0.01, 15).startswith("Crawling"))

    def test_overshooting_but_closing_in(self):
        self.assertTrue(verdict(0.95, 40).startswith("Overshooting"))

    def test_diverged_is_detected_by_growing_distance_not_magnitude(self):
        """lr=1.05 over 15 steps is only ~12 away - the old magnitude
        check called this 'oscillating, it'll get there eventually'."""
        self.assertTrue(verdict(1.05, 15).startswith("Diverged"))

    def test_exact_lr_one_is_a_perfect_cycle(self):
        self.assertTrue(verdict(1.0, 20).startswith("Stuck in a perfect cycle"))

    def test_starting_on_the_minimum_counts_as_converged(self):
        self.assertTrue(verdict(0.1, 5, start=gd.MINIMUM).startswith("Converged"))


if __name__ == "__main__":
    unittest.main()
