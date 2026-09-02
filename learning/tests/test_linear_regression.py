"""Linear regression: the split, the metrics, and the one rule that
matters most - held-out students never influence a weight update."""

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

LEARNING_DIR = Path(__file__).resolve().parent.parent
if str(LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(LEARNING_DIR))

import phase1_linear_regression as lr  # noqa: E402


class TheSplit(unittest.TestCase):
    def test_sizes(self):
        self.assertEqual(len(lr.train_hours), 15)
        self.assertEqual(len(lr.test_hours), 5)

    def test_disjoint_and_complete(self):
        together = np.concatenate([lr.train_hours, lr.test_hours])
        self.assertEqual(sorted(together.tolist()), sorted(lr.hours.tolist()))
        self.assertEqual(len(set(together.tolist())), 20)

    def test_split_is_random_not_a_slice(self):
        """A slice off the sorted end would hold out only the highest
        hours and measure extrapolation instead of generalisation."""
        self.assertLess(lr.test_hours.min(), lr.train_hours.max())
        self.assertGreater(lr.test_hours.max(), lr.train_hours.min())

    def test_split_is_reproducible(self):
        first = lr.split_data()
        second = lr.split_data()
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_different_seed_gives_a_different_split(self):
        other_test_hours = lr.split_data(seed=1)[2]
        self.assertFalse(np.array_equal(other_test_hours, lr.test_hours))


class HeldOutDataNeverTrains(unittest.TestCase):
    def test_weights_are_identical_when_test_data_is_replaced(self):
        """If test data ever leaked into training, changing it would move
        the weights. It must not move them by a single bit."""
        history_a, m_a, b_a = lr.train(0.01, 200, verbose=False)

        poisoned = lr.test_scores * 0 + 999.0
        with mock.patch.object(lr, "test_scores", poisoned):
            history_b, m_b, b_b = lr.train(0.01, 200, verbose=False)

        self.assertEqual(m_a, m_b)
        self.assertEqual(b_a, b_b)
        self.assertEqual([row.loss for row in history_a],
                         [row.loss for row in history_b])
        # ...while the recorded test loss, which is allowed to look, differs.
        self.assertNotEqual(history_a[-1].test_loss, history_b[-1].test_loss)

    def test_history_has_both_losses_and_a_final_row(self):
        history, m, b = lr.train(0.01, 50, verbose=False)
        self.assertEqual(len(history), 51)
        final = history[-1]
        self.assertEqual((final.m, final.b), (m, b))
        self.assertAlmostEqual(final.loss, lr.mse_loss(m, b))
        self.assertAlmostEqual(final.test_loss,
                               lr.mse_loss(m, b, lr.test_hours, lr.test_scores))

    def test_history_rows_still_support_positional_access(self):
        row = lr.train(0.01, 5, verbose=False)[0][0]
        self.assertEqual(row[0], row.step)
        self.assertEqual(row[3], row.loss)


class Metrics(unittest.TestCase):
    def test_perfect_line_scores_perfectly(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = 2 * x + 1
        result = lr.evaluate(2.0, 1.0, x, y)
        self.assertEqual(result["mse"], 0.0)
        self.assertEqual(result["rmse"], 0.0)
        self.assertEqual(result["r2"], 1.0)

    def test_hand_computed_mse_and_rmse(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        result = lr.evaluate(0.0, 0.0, x, y)      # predicts 0 everywhere
        self.assertAlmostEqual(result["mse"], (1 + 4 + 9) / 3)
        self.assertAlmostEqual(result["rmse"], math.sqrt((1 + 4 + 9) / 3))

    def test_predicting_the_mean_gives_r2_of_zero(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        result = lr.evaluate(0.0, 2.0, x, y)      # flat line at the mean
        self.assertAlmostEqual(result["r2"], 0.0)

    def test_r2_goes_negative_for_a_line_worse_than_the_mean(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        result = lr.evaluate(-5.0, 0.0, x, y)
        self.assertLess(result["r2"], 0.0)

    def test_infinite_loss_does_not_crash_rmse(self):
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        result = lr.evaluate(float("inf"), 0.0, x, y)
        self.assertTrue(math.isnan(result["rmse"]) or math.isinf(result["rmse"]))


class ClosedForm(unittest.TestCase):
    def test_matches_numpy_least_squares(self):
        m, b = lr.closed_form_solution()
        ref_m, ref_b = np.polyfit(lr.train_hours, lr.train_scores, 1)
        self.assertAlmostEqual(m, ref_m, places=9)
        self.assertAlmostEqual(b, ref_b, places=9)

    def test_gradient_descent_approaches_but_never_beats_it(self):
        best_loss = lr.mse_loss(*lr.closed_form_solution())
        _, m, b = lr.train(0.01, 3000, verbose=False)
        trained_loss = lr.mse_loss(m, b)
        self.assertGreaterEqual(trained_loss, best_loss - 1e-9)
        self.assertLess(trained_loss, best_loss * 1.01)

    def test_defaults_to_the_training_split(self):
        self.assertEqual(lr.closed_form_solution(),
                         lr.closed_form_solution(lr.train_hours, lr.train_scores))


class Diagnoses(unittest.TestCase):
    def test_converged_run(self):
        history, _, _ = lr.train(0.01, 3000, verbose=False)
        self.assertTrue(lr.diagnose(history).startswith("Converged"))

    def test_rising_loss_is_blamed_on_too_large_a_rate(self):
        """The old verdict said 'too small, train longer' - the opposite."""
        history, _, _ = lr.train(0.03, 200, verbose=False)
        self.assertTrue(lr.diagnose(history).startswith("Diverging"))

    def test_blow_up_is_named(self):
        history, _, _ = lr.train(0.5, 200, verbose=False)
        self.assertIn("blew up", lr.diagnose(history))

    def test_generalisation_verdicts(self):
        self.assertIn("genuinely learned", lr.generalization_verdict(10.0, 12.0))
        self.assertIn("noticeably worse", lr.generalization_verdict(10.0, 25.0))
        self.assertIn("overfitting", lr.generalization_verdict(10.0, 100.0))
        self.assertIn("finite", lr.generalization_verdict(10.0, float("nan")))


if __name__ == "__main__":
    unittest.main()
