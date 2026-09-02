"""Overfitting demo: scaling that only ever sees training data, and a
degree sweep whose test error genuinely turns upward."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

LEARNING_DIR = Path(__file__).resolve().parent.parent
if str(LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(LEARNING_DIR))

import phase1_linear_regression as lr  # noqa: E402
import phase1_overfitting as of  # noqa: E402


class ScalingIsFittedOnTrainingDataOnly(unittest.TestCase):
    def test_statistics_come_from_the_training_split(self):
        self.assertAlmostEqual(of.X_MEAN, float(np.mean(lr.train_hours)))
        self.assertAlmostEqual(of.X_STD, float(np.std(lr.train_hours)))

    def test_and_not_from_all_the_data(self):
        """Even a mean learned from the test set is a leak."""
        self.assertNotAlmostEqual(of.X_MEAN, float(np.mean(lr.hours)), places=3)

    def test_training_data_scales_to_zero_mean_unit_variance(self):
        z = of.scale(lr.train_hours)
        self.assertAlmostEqual(float(z.mean()), 0.0, places=9)
        self.assertAlmostEqual(float(z.std()), 1.0, places=9)


class Features(unittest.TestCase):
    def test_shape_and_intercept_column(self):
        features = of.polynomial_features(lr.train_hours, 4)
        self.assertEqual(features.shape, (len(lr.train_hours), 5))
        np.testing.assert_array_equal(features[:, 0], np.ones(len(lr.train_hours)))

    def test_degree_one_agrees_with_the_straight_line_solution(self):
        coeffs = of.fit_polynomial(lr.train_hours, lr.train_scores, 1)
        m, b = lr.closed_form_solution()
        np.testing.assert_allclose(of.predict_polynomial(lr.train_hours, coeffs),
                                   lr.predict(lr.train_hours, m, b), rtol=1e-8)


class TheSweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = of.sweep()

    def test_covers_every_degree(self):
        self.assertEqual([r.degree for r in self.results], list(range(1, of.MAX_DEGREE + 1)))

    def test_training_error_falls_with_more_freedom(self):
        self.assertLess(self.results[-1].train_mse, self.results[0].train_mse)

    def test_test_error_turns_upward(self):
        """The U-curve: the most flexible model is worse on unseen data
        than the best one, and the best one is not the most flexible."""
        best = of.best_result(self.results)
        self.assertLess(best.degree, self.results[-1].degree)
        self.assertGreater(self.results[-1].test_mse, best.test_mse)

    def test_best_degree_is_chosen_on_test_error(self):
        best = of.best_result(self.results)
        self.assertEqual(best.test_mse, min(r.test_mse for r in self.results))

    def test_verdict_names_the_gap(self):
        text = of.diagnose(self.results)
        self.assertIn("worse than at degree", text)
        self.assertIn("never judged on its training data", text)


class PlotsAreTidy(unittest.TestCase):
    def test_plots_write_where_told_and_close_their_figures(self):
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(of, "OUT_DIR", Path(tmp)):
            of.make_plots()
            written = sorted(p.name for p in Path(tmp).glob("*.png"))
        self.assertEqual(written, ["overfitting_curve.png", "overfitting_fits.png"])
        self.assertEqual(plt.get_fignums(), [], "figures must be closed after saving")


if __name__ == "__main__":
    unittest.main()
