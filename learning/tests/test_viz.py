"""The two plotting scripts: they save where told and clean up after
themselves, so a menu can run them repeatedly without leaking figures."""

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

LEARNING_DIR = Path(__file__).resolve().parent.parent
if str(LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(LEARNING_DIR))


class VizScripts(unittest.TestCase):
    def run_into_temp(self, module_name, expected):
        import matplotlib.pyplot as plt
        module = importlib.import_module(module_name)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(module, "OUT_DIR", Path(tmp)):
            module.run()
            written = sorted(p.name for p in Path(tmp).glob("*.png"))
        self.assertEqual(written, expected)
        self.assertEqual(plt.get_fignums(), [], f"{module_name} leaked figures")

    def test_gradient_descent_viz(self):
        self.run_into_temp("phase1_gradient_descent_viz",
                           ["gradient_descent_convergence.png", "gradient_descent_paths.png"])

    def test_linear_regression_viz(self):
        self.run_into_temp("phase1_linear_regression_viz",
                           ["linreg_fit.png", "linreg_loss.png"])


if __name__ == "__main__":
    unittest.main()
