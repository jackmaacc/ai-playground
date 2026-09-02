"""Importing a module must never train, plot, launch, or phone home.

Every check reloads the module so the module body runs again under
observation, then asserts that nothing outside the process changed.
"""

import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

LEARNING_DIR = Path(__file__).resolve().parent.parent
if str(LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(LEARNING_DIR))

import config  # noqa: E402


def png_snapshot():
    return {p.name: p.stat().st_mtime_ns for p in LEARNING_DIR.glob("*.png")}


class ImportsHaveNoSideEffects(unittest.TestCase):
    def test_viz_and_overfitting_modules_do_not_write_plots_on_import(self):
        before = png_snapshot()
        for name in ("phase1_gradient_descent_viz", "phase1_linear_regression_viz",
                     "phase1_overfitting"):
            importlib.reload(importlib.import_module(name))
        self.assertEqual(png_snapshot(), before)

    def test_manager_and_services_do_not_create_runtime_state_on_import(self):
        existed = config.RUNTIME_DIR.exists()
        for name in ("services", "manager"):
            importlib.reload(importlib.import_module(name))
        self.assertEqual(config.RUNTIME_DIR.exists(), existed)

    def test_model_modules_do_not_call_the_api_on_import(self):
        import requests

        with mock.patch.object(requests, "get", side_effect=AssertionError("network on import")), \
                mock.patch.object(requests, "post", side_effect=AssertionError("network on import")):
            for name in ("model_playground", "lessons"):
                importlib.reload(importlib.import_module(name))

    def test_main_does_not_import_the_web_ui(self):
        """Building the Gradio app just to check whether gradio exists was
        the old behaviour; main must stay lightweight."""
        already_loaded = "webui" in sys.modules
        importlib.reload(importlib.import_module("main"))
        if not already_loaded:
            self.assertNotIn("webui", sys.modules)

    def test_no_module_launches_a_process_on_import(self):
        import subprocess

        with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("Popen on import")):
            for name in ("config", "services", "manager", "main", "model_playground"):
                importlib.reload(importlib.import_module(name))


class EntryPoints(unittest.TestCase):
    def test_every_script_guards_its_main_block(self):
        scripts = [p for p in LEARNING_DIR.glob("*.py") if p.name != "__init__.py"]
        for script in scripts:
            text = script.read_text(encoding="utf-8", errors="replace")
            if "def main(" in text or "def run(" in text or "def demo(" in text:
                self.assertIn('if __name__ == "__main__":', text, script.name)


if __name__ == "__main__":
    unittest.main()
