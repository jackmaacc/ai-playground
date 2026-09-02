"""Configuration and portability tests.

Runs with the standard library alone:
    python -m unittest discover -s learning/tests -t learning
and is collected unchanged by pytest once that is installed.
"""

import importlib
import os
import sys
import unittest
from pathlib import Path

# Make the learning/ modules importable however this suite is launched.
LEARNING_DIR = Path(__file__).resolve().parent.parent
if str(LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(LEARNING_DIR))

import config  # noqa: E402


class PathsAreDerived(unittest.TestCase):
    """The repository must work after being moved or copied."""

    def test_repo_root_is_parent_of_learning_dir(self):
        self.assertEqual(config.LEARNING_DIR.name, "learning")
        self.assertEqual(config.REPO_ROOT, config.LEARNING_DIR.parent)

    def test_paths_are_absolute_and_resolved(self):
        self.assertTrue(config.REPO_ROOT.is_absolute())
        self.assertEqual(config.REPO_ROOT, config.REPO_ROOT.resolve())

    def test_app_dirs_sit_under_the_repo(self):
        for path in (config.CHAT_LLM_DIR, config.IMAGE_GEN_DIR, config.RUNTIME_DIR):
            self.assertEqual(path.parent, config.REPO_ROOT
                             if path.parent == config.REPO_ROOT else path.parent)
            self.assertTrue(str(path).startswith(str(config.REPO_ROOT)))

    def test_no_hardcoded_user_path_in_source(self):
        """No learning/ module may contain someone's home directory.

        This is the regression guard for the bug this milestone fixed:
        main.py previously pinned C:\\Users\\jackm\\... which broke the
        moment the repository moved.
        """
        offenders = []
        for path in sorted(LEARNING_DIR.glob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            for marker in ("C:\\Users\\", "C:/Users/"):
                if marker in text:
                    offenders.append(f"{path.name} contains {marker!r}")
        self.assertEqual(offenders, [], "; ".join(offenders))


class NetworkDefaults(unittest.TestCase):
    """Loopback unless someone deliberately opts out."""

    def test_defaults_to_loopback(self):
        self.assertEqual(config.LOOPBACK, "127.0.0.1")
        self.assertFalse(config.is_public_bind())
        self.assertIsNone(config.public_bind_warning())

    def test_url_for_builds_a_clickable_address(self):
        self.assertEqual(config.url_for(7862, "127.0.0.1"), "http://127.0.0.1:7862")

    def test_wildcard_bind_uses_loopback_for_browser_url(self):
        original_bind = os.environ.get("AIPLAY_BIND_HOST")
        original_browse = os.environ.get("AIPLAY_BROWSE_HOST")
        os.environ["AIPLAY_BIND_HOST"] = "0.0.0.0"
        os.environ.pop("AIPLAY_BROWSE_HOST", None)
        try:
            reloaded = importlib.reload(config)
            self.assertEqual(reloaded.url_for(7862), "http://127.0.0.1:7862")
        finally:
            if original_bind is None:
                os.environ.pop("AIPLAY_BIND_HOST", None)
            else:
                os.environ["AIPLAY_BIND_HOST"] = original_bind
            if original_browse is None:
                os.environ.pop("AIPLAY_BROWSE_HOST", None)
            else:
                os.environ["AIPLAY_BROWSE_HOST"] = original_browse
            importlib.reload(config)

    def test_model_urls_derive_from_the_api_port(self):
        self.assertTrue(config.MODEL_CHAT_URL.endswith("/v1/chat/completions"))
        self.assertTrue(config.MODEL_LIST_URL.endswith("/v1/models"))
        self.assertIn(str(config.CHAT_LLM_API_PORT), config.MODEL_API_BASE_URL)

    def test_public_bind_is_detected_and_warned_about(self):
        original = os.environ.get("AIPLAY_BIND_HOST")
        os.environ["AIPLAY_BIND_HOST"] = "0.0.0.0"
        try:
            reloaded = importlib.reload(config)
            self.assertTrue(reloaded.is_public_bind())
            warning = reloaded.public_bind_warning()
            self.assertIsNotNone(warning)
            self.assertIn("WARNING", warning)
            self.assertIn("authentication", warning)
        finally:
            if original is None:
                os.environ.pop("AIPLAY_BIND_HOST", None)
            else:
                os.environ["AIPLAY_BIND_HOST"] = original
            importlib.reload(config)


class EnvironmentOverrides(unittest.TestCase):
    """Overridable, but working without any variable set."""

    def _reload_with(self, name, value):
        original = os.environ.get(name)
        os.environ[name] = value
        try:
            return importlib.reload(config)
        finally:
            if original is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original

    def test_port_override_is_honoured(self):
        try:
            reloaded = self._reload_with("AIPLAY_LEARNING_WEB_PORT", "9999")
            self.assertEqual(reloaded.LEARNING_WEB_PORT, 9999)
        finally:
            importlib.reload(config)

    def test_nonsense_port_override_is_rejected_with_its_name(self):
        try:
            with self.assertRaisesRegex(config.ConfigurationError,
                                        "AIPLAY_LEARNING_WEB_PORT"):
                self._reload_with("AIPLAY_LEARNING_WEB_PORT", "not-a-port")
        finally:
            importlib.reload(config)

    def test_out_of_range_port_is_rejected(self):
        try:
            with self.assertRaisesRegex(config.ConfigurationError,
                                        "between 1 and 65535"):
                self._reload_with("AIPLAY_IMAGE_GEN_PORT", "70000")
        finally:
            importlib.reload(config)

    def test_default_ports_match_the_documented_ones(self):
        self.assertEqual(config.CHAT_LLM_UI_PORT, 7860)
        self.assertEqual(config.CHAT_LLM_API_PORT, 5000)
        self.assertEqual(config.IMAGE_GEN_PORT, 7861)
        self.assertEqual(config.LEARNING_WEB_PORT, 7862)


class GenerationDefaults(unittest.TestCase):
    def test_defaults_are_conservative(self):
        defaults = config.GENERATION_DEFAULTS
        self.assertLessEqual(defaults["max_tokens"], 512,
                             "a high cap turns a repetition loop into a wall of text")
        self.assertGreater(defaults["repetition_penalty"], 1.0,
                           "some repetition penalty should be on by default")
        self.assertTrue(0 < defaults["temperature"] <= 2.0)
        self.assertTrue(0 < defaults["top_p"] <= 1.0)

    def test_describe_reports_the_essentials(self):
        details = config.describe()
        for key in ("repo_root", "bind_host", "public_bind", "ports", "model_api"):
            self.assertIn(key, details)


if __name__ == "__main__":
    unittest.main()
