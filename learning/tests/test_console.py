"""console.py: a typo never ends a session, and Ctrl+C backs out one level."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

LEARNING_DIR = Path(__file__).resolve().parent.parent
if str(LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(LEARNING_DIR))

import console  # noqa: E402


def typing(*answers):
    """Patch input() to return these answers in turn."""
    return mock.patch("builtins.input", side_effect=list(answers))


class AskNumber(unittest.TestCase):
    def test_junk_is_re_asked_then_a_number_is_accepted(self):
        with typing("abc", "", "5"), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(console.ask_float("x"), 5.0)
        self.assertIn("isn't a number", out.getvalue())

    def test_blank_takes_the_default(self):
        with typing(""), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(console.ask_float("x", default=0.25), 0.25)

    def test_bounds_are_enforced(self):
        with typing("0", "99", "3"), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(console.ask_int("x", minimum=1, maximum=10), 3)
        self.assertIn("at least 1", out.getvalue())
        self.assertIn("at most 10", out.getvalue())

    def test_int_rejects_decimals(self):
        with typing("1.5", "2"), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(console.ask_int("x"), 2)
        self.assertIn("whole number", out.getvalue())

    def test_default_is_shown_in_the_prompt(self):
        with mock.patch("builtins.input", return_value="") as fake:
            console.ask_float("learning rate", default=0.01)
        self.assertIn("[0.01]", fake.call_args[0][0])


class AskText(unittest.TestCase):
    def test_empty_is_re_asked_unless_allowed(self):
        with typing("", "hello"), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(console.ask_text("say"), "hello")
        with typing(""):
            self.assertEqual(console.ask_text("say", allow_empty=True), "")


class BackingOut(unittest.TestCase):
    def test_eof_becomes_back(self):
        with mock.patch("builtins.input", side_effect=EOFError), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(console.Back):
                console.ask_float("x")

    def test_ctrl_c_becomes_back(self):
        with mock.patch("builtins.input", side_effect=KeyboardInterrupt), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(console.Back):
                console.ask_text("x")


class RunMenu(unittest.TestCase):
    def test_invalid_choice_re_asks_then_runs_the_action_then_backs_out(self):
        calls = []
        with typing("9", "1", "2"), contextlib.redirect_stdout(io.StringIO()) as out:
            console.run_menu("t", [("do it", lambda: calls.append("ran"))])
        self.assertEqual(calls, ["ran"])
        self.assertIn("Not a valid option", out.getvalue())

    def test_back_raised_inside_an_action_returns_to_the_menu(self):
        def bail():
            raise console.Back

        with typing("1", "2"), contextlib.redirect_stdout(io.StringIO()):
            console.run_menu("t", [("bail", bail)])   # must not propagate

    def test_eof_at_the_menu_prompt_exits_cleanly(self):
        with mock.patch("builtins.input", side_effect=EOFError), \
                contextlib.redirect_stdout(io.StringIO()):
            console.run_menu("t", [("never", lambda: None)])


if __name__ == "__main__":
    unittest.main()
