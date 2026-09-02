"""Web UI handlers, run only where gradio is installed.

The plain system Python has no gradio, so this file skips itself there.
Run it under a bundled interpreter, for example:

    <repo>/chat-llm/installer_files/env/python.exe -m unittest \
        discover -s learning/tests -t learning -p "test_webui.py"

Everything is still offline: the model API is mocked.
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

LEARNING_DIR = Path(__file__).resolve().parent.parent
if str(LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(LEARNING_DIR))

HAS_GRADIO = importlib.util.find_spec("gradio") is not None


@unittest.skipUnless(HAS_GRADIO, "gradio is not installed for this interpreter")
class ChatHandlers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import webui
        import model_playground as mp
        cls.webui, cls.mp = webui, mp

    def stream(self, pieces, finish="stop"):
        lines = [f'data: {{"choices":[{{"delta":{{"content":"{p}"}}}}]}}' for p in pieces]
        lines.append(f'data: {{"choices":[{{"delta":{{}},"finish_reason":"{finish}"}}]}}')
        lines.append("data: [DONE]")

        class Response:
            status_code, ok, closed = 200, True, False

            def raise_for_status(self):
                pass

            def iter_lines(self, decode_unicode=False):
                yield from lines

            def close(self):
                self.closed = True

        return Response()

    def test_chat_streams_then_records_history(self):
        with mock.patch.object(self.mp.requests, "post", return_value=self.stream(["Hi", "!"])):
            updates = list(self.webui.chat_stream("hello", [], 0.7, 0.9, 20, 1.1, 150))
        history, text, notes, context = updates[-1]
        self.assertEqual(text, "Hi!")
        self.assertEqual([m["role"] for m in history], ["user", "assistant"])
        self.assertIn("Finished normally", notes)
        self.assertIn("estimate", context)

    def test_cut_off_reply_is_flagged_in_notes(self):
        with mock.patch.object(self.mp.requests, "post",
                               return_value=self.stream(["a"], finish="length")):
            *_, (history, text, notes, context) = self.webui.chat_stream(
                "hello", [], 0.7, 0.9, 20, 1.1, 16)
        self.assertIn("CUT OFF", notes)

    def test_empty_prompt_does_not_call_the_model(self):
        with mock.patch.object(self.mp.requests, "post") as post:
            updates = list(self.webui.chat_stream("   ", [], 0.7, 0.9, 20, 1.1, 150))
        post.assert_not_called()
        self.assertIn("Type a prompt", updates[-1][2])

    def test_clear_conversation_resets_everything(self):
        history, text, notes, context = self.webui.clear_conversation()
        self.assertEqual(history, [])
        self.assertEqual(text, "")
        self.assertIn("cleared", notes.lower())

    def test_stop_with_nothing_running_still_asks_the_server(self):
        with mock.patch.object(self.mp, "request_server_stop", return_value="accepted"):
            message = self.webui.stop_generation()
        self.assertIn("Nothing is generating", message)
        self.assertIn("accepted", message)

    def test_slider_defaults_come_from_the_shared_settings_table(self):
        """The old duplication bug: web defaults drifting from the spec."""
        expected = {key: spec["default"] for key, spec in self.mp.SETTINGS_SPEC.items()}
        self.assertEqual(self.webui.defaults, expected)


if __name__ == "__main__":
    unittest.main()
