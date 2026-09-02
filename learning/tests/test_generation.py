"""Generation result model, safeguards, and cancellation.

Everything here is offline. The API is mocked, so these tests prove that
we PARSE and DETECT correctly - they deliberately prove nothing about why
a real model might start repeating, which needs a live experiment.
"""

import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

LEARNING_DIR = Path(__file__).resolve().parent.parent
if str(LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(LEARNING_DIR))

import config  # noqa: E402
import generation as gen  # noqa: E402
import model_playground as mp  # noqa: E402


def chat_payload(text="hello", finish_reason="stop", usage=True):
    payload = {"choices": [{"message": {"content": text},
                            "finish_reason": finish_reason}]}
    if usage:
        payload["usage"] = {"prompt_tokens": 10, "completion_tokens": 5,
                            "total_tokens": 15}
    return payload


class ParsingResponses(unittest.TestCase):
    def test_reads_text_finish_reason_and_usage(self):
        result = gen.parse_response(chat_payload("hi there"), {"max_tokens": 150})
        self.assertEqual(result.text, "hi there")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.total_tokens, 15)
        self.assertTrue(result.finished_naturally)
        self.assertFalse(result.truncated)

    def test_length_finish_reason_means_truncated(self):
        result = gen.parse_response(chat_payload(finish_reason="length"))
        self.assertTrue(result.truncated)
        self.assertFalse(result.finished_naturally)

    def test_missing_usage_is_tolerated(self):
        result = gen.parse_response(chat_payload(usage=False))
        self.assertEqual(result.total_tokens, 0)
        self.assertEqual(result.text, "hello")

    def test_raw_completion_shape_is_supported(self):
        payload = {"choices": [{"text": "raw", "finish_reason": "length"}]}
        self.assertEqual(gen.parse_response(payload).text, "raw")

    def test_empty_string_reply_is_valid_not_malformed(self):
        self.assertEqual(gen.parse_response(chat_payload("")).text, "")

    def test_malformed_payloads_raise(self):
        for bad in ([], "nope", {}, {"choices": []}, {"choices": [{}]}):
            with self.assertRaises(gen.MalformedResponse):
                gen.parse_response(bad)


class TruncationWarnings(unittest.TestCase):
    def test_natural_ending_produces_no_warning(self):
        result = gen.parse_response(chat_payload(), {"max_tokens": 150})
        self.assertIsNone(gen.truncation_warning(result))

    def test_cut_off_reply_is_explained(self):
        result = gen.parse_response(chat_payload(finish_reason="length"),
                                    {"max_tokens": 150})
        warning = gen.truncation_warning(result)
        self.assertIn("CUT OFF", warning)
        self.assertIn("150", warning)

    def test_cancelled_reply_is_described_as_deliberate(self):
        result = gen.parse_response(chat_payload(finish_reason="length"),
                                    {"max_tokens": 150}, cancelled=True)
        self.assertIn("You stopped this one", gen.truncation_warning(result))


class RepetitionDetection(unittest.TestCase):
    """Detection only. These say nothing about causes."""

    def test_a_real_loop_is_detected(self):
        looping = "You're doing great! You've prepared well, and you've got this. " * 8
        report = gen.repetition_report(looping)
        self.assertTrue(report.looping)
        self.assertGreaterEqual(report.worst_count, 3)
        self.assertIn("repetition loop", report.detail)

    def test_ordinary_prose_is_not_flagged(self):
        prose = (
            "Gradient descent walks downhill using the slope of the loss. "
            "Each step is scaled by the learning rate, which decides how far "
            "the parameters move before the slope is measured again. Too "
            "large and it overshoots the minimum entirely; too small and it "
            "crawls without arriving anywhere useful in reasonable time."
        )
        report = gen.repetition_report(prose)
        self.assertFalse(report.looping)
        self.assertGreater(report.unique_ratio, 0.9)

    def test_short_text_is_not_judged(self):
        report = gen.repetition_report("too short")
        self.assertFalse(report.looping)
        self.assertIn("too short", report.detail)

    def test_repeat_threshold_is_configurable(self):
        twice = "the same six word phrase here. " * 2
        self.assertFalse(gen.repetition_report(twice, min_repeats=3).looping)
        self.assertTrue(gen.repetition_report(twice, min_repeats=2).looping)


class ContextAccounting(unittest.TestCase):
    def test_reports_fraction_used(self):
        usage = gen.context_usage(16384, 32768)
        self.assertAlmostEqual(usage.fraction, 0.5)
        self.assertIn("50%", usage.detail)

    def test_nearly_full_context_is_called_out(self):
        usage = gen.context_usage(31000, 32768)
        self.assertGreater(usage.fraction, 0.9)
        self.assertIn("Clear the conversation", usage.detail)

    def test_zero_limit_does_not_divide_by_zero(self):
        self.assertEqual(gen.context_usage(10, 0).limit, 1)


class ReviewCombinesWarnings(unittest.TestCase):
    def test_truncated_and_looping_reply_gets_both_notes(self):
        looping = "keep going and keep breathing right now. " * 8
        payload = {"choices": [{"message": {"content": looping},
                                "finish_reason": "length"}],
                   "usage": {"total_tokens": 30000}}
        result = gen.parse_response(payload, {"max_tokens": 150})
        notes = gen.review(result, context_limit=32768)
        self.assertTrue(any("CUT OFF" in n for n in notes))
        self.assertTrue(any("repetition loop" in n for n in notes))
        self.assertTrue(any("Clear the conversation" in n for n in notes))

    def test_clean_reply_gets_no_notes(self):
        result = gen.parse_response(chat_payload("A short, clean answer."))
        self.assertEqual(gen.review(result, context_limit=32768), [])


class FakeResponse:
    def __init__(self, payload=None, status=200, lines=()):
        self._payload = payload
        self.status_code = status
        self.ok = status < 400
        self._lines = list(lines)
        self.closed = False

    def raise_for_status(self):
        if not self.ok:
            import requests
            error = requests.exceptions.HTTPError("boom")
            error.response = self
            raise error

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def iter_lines(self, decode_unicode=False):
        for line in self._lines:
            yield line

    def close(self):
        self.closed = True


class GenerateOnce(unittest.TestCase):
    def test_returns_a_generation_with_finish_reason(self):
        with mock.patch.object(mp.requests, "post",
                               return_value=FakeResponse(chat_payload("hi", "length"))):
            result = mp.generate_once("prompt")
        self.assertEqual(result.text, "hi")
        self.assertTrue(result.truncated)

    def test_connection_error_becomes_a_readable_model_error(self):
        import requests
        with mock.patch.object(mp.requests, "post",
                               side_effect=requests.exceptions.ConnectionError()):
            with self.assertRaises(mp.ModelError) as caught:
                mp.generate_once("prompt")
        self.assertIn("Can't reach the model API", str(caught.exception))

    def test_timeout_becomes_a_readable_model_error(self):
        import requests
        with mock.patch.object(mp.requests, "post",
                               side_effect=requests.exceptions.Timeout()):
            with self.assertRaises(mp.ModelError) as caught:
                mp.generate_once("prompt")
        self.assertIn("didn't reply within", str(caught.exception))

    def test_http_error_mentions_the_status(self):
        with mock.patch.object(mp.requests, "post",
                               return_value=FakeResponse(status=500)):
            with self.assertRaises(mp.ModelError) as caught:
                mp.generate_once("prompt")
        self.assertIn("500", str(caught.exception))

    def test_malformed_json_is_reported_not_raised_raw(self):
        with mock.patch.object(mp.requests, "post",
                               return_value=FakeResponse({"nonsense": True})):
            with self.assertRaises(mp.ModelError) as caught:
                mp.generate_once("prompt")
        self.assertIn("Couldn't make sense", str(caught.exception))

    def test_lock_is_released_after_a_failure(self):
        """A failed generation must not wedge the app into 'busy' forever."""
        import requests
        with mock.patch.object(mp.requests, "post",
                               side_effect=requests.exceptions.ConnectionError()):
            with self.assertRaises(mp.ModelError):
                mp.generate_once("prompt")
        with mock.patch.object(mp.requests, "post",
                               return_value=FakeResponse(chat_payload("ok"))):
            self.assertEqual(mp.generate_once("prompt").text, "ok")

    def test_call_model_keeps_its_old_two_tuple_contract(self):
        with mock.patch.object(mp.requests, "post",
                               return_value=FakeResponse(chat_payload("legacy"))):
            text, params = mp.call_model("prompt")
        self.assertEqual(text, "legacy")
        self.assertIn("temperature", params)


class DuplicateGenerationIsRefused(unittest.TestCase):
    def test_second_generation_while_one_runs_is_refused(self):
        started, release = threading.Event(), threading.Event()

        def slow_post(*args, **kwargs):
            started.set()
            release.wait(timeout=5)
            return FakeResponse(chat_payload("done"))

        with mock.patch.object(mp.requests, "post", side_effect=slow_post):
            worker = threading.Thread(target=lambda: mp.generate_once("first"))
            worker.start()
            started.wait(timeout=5)
            try:
                with self.assertRaises(mp.GenerationBusy):
                    mp.generate_once("second")
            finally:
                release.set()
                worker.join(timeout=5)

    def test_busy_error_is_a_model_error_so_existing_handlers_catch_it(self):
        self.assertTrue(issubclass(mp.GenerationBusy, mp.ModelError))


class Streaming(unittest.TestCase):
    def stream_lines(self, pieces, finish="stop"):
        lines = [f'data: {{"choices":[{{"delta":{{"content":"{piece}"}}}}]}}'
                 for piece in pieces]
        lines.append(f'data: {{"choices":[{{"delta":{{}},"finish_reason":"{finish}"}}]}}')
        lines.append("data: [DONE]")
        return lines

    def test_yields_progressively_and_finishes_with_a_generation(self):
        response = FakeResponse(lines=self.stream_lines(["Hel", "lo", " there"]))
        with mock.patch.object(mp.requests, "post", return_value=response):
            updates = list(mp.stream_generation("prompt"))
        texts = [text for text, _ in updates]
        self.assertEqual(texts[0], "Hel")
        self.assertEqual(texts[-1], "Hello there")
        self.assertIsNotNone(updates[-1][1])
        self.assertEqual(updates[-1][1].finish_reason, "stop")

    def test_cancelling_stops_early_and_closes_the_connection(self):
        """Closing the connection is what makes the SERVER stop working."""
        response = FakeResponse(lines=self.stream_lines(["a", "b", "c", "d", "e"]))
        token = mp.CancelToken()

        with mock.patch.object(mp.requests, "post", return_value=response):
            collected = []
            for text, final in mp.stream_generation("prompt", cancel_token=token):
                collected.append(text)
                if final is None and len(collected) == 2:
                    token.cancel()

        self.assertTrue(response.closed, "must close the stream to cancel server work")
        self.assertLess(len(collected), 6)

    def test_stream_marks_the_result_cancelled(self):
        response = FakeResponse(lines=self.stream_lines(["a", "b", "c"]))
        token = mp.CancelToken()
        token.cancel()
        with mock.patch.object(mp.requests, "post", return_value=response):
            updates = list(mp.stream_generation("prompt", cancel_token=token))
        self.assertTrue(updates[-1][1].cancelled)

    def test_stream_releases_the_lock_when_cancelled(self):
        response = FakeResponse(lines=self.stream_lines(["a"]))
        token = mp.CancelToken()
        token.cancel()
        with mock.patch.object(mp.requests, "post", return_value=response):
            list(mp.stream_generation("prompt", cancel_token=token))
        with mock.patch.object(mp.requests, "post",
                               return_value=FakeResponse(chat_payload("after"))):
            self.assertEqual(mp.generate_once("prompt").text, "after")


class CancellationReporting(unittest.TestCase):
    """Revision 4: say what we actually know, never assume success."""

    def test_reports_server_accepting_the_stop_endpoint(self):
        with mock.patch.object(mp.requests, "post", return_value=FakeResponse(status=200)), \
             mock.patch.object(mp.requests, "get", return_value=FakeResponse(status=200)):
            outcome = mp.cancel_generation(mp.CancelToken())
        self.assertEqual(outcome.stop_endpoint, "accepted")
        self.assertIn("accepted the stop request", outcome.detail)

    def test_missing_stop_endpoint_falls_back_to_connection_close(self):
        with mock.patch.object(mp.requests, "post", return_value=FakeResponse(status=404)), \
             mock.patch.object(mp.requests, "get", return_value=FakeResponse(status=200)):
            outcome = mp.cancel_generation(mp.CancelToken())
        self.assertEqual(outcome.stop_endpoint, "not available on this server build")
        self.assertTrue(outcome.server_responsive)
        self.assertIn("generation slot is free", outcome.detail)

    def test_unreachable_server_is_reported_as_unconfirmed(self):
        import requests
        with mock.patch.object(mp.requests, "post",
                               side_effect=requests.exceptions.ConnectionError()), \
             mock.patch.object(mp.requests, "get",
                               side_effect=requests.exceptions.ConnectionError()):
            outcome = mp.cancel_generation(mp.CancelToken())
        self.assertIsNone(outcome.server_responsive)
        self.assertIn("Could not confirm", outcome.detail)

    def test_cancel_sets_the_token(self):
        token = mp.CancelToken()
        self.assertFalse(token.cancelled)
        with mock.patch.object(mp.requests, "post", return_value=FakeResponse(status=404)), \
             mock.patch.object(mp.requests, "get", return_value=FakeResponse(status=200)):
            mp.cancel_generation(token)
        self.assertTrue(token.cancelled)


class SafeDefaults(unittest.TestCase):
    def test_settings_come_from_config(self):
        for key, value in config.GENERATION_DEFAULTS.items():
            self.assertEqual(mp.SETTINGS_SPEC[key]["default"], value)

    def test_urls_come_from_config(self):
        self.assertEqual(mp.CHAT_URL, config.MODEL_CHAT_URL)
        self.assertEqual(mp.MODELS_URL, config.MODEL_LIST_URL)


if __name__ == "__main__":
    unittest.main()
