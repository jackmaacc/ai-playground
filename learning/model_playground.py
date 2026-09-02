"""
Interactive playground for the REAL local model (Qwen2.5-3B, running in
text-generation-webui with --api enabled at http://127.0.0.1:5000).

Unlike phase1_interactive.py (which trains a tiny model from scratch),
this talks to your actual chat model over its API and lets you feel the
effect of sampling settings directly - including running the same prompt
side-by-side at two different settings, which is the fastest way to
build intuition for what each one actually does.

This is also the API client for the rest of the project: lessons.py and
webui.py both call call_model() from here rather than talking to the
server themselves.

Requires: text-generation-webui running with --api (already set in
chat-llm/user_data/CMD_FLAGS.txt) and a model loaded.

Run with: python model_playground.py
"""

import json
import threading
from typing import NamedTuple, Optional

import requests

import config
import console
import generation as gen

console.use_utf8_output()

# All of these now come from config.py, which derives them from one place
# and lets AIPLAY_* environment variables override them. The names are
# kept because other modules import them.
BASE_URL = config.MODEL_API_BASE_URL
CHAT_URL = config.MODEL_CHAT_URL
MODELS_URL = config.MODEL_LIST_URL

# text-generation-webui's own endpoint for aborting whatever it is
# currently generating. Best effort: not every build exposes it.
STOP_URL = f"{BASE_URL}/v1/internal/stop-generation"

# How long to wait for a reply. A 3B model on CPU can be slow, but if it's
# taking minutes something is wrong rather than slow.
TIMEOUT_SECONDS = config.GENERATION_TIMEOUT

OFFLINE_HINT = (
    f"Can't reach the model API at {BASE_URL} - is text-generation-webui\n"
    "running? (start_windows.bat in chat-llm/)"
)


class ModelError(RuntimeError):
    """The model API couldn't be reached, or sent back something unusable.

    Wrapping the various requests exceptions in one type means callers get
    a sentence they can show a human, instead of a traceback ending a
    session halfway through a lesson.
    """


# Every knob in one table: its default, how to parse what you type, and
# what it actually does. Keeping these together is what stops the parsing
# and the help text from drifting apart - and fixes the old bug where
# typing "1.5" for top_k quietly produced an invalid float.
SETTINGS_SPEC = {
    "temperature": {
        "cast": float,
        "default": config.GENERATION_DEFAULTS["temperature"],
        "help": (
            "Reshapes the model's probability distribution over next tokens before "
            "picking one. Low (0.1-0.3) = flattens toward the single most likely "
            "token -> repetitive, predictable. High (1.2+) = flattens the "
            "DIFFERENCES between options -> more surprising, sometimes incoherent."
        ),
    },
    "top_p": {
        "cast": float,
        "default": config.GENERATION_DEFAULTS["top_p"],
        "help": (
            "Nucleus sampling. Only keep the smallest set of tokens whose "
            "probabilities add up to top_p, then sample from just those. "
            "1.0 = consider everything (riskier). 0.5 = only the most confident "
            "half of the options (safer)."
        ),
    },
    "top_k": {
        "cast": int,
        "default": config.GENERATION_DEFAULTS["top_k"],
        "help": (
            "Blunter than top_p: keep only the K most-likely tokens, discard the "
            "rest no matter how confident the model was. top_k=1 means always pick "
            "the single most likely token (fully deterministic). 0 disables it."
        ),
    },
    "repetition_penalty": {
        "cast": float,
        "default": config.GENERATION_DEFAULTS["repetition_penalty"],
        "help": (
            "Punishes tokens the model has already used recently, so it doesn't "
            "loop ('I think that I think that...'). Too high (>1.3) makes it "
            "avoid words it actually needs to repeat, and output gets weird."
        ),
    },
    "max_tokens": {
        "cast": int,
        "default": config.GENERATION_DEFAULTS["max_tokens"],
        "help": (
            "Not a sampling parameter - just a hard cap on how many tokens the "
            "response is allowed to be."
        ),
    },
}

# The sampling knobs proper - max_tokens is a length limit, not a way of
# choosing between tokens, so comparisons leave it out.
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "repetition_penalty")

# Kept as its own name because webui.py renders it directly as help text.
EXPLANATIONS = {key: spec["help"] for key, spec in SETTINGS_SPEC.items()}

# Current settings, adjustable through the menu.
settings = {key: spec["default"] for key, spec in SETTINGS_SPEC.items()}


class GenerationBusy(ModelError):
    """A generation is already running.

    Two generations at once on a single-slot local server means both crawl
    and neither can be cancelled cleanly, so we refuse the second rather
    than queue it.
    """


class CancelToken:
    """A flag the UI can set from another thread to stop a generation."""

    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    @property
    def cancelled(self):
        return self._event.is_set()


class CancelOutcome(NamedTuple):
    """What we actually know about a cancellation.

    Deliberately honest: stopping the stream is something we can be sure
    of, but whether the SERVER stopped working is something we can only
    ask about, so it is reported separately rather than assumed.
    """
    stream_closed: bool
    stop_endpoint: str
    server_responsive: Optional[bool]
    detail: str


# Only one generation at a time. Non-blocking: callers get told "busy"
# instead of silently piling up.
_generation_lock = threading.Lock()


def _build_payload(params, prompt=None, messages=None, stream=False):
    payload = {"messages": messages or [{"role": "user", "content": prompt}]}
    payload.update({key: params[key] for key in SETTINGS_SPEC})
    if stream:
        payload["stream"] = True
    return payload


def _as_model_error(error):
    """Turn any requests failure into one clear sentence."""
    if isinstance(error, requests.exceptions.ConnectionError):
        return ModelError(OFFLINE_HINT)
    if isinstance(error, requests.exceptions.Timeout):
        return ModelError(
            f"The model didn't reply within {TIMEOUT_SECONDS}s. Try a smaller "
            "max_tokens, or check whether the server is stuck loading."
        )
    if isinstance(error, requests.exceptions.HTTPError):
        code = getattr(getattr(error, "response", None), "status_code", "?")
        return ModelError(
            f"The API rejected that request (HTTP {code}). "
            "Is a model actually loaded in text-generation-webui?"
        )
    return ModelError(f"Couldn't reach the model: {error}")


def request_server_stop():
    """Ask the server to abort what it is generating. Best effort.

    Returns a short string describing what happened, which the UI shows
    verbatim - we would rather say "not available" than imply success.
    """
    try:
        response = requests.post(STOP_URL, timeout=config.HEALTH_TIMEOUT)
    except requests.exceptions.RequestException as error:
        return f"could not be reached ({type(error).__name__})"
    if response.status_code == 404:
        return "not available on this server build"
    if response.ok:
        return "accepted"
    return f"refused (HTTP {response.status_code})"


def confirm_server_idle(timeout=None):
    """Is the server answering promptly again?

    A quick reply means the generation slot is free, which is the best
    evidence available from outside that our cancellation landed. True,
    False, or None when we could not tell.
    """
    try:
        response = requests.get(MODELS_URL, timeout=timeout or config.HEALTH_TIMEOUT)
        return bool(response.ok)
    except requests.exceptions.Timeout:
        return False
    except requests.exceptions.RequestException:
        return None


def generate_once(prompt=None, overrides=None, messages=None):
    """One reply, as a Generation carrying finish_reason and token counts.

    This is the honest replacement for call_model(): the reason the model
    stopped is the thing you need in order to know whether the answer is
    complete.
    """
    params = {**settings, **(overrides or {})}
    payload = _build_payload(params, prompt, messages)

    if not _generation_lock.acquire(blocking=False):
        raise GenerationBusy(
            "Already generating. Wait for it to finish, or press Stop."
        )
    try:
        response = requests.post(CHAT_URL, json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        body = response.json()
    except requests.exceptions.RequestException as error:
        raise _as_model_error(error)
    except ValueError as error:
        raise ModelError(f"The API's reply was not valid JSON: {error}")
    finally:
        _generation_lock.release()

    try:
        return gen.parse_response(body, params)
    except gen.MalformedResponse as error:
        raise ModelError(f"Couldn't make sense of the API's reply: {error}")


def stream_generation(prompt=None, overrides=None, messages=None, cancel_token=None):
    """Yield the reply piece by piece, and stop promptly when cancelled.

    Cancellation works by closing the HTTP connection. That is not just a
    UI trick: the server notices the client has gone and abandons the
    task, which is what makes Stop actually free the GPU rather than
    merely hiding the output. We additionally ask the server's own
    stop endpoint, and report separately what each attempt achieved.

    Yields (text_so_far, generation_or_None). The final item carries the
    finished Generation.
    """
    params = {**settings, **(overrides or {})}
    payload = _build_payload(params, prompt, messages, stream=True)
    token = cancel_token or CancelToken()

    if not _generation_lock.acquire(blocking=False):
        raise GenerationBusy(
            "Already generating. Wait for it to finish, or press Stop."
        )

    collected = []
    finish_reason = ""
    response = None
    try:
        try:
            response = requests.post(CHAT_URL, json=payload, stream=True,
                                     timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.exceptions.RequestException as error:
            raise _as_model_error(error)

        for line in response.iter_lines(decode_unicode=True):
            if token.cancelled:
                break
            if not line:
                continue
            if line.startswith("data:"):
                line = line[len("data:"):].strip()
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except ValueError:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            piece = (choice.get("delta") or {}).get("content") or choice.get("text") or ""
            finish_reason = choice.get("finish_reason") or finish_reason
            if piece:
                collected.append(piece)
                yield "".join(collected), None

        cancelled = token.cancelled
        text = "".join(collected)
        yield text, gen.Generation(
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            params=params,
            cancelled=cancelled,
        )
    finally:
        if response is not None:
            # Closing the connection is what tells the server to stop.
            response.close()
        _generation_lock.release()


def cancel_generation(token):
    """Stop the running generation and report what we could confirm."""
    token.cancel()
    endpoint = request_server_stop()
    responsive = confirm_server_idle()

    if endpoint == "accepted":
        detail = "The server accepted the stop request."
    elif responsive is True:
        detail = ("Closed the connection; the server answered again "
                  "immediately, so its generation slot is free.")
    elif responsive is False:
        detail = ("Closed the connection, but the server is still slow to "
                  "answer - it may still be finishing the last batch.")
    else:
        detail = ("Closed the connection. Could not confirm what the server "
                  "did, because it is not answering at all.")

    return CancelOutcome(
        stream_closed=True,
        stop_endpoint=endpoint,
        server_responsive=responsive,
        detail=detail,
    )


def call_model(prompt, overrides=None):
    """Send one prompt and return (reply_text, settings_actually_used).

    Kept exactly as it was so lessons.py and webui.py keep working. New
    code should prefer generate_once(), which also tells you why the model
    stopped.
    """
    result = generate_once(prompt, overrides)
    return result.text, result.params


def api_reachable():
    try:
        requests.get(MODELS_URL, timeout=5)
        return True
    except requests.exceptions.RequestException:
        return False


def format_settings(params, keys=SAMPLING_KEYS):
    return "(" + ", ".join(f"{key}={params[key]}" for key in keys) + ")"


def print_settings():
    for key, value in settings.items():
        print(f"  {key}: {value}")


def ask_setting(key, current):
    """Prompt for one setting, parsed the way that setting needs."""
    spec = SETTINGS_SPEC[key]
    return console.ask_number(f"  {key}", default=current, cast=spec["cast"])


def do_chat():
    print("\n--- Chat (current settings shown below) ---")
    print_settings()
    prompt = console.ask_text("\nyour prompt: ")

    print("\nthinking...")
    try:
        result = generate_once(prompt)
    except ModelError as error:
        print(f"\n{error}")
        return

    print(f"\nmodel: {result.text}")
    for note in gen.review(result, config.MODEL_CONTEXT_TOKENS):
        print(f"\n[!] {note}")


def do_compare():
    print("\n--- Compare: same prompt, two settings ---")
    prompt = console.ask_text("prompt to compare: ")

    print("\nSetting A is your current settings:")
    print_settings()
    print("\nSetting B - press Enter to keep a value, or type a new one:")
    overrides = {key: ask_setting(key, settings[key]) for key in SAMPLING_KEYS}

    print("\ngenerating both responses...")
    try:
        reply_a, used_a = call_model(prompt)
        reply_b, used_b = call_model(prompt, overrides)
    except ModelError as error:
        print(f"\n{error}")
        return

    print(f"\n=== Setting A {format_settings(used_a)} ===\n{reply_a}")
    print(f"\n=== Setting B {format_settings(used_b)} ===\n{reply_b}")
    print(
        "\nSame prompt, same model weights - any difference above came purely "
        "from how the next token was SELECTED at each step, not from the model "
        "'thinking' differently."
    )


def do_adjust():
    print("\n--- Adjust settings (Enter keeps the current value) ---")
    for key in SETTINGS_SPEC:
        settings[key] = ask_setting(key, settings[key])
    print("\nUpdated:")
    print_settings()


def do_explain():
    print("\n--- What each setting does ---")
    for key, spec in SETTINGS_SPEC.items():
        print(f"\n{key} (current: {settings[key]}):\n  {spec['help']}")


def run():
    if not api_reachable():
        print(OFFLINE_HINT)
        return

    console.run_menu(
        "Model playground - talking to your real Qwen2.5-3B model",
        [
            ("Chat", do_chat),
            ("Compare the same prompt at two settings", do_compare),
            ("Adjust settings", do_adjust),
            ("Explain what each setting does", do_explain),
        ],
        back_label="Quit",
    )


if __name__ == "__main__":
    if not api_reachable():
        print(OFFLINE_HINT)
        raise SystemExit(1)
    run()
