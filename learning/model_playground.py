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

import requests

import console

console.use_utf8_output()

# One base URL, derived everywhere else - so pointing this at a different
# machine or port is a one-line change.
BASE_URL = "http://127.0.0.1:5000"
CHAT_URL = f"{BASE_URL}/v1/chat/completions"
MODELS_URL = f"{BASE_URL}/v1/models"

# How long to wait for a reply. A 3B model on CPU can be slow, but if it's
# taking minutes something is wrong rather than slow.
TIMEOUT_SECONDS = 120

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
        "default": 0.7,
        "help": (
            "Reshapes the model's probability distribution over next tokens before "
            "picking one. Low (0.1-0.3) = flattens toward the single most likely "
            "token -> repetitive, predictable. High (1.2+) = flattens the "
            "DIFFERENCES between options -> more surprising, sometimes incoherent."
        ),
    },
    "top_p": {
        "cast": float,
        "default": 0.9,
        "help": (
            "Nucleus sampling. Only keep the smallest set of tokens whose "
            "probabilities add up to top_p, then sample from just those. "
            "1.0 = consider everything (riskier). 0.5 = only the most confident "
            "half of the options (safer)."
        ),
    },
    "top_k": {
        "cast": int,
        "default": 20,
        "help": (
            "Blunter than top_p: keep only the K most-likely tokens, discard the "
            "rest no matter how confident the model was. top_k=1 means always pick "
            "the single most likely token (fully deterministic). 0 disables it."
        ),
    },
    "repetition_penalty": {
        "cast": float,
        "default": 1.1,
        "help": (
            "Punishes tokens the model has already used recently, so it doesn't "
            "loop ('I think that I think that...'). Too high (>1.3) makes it "
            "avoid words it actually needs to repeat, and output gets weird."
        ),
    },
    "max_tokens": {
        "cast": int,
        "default": 150,
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


def call_model(prompt, overrides=None):
    """Send one prompt and return (reply_text, settings_actually_used).

    Raises ModelError - never a raw requests exception - so callers can
    print one clear line and carry on.
    """
    params = {**settings, **(overrides or {})}
    payload = {"messages": [{"role": "user", "content": prompt}]}
    payload.update({key: params[key] for key in SETTINGS_SPEC})

    try:
        response = requests.post(CHAT_URL, json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"], params
    except requests.exceptions.ConnectionError:
        raise ModelError(OFFLINE_HINT)
    except requests.exceptions.Timeout:
        raise ModelError(
            f"The model didn't reply within {TIMEOUT_SECONDS}s. Try a smaller "
            "max_tokens, or check whether the server is stuck loading."
        )
    except requests.exceptions.HTTPError as error:
        raise ModelError(
            f"The API rejected that request (HTTP {error.response.status_code}). "
            "Is a model actually loaded in text-generation-webui?"
        )
    except (KeyError, IndexError, ValueError) as error:
        raise ModelError(f"Couldn't make sense of the API's reply: {error}")


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
        reply, _ = call_model(prompt)
    except ModelError as error:
        print(f"\n{error}")
        return
    print(f"\nmodel: {reply}")


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
