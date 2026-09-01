"""
Interactive playground for the REAL local model (Qwen2.5-3B, running in
text-generation-webui with --api enabled at http://127.0.0.1:5000).

Unlike phase1_interactive.py (which trains a tiny model from scratch),
this talks to your actual chat model over its API and lets you feel the
effect of sampling settings directly - including running the same prompt
side-by-side at two different settings, which is the fastest way to
build intuition for what each one actually does.

Requires: text-generation-webui running with --api (already set in
chat-llm/user_data/CMD_FLAGS.txt) and a model loaded.

Run with: python model_playground.py
"""

import sys

import requests

# Windows' console defaults to a codepage that can't print every character
# the model might generate (e.g. CJK punctuation) - force UTF-8 so a reply
# never crashes the print statement.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API_URL = "http://127.0.0.1:5000/v1/chat/completions"

# Current settings, adjustable through the menu. These map directly onto
# what we covered earlier: temperature reshapes the probability
# distribution, top_p/top_k cut it off, repetition_penalty discourages
# looping.
settings = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 20,
    "repetition_penalty": 1.1,
    "max_tokens": 150,
}

EXPLANATIONS = {
    "temperature": (
        "Reshapes the model's probability distribution over next tokens before "
        "picking one. Low (0.1-0.3) = flattens toward the single most likely "
        "token -> repetitive, predictable. High (1.2+) = flattens the "
        "DIFFERENCES between options -> more surprising, sometimes incoherent."
    ),
    "top_p": (
        "Nucleus sampling. Only keep the smallest set of tokens whose "
        "probabilities add up to top_p, then sample from just those. "
        "1.0 = consider everything (riskier). 0.5 = only the most confident "
        "half of the options (safer)."
    ),
    "top_k": (
        "Blunter than top_p: keep only the K most-likely tokens, discard the "
        "rest no matter how confident the model was. top_k=1 means always pick "
        "the single most likely token (fully deterministic). 0 disables it."
    ),
    "repetition_penalty": (
        "Punishes tokens the model has already used recently, so it doesn't "
        "loop ('I think that I think that...'). Too high (>1.3) makes it "
        "avoid words it actually needs to repeat, and output gets weird."
    ),
    "max_tokens": (
        "Not a sampling parameter - just a hard cap on how many tokens the "
        "response is allowed to be."
    ),
}


def call_model(prompt, overrides=None):
    params = {**settings, **(overrides or {})}
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "top_k": params["top_k"],
        "repetition_penalty": params["repetition_penalty"],
        "max_tokens": params["max_tokens"],
    }
    response = requests.post(API_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"], params


def do_chat():
    print("\n--- Chat (current settings shown below) ---")
    print_settings()
    prompt = input("\nyour prompt: ")
    print("\nthinking...")
    reply, used = call_model(prompt)
    print(f"\nmodel: {reply}")


def do_compare():
    print("\n--- Compare: same prompt, two settings ---")
    prompt = input("prompt to compare: ")

    print("\nSetting A is your current settings:")
    print_settings()
    print("\nSetting B - enter values to change (blank = keep current):")
    overrides = {}
    for key in ("temperature", "top_p", "top_k", "repetition_penalty"):
        raw = input(f"  {key} [{settings[key]}]: ").strip()
        if raw:
            overrides[key] = float(raw) if "." in raw or key != "top_k" else int(raw)

    print("\ngenerating both responses...")
    reply_a, used_a = call_model(prompt)
    reply_b, used_b = call_model(prompt, overrides)

    print(f"\n=== Setting A {trim(used_a)} ===\n{reply_a}")
    print(f"\n=== Setting B {trim(used_b)} ===\n{reply_b}")
    print(
        "\nSame prompt, same model weights - any difference above came purely "
        "from how the next token was SELECTED at each step, not from the model "
        "'thinking' differently."
    )


def do_adjust():
    print("\n--- Adjust settings ---")
    for key in settings:
        raw = input(f"{key} [{settings[key]}] (blank = keep): ").strip()
        if raw:
            settings[key] = float(raw) if "." in raw or key not in ("top_k", "max_tokens") else int(raw)
    print("\nUpdated:")
    print_settings()


def do_explain():
    print("\n--- What each setting does ---")
    for key, text in EXPLANATIONS.items():
        print(f"\n{key} (current: {settings[key]}):\n  {text}")


def print_settings():
    for key, value in settings.items():
        print(f"  {key}: {value}")


def trim(d):
    return "(" + ", ".join(f"{k}={v}" for k, v in d.items() if k != "max_tokens") + ")"


def menu():
    print("\n" + "=" * 55)
    print("Model playground - talking to your real Qwen2.5-3B model")
    print("=" * 55)
    print("1) Chat")
    print("2) Compare the same prompt at two settings")
    print("3) Adjust settings")
    print("4) Explain what each setting does")
    print("5) Quit")


if __name__ == "__main__":
    try:
        requests.get("http://127.0.0.1:5000/v1/models", timeout=5)
    except requests.exceptions.ConnectionError:
        print("Can't reach the model API at http://127.0.0.1:5000 - is")
        print("text-generation-webui running? (start_windows.bat in chat-llm/)")
        raise SystemExit(1)

    while True:
        menu()
        choice = input("choose (1-5): ").strip()
        if choice == "1":
            do_chat()
        elif choice == "2":
            do_compare()
        elif choice == "3":
            do_adjust()
        elif choice == "4":
            do_explain()
        elif choice == "5":
            break
        else:
            print("Not a valid option, try again.")
