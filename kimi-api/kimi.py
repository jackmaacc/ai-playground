"""
Command-line client for Moonshot AI's Kimi (K2) API.

The Kimi API speaks the same wire protocol as OpenAI's chat-completions
endpoint, so we use the `openai` Python SDK and just point it at a
different base URL. That's the single most useful fact about hosted LLM
APIs today: nearly all of them (Moonshot, DeepSeek, Groq, Together,
your local text-generation-webui in ../learning) accept the same JSON
shape, so one client works everywhere and switching providers is a URL
and a key.

Three ways to use it:

    python kimi.py                      interactive chat (remembers the conversation)
    python kimi.py "explain attention"  one-shot question, answer printed, exit
    python kimi.py --models             list the models your key can use

Setup: copy .env.example to .env and paste in your key (see README.md).
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI

DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"

# Moonshot's own recommended temperature for K2 - it's lower than the 1.0
# most APIs default to, because K2 was tuned expecting it.
DEFAULT_TEMPERATURE = 0.6

# If no model is chosen, take the first of these that the API says exists.
# Ordered newest/most capable first. Anything not listed here still works
# via --model, and --models shows you the live list, so this never goes
# stale in a way that breaks anything - it just picks a default.
PREFERRED_MODELS = (
    "kimi-k2.5",
    "kimi-k2-thinking",
    "kimi-k2-0905-preview",
    "kimi-k2-turbo-preview",
    "kimi-k2-0711-preview",
    "kimi-latest",
)

SYSTEM_PROMPT = (
    "You are Kimi, a helpful assistant. Explain things from first principles "
    "and show intermediate steps when reasoning about maths or code."
)


def make_client():
    """Build the SDK client from .env / environment, failing with a sentence
    rather than a traceback if the key is missing."""
    load_dotenv()  # reads ./.env if present; real env vars still win
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        sys.exit(
            "No MOONSHOT_API_KEY found.\n"
            "  cp .env.example .env   then paste your key into .env\n"
            "  (keys come from https://platform.moonshot.ai/console/api-keys)"
        )
    base_url = os.environ.get("MOONSHOT_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def list_models(client):
    """Return the model ids this key is allowed to call, sorted."""
    return sorted(model.id for model in client.models.list())


def pick_model(client, requested=None):
    if requested:
        return requested
    available = list_models(client)
    for name in PREFERRED_MODELS:
        if name in available:
            return name
    if available:
        return available[0]
    sys.exit("The API returned an empty model list - check your key's permissions.")


def stream_reply(client, model, messages, temperature):
    """Send the conversation and print the reply token-by-token as it
    arrives. Returns the full reply text so it can be appended to history.

    Streaming isn't just cosmetic: with a "thinking" model the visible
    answer can take a while, and seeing tokens arrive tells you it's alive.
    Thinking models also send their reasoning in a separate
    `reasoning_content` field before the answer proper; we show it dimmed
    so you can watch the model work without it polluting the answer.
    """
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )

    reply_parts = []
    in_reasoning = False
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            if not in_reasoning:
                print("\033[2m[thinking] ", end="")
                in_reasoning = True
            print(reasoning, end="", flush=True)

        if delta.content:
            if in_reasoning:
                print("\033[0m\n")  # end the dim block before the answer
                in_reasoning = False
            print(delta.content, end="", flush=True)
            reply_parts.append(delta.content)

    if in_reasoning:
        print("\033[0m")
    print()
    return "".join(reply_parts)


def chat_loop(client, model, temperature):
    print(f"Chatting with {model} (temperature {temperature}). Ctrl+C or 'quit' to exit.\n")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            user_text = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit", "q"}:
            return

        messages.append({"role": "user", "content": user_text})
        print("\nkimi: ", end="", flush=True)
        try:
            reply = stream_reply(client, model, messages, temperature)
        except KeyboardInterrupt:
            print("\n(interrupted - that turn was dropped)\n")
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": reply})
        print()


def main():
    parser = argparse.ArgumentParser(description="Talk to Kimi K2 via the Moonshot API.")
    parser.add_argument("prompt", nargs="?", help="ask one question and exit (omit for interactive chat)")
    parser.add_argument("--model", default=os.environ.get("MOONSHOT_MODEL"), help="model id (default: newest K2 your key can see)")
    parser.add_argument("--models", action="store_true", help="list available models and exit")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    args = parser.parse_args()

    client = make_client()
    try:
        if args.models:
            print("\n".join(list_models(client)))
            return

        model = pick_model(client, args.model)
        if args.prompt:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": args.prompt},
            ]
            stream_reply(client, model, messages, args.temperature)
        else:
            chat_loop(client, model, args.temperature)

    except AuthenticationError:
        sys.exit("The API rejected the key (401). Check MOONSHOT_API_KEY in .env, and that the base URL matches where the key was issued (.ai vs .cn).")
    except APIConnectionError as error:
        sys.exit(f"Couldn't reach the API at {client.base_url}: {error}")
    except APIStatusError as error:
        sys.exit(f"API error {error.status_code}: {error.message}")


if __name__ == "__main__":
    main()
