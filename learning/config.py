"""
One place that knows where things are, what they listen on, and what the
safe defaults are.

Everything is derived from this file's own location, so the repository can
be copied to another folder, another drive, or another Windows account and
still work. Nothing here contains a machine-specific absolute path.

Every value can be overridden with an environment variable (all prefixed
AIPLAY_) but the defaults are chosen so that nothing needs setting up for
normal local use.

Nothing in this module starts a process, opens a socket, or writes a file
at import time - it only computes values.
"""

import os
from pathlib import Path

# --- Where things are -------------------------------------------------
# config.py lives in learning/, so the repository root is one level up.
# resolve() first so symlinks and relative launches behave.
LEARNING_DIR = Path(__file__).resolve().parent
REPO_ROOT = LEARNING_DIR.parent

CHAT_LLM_DIR = REPO_ROOT / "chat-llm"
IMAGE_GEN_DIR = REPO_ROOT / "image-gen"

# Machine-specific runtime state (which app we started, its pid). Never
# committed - see .gitignore.
RUNTIME_DIR = LEARNING_DIR / ".runtime"
SERVICE_STATE_FILE = RUNTIME_DIR / "services.json"
LOG_DIR = RUNTIME_DIR / "logs"


try:
    # On importlib.reload() this name already exists in the module's
    # namespace. Keep that same class object, otherwise every reload mints
    # a new ConfigurationError and `except ConfigurationError` in code
    # that imported the old one silently stops matching.
    ConfigurationError  # noqa: B018
except NameError:
    class ConfigurationError(ValueError):
        """An AIPLAY_* variable is set to something unusable.

        Raised loudly, naming the variable, rather than silently falling
        back to a default - a typo in a port number should not quietly
        start the app somewhere you were not expecting.
        """


def _env(name, default):
    """Read AIPLAY_<name>, falling back to the default."""
    return os.environ.get(f"AIPLAY_{name}", default)


def _env_int(name, default):
    raw = _env(name, None)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigurationError(
            f"AIPLAY_{name} is set to {raw!r}, which is not a whole number."
        ) from None


def _env_port(name, default):
    port = _env_int(name, default)
    if not 1 <= port <= 65535:
        raise ConfigurationError(
            f"AIPLAY_{name} is {port}; a port must be between 1 and 65535."
        )
    return port


# --- Networking -------------------------------------------------------
# Loopback only. Anything else means other machines on the network can
# reach these apps, and none of them are built to be exposed - no
# authentication, and the model will answer anyone who asks.
LOOPBACK = "127.0.0.1"
BIND_HOST = _env("BIND_HOST", LOOPBACK)

# A wildcard bind (0.0.0.0) is an instruction to the server, not an
# address a browser can open. Links shown to the user go to loopback in
# that case, unless AIPLAY_BROWSE_HOST says otherwise.
_WILDCARDS = ("0.0.0.0", "::", "")
BROWSE_HOST = _env("BROWSE_HOST", None) or (
    LOOPBACK if BIND_HOST in _WILDCARDS else BIND_HOST
)


def is_public_bind():
    """True when the configured host is reachable from outside this machine."""
    return BIND_HOST not in (LOOPBACK, "localhost", "::1")


def public_bind_warning():
    """The warning to show when someone opts out of loopback. Returns None
    when the binding is safe, so callers can just print the result."""
    if not is_public_bind():
        return None
    return (
        f"WARNING: services are set to bind {BIND_HOST}, not {LOOPBACK}.\n"
        "That exposes them to your whole network. None of these apps have\n"
        "any authentication, so anyone who can reach the port can use your\n"
        "model, read your chats, and generate images on your GPU.\n"
        f"Unset AIPLAY_BIND_HOST to go back to {LOOPBACK}-only."
    )


# --- Ports ------------------------------------------------------------
# Defaults match how the apps land when started in the documented order.
# They are not guaranteed to be free; services.py detects conflicts and
# explains them rather than assuming.
CHAT_LLM_UI_PORT = _env_port("CHAT_LLM_UI_PORT", 7860)
CHAT_LLM_API_PORT = _env_port("CHAT_LLM_API_PORT", 5000)
IMAGE_GEN_PORT = _env_port("IMAGE_GEN_PORT", 7861)
LEARNING_WEB_PORT = _env_port("LEARNING_WEB_PORT", 7862)


def url_for(port, host=None):
    """The clickable address for a port - something a browser can open."""
    return f"http://{host or BROWSE_HOST}:{port}"


# --- The local model API ---------------------------------------------
MODEL_API_BASE_URL = _env("MODEL_API_URL", url_for(CHAT_LLM_API_PORT, LOOPBACK))
MODEL_CHAT_URL = f"{MODEL_API_BASE_URL}/v1/chat/completions"
MODEL_LIST_URL = f"{MODEL_API_BASE_URL}/v1/models"

# Forge's API is only present when it was started with --api. We never
# assume it is there; the image lessons check first and explain if not.
IMAGE_API_BASE_URL = _env("IMAGE_API_URL", url_for(IMAGE_GEN_PORT, LOOPBACK))

DEFAULT_MODEL_NAME = _env("MODEL_NAME", "qwen2.5-3b-instruct-q4_k_m.gguf")

# How much the model can hold in mind at once. Used only to show how full
# the conversation is; the server enforces the real limit. The bundled
# Qwen build reports 32768 at startup.
MODEL_CONTEXT_TOKENS = _env_int("MODEL_CONTEXT_TOKENS", 32768)

# --- Timeouts (seconds) ----------------------------------------------
# HEALTH_TIMEOUT is deliberately short: it answers "is it up?", and a slow
# answer is a no. GENERATION_TIMEOUT is long because a local model on a
# small GPU genuinely takes a while.
HEALTH_TIMEOUT = _env_int("HEALTH_TIMEOUT", 5)
GENERATION_TIMEOUT = _env_int("GENERATION_TIMEOUT", 120)
STARTUP_TIMEOUT = _env_int("STARTUP_TIMEOUT", 300)
SHUTDOWN_TIMEOUT = _env_int("SHUTDOWN_TIMEOUT", 20)

# --- Generation defaults ---------------------------------------------
# Conservative on purpose. MAX_TOKENS especially: a runaway generation is
# the failure this project actually hit, and a low cap turns a long wall
# of repeated text into a short answer plus a visible warning.
GENERATION_DEFAULTS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 20,
    "repetition_penalty": 1.1,
    "max_tokens": _env_int("MAX_TOKENS", 150),
}


# --- Interpreters -----------------------------------------------------
# The two bundled apps ship their own Python. We look for them rather than
# hardcoding a path, and return None when they are absent so callers can
# say something useful instead of crashing.
def chat_llm_python():
    candidate = CHAT_LLM_DIR / "installer_files" / "env" / "python.exe"
    return candidate if candidate.exists() else None


def image_gen_python():
    candidate = IMAGE_GEN_DIR / "venv" / "Scripts" / "python.exe"
    return candidate if candidate.exists() else None


def gradio_python():
    """An interpreter that can run the learning web UI.

    The plain system Python usually lacks gradio, while both bundled apps
    ship it, so prefer those. Returns None if nothing suitable is found.
    """
    for candidate in (chat_llm_python(), image_gen_python()):
        if candidate is not None:
            return candidate
    return None


def describe():
    """A plain-language summary, used by the manager and by tests."""
    return {
        "repo_root": str(REPO_ROOT),
        "bind_host": BIND_HOST,
        "public_bind": is_public_bind(),
        "ports": {
            "chat_llm_ui": CHAT_LLM_UI_PORT,
            "chat_llm_api": CHAT_LLM_API_PORT,
            "image_gen": IMAGE_GEN_PORT,
            "learning_web": LEARNING_WEB_PORT,
        },
        "model_api": MODEL_API_BASE_URL,
        "chat_llm_python": str(chat_llm_python() or ""),
        "image_gen_python": str(image_gen_python() or ""),
    }
