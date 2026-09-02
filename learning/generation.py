"""
What came back from the model, and whether you should trust it.

The old code kept only the text of a reply and threw everything else
away. That hid the single most useful fact the API tells you: WHY the
model stopped. A reply that ends because the model finished a thought and
a reply that ends because it hit a token ceiling look identical as
strings, and mean completely different things.

This module is pure logic - no network, no UI. It turns a raw API
response into something honest, and it measures repetition.

On repetition: everything here DETECTS repetition in a piece of text. It
does not explain what caused it. Working out why a particular model
started looping needs a live experiment against that model, not an
offline function, and this module makes no claim about causes.
"""

import re
from collections import Counter
from typing import NamedTuple, Optional

# "length" is the OpenAI-compatible way of saying "I was cut off".
FINISH_TRUNCATED = "length"
FINISH_NATURAL = "stop"


class Generation(NamedTuple):
    """One reply, with the context needed to judge it."""
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    params: dict
    cancelled: bool = False

    @property
    def truncated(self):
        """True when the model was still talking when we cut it off."""
        return self.finish_reason == FINISH_TRUNCATED

    @property
    def finished_naturally(self):
        return self.finish_reason == FINISH_NATURAL


class MalformedResponse(ValueError):
    """The API replied, but not with anything we can use."""


def parse_response(payload, params=None, cancelled=False):
    """Turn the API's JSON into a Generation.

    Tolerant about what is missing (older builds omit `usage`), strict
    about what must be present (there has to be some text).
    """
    params = params or {}
    if not isinstance(payload, dict):
        raise MalformedResponse("the API did not return a JSON object")

    choices = payload.get("choices")
    if not choices:
        raise MalformedResponse("the API returned no choices")

    first = choices[0] or {}
    message = first.get("message") or {}
    # Chat replies carry "message.content"; raw completions carry "text".
    text = message.get("content")
    if text is None:
        text = first.get("text")
    if text is None:
        raise MalformedResponse("the API returned a choice with no text")

    usage = payload.get("usage") or {}
    return Generation(
        text=text,
        finish_reason=first.get("finish_reason") or "",
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        total_tokens=int(usage.get("total_tokens") or 0),
        params=dict(params),
        cancelled=cancelled,
    )


def truncation_warning(generation):
    """Explain a cut-off reply, or return None when it ended properly."""
    if generation.cancelled:
        return "You stopped this one, so the reply is incomplete on purpose."
    if not generation.truncated:
        return None
    cap = generation.params.get("max_tokens")
    used = (f" ({generation.completion_tokens} tokens used)"
            if generation.completion_tokens else "")
    return (
        "This reply was CUT OFF, not finished. The model hit the "
        f"{cap}-token limit and was still going{used}. "
        "Raise max_tokens if you want the rest - but a reply that keeps "
        "hitting the ceiling is often a sign the model is repeating itself "
        "rather than having a lot to say."
    )


# --- Repetition detection --------------------------------------------

_WORD = re.compile(r"[a-z0-9']+")


def _normalise(text):
    return _WORD.findall(text.lower())


class RepetitionReport(NamedTuple):
    looping: bool
    unique_ratio: float          # 1.0 = nothing repeats, lower = more repeats
    worst_phrase: Optional[str]
    worst_count: int
    detail: str


def repetition_report(text, phrase_words=6, min_repeats=3):
    """Measure how much a piece of text repeats itself.

    Slides a window of `phrase_words` words across the text and counts how
    often each window recurs. Degenerate loops show up dramatically: real
    prose almost never repeats an exact six-word phrase three times.

    Reports what it measured. It does NOT say why the model did it.
    """
    words = _normalise(text)
    if len(words) < phrase_words * 2:
        return RepetitionReport(False, 1.0, None, 0,
                                "too short to judge repetition")

    phrases = [" ".join(words[i:i + phrase_words])
               for i in range(len(words) - phrase_words + 1)]
    counts = Counter(phrases)
    worst_phrase, worst_count = counts.most_common(1)[0]
    unique_ratio = len(counts) / len(phrases)

    looping = worst_count >= min_repeats
    if looping:
        detail = (
            f'The phrase "{worst_phrase}" appears {worst_count} times. '
            "That is a repetition loop: the model is cycling instead of "
            "adding anything new."
        )
    elif unique_ratio < 0.75:
        detail = (
            f"Noticeably repetitive ({unique_ratio:.0%} of phrases are "
            "distinct), though nothing is looping outright."
        )
    else:
        detail = f"No significant repetition ({unique_ratio:.0%} of phrases distinct)."

    return RepetitionReport(looping, unique_ratio, worst_phrase, worst_count, detail)


# --- Context accounting ----------------------------------------------

class ContextUsage(NamedTuple):
    used: int
    limit: int
    fraction: float
    detail: str


def context_usage(total_tokens, limit):
    """How full the model's context window is.

    Worth showing because a conversation is re-sent in full on every turn:
    the cost and the slowdown grow with the whole history, not with your
    latest message.
    """
    limit = max(int(limit or 0), 1)
    used = max(int(total_tokens or 0), 0)
    fraction = used / limit

    if fraction >= 0.9:
        detail = (f"{used} of {limit} tokens ({fraction:.0%}) - nearly full. "
                  "Clear the conversation, or the oldest turns start falling "
                  "out and the model will seem to forget things.")
    elif fraction >= 0.6:
        detail = (f"{used} of {limit} tokens ({fraction:.0%}) - filling up. "
                  "Every new message re-sends this whole history.")
    else:
        detail = f"{used} of {limit} tokens ({fraction:.0%}) used."
    return ContextUsage(used, limit, fraction, detail)


def review(generation, context_limit=None):
    """Every warning worth showing about one reply, most important first."""
    notes = []

    warning = truncation_warning(generation)
    if warning:
        notes.append(warning)

    repetition = repetition_report(generation.text)
    if repetition.looping:
        notes.append(repetition.detail)

    if context_limit and generation.total_tokens:
        usage = context_usage(generation.total_tokens, context_limit)
        if usage.fraction >= 0.6:
            notes.append(usage.detail)

    return notes
