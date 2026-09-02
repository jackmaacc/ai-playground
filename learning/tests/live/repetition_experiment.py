"""
Does the local model repeat itself, and under which conditions?

An experiment, not a test, and not an explanation. It runs the same
prompt several times under each of a few conditions and reports how much
each reply repeated, using the same detector the app uses. It makes NO
claim about causes: if a condition shows more looping, that is one
observation on one model with one prompt, and the honest next step is to
change one variable and run it again.

Requires the model API to be running. Run by hand:

    python learning/tests/live/repetition_experiment.py [--samples 3]

The offline test suite never imports this file.
"""

import argparse
import sys
from pathlib import Path

LEARNING_DIR = Path(__file__).resolve().parents[2]
if str(LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(LEARNING_DIR))

import console  # noqa: E402
import generation as gen  # noqa: E402
import model_playground as mp  # noqa: E402

console.use_utf8_output()

PROMPT = "I have an exam tomorrow and I am nervous. Reassure me."

# A history that already contains a loop, so we can observe whether the
# model continues it. Observing that it does would NOT prove the history
# caused the original loop - only that this model, given this history,
# produced this output.
LOOPING_HISTORY = [
    {"role": "user", "content": PROMPT},
    {"role": "assistant", "content": (
        "You are doing great! You have prepared well, and you have got this. "
        "You are doing amazing! Just keep going and keep breathing. ") * 12},
    {"role": "user", "content": "thanks, anything else?"},
]

CONDITIONS = [
    ("fresh prompt, penalty 1.0",     dict(repetition_penalty=1.0), None),
    ("fresh prompt, penalty 1.15",    dict(repetition_penalty=1.15), None),
    ("looping history, penalty 1.0",  dict(repetition_penalty=1.0), LOOPING_HISTORY),
    ("looping history, penalty 1.15", dict(repetition_penalty=1.15), LOOPING_HISTORY),
]


def run_condition(label, overrides, history, samples, max_tokens):
    print(f"\n=== {label}  ({samples} samples, max_tokens={max_tokens}) ===")
    settings = {"temperature": 0.7, "top_p": 0.85, "top_k": 30,
                "max_tokens": max_tokens, **overrides}
    loops = 0
    for index in range(1, samples + 1):
        try:
            if history:
                result = mp.generate_once(messages=history, overrides=settings)
            else:
                result = mp.generate_once(PROMPT, settings)
        except mp.ModelError as error:
            print(f"  run {index}: {error}")
            return None
        report = gen.repetition_report(result.text)
        loops += int(report.looping)
        flag = "LOOP" if report.looping else "ok  "
        print(f"  run {index}: {flag}  finish={result.finish_reason:6} "
              f"tokens={result.completion_tokens:4}  distinct={report.unique_ratio:.0%}")
    print(f"  -> {loops} of {samples} replies looped")
    return loops


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=400)
    args = parser.parse_args()

    if not mp.api_reachable():
        print(mp.OFFLINE_HINT)
        return 1

    print("Measuring repetition across conditions. This reports what happened;")
    print("it does not say why. Vary one thing at a time to learn anything.")
    results = {}
    for label, overrides, history in CONDITIONS:
        results[label] = run_condition(label, overrides, history,
                                       args.samples, args.max_tokens)

    print("\n=== summary (loops out of samples) ===")
    for label, loops in results.items():
        shown = "n/a (error)" if loops is None else f"{loops}/{args.samples}"
        print(f"  {label:<34}{shown}")
    print("\nA difference between conditions is an observation about this model,")
    print("this prompt and these runs - not a general rule, and not a cause.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
