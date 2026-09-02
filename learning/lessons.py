"""
Guided lessons for teaching yourself how to work with your local Qwen
model - a structured sequence, not a free-play sandbox (that's
model_playground.py, once you're past this).

Each lesson: a short recap of the concept, a real prompt run at two
settings on your actual model, a question for YOU to answer first, then
an explanation of what's actually happening under the hood.

The LESSONS list is also what drives the "Guided Lessons" tab in
webui.py, so a lesson added here shows up in both places.

Run with: python lessons.py
"""

import console
from model_playground import ModelError, OFFLINE_HINT, api_reachable, call_model

console.use_utf8_output()


LESSONS = [
    dict(
        number=1,
        title="Temperature",
        concept=(
            "Temperature reshapes the model's probability distribution over the "
            "next token before it picks one. Low temperature flattens TOWARD the "
            "single most likely choice every time. High temperature flattens the "
            "DIFFERENCES between choices, making unlikely options more pickable."
        ),
        prompt="Write a two-sentence story about a robot who learns to paint.",
        setting_a={"temperature": 0.2, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0},
        setting_b={"temperature": 1.4, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0},
        reveal=(
            "A (temp=0.2) is close to 'greedy' - almost always the single most "
            "probable next word, which is why it reads safe and predictable. "
            "B (temp=1.4) let genuinely less-likely words compete - more "
            "surprising, sometimes at the cost of coherence. Same weights, same "
            "prompt - the ONLY thing that changed was how the next token was "
            "selected from the model's own probability distribution."
        ),
    ),
    dict(
        number=2,
        title="Top-p (nucleus sampling)",
        concept=(
            "Top-p keeps only the smallest set of tokens whose probabilities add "
            "up to p, then samples from just that set. It adapts: if the model is "
            "very confident, the set is tiny even at high p. If it's unsure, the "
            "set is bigger."
        ),
        prompt="Complete this sentence: The strangest thing I've ever seen was",
        setting_a={"temperature": 1.0, "top_p": 0.3, "top_k": 0, "repetition_penalty": 1.0},
        setting_b={"temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0},
        reveal=(
            "Same temperature both times - the only change was how much of the "
            "probability distribution was even eligible to be picked from. "
            "top_p=0.3 (A) restricted choices to a narrow, high-confidence set. "
            "top_p=1.0 (B) let the full distribution compete, including its long "
            "tail of unlikely continuations."
        ),
    ),
    dict(
        number=3,
        title="Top-k",
        concept=(
            "Top-k is blunter than top-p: keep only the K most-likely tokens, "
            "full stop, regardless of how confident the model actually was. "
            "top_k=1 means ALWAYS pick the single most likely token - fully "
            "deterministic, no randomness at all."
        ),
        prompt="Name a color.",
        setting_a={"temperature": 1.0, "top_p": 1.0, "top_k": 1, "repetition_penalty": 1.0},
        setting_b={"temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0},
        reveal=(
            "A (top_k=1) is greedy decoding - if you ran it 5 more times you'd "
            "get the exact same answer every time, because only one token is "
            "ever eligible. B (top_k=0, disabled) lets temperature actually do "
            "something, since nothing is artificially cut off first."
        ),
    ),
    dict(
        number=4,
        title="Repetition penalty",
        concept=(
            "Repetition penalty pushes DOWN the probability of tokens the model "
            "has already used recently, so it doesn't get stuck looping the same "
            "phrase. Set too high, it starts avoiding words it actually needs."
        ),
        prompt="Write a short paragraph about how vast and blue the ocean is.",
        setting_a={"temperature": 0.8, "top_p": 0.9, "top_k": 0, "repetition_penalty": 1.0},
        setting_b={"temperature": 0.8, "top_p": 0.9, "top_k": 0, "repetition_penalty": 1.3},
        reveal=(
            "With a small model and short prompt the effect may be subtle this "
            "time - that's a real, useful result too. Repetition penalty matters "
            "most on longer generations, where a model left unpenalized can "
            "genuinely spiral into repeating itself. If A and B look similar "
            "here, try this same comparison in model_playground.py with a much "
            "longer max_tokens and see if it shows up more."
        ),
    ),
]


def run_lesson(lesson):
    print("\n" + "#" * 60)
    print(f"# Lesson {lesson['number']}: {lesson['title']}")
    print("#" * 60)
    print(f"\n{lesson['concept']}")

    console.pause("[press Enter to run the experiment]")
    print(f"\nPrompt: \"{lesson['prompt']}\"\n")

    try:
        print("Generating response A...")
        reply_a, _ = call_model(lesson["prompt"], lesson["setting_a"])
        print("Generating response B...")
        reply_b, _ = call_model(lesson["prompt"], lesson["setting_b"])
    except ModelError as error:
        print(f"\n{error}")
        return

    label_a = ", ".join(f"{k}={v}" for k, v in lesson["setting_a"].items())
    label_b = ", ".join(f"{k}={v}" for k, v in lesson["setting_b"].items())

    print(f"\n=== A ({label_a}) ===\n{reply_a}")
    print(f"\n=== B ({label_b}) ===\n{reply_b}")

    # Answering before reading the explanation is the point of the
    # exercise - it forces a prediction you can actually be wrong about.
    console.ask_text(
        "\nWhat do YOU notice is different between A and B? "
        "(type anything, press Enter)\n> ",
        allow_empty=True,
    )

    print(f"\n--- What's actually happening ---\n{lesson['reveal']}")


def run_all():
    for lesson in LESSONS:
        run_lesson(lesson)
        console.pause("[press Enter to continue]")

    print("\n" + "=" * 60)
    print("Done with all lessons.")
    print("=" * 60)
    print(
        "\nNext: the free-play playground - now with the vocabulary and "
        "intuition to predict what a setting will do before you run it, "
        "instead of guessing. Also worth checking off in LEARNING_PATH.md "
        "(Phase 0)."
    )


def run():
    if not api_reachable():
        print(OFFLINE_HINT)
        return

    entries = [("Run all lessons in order", run_all)]
    entries += [
        # The default argument binds this loop's lesson; without it every
        # entry would close over the last one.
        (f"Lesson {lesson['number']}: {lesson['title']}", lambda chosen=lesson: run_lesson(chosen))
        for lesson in LESSONS
    ]

    console.run_menu(
        "Guided lessons: your local model's sampling settings",
        entries,
        subtitle=f"{len(LESSONS)} lessons, run against your actual Qwen model",
    )


# Kept because main.py calls lessons.main().
main = run


if __name__ == "__main__":
    run()
