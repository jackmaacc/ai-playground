"""
Shared console helpers for the terminal side of the learning path.

Nothing in here is machine learning - it's the plumbing, kept in one place
so the other files can stay focused on the actual maths instead of
re-implementing input parsing five times over.

Two things it guarantees, both of which the hand-rolled `input()` calls
this replaces got wrong:

  - a typo never kills a session you're halfway through. `float(input(...))`
    raises ValueError and dumps you back to the shell; these prompts just
    ask again.
  - Ctrl+C / Ctrl+D backs out one level instead of printing a traceback.

webui.py doesn't use any of this - Gradio widgets do the same job there.
"""

import sys


def use_utf8_output():
    """Windows' console defaults to a codepage that can't print every
    character a model might generate (e.g. CJK punctuation). Force UTF-8 so
    a reply never crashes the print statement."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        # stdout has been replaced with something that can't be reconfigured
        # (a pipe, a test harness). Not worth failing the program over.
        pass


class Back(Exception):
    """Raised when the user backs out of a prompt with Ctrl+C or Ctrl+D.

    run_menu() catches it and simply redraws the menu, so an interrupt
    means "take me up one level", not "throw away the session".
    """


def _ask(prompt):
    """The single place this project reads a line from the user."""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()  # move off the ^C line so the next output isn't ragged
        raise Back


def ask_text(prompt, allow_empty=False):
    while True:
        value = _ask(prompt).strip()
        if value or allow_empty:
            return value
        print("  Nothing entered - type something, or press Ctrl+C to go back.")


def ask_number(prompt, default=None, cast=float, minimum=None, maximum=None):
    """Prompt for a number, showing the default in brackets, re-asking on
    anything unparseable or out of range. Blank input takes the default."""
    label = f"{prompt} [{default}]: " if default is not None else f"{prompt}: "
    while True:
        raw = _ask(label).strip()

        if not raw:
            if default is not None:
                return default
            print("  Please enter a number.")
            continue

        try:
            value = cast(raw)
        except ValueError:
            kind = "whole number" if cast is int else "number"
            print(f"  '{raw}' isn't a {kind}.")
            continue

        if minimum is not None and value < minimum:
            print(f"  Needs to be at least {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"  Needs to be at most {maximum}.")
            continue
        return value


def ask_float(prompt, default=None, minimum=None, maximum=None):
    return ask_number(prompt, default, float, minimum, maximum)


def ask_int(prompt, default=None, minimum=None, maximum=None):
    return ask_number(prompt, default, int, minimum, maximum)


def pause(message="[press Enter to continue]"):
    _ask(f"\n{message}")


def heading(title, subtitle=None):
    width = max(len(title), len(subtitle or ""), 50)
    print("\n" + "=" * width)
    print(title)
    if subtitle:
        print(subtitle)
    print("=" * width)


def run_menu(title, entries, back_label="Back", subtitle=None):
    """Render a numbered menu and keep running the user's choices until
    they back out.

    entries: list of (label, zero-argument function) pairs. `back_label` is
    appended as the final numbered option.

    Every menu in this project used to hand-roll this same
    print/input/if-elif ladder; now they just describe their options.
    """
    while True:
        heading(title, subtitle)
        for number, (label, _) in enumerate(entries, start=1):
            print(f"{number}) {label}")
        back = len(entries) + 1
        print(f"{back}) {back_label}")

        try:
            choice = _ask(f"choose (1-{back}): ").strip()
        except Back:
            return

        if choice == str(back):
            return
        if not (choice.isdigit() and 1 <= int(choice) <= len(entries)):
            print("Not a valid option, try again.")
            continue

        try:
            entries[int(choice) - 1][1]()
        except Back:
            pass  # backed out of that action - redraw the menu
        except KeyboardInterrupt:
            print("\n(interrupted)")
