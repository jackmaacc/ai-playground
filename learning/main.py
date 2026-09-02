"""
Single entry point for the whole learning path (see LEARNING_PATH.md).

Ties together everything built so far into one interactive program instead
of separate scripts you have to remember to run individually:

  - Gradient descent from scratch          (phase1_gradient_descent.py)
  - Linear regression trainer              (phase1_interactive.py)
  - Visualizations (PNGs) for both of those
  - Guided lessons on sampling settings, run against your real Qwen model
    (lessons.py)
  - Free-play model playground             (model_playground.py)
  - The same thing in a browser            (webui.py)

Each topic is still a standalone script too - main.py just imports and
calls into them, it doesn't duplicate their logic. Run any of them
directly if you only want that one thing.

Run with: python main.py
"""

from pathlib import Path

import console

console.use_utf8_output()

import model_playground
import phase1_gradient_descent
import phase1_interactive

# The web UI needs gradio, which lives in text-generation-webui's
# environment rather than the plain system Python this usually runs on.
WEBUI_PYTHON = Path(r"C:\Users\jackm\ai-playground\chat-llm\installer_files\env\python.exe")


def generate_gradient_descent_plots():
    # Imported here rather than at the top so that starting the program
    # doesn't pay for matplotlib unless you actually ask for a plot.
    import phase1_gradient_descent_viz

    phase1_gradient_descent_viz.run()


def generate_linear_regression_plots():
    import phase1_linear_regression_viz

    phase1_linear_regression_viz.run()


def generate_all_plots():
    generate_gradient_descent_plots()
    generate_linear_regression_plots()


def run_visualizations():
    console.run_menu(
        "Generate visualizations (PNGs saved next to the scripts)",
        [
            ("Gradient descent paths + convergence", generate_gradient_descent_plots),
            ("Linear regression fit + loss curve", generate_linear_regression_plots),
            ("Both", generate_all_plots),
        ],
    )


def run_guided_lessons():
    import lessons

    lessons.run()


def run_llm_menu():
    if not model_playground.api_reachable():
        print(f"\n{model_playground.OFFLINE_HINT}")
        print("\nEverything else in this program works without it - only the")
        print("LLM options need the model running.")
        return

    console.run_menu(
        "Real Qwen model - guided lessons or free play",
        [
            ("Guided lessons (teaches the settings one at a time)", run_guided_lessons),
            ("Free-play playground (chat, compare, adjust, explain)", model_playground.run),
        ],
    )


def launch_web_ui():
    """Start the browser version, if this interpreter can run it."""
    try:
        import webui
    except ImportError:
        # Almost always a missing gradio: the system Python doesn't have
        # it, text-generation-webui's environment does.
        print("\nThe web UI needs gradio, which isn't installed for this Python.")
        print("Run it with text-generation-webui's interpreter instead:\n")
        print(f'  & "{WEBUI_PYTHON}" webui.py')
        return

    print("\nStarting the web UI - press Ctrl+C here to stop it.")
    webui.launch()


def main():
    console.run_menu(
        "AI/ML Learning Path - interactive menu",
        [
            ("Gradient descent basics           (Phase 0/1 foundations)", phase1_gradient_descent.run),
            ("Linear regression trainer         (Phase 1 project)", phase1_interactive.run),
            ("Generate visualizations (PNGs)    (Phase 1)", run_visualizations),
            ("LLM lessons / playground          (Phase 0/2)", run_llm_menu),
            ("Open the web UI in a browser", launch_web_ui),
        ],
        back_label="Quit",
        subtitle="(see LEARNING_PATH.md for the full roadmap)",
    )


if __name__ == "__main__":
    main()
