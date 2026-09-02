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

import config
import console

console.use_utf8_output()

import manager
import model_playground
import phase1_gradient_descent
import phase1_interactive
import phase1_overfitting


def generate_gradient_descent_plots():
    # Imported here rather than at the top so that starting the program
    # doesn't pay for matplotlib unless you actually ask for a plot.
    import phase1_gradient_descent_viz

    phase1_gradient_descent_viz.run()


def generate_linear_regression_plots():
    import phase1_linear_regression_viz

    phase1_linear_regression_viz.run()


def generate_overfitting_plots():
    phase1_overfitting.make_plots()


def generate_all_plots():
    generate_gradient_descent_plots()
    generate_linear_regression_plots()
    generate_overfitting_plots()


def run_visualizations():
    console.run_menu(
        "Generate visualizations (PNGs saved next to the scripts)",
        [
            ("Gradient descent paths + convergence", generate_gradient_descent_plots),
            ("Linear regression fit + loss curve", generate_linear_regression_plots),
            ("Overfitting U-curve + the fits themselves", generate_overfitting_plots),
            ("All of them", generate_all_plots),
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
    # Check for gradio without importing webui, which would build the
    # whole interface just to find out whether it could.
    import importlib.util

    if importlib.util.find_spec("gradio") is None:
        # The plain system Python usually lacks gradio; the bundled apps
        # ship it. config finds one rather than hardcoding a path.
        interpreter = config.gradio_python()
        print("\nThe web UI needs gradio, which isn't installed for this Python.")
        if interpreter:
            print("Run it with an interpreter that has it:\n")
            print(f'  & "{interpreter}" "{config.LEARNING_DIR / "webui.py"}"')
        else:
            print("No interpreter with gradio was found in this repository.")
        print("\nOr use the app manager (option above), which launches it for you.")
        return

    import webui

    print("\nStarting the web UI - press Ctrl+C here to stop it.")
    webui.launch()


def main():
    console.run_menu(
        "AI/ML Learning Path - interactive menu",
        [
            ("Gradient descent basics           (Phase 0/1 foundations)", phase1_gradient_descent.run),
            ("Linear regression trainer         (Phase 1 project)", phase1_interactive.run),
            ("Overfitting: fitting vs learning  (Phase 1)", phase1_overfitting.run),
            ("Generate visualizations (PNGs)    (Phase 1)", run_visualizations),
            ("LLM lessons / playground          (Phase 0/2)", run_llm_menu),
            ("App manager - start/stop chat-llm, image-gen, web UI", manager.run),
            ("Open the web UI in a browser", launch_web_ui),
        ],
        back_label="Quit",
        subtitle="(see LEARNING_PATH.md for the full roadmap)",
    )


if __name__ == "__main__":
    main()
