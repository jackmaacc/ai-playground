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

Each topic is still a standalone script too - main.py just imports and
calls into them, it doesn't duplicate their logic. Run any of them
directly if you only want that one thing.

Run with: python main.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import phase1_gradient_descent as gradient_descent
import phase1_interactive as linear_regression
import model_playground


def run_visualizations():
    print("\n" + "=" * 50)
    print("Generate visualizations (saves PNGs into learning/)")
    print("=" * 50)
    print("1) Gradient descent paths + convergence")
    print("2) Linear regression fit + loss curve")
    print("3) Both")
    print("4) Back")
    choice = input("choose (1-4): ").strip()

    if choice in ("1", "3"):
        import phase1_gradient_descent_viz
        phase1_gradient_descent_viz.run()
    if choice in ("2", "3"):
        import phase1_linear_regression_viz
        phase1_linear_regression_viz.run()
    if choice not in ("1", "2", "3", "4"):
        print("Not a valid option, try again.")


def run_llm_playground():
    if not model_playground.api_reachable():
        print("\nCan't reach the model API at http://127.0.0.1:5000 - is")
        print("text-generation-webui running? (start_windows.bat in chat-llm/)")
        print("The rest of the program works fine without it; only the LLM")
        print("options need the model running.")
        return

    print("\n" + "=" * 50)
    print("Real Qwen model - guided lessons or free play")
    print("=" * 50)
    print("1) Guided lessons (structured, teaches you the settings one by one)")
    print("2) Free-play playground (chat, compare, adjust, explain)")
    print("3) Back")
    choice = input("choose (1-3): ").strip()

    if choice == "1":
        import lessons
        lessons.main()
    elif choice == "2":
        model_playground.run()
    elif choice == "3":
        return
    else:
        print("Not a valid option, try again.")


def main_menu():
    print("\n" + "=" * 55)
    print("AI/ML Learning Path - interactive menu")
    print("(see LEARNING_PATH.md for the full roadmap)")
    print("=" * 55)
    print("1) Gradient descent basics           (Phase 0/1 foundations)")
    print("2) Linear regression trainer         (Phase 1 project)")
    print("3) Generate visualizations (PNGs)    (Phase 1)")
    print("4) LLM lessons / playground - real Qwen model  (Phase 0/2)")
    print("5) Quit")


def main():
    while True:
        main_menu()
        choice = input("choose (1-5): ").strip()
        if choice == "1":
            gradient_descent.run()
        elif choice == "2":
            linear_regression.run()
        elif choice == "3":
            run_visualizations()
        elif choice == "4":
            run_llm_playground()
        elif choice == "5":
            break
        else:
            print("Not a valid option, try again.")


if __name__ == "__main__":
    main()
