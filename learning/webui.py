"""
Web UI for the whole learning path - same idea as main.py's menu, but as
a browser interface instead of typing numbers into a terminal.

Every tab here is a front-end for logic that lives elsewhere: the maths
comes from phase1_gradient_descent.py and phase1_linear_regression.py,
the model calls from model_playground.py, the lesson text from lessons.py.
Nothing is reimplemented, so the browser and the terminal always teach the
same thing.

Runs on the chat-llm venv (already has gradio + requests installed for
text-generation-webui itself). Launch with:

    C:\\Users\\jackm\\ai-playground\\chat-llm\\installer_files\\env\\python.exe webui.py

Opens at http://127.0.0.1:7862
"""

import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, we're handing figures to gradio
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from lessons import LESSONS
from model_playground import (
    EXPLANATIONS,
    OFFLINE_HINT,
    ModelError,
    api_reachable,
    call_model,
)
from phase1_gradient_descent import MINIMUM, f
from phase1_gradient_descent import diagnose as diagnose_descent
from phase1_gradient_descent import gradient_descent
from phase1_linear_regression import closed_form_solution, hours, predict, scores
from phase1_linear_regression import diagnose as diagnose_regression
from phase1_linear_regression import train, true_b, true_m

PORT = 7862


def generate(prompt, overrides):
    """One model call, with failures turned into text the page can show.

    Without this an unreachable server throws a raw exception into
    Gradio's error box mid-lesson; this explains what to do instead.
    """
    if not prompt.strip():
        return "Type a prompt first."
    try:
        reply, _ = call_model(prompt, overrides)
    except ModelError as error:
        return f"[{error}]"
    return reply


# ---------------------------------------------------------------- LLM tab

def do_generate(prompt, temperature, top_p, top_k, repetition_penalty, max_tokens):
    return generate(prompt, {
        "temperature": temperature, "top_p": top_p, "top_k": int(top_k),
        "repetition_penalty": repetition_penalty, "max_tokens": int(max_tokens),
    })


SETTINGS_HELP = "\n\n".join(f"**{k}**: {v}" for k, v in EXPLANATIONS.items())


# ---------------------------------------------------------------- Compare tab

def do_compare(prompt, ta, pa, ka, ra, tb, pb, kb, rb):
    setting_a = {"temperature": ta, "top_p": pa, "top_k": int(ka), "repetition_penalty": ra, "max_tokens": 150}
    setting_b = {"temperature": tb, "top_p": pb, "top_k": int(kb), "repetition_penalty": rb, "max_tokens": 150}
    return generate(prompt, setting_a), generate(prompt, setting_b)


# ---------------------------------------------------------------- Lessons tab

LESSON_TITLES = [f"{lesson['number']}. {lesson['title']}" for lesson in LESSONS]


def _lesson_for(choice):
    return LESSONS[LESSON_TITLES.index(choice)]


def lesson_concept(choice):
    lesson = _lesson_for(choice)
    return f"### {lesson['title']}\n\n{lesson['concept']}\n\n**Prompt:** \"{lesson['prompt']}\""


def run_lesson_experiment(choice):
    lesson = _lesson_for(choice)
    label_a = ", ".join(f"{k}={v}" for k, v in lesson["setting_a"].items())
    label_b = ", ".join(f"{k}={v}" for k, v in lesson["setting_b"].items())
    reply_a = generate(lesson["prompt"], lesson["setting_a"])
    reply_b = generate(lesson["prompt"], lesson["setting_b"])
    return f"**A** ({label_a})\n\n{reply_a}", f"**B** ({label_b})\n\n{reply_b}"


def reveal_lesson(choice):
    return _lesson_for(choice)["reveal"]


# ---------------------------------------------------------------- Gradient descent tab

def run_gradient_descent(start_x, learning_rate, steps):
    # The algorithm itself lives in phase1_gradient_descent.py - this only
    # draws the trajectory it hands back.
    history = gradient_descent(start_x, learning_rate, int(steps), verbose=False)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    # A diverged run spans astronomical values; clip the drawing window to
    # the interesting region so the curve doesn't collapse to a flat line.
    finite = [x for x in history if abs(x) < 1e6]
    lo = min(min(finite, default=MINIMUM), MINIMUM) - 1
    hi = max(max(finite, default=MINIMUM), MINIMUM) + 1
    curve_x = np.linspace(lo, hi, 300)
    ax.plot(curve_x, f(curve_x), color="lightgray", linewidth=2)
    ax.plot(history, [f(x) for x in history], "o-", color="crimson", markersize=4)
    ax.scatter([MINIMUM], [f(MINIMUM)], color="green", marker="*", s=150, zorder=3,
               label="true minimum")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    fig.tight_layout()

    summary = f"Final x = {history[-1]:.5f}\n\n{diagnose_descent(history)}"
    return fig, summary


# ---------------------------------------------------------------- Linear regression tab

def run_linear_regression(learning_rate, steps):
    history, m, b = train(learning_rate, int(steps), verbose=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.scatter(hours, scores, color="black", zorder=3, label="real data")
    xs = np.linspace(0, 10, 100)
    ax1.plot(xs, predict(xs, m, b), color="crimson", linewidth=2, label="learned fit")

    best_m, best_b = closed_form_solution()
    ax1.plot(xs, predict(xs, best_m, best_b), color="green", linestyle="--",
             linewidth=1.5, label="best possible")
    ax1.set_xlabel("hours studied")
    ax1.set_ylabel("exam score")
    ax1.legend()

    losses = [row[3] for row in history]
    ax2.plot(range(len(losses)), losses, color="crimson")
    ax2.set_xlabel("step")
    ax2.set_ylabel("loss (MSE)")
    fig.tight_layout()

    summary = (
        f"Learned:  score = {m:.3f} * hours + {b:.3f}\n"
        f"Best possible: score = {best_m:.3f} * hours + {best_b:.3f}\n"
        f"Truth:    score = {true_m} * hours + {true_b}\n"
        f"Final loss: {losses[-1]:.3f} (started at {losses[0]:.1f})\n\n"
        f"{diagnose_regression(history)}"
    )
    return fig, summary


# ---------------------------------------------------------------- Build the app

with gr.Blocks(title="AI Learning Path") as demo:
    gr.Markdown("# AI Learning Path\nSee `LEARNING_PATH.md` for the full roadmap.")

    with gr.Tab("Chat with Qwen"):
        gr.Markdown("Play with sampling settings against your real local model.")
        prompt_in = gr.Textbox(label="Prompt", lines=2)
        with gr.Row():
            temp = gr.Slider(0.05, 2.0, value=0.7, step=0.05, label="temperature")
            top_p = gr.Slider(0.05, 1.0, value=0.9, step=0.05, label="top_p")
            top_k = gr.Slider(0, 100, value=20, step=1, label="top_k")
            rep_pen = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="repetition_penalty")
            max_tok = gr.Slider(16, 500, value=150, step=16, label="max_tokens")
        gen_btn = gr.Button("Generate", variant="primary")
        output = gr.Textbox(label="Response", lines=6)
        gen_btn.click(do_generate, [prompt_in, temp, top_p, top_k, rep_pen, max_tok], output)
        with gr.Accordion("What do these settings mean?", open=False):
            gr.Markdown(SETTINGS_HELP)

    with gr.Tab("Compare A/B"):
        gr.Markdown("Same prompt, two settings, side by side - the fastest way to see what a setting actually does.")
        cmp_prompt = gr.Textbox(label="Prompt", lines=2)
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Setting A**")
                ta = gr.Slider(0.05, 2.0, value=0.3, step=0.05, label="temperature")
                pa = gr.Slider(0.05, 1.0, value=0.9, step=0.05, label="top_p")
                ka = gr.Slider(0, 100, value=20, step=1, label="top_k")
                ra = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="repetition_penalty")
            with gr.Column():
                gr.Markdown("**Setting B**")
                tb = gr.Slider(0.05, 2.0, value=1.4, step=0.05, label="temperature")
                pb = gr.Slider(0.05, 1.0, value=0.9, step=0.05, label="top_p")
                kb = gr.Slider(0, 100, value=20, step=1, label="top_k")
                rb = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="repetition_penalty")
        cmp_btn = gr.Button("Compare", variant="primary")
        with gr.Row():
            out_a = gr.Textbox(label="Response A", lines=6)
            out_b = gr.Textbox(label="Response B", lines=6)
        cmp_btn.click(do_compare, [cmp_prompt, ta, pa, ka, ra, tb, pb, kb, rb], [out_a, out_b])

    with gr.Tab("Guided Lessons"):
        lesson_pick = gr.Dropdown(LESSON_TITLES, value=LESSON_TITLES[0], label="Lesson")
        concept_md = gr.Markdown(lesson_concept(LESSON_TITLES[0]))
        lesson_pick.change(lesson_concept, lesson_pick, concept_md)
        gr.Markdown("_Predict what will differ before you press Reveal - being wrong is the useful bit._")
        run_btn = gr.Button("Run the experiment", variant="primary")
        with gr.Row():
            lesson_a = gr.Markdown()
            lesson_b = gr.Markdown()
        run_btn.click(run_lesson_experiment, lesson_pick, [lesson_a, lesson_b])
        reveal_btn = gr.Button("Reveal explanation")
        reveal_md = gr.Markdown()
        reveal_btn.click(reveal_lesson, lesson_pick, reveal_md)

    with gr.Tab("Gradient Descent (toy)"):
        gr.Markdown("Minimizing f(x) = (x-3)^2 by hand - the mechanics behind all model training.")
        with gr.Row():
            gd_start = gr.Slider(-10, 10, value=0, step=0.5, label="start_x")
            gd_lr = gr.Slider(0.001, 1.2, value=0.1, step=0.001, label="learning_rate")
            gd_steps = gr.Slider(1, 100, value=15, step=1, label="steps")
        gd_btn = gr.Button("Run", variant="primary")
        gd_plot = gr.Plot()
        gd_summary = gr.Textbox(label="Result", lines=4)
        gd_btn.click(run_gradient_descent, [gd_start, gd_lr, gd_steps], [gd_plot, gd_summary])

    with gr.Tab("Linear Regression (toy)"):
        gr.Markdown("Fitting score = m*hours + b to real-shaped data with gradient descent.")
        with gr.Row():
            lrg_lr = gr.Slider(0.0001, 0.05, value=0.01, step=0.0001, label="learning_rate")
            lrg_steps = gr.Slider(10, 2000, value=500, step=10, label="steps")
        lrg_btn = gr.Button("Run", variant="primary")
        lrg_plot = gr.Plot()
        lrg_summary = gr.Textbox(label="Result", lines=7)
        lrg_btn.click(run_linear_regression, [lrg_lr, lrg_steps], [lrg_plot, lrg_summary])


def launch(open_browser=True):
    if not api_reachable():
        print(f"Warning: {OFFLINE_HINT}")
        print("The Chat/Compare/Lessons tabs won't work until it's running.")
        print("The Gradient Descent and Linear Regression tabs work regardless.")
    demo.launch(server_name="127.0.0.1", server_port=PORT, inbrowser=open_browser)


if __name__ == "__main__":
    launch()
