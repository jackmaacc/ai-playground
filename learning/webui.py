"""
Web UI for the whole learning path - same idea as main.py's menu, but as
a browser interface instead of typing numbers into a terminal.

Runs on the chat-llm venv (already has gradio + requests installed for
text-generation-webui itself). Launch with:

    C:\\Users\\jackm\\ai-playground\\chat-llm\\installer_files\\env\\python.exe webui.py

Opens at http://127.0.0.1:7862
"""

import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, we're handing figures to gradio
import matplotlib.pyplot as plt
import gradio as gr

from model_playground import call_model, EXPLANATIONS, api_reachable
from lessons import LESSONS
from phase1_gradient_descent import f, f_prime
from phase1_linear_regression import hours as lr_hours, scores as lr_scores, predict as lr_predict, mse_loss, gradients as lr_gradients, true_m, true_b


# ---------------------------------------------------------------- LLM tab

def do_generate(prompt, temperature, top_p, top_k, repetition_penalty, max_tokens):
    if not prompt.strip():
        return "Type a prompt first."
    reply, _ = call_model(prompt, {
        "temperature": temperature, "top_p": top_p, "top_k": int(top_k),
        "repetition_penalty": repetition_penalty, "max_tokens": int(max_tokens),
    })
    return reply


SETTINGS_HELP = "\n\n".join(f"**{k}**: {v}" for k, v in EXPLANATIONS.items())


# ---------------------------------------------------------------- Compare tab

def do_compare(prompt, ta, pa, ka, ra, tb, pb, kb, rb):
    if not prompt.strip():
        return "Type a prompt first.", ""
    reply_a, _ = call_model(prompt, {"temperature": ta, "top_p": pa, "top_k": int(ka), "repetition_penalty": ra, "max_tokens": 150})
    reply_b, _ = call_model(prompt, {"temperature": tb, "top_p": pb, "top_k": int(kb), "repetition_penalty": rb, "max_tokens": 150})
    return reply_a, reply_b


# ---------------------------------------------------------------- Lessons tab

LESSON_TITLES = [f"{l['number']}. {l['title']}" for l in LESSONS]


def lesson_concept(choice):
    lesson = LESSONS[LESSON_TITLES.index(choice)]
    return f"### {lesson['title']}\n\n{lesson['concept']}\n\n**Prompt:** \"{lesson['prompt']}\""


def run_lesson_experiment(choice):
    lesson = LESSONS[LESSON_TITLES.index(choice)]
    reply_a, _ = call_model(lesson["prompt"], lesson["setting_a"])
    reply_b, _ = call_model(lesson["prompt"], lesson["setting_b"])
    label_a = ", ".join(f"{k}={v}" for k, v in lesson["setting_a"].items())
    label_b = ", ".join(f"{k}={v}" for k, v in lesson["setting_b"].items())
    return f"**A** ({label_a})\n\n{reply_a}", f"**B** ({label_b})\n\n{reply_b}"


def reveal_lesson(choice):
    lesson = LESSONS[LESSON_TITLES.index(choice)]
    return lesson["reveal"]


# ---------------------------------------------------------------- Gradient descent tab

def run_gradient_descent(start_x, learning_rate, steps):
    plt.close("all")  # each Run click makes a new figure; clean up past ones
    x = start_x
    xs_hist = [x]
    for _ in range(int(steps)):
        x = x - learning_rate * f_prime(x)
        xs_hist.append(x)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    lo = min(min(xs_hist), 3) - 1
    hi = max(max(xs_hist), 3) + 1
    import numpy as np
    curve_x = np.linspace(lo, hi, 300)
    ax.plot(curve_x, f(curve_x), color="lightgray", linewidth=2)
    ax.plot(xs_hist, [f(v) for v in xs_hist], "o-", color="crimson", markersize=4)
    ax.scatter([3], [f(3)], color="green", marker="*", s=150, zorder=3, label="true minimum")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    fig.tight_layout()

    final_x = xs_hist[-1]
    distance = abs(final_x - 3)
    if distance != distance:
        verdict = "Diverged to NaN - learning rate too big, the update blew up."
    elif distance > 0.5:
        verdict = f"Still {distance:.3f} from the true minimum after {int(steps)} steps - try a different learning rate or more steps."
    else:
        verdict = f"Converged: within {distance:.4f} of the true minimum (x=3)."

    return fig, f"Final x = {final_x:.5f}\n{verdict}"


# ---------------------------------------------------------------- Linear regression tab

def run_linear_regression(learning_rate, steps):
    plt.close("all")  # each Run click makes a new figure; clean up past ones
    m, b = 0.0, 0.0
    loss_hist = []
    for _ in range(int(steps)):
        loss_hist.append(mse_loss(m, b))
        gm, gb = lr_gradients(m, b)
        m -= learning_rate * gm
        b -= learning_rate * gb

    import numpy as np
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.scatter(lr_hours, lr_scores, color="black", zorder=3, label="real data")
    xs = np.linspace(0, 10, 100)
    ax1.plot(xs, lr_predict(xs, m, b), color="crimson", linewidth=2, label="learned fit")
    ax1.set_xlabel("hours studied")
    ax1.set_ylabel("exam score")
    ax1.legend()

    ax2.plot(range(len(loss_hist)), loss_hist, color="crimson")
    ax2.set_xlabel("step")
    ax2.set_ylabel("loss (MSE)")
    fig.tight_layout()

    summary = (
        f"Learned: score = {m:.3f} * hours + {b:.3f}\n"
        f"Truth:   score = {true_m} * hours + {true_b}\n"
        f"Final loss: {loss_hist[-1]:.3f} (started at {loss_hist[0]:.1f})"
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
        gd_summary = gr.Textbox(label="Result", lines=2)
        gd_btn.click(run_gradient_descent, [gd_start, gd_lr, gd_steps], [gd_plot, gd_summary])

    with gr.Tab("Linear Regression (toy)"):
        gr.Markdown("Fitting score = m*hours + b to real-shaped data with gradient descent.")
        with gr.Row():
            lrg_lr = gr.Slider(0.0001, 0.05, value=0.01, step=0.0001, label="learning_rate")
            lrg_steps = gr.Slider(10, 2000, value=500, step=10, label="steps")
        lrg_btn = gr.Button("Run", variant="primary")
        lrg_plot = gr.Plot()
        lrg_summary = gr.Textbox(label="Result", lines=4)
        lrg_btn.click(run_linear_regression, [lrg_lr, lrg_steps], [lrg_plot, lrg_summary])


if __name__ == "__main__":
    if not api_reachable():
        print("Warning: can't reach the model API at http://127.0.0.1:5000 -")
        print("the Chat/Compare/Lessons tabs won't work until text-generation-webui")
        print("is running. The Gradient Descent and Linear Regression tabs work regardless.")
    demo.launch(server_name="127.0.0.1", server_port=7862, inbrowser=True)
