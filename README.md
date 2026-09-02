# ai-playground

A local, offline AI learning environment. Three applications live side by
side in this repository:

| Folder | What it is | What it's for |
|---|---|---|
| `chat-llm/` | [text-generation-webui](https://github.com/oobabooga/text-generation-webui) running a local Qwen chat model | Talking to a language model that runs on your own GPU, and the API the learning program uses |
| `image-gen/` | [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) | Generating images locally |
| `learning/` | A custom learning program, in the terminal and in the browser | Learning how machine learning actually works, by building it and by experimenting on the two apps above |

`chat-llm/` and `image-gen/` are large third-party projects. They are **not
tracked in Git** (see the `.gitignore`), and this repository never edits
their internal source. Everything custom lives in `learning/`.

The roadmap this follows is in [`LEARNING_PATH.md`](LEARNING_PATH.md).

---

## Hardware assumptions

- **Windows** (the launchers are `.bat` files and the app manager uses Windows process APIs).
- **An NVIDIA GPU with at least 8 GB of memory.** This was built on an RTX 3050 8 GB. The Qwen model and Stable Diffusion each fit on their own; running both at once is tight (see *CUDA out of memory* below).
- **About 32 GB of system RAM** is comfortable. Less works, but model loading gets slow.
- **Roughly 20 GB of disk** once the two apps and their models are installed.

No GPU at all? The learning program's maths lessons (gradient descent, regression, overfitting) run fine on CPU. Only the language-model and image parts need the GPU.

## Installation requirements

Three different Pythons are involved, and that is normal:

| Python | Where | Used for |
|---|---|---|
| System Python 3.13 | on your PATH | the terminal learning program, the app manager, the tests |
| `chat-llm/installer_files/env/python.exe` | bundled with chat-llm | chat-llm itself, and the browser learning program (it has gradio) |
| `image-gen/venv/Scripts/python.exe` | bundled with image-gen | image-gen itself |

The system Python needs only three small packages, which are probably already
installed:

```
python -m pip install -r requirements.txt
```

There are separate files for the plotting extras (`requirements-plot.txt`),
the web UI (`requirements-web.txt` — usually not needed, the bundled
interpreters already have it) and testing (`requirements-dev.txt`, optional).
Each file explains which Python it is for.

The two big apps install themselves the first time you run their launcher
(`chat-llm/start_windows.bat`, `image-gen/webui-user.bat`). That first run
downloads a lot and takes a while.

## Where the models go

Model files are **deliberately not stored in Git**. They are gigabytes each,
and they are downloads, not source code. Put them here:

| Kind | Folder | Example |
|---|---|---|
| Language model (GGUF) | `chat-llm/user_data/models/` | `qwen2.5-3b-instruct-q4_k_m.gguf` |
| Stable Diffusion checkpoint | `image-gen/models/Stable-diffusion/` | `v1-5-pruned-emaonly.safetensors` |

chat-llm loads the model named in `chat-llm/user_data/CMD_FLAGS.txt` at
startup. Forge picks the first checkpoint it finds if none is selected.

---

## Starting and stopping the apps

### The easy way: the app manager

```
python learning/main.py
```

then choose **App manager**. It shows what is running, on which port, with
which model, and how much GPU memory is in use, and it has Start / Stop /
Restart / Open for each app. You can also run it directly:

```
python learning/manager.py
```

Two rules it follows, so it never breaks anything else on your computer:

- **It only stops apps that it started.** If you launched chat-llm yourself, the manager will show it as running but will tell you to close it from wherever you opened it.
- **It never stops something just because it is using a port** the manager expected. It tells you what is on the port and leaves it alone.

### The manual way

Each app has its own launcher. Run it from its own folder:

| App | Start | Stop |
|---|---|---|
| chat-llm | `chat-llm\start_windows.bat` | close its window, or Ctrl+C in it |
| image-gen | `image-gen\webui-user.bat` | close its window, or Ctrl+C in it |
| learning web UI | `chat-llm\installer_files\env\python.exe learning\webui.py` | Ctrl+C |

### Startup order

**Start chat-llm first.** Both chat-llm and image-gen prefer port 7860, and
whichever starts first takes it; the other moves to 7861. Starting chat-llm
first keeps the addresses below true. The learning web UI is pinned to 7862
and never collides.

### Default addresses

| App | Address |
|---|---|
| chat-llm chat UI | http://127.0.0.1:7860 |
| chat-llm API (used by the learning program) | http://127.0.0.1:5000 |
| image-gen | http://127.0.0.1:7861 |
| learning web UI | http://127.0.0.1:7862 |

Everything listens on `127.0.0.1` only — reachable from this computer and
nothing else. That is deliberate: none of these apps have any login, so
exposing them to your network would let anyone on it use your GPU and read
your chats. If you truly need that, set the `AIPLAY_BIND_HOST` environment
variable; the manager will warn you every time.

Every port and path can be changed with an `AIPLAY_*` environment variable
(see `learning/config.py`), but nothing needs setting for normal use.

## Selecting models

- **chat-llm:** the *Model* tab in its web page lists everything in `user_data/models/`. Choose one and click Load. To change what loads automatically at startup, edit the `--model` line in `chat-llm/user_data/CMD_FLAGS.txt`.
- **image-gen:** the *Checkpoint* dropdown at the top-left of its page. If you have just added a file, click the refresh button next to the dropdown — no restart needed.

---

## The learning program

### In the terminal

```
python learning/main.py
```

A numbered menu. Type a number, press Enter. Typos are forgiven; Ctrl+C
backs up one level instead of quitting. The topics, in order:

1. **Gradient descent** — the one loop every trained model is built from. Try a learning rate that is too big, too small, and exactly 1.0, and read what happened.
2. **Linear regression** — the same loop, now learning from data. It trains on 15 students and is scored on 5 it never saw, and it tells you the difference.
3. **Overfitting** — give a model more freedom and watch it get better on the data it has seen and worse on everything else.
4. **Visualisations** — the same lessons as pictures (PNGs saved into `learning/`).
5. **LLM lessons and playground** — needs chat-llm running. Guided lessons on temperature, top-p, top-k and repetition penalty, each run several times so you can tell a real effect from luck; then a free-play chat.
6. **App manager** — described above.

Every lesson is also a standalone script (`python learning/phase1_gradient_descent.py` and so on), and the maths lives in files you can read: `phase1_gradient_descent.py`, `phase1_linear_regression.py`, `phase1_overfitting.py`.

### In the browser

Same lessons, with sliders and plots. Start it from the app manager, or by
hand with one of the bundled interpreters:

```
chat-llm\installer_files\env\python.exe learning\webui.py
```

Then open http://127.0.0.1:7862. The *Chat with Qwen* tab streams replies as
they are generated, has a **Stop** button that really stops the model (not
just the display), shows how full the conversation is, and warns you when a
reply was cut off or is repeating itself.

### Running the tests

```
python -m unittest discover -s learning/tests -t learning
```

About 150 tests, none of which need a model, a GPU, a browser, or the
internet. The ones for the web UI skip themselves unless gradio is present
— run them under the chat-llm interpreter to include them.

---

## Chat mode, Notebook mode, and raw completion

chat-llm's web page has several tabs that send text to the same model in
different ways, and mixing them up is the most common source of confusion:

- **Chat** (with mode *chat-instruct* or *instruct*) wraps your message in the model's conversation format, so the model knows where your turn ends and where its reply should stop. **Use this for talking to the model.**
- **Notebook** and **Default** send your text to the model *raw*, with no wrapping. The model just continues the text. It has no idea a conversation is happening and nothing tells it to stop, so it writes until it hits the length limit — and the output is pasted back into the box, so the next run is longer still. This is the right tool for "continue this story" and the wrong tool for a conversation.
- The **API** the learning program uses (`/v1/chat/completions`) behaves like Chat mode: the wrapping is applied for you.

---

## Common problems and what to do

### "You do not have any model! Please download at least one model in [models/Stable-diffusion]"

image-gen has no checkpoint. Download a Stable Diffusion checkpoint (`.safetensors`) into `image-gen/models/Stable-diffusion/` and restart it. SD 1.5 (about 4 GB) runs comfortably on an 8 GB card. Nothing in this repository downloads models for you.

### "Can't reach the model API at http://127.0.0.1:5000"

chat-llm is not running, or is still loading. Start it (app manager, or `chat-llm\start_windows.bat`) and wait until its window says the API is at `http://127.0.0.1:5000/v1`. Everything in the learning program that doesn't need the model keeps working meanwhile.

### The API answers, but every reply fails (HTTP 500) — the manager says DEGRADED

chat-llm is really two processes: a web server you talk to, and a model
process behind it that does the generating. The model process can stop
while the web server keeps running, and then `/v1/models` still says
"fine" while every chat request fails. The app manager checks by actually
generating one token, so it shows **DEGRADED** and says *"its model
backend process has exited"*. Fix: **Restart** chat-llm from the manager
(Start will refuse, on purpose, because the old process is still there).
This has happened on this machine; the logs did not record why the model
process exited, so treat the cause as unknown rather than assuming it.

### "Port already in use"

Something else is listening on the port. The app manager's *Show status* names the program holding it. Either close that program, or set a different port with an environment variable, e.g. `AIPLAY_LEARNING_WEB_PORT=7870`. The manager will not stop the other program for you.

### CUDA out of memory

The GPU is full. The usual cause is running chat-llm and image-gen at the same time on an 8 GB card. Stop one (the manager shows GPU memory use), or use a smaller model or a smaller image size.

### The model keeps repeating itself

You'll see the same sentences cycling, and the reply runs until the length limit. Things that help, in order of cheapness:

1. **Clear the conversation** (the *Clear conversation* button in the learning web UI, or *New chat* in chat-llm) and ask again.
2. Lower **max_tokens**, so a runaway costs seconds instead of minutes.
3. Raise **repetition_penalty** a little (1.1–1.2).
4. Make sure you are in **Chat** mode, not Notebook (see above).

The learning program detects repetition and says so. It does *not* claim to know why it happened in your case — that depends on the model, the prompt, the settings and the history together. `learning/tests/live/repetition_experiment.py` runs a small controlled comparison against your own model if you want to investigate.

### "This reply was CUT OFF, not finished"

The model hit the `max_tokens` limit while still talking. If you wanted the rest, raise the limit and ask again. If it happens every time, the model is probably repeating itself rather than having more to say — see the previous entry.

### "Error loading script: soft_inpainting.py … No module named 'joblib'"

An optional image-gen extension is missing a package. image-gen still starts and works; only that one extension is unavailable. This repository does not fix third-party environments automatically. If you want the extension: `image-gen\venv\Scripts\python.exe -m pip install joblib`.

### "INCOMPATIBLE PYTHON VERSION … tested with 3.10.6 Python, but you have 3.13"

image-gen prints this at startup because its bundled environment was built on Python 3.13, newer than the version it was tested with. On this machine it runs anyway. If you ever hit an install failure inside image-gen, that warning is the first suspect, and the fix (a Python 3.10 environment) is documented in the message itself. This repository reports the mismatch but does not change image-gen's environment.

---

## What is and isn't in Git

Tracked: everything in `learning/`, this README, the roadmap, the requirements files.

Never tracked, on purpose: the two third-party apps, model weights, virtual
environments, certificates, logs, and the manager's runtime state
(`learning/.runtime/`). If `git status` ever shows a multi-gigabyte file,
something is wrong — check `.gitignore` before committing.
