# AI/ML Engineer Learning Path

Goal: work in the AI industry as an ML/AI engineer. Project-driven — each item gets learned by building it, mostly on this local AI stack (`chat-llm/`, `image-gen/`).

Run `python learning/main.py` for one interactive menu covering everything built so far: gradient descent, linear regression with a held-out test set, overfitting, the LLM sampling lessons and playground against the real Qwen model, and an app manager for the three applications. There is a browser version too (`learning/webui.py`), and every piece also runs standalone. See `README.md` for setup.

**Current status (reviewed 2026-09-02 against the code):** Phase 0 has its numpy and linear-algebra pieces and the sampling project done; Phase 1's classical-ML half is done in raw numpy. Nothing in Phase 2 has started. One Phase 3 habit — reproducibility — arrived early as a by-product of building the tooling. (The repository also has a software test suite; that is not the same thing as a model-evaluation pipeline, and isn't ticked as one.)

How to read the boxes: a `[x]` means the implementation genuinely covers the topic, and names the file(s) that back it up. Compound topics are split so a half-finished one isn't ticked. Boxes describe what the *repository* does, not skills a person has finished acquiring. `learning/tests/test_roadmap.py` is a consistency check only: it verifies that every cited evidence file exists and that known-unbuilt topics stay unticked — it cannot judge whether a file actually proves a learning claim; that is a human's job.

## Phase 0 — Foundations (pre-calculus)
- Python for ML
  - [x] numpy arrays and vectorised maths — every loss, gradient and metric is a numpy expression, no Python loops over data (evidence: learning/phase1_linear_regression.py, learning/phase1_overfitting.py)
  - [ ] pandas basics — not touched yet; the data so far is 20 generated points, not a file
  - [ ] virtual environments — the repository *documents* which of three interpreters runs what (README.md, requirements-web.txt), but that is a repository capability, not this skill completed; unticked until a venv has been built and used unassisted
- Linear algebra for ML
  - [x] vectors, dot products, matrix multiplication — a design matrix built with `np.vander`, predictions as a matrix product, a system solved with `lstsq` (evidence: learning/phase1_overfitting.py)
  - [ ] tensor shapes — arrives with PyTorch in Phase 1
- [ ] Probability refresher: distributions, expected value, conditional probability — *partially touched* through the sampling lessons only (a probability distribution over the next token; top-p as a cumulative probability). Expected value and conditional probability not covered.
- [ ] Git / command-line fluency — plenty of Git and terminal work has happened *on* this repository (commits, pushes, ignore rules, secret checks), but much of it assisted. A skill box, not a repository box: tick it after a session done unassisted.
- [ ] **Project**: tokenization & embeddings walkthrough on the local Qwen model — see text become vectors. **This is the next lesson** (see below).
- [x] **Project (added)**: LLM sampling experiments on the running Qwen model — temperature, top-p, top-k, repetition penalty. Each setting is run several times and every lesson opens with a control (same setting twice) so a real effect can be told from the sampler's own randomness (evidence: learning/lessons.py, learning/model_playground.py)

## Phase 1 — Core ML + intro deep learning (calculus-based; can start earlier since some basic calc is already known — introduce as relevant, not strictly gated)
- Classical ML
  - [x] regression — a straight line fitted by gradient descent, checked against the exact closed-form answer (evidence: learning/phase1_linear_regression.py, learning/tests/test_linear_regression.py)
  - [ ] classification — not started
  - [x] train/test splits — a random (not sliced) hold-out; a test proves the held-out points never influence a weight update (evidence: learning/phase1_linear_regression.py, learning/tests/test_linear_regression.py)
  - [x] overfitting — a polynomial-degree sweep where training error keeps falling while test error turns upward; scaling fitted on training data only (evidence: learning/phase1_overfitting.py, learning/tests/test_overfitting.py)
  - [x] eval metrics — MSE, RMSE (in exam points), R² (evidence: learning/phase1_linear_regression.py)
  - [ ] scikit-learn — deliberately not used yet; everything above is raw numpy so the maths is visible. Worth a short pass later to see the same things as library calls.
- Gradient descent and backpropagation
  - [x] gradient descent — on a toy function and on real data; converging, crawling, oscillating and diverging are each recognised and explained (evidence: learning/phase1_gradient_descent.py, learning/tests/test_gradient_descent.py)
  - [ ] backpropagation — the gradients so far are derived by hand for a two-parameter model. The chain rule through layers only comes with the neural network project.
- [ ] **Project**: tiny neural network from scratch in raw numpy
- [ ] PyTorch fundamentals: tensors, autograd, `nn.Module`, training loops
- [ ] **Project**: rebuild the from-scratch NN in PyTorch, compare

## Phase 2 — Modern deep learning (transformers, fine-tuning, diffusion)
- [ ] Transformers deep dive: attention, multi-head attention, positional encoding (against the running Qwen model) — the sampling lessons cover only the final decoding step, not the model itself
- [ ] **Project**: LoRA fine-tune the local Qwen model on a small custom dataset
- [ ] CNNs and diffusion model basics
- [ ] **Project**: explore Forge's denoising process, possibly a small image-side LoRA — a first, practical pass (steps, samplers, CFG, seeds) is planned as image-generation lessons in the learning program; the maths of diffusion is separate

## Phase 3 — Industry-ready engineering
- MLOps basics
  - [ ] experiment tracking
  - [x] reproducibility — seeded data and seeded splits; tests assert that a re-run gives bit-identical weights (evidence: learning/phase1_linear_regression.py, learning/tests/test_linear_regression.py)
  - [ ] model versioning
  - [ ] eval pipelines — not started. The repository has a software test suite (`learning/tests`, ~150 offline unit tests), but unit tests of code are not a pipeline that evaluates a *model's* quality on held-out data. That is a different thing and still to build.
- [ ] Hosted APIs / cloud tooling (contrast with local-only setup)
- [ ] Reading & summarizing real papers (start at blog-post level)
- [ ] Portfolio: 2-3 polished documented projects on GitHub — *in progress*: this repository now has a README, tests and dependency files, and will count as one of the two or three once the tokenization project lands
- [ ] Interview prep: DS&A fundamentals + ML system design (deferred until closer to job search)

## Next lesson

**Tokenization & embeddings on the local Qwen model — top priority.** Everything built so far about the LLM concerns how the *last* step works (choosing a token from a probability distribution). Nothing yet shows the *first* step: text becoming token ids, and token ids becoming vectors. That gap is why "context window", "why does it cost per token" and "why is it bad at spelling" are still abstract. A concrete plan:

1. Show a sentence turning into token ids and back, and a few surprises (one word becoming three tokens; spaces belonging to the token after them).
2. Count tokens for a whole conversation, and connect that number to the context-size indicator the chat tab already shows.
3. Pull the embedding vector for a handful of tokens and measure distances between them, to make "vectors" mean something.

A practical note (2026-09-02): the lesson can use chat-llm's own API. `/v1/internal/encode`, `/v1/internal/decode` and `/v1/internal/token-count` all work against the running Qwen model — "Hello world, tokens become vectors." encodes to 7 ids and decodes back exactly. (An earlier probe got HTTP 500 from them, but so did chat itself: the model backend behind the web server had exited. Once the backend was healthy they worked. Lesson learned, and now built into the app manager: check that the model can *generate*, not just that the web server answers.) No tokenizer files need loading; embeddings are the part still to work out.

After that, in rough order of payoff:

- **Tiny neural network from scratch** — this is where backpropagation is actually learned; everything needed (loss, gradients, the update loop) already exists.
- **k-fold cross-validation** — the current train/test split is a single random draw of 5 students; the U-curve is visibly lumpy because of it. Cross-validation is the fix, and a small one.
- **Multi-turn chat and statelessness** — a lesson making it obvious that the whole history is re-sent every turn (the chat tab's context indicator already hints at it).
- **Image-generation lessons** — checkpoints, seeds, steps, samplers, CFG, resolution vs VRAM, using Forge.

---
*Roadmap agreed 2026-09-01. Status reviewed against the code 2026-09-02. Update the checkboxes as items are completed and keep the evidence pointers honest; add notes inline if a topic needs revisiting.*
