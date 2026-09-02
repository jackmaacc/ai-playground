# Plan: honest evaluation + sampling rigor

**Scope:** improvements #1 and #2 from the review of `learning/`.
**Status:** awaiting approval. No code written yet.

## Objective

Two changes, both aimed at a habit rather than a feature:

1. **Stop measuring the model on the data it trained on.** Today `mse_loss`
   scores the fit against all 20 points it just learned from, and `diagnose()`
   calls that "the best any straight line can do on this data". A learner
   reasonably concludes lower training loss = better model. That is the
   single most expensive wrong instinct in applied ML.
2. **Stop drawing conclusions from one sample.** `lessons.py` generates one
   response at setting A and one at setting B and asks what differs. With a
   stochastic sampler, two runs at *identical* settings can differ as much as
   A differs from B. Lesson 4's own reveal admits the effect "may be subtle".

Both map to items already on `LEARNING_PATH.md` (Phase 1: "train/test splits,
overfitting, eval metrics").

## Out of scope

Deferred deliberately: the tokenization walkthrough (#3), multi-turn chat and
statelessness (#4), the webui slider/`SETTINGS_SPEC` DRY leak (#5), and
portfolio hygiene — README, requirements.txt, hardcoded `C:\Users\jackm`
paths, tests (#6).

---

## Findings that constrain the design

Investigated before planning; each one changes what the implementation can do.

### F1. Seven call sites index the history tuple positionally

`history` entries are `(step, m, b, loss)` and are read by index in four files:

| Location | Access |
|---|---|
| `phase1_interactive.py:36,56,121` | `history[...][3]` |
| `phase1_linear_regression.py:129` | `history[...][3]` |
| `phase1_linear_regression_viz.py:58,59` | `row[0]`, `row[3]` |
| `webui.py:150` | `row[3]` |
| **`phase1_linear_regression_viz.py:39`** | **`_, m, b, _ = history[step_idx]`** |

That last one is a fixed-width unpack. Adding a fifth field raises
`ValueError: too many values to unpack`. It is the only hard breakage.

### F2. Gradient descent cannot fit high-degree polynomials

Raw polynomial features on `hours ∈ [0, 10]` produce a Vandermonde matrix
whose condition number is astronomical by degree 9. No learning rate
converges: too small and it crawls, large enough to move and it diverges.
Attempting the overfitting demo with the existing `train()` loop would
produce garbage, and — worse — the learner would blame the overfitting
concept for what is actually a conditioning problem.

**Therefore the polynomial demo must solve by least squares
(`np.linalg.lstsq`), not gradient descent.** This is also a genuine teaching
point: the closed form exists here, which is exactly why we can isolate
overfitting without the optimiser confounding it.

### F3. The test split must be random, not a slice

`hours = np.linspace(0, 10, 20)` is evenly spaced and sorted. Holding out the
last 5 points means testing only on 7.9–10 hours — measuring *extrapolation*,
not generalization, and the test error would look alarming for the wrong
reason. Split must be a seeded shuffle.

### F4. `scores` is clipped at 100

`np.clip(scores, 0, 100)` combined with `5*10 + 50 = 100` means the top of the
range is flattened. Mild, and arguably realistic, but it means the true
relationship is not perfectly linear at the boundary — worth a comment so the
residual error is not mistaken for a bug.

### F5. Three samples per setting triples lesson runtime

4 lessons x 2 settings x 3 samples = 24 generations, up from 8. Local model so
no monetary cost, but noticeably slower. Sample count must be configurable.

### F6. `webui.py` runs on a different interpreter

It executes under `chat-llm/installer_files/env/python.exe`. Any new
dependency must exist there. Everything planned uses numpy + matplotlib only —
**no new dependencies**.

---

## Design decisions

### D1. History becomes a NamedTuple, gaining `test_loss`

| Option | Verdict |
|---|---|
| Parallel list of test losses | Rejected — two lists to keep in sync |
| Append 5th element to the plain tuple | Works, but `row[4]` is unreadable |
| **`NamedTuple(step, m, b, loss, test_loss)`** | **Chosen** |

A NamedTuple *is* a tuple, so all index-based reads in F1 keep working
untouched. It adds `row.test_loss` for readability, and introduces the learner
to a useful stdlib type. Requires exactly one fix: the unpack at
`phase1_linear_regression_viz.py:39`.

### D2. Data passed explicitly, defaulting to the full set

`mse_loss(m, b)` and `gradients(m, b)` close over module-level `hours`/`scores`.
They gain optional data parameters defaulting to the existing globals, so
`webui.py` and `phase1_interactive.py` keep working unchanged, while
`mse_loss(m, b, test_hours, test_scores)` becomes expressible.

Rejected: a full `Dataset`/`X, y` refactor. Cleaner in the abstract, but it
breaks all three importers and obscures the file whose entire virtue is being
the simplest readable example.

### D3. The overfitting demo lives in its own file

`phase1_linear_regression.py` earns its keep by being the simplest possible
model. Polynomial features, `lstsq`, and a degree sweep belong in a new
`phase1_overfitting.py` that imports the shared data and split.

### D4. Lessons lead with a control, not a comparison

The fix is not merely "more samples" — it is running the *same* setting twice
first, so within-setting variance is visible before any across-setting claim.
Structure per lesson: control pair → prediction → A/B at N samples → reveal.
This matches the existing predict-before-reveal style.

---

## Workstream A — honest evaluation

**`phase1_linear_regression.py`**
- `split_data(test_fraction=0.25, seed=0)` → seeded shuffle (F3), returns train
  and test arrays.
- `mse_loss` / `gradients` / `predict` take optional data args (D2).
- `evaluate(m, b, data)` → MSE, **RMSE** (headline: "off by N points on the
  exam", in the units of the target), and **R²**.
- `train()` fits on the train split only and records `test_loss` each step (D1).
- `diagnose()` gains a generalization sentence: compares train vs test and
  names the gap, replacing "best any line can do on this data".

**`phase1_interactive.py`**
- Training reports train and test metrics side by side.
- "Explain" gains: what the gap means, and why a model that has memorised its
  training data is worthless on a new student.

**`phase1_linear_regression_viz.py`**
- Fix the 4-element unpack (F1).
- Loss plot draws train and test curves together.

## Workstream B — the overfitting curve

**New `phase1_overfitting.py`**
- `polynomial_features(x, degree)` → Vandermonde design matrix.
- `fit_polynomial(x, y, degree)` → `np.linalg.lstsq` (F2), with a comment
  explaining why gradient descent is the wrong tool here.
- Sweep degree 1..12 over the same split; record train and test error.
- Two plots: the **U-curve** (train error falling monotonically while test
  error turns upward), and the fitted curves themselves at low/good/absurd
  degree, so the wiggle is visible next to the data.
- A `diagnose`-style verdict naming the degree where test error bottoms out.

## Workstream C — sampling rigor

**`lessons.py`**
- `call_model_samples(prompt, settings, n)` helper.
- `run_lesson` restructured per D4: control pair first, then A/B at `n` samples.
- Per-lesson `samples` key (default 3, configurable — F5). Adding a key is
  backward compatible with `webui.py`'s `LESSONS` import.
- New lesson 0 or a preamble making the control explicit: *"same settings,
  twice — this is how much varies when nothing changed."*

**`webui.py`**
- Lessons tab renders the control pair and N samples per setting.

**Optional, needs verification first:** pin `seed` to show that identical
settings + identical seed reproduce byte-identical output. Whether
text-generation-webui's OpenAI-compatible endpoint honours `seed` is
**unverified** — probe it before committing to a lesson built on it.

## Entry points

`main.py` gains an overfitting menu entry; `webui.py` gains a tab. Both follow
the existing pattern — front-ends call shared functions, no logic duplicated.

---

## Risks

| Risk | Mitigation |
|---|---|
| The 4-element unpack breaks (F1) | Known line; fix in the same commit |
| Polynomial fit misinterpreted as GD failure (F2) | Use `lstsq`; comment why |
| Slice-split measures extrapolation (F3) | Seeded shuffle |
| 20 points is small; one split is luck | Fixed seed for reproducibility; note that real work uses cross-validation |
| Lessons 3x slower (F5) | Configurable `samples`, default 3 |
| `seed` may be unsupported | Probe before building; drop if absent |

## Verification

- Existing smoke tests still pass: every `main.py` menu path, both viz scripts,
  `webui.py` import under the chat-llm interpreter.
- Assert train loss ≤ test loss in the typical case, and that the U-curve
  actually turns (test error at degree 12 > test error at its minimum).
- Confirm all four PNGs still regenerate.
- Confirm index-based history reads still work after the NamedTuple swap.

## Effort

Workstream A ~1 session, B ~1 session, C ~1 session including a live-model
run-through. A and B are pure numpy and testable without the model server; C
needs Qwen running.

## Follow-ups deferred

#3 tokenization walkthrough, #4 multi-turn/statelessness, #5 slider DRY leak,
#6 README + requirements + relative paths + `diagnose()` tests.
