# AI/ML Engineer Learning Path

Goal: work in the AI industry as an ML/AI engineer. Project-driven — each item gets learned by building it, mostly on this local AI stack (`chat-llm/`, `image-gen/`).

Run `python learning/main.py` for one interactive menu covering everything built so far (gradient descent, linear regression, and the LLM lessons/playground against the real Qwen model). Each piece also runs standalone if you only want one topic.

Current status: **Phase 0, just starting.**

## Phase 0 — Foundations (now, pre-calculus)
- [ ] Python for ML: numpy arrays/vectors, pandas basics, venvs
- [ ] Linear algebra for ML: vectors, matrices, dot products, matrix multiplication, tensor shapes
- [ ] Probability refresher: distributions, expected value, conditional probability
- [ ] Git / command-line fluency
- [ ] **Project**: tokenization & embeddings walkthrough on the local Qwen model — see text become vectors

## Phase 1 — Core ML + intro deep learning (calculus-based; can start earlier since some basic calc is already known — introduce as relevant, not strictly gated)
- [ ] Classical ML: regression, classification, train/test splits, overfitting, eval metrics (scikit-learn)
- [ ] Gradient descent and backpropagation
- [ ] **Project**: tiny neural network from scratch in raw numpy
- [ ] PyTorch fundamentals: tensors, autograd, `nn.Module`, training loops
- [ ] **Project**: rebuild the from-scratch NN in PyTorch, compare

## Phase 2 — Modern deep learning (transformers, fine-tuning, diffusion)
- [ ] Transformers deep dive: attention, multi-head attention, positional encoding (against the running Qwen model)
- [ ] **Project**: LoRA fine-tune the local Qwen model on a small custom dataset
- [ ] CNNs and diffusion model basics
- [ ] **Project**: explore Forge's denoising process, possibly a small image-side LoRA

## Phase 3 — Industry-ready engineering
- [ ] MLOps basics: experiment tracking, reproducibility, model versioning, eval pipelines
- [ ] Hosted APIs / cloud tooling (contrast with local-only setup)
- [ ] Reading & summarizing real papers (start at blog-post level)
- [ ] Portfolio: 2-3 polished documented projects on GitHub
- [ ] Interview prep: DS&A fundamentals + ML system design (deferred until closer to job search)

---
*Roadmap agreed 2026-09-01. Update the checkboxes as items are completed; add notes inline if a topic needs revisiting.*
