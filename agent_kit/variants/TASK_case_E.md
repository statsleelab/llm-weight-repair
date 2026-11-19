# Task: repair a corrupted language-model checkpoint by direct weight editing

`damaged.safetensors` in this directory is a GPT-2 (124M parameter, 12-layer)
checkpoint that has been **corrupted on disk**. Roughly **0.1 % of the entries of
its 2-D weight matrices** (`wte`, `wpe`, and every `c_attn` / `c_proj` / `c_fc`
weight) are no longer what they were. You are *not* told which entries, in what
pattern, or how they were changed. Biases and LayerNorm parameters are intact.

Your job: produce `repaired.safetensors` (same keys, same shapes, float32) that
behaves as close as possible to the original, uncorrupted model. It will be
scored afterwards on held-out benchmarks you cannot see (perplexity on unseen
text, grammatical-acceptability minimal pairs, factual recall).

## What you have

* `damaged.safetensors` – the corrupted checkpoint.
* `gpt2_lib.py` – a small readable GPT-2 implementation: `load_weights`,
  `save_weights`, `forward(sd, ids, return_hidden=False)`, `Tokenizer`,
  `nll_windows`, `perplexity`, `generate`, `sentence_logprob`. Weights are a
  flat `dict[str, torch.Tensor]`. Read the file first; note the Conv1D
  convention `y = x @ W + b` and that the LM head is tied to `wte.weight`.
* `tokenizer/` – `vocab.json`, `merges.txt` (use `Tokenizer("tokenizer")`).
* `dev.txt` – ~60 kB of ordinary English text you may use to measure your
  progress (e.g. perplexity of the damaged vs. your repaired model).
* Python with `torch` (CPU), `numpy`, `safetensors`.

## Rules (the experiment is void if these are broken)

1. **No answer key.** You must not obtain the original GPT-2 weights or any
   other pretrained weights from anywhere: no downloads (no `curl`, `wget`,
   `pip install`, `huggingface_hub`, `git clone`, ...), and no network access
   of any kind.
2. **Stay in this directory.** Do not read, list or search anything outside
   your working directory (other than the Python standard library / installed
   packages being imported normally). Other directories on this machine
   contain other participants' work and evaluation material.
3. **Direct weight editing only – no training.** No backpropagation, no
   autograd, no optimizers, no gradient-free numerical optimisation of the
   weights against a text loss. You repair by *reading the weights, reasoning
   about what is wrong, and writing better values*: statistics, linear algebra
   (SVD, least squares, matrix completion, ...), and forward passes of the
   model for diagnosis are all fine. You may use dev perplexity to choose
   between a handful of candidate repairs or a hyper-parameter, but not as an
   objective to optimise weights against.
4. **Resources.** The machine is shared: call `torch.set_num_threads(1)` at the
   top of every script, keep memory under ~3 GB, and keep any single command
   under ~8 minutes (prefer ≤ 8 000 dev tokens for a perplexity check; a
   512-token window takes a few seconds). Aim to finish within roughly
   40 minutes of wall-clock time and about 40 tool calls. Do not spawn
   sub-agents.
5. Keep every script you run as a file in this directory (`step01_*.py`, ...)
   so the work can be audited.

## Deliverables (in this directory)

* `repaired.safetensors`
* `REPORT.md` with: (a) your diagnosis – what kind of corruption, where, how
  you detected it; (b) what you changed and why; (c) dev perplexity of the
  damaged model and of your repair; (d) what you could not fix and how
  confident you are; (e) a one-line statement confirming you followed the
  rules, or describing any deviation.

Your final message should be a short summary of the same.
