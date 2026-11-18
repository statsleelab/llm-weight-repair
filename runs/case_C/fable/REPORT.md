# Repair report

## (a) Diagnosis
* Every 2-D weight matrix (`wte`, `wpe`, all 48 `c_attn`/`c_proj`/`c_fc` weights) contains **exactly one
  contiguous rectangular block of exact zeros** covering ~10 % of the rows x ~10 % of the columns
  (= 1.00 % of the entries). E.g. `wte`: tokens 25013-30038 x dims 240-316; `wpe`: positions 734-835 x
  dims 123-199; `h.0.attn.c_attn`: rows 622-698 x cols 1539-1768. Full list in `step03_log.json`.
* Detected by per-matrix statistics (`step01_stats.py`: zero fraction 0.0100 in every matrix, no NaN/Inf,
  no anomalous outliers beyond GPT-2's natural heavy tails) and by the zero-mask geometry
  (`step02_pattern.py`: zeros are ~99 % adjacent to other zeros; row/column counts show a single
  rectangle; `step03` asserts #zeros == |rows| x |cols| for each matrix).
* I found no evidence of any other kind of corruption (non-zero entries were left untouched).
* Ablation (`step05_wte.py`): almost all of the functional damage comes from the `wte` block
  (dev ppl 274 -> 27.7 when only `wte` is repaired; repairing only the layer matrices gives 274 -> 264).
  The zeroed embedding dims carry a large common (mean) component, so zeroing them mis-scores ~5 000
  tokens in both the input embedding and the tied output head.

## (b) What I changed
Only the entries inside the zeroed blocks were rewritten; everything else is bit-identical to the damaged file.
* **All matrices (`step03_complete.py`)**: regression / Nystrom-style block completion. With
  M = [[A, B], [C, D]] and A missing, A_hat = B f(D) C, a ridge-regularised pseudo-inverse of D
  (equivalently: ridge-regress the missing columns on the intact columns using the intact rows).
  The ridge strength was chosen per matrix by hiding same-sized blocks in the intact part of the same
  matrix and measuring reconstruction error (weight space only, no text).
  Simulated relative squared error: `wpe` 0.017 (near-perfect, it is very low-rank), `wte` 0.59,
  transformer matrices 0.72-0.96 (only 5-28 % of the variance is predictable; the estimate is
  correspondingly shrunk toward zero).
* **`wte` (`step06_final.py`)**: OLS of the 77 missing dims on the 691 intact dims (fit on the ~45 k intact
  tokens) plus a kNN residual correction (softmax-weighted mean OLS residual of the 16 nearest intact
  tokens by cosine similarity on the intact dims, shrunk by 0.85). k / temperature / shrink selected on
  two simulated blocks (neighbouring token bands, same columns): rel. error 0.55 (OLS) -> 0.42.
* Tried and rejected (`step04_augtest.py`): augmenting the regression with the paired matrix along the
  shared neuron / head axis (c_fc <-> mlp c_proj, attn c_proj <-> V part of c_attn) – no gain in simulation.

## (c) Dev perplexity (first 8192 tokens of dev.txt, 512-token windows)
| model | ppl | top-1 acc |
|---|---|---|
| damaged | 255.86 | 0.290 |
| ridge completion only | 25.68 | 0.412 |
| **repaired.safetensors** (final) | **25.56** | 0.412 |

Dev perplexity was used only to compare these candidates; all hyper-parameters were chosen in weight space.

## (d) What could not be fixed / confidence
* High confidence in the diagnosis (one zeroed rectangle per matrix) and that the `wpe` block is
  essentially restored (~98 % of variance).
* `wte` block: an estimated ~58 % of its variance restored; the rest (token-specific detail of ~5 000
  mid/low-frequency tokens in 77 dims) is not recoverable from the surviving weights, so those tokens
  will remain somewhat degraded (may matter for factual recall involving them).
* Transformer-layer blocks: only ~5-28 % of variance restored; the blocks are mostly unrecoverable by
  weight statistics, but their functional impact is small (dev ppl changes by < 1 with or without them).
* I cannot know the true original perplexity; the residual gap to it is presumably small but not zero.

## (e) Rules
I followed all rules: no network/downloads, nothing read or written outside this directory, no
gradients/optimizers/loss-driven weight optimisation (only closed-form linear algebra and kNN on the
weights, forward passes for evaluation), `torch.set_num_threads(1)` in every script, no sub-agents,
all scripts kept as `step01`-`step06`.
