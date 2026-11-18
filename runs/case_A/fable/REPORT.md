# Repair report

## (a) Diagnosis

**Corruption: about 1.00 % of the entries of every 2-D weight matrix were overwritten with exact zeros, at independent random positions.**

- `step01_stats.py`: every 2-D matrix (`wte`, `wpe`, all 48 `c_attn`/`c_proj`/`c_fc` weights) has 0.99–1.03 % exact zeros (1,244,026 entries in total). No NaN/Inf. No bias or LayerNorm entry is zero.
- `step02_mask_and_ppl.py`: zeros per row and per column are Poisson-distributed (std matches sqrt(mean)), and adjacent-pair frequency is 1e-4 = p², so the mask is i.i.d., not structured by row, column or block.
- `step03`–`step05`: I looked for a second kind of corruption (out-of-scale values). The large entries I found (e.g. ±8…17 in columns 447/138 of several `c_proj` matrices) form coherent row/column patterns and look native to the model; I found no evidence of non-zero corruption, but I cannot rule out small-magnitude changes (see (d)).
- Damaged model: dev perplexity **1.1e29** (top-1 accuracy 3.9 %), and it greedily generates " Dear Dear Dear …".
- `step10_ablate.py` localised the catastrophe: **essentially all of it comes from `wte`.** Repairing only `wte` takes perplexity from ~1e28 to 24.8; repairing everything except `wte` leaves it at 4e27. The reason is that a handful of embedding columns hold a large constant offset shared by all tokens (e.g. column 36: mean −0.317, token-to-token std 0.002) and the final LayerNorm gain on those dimensions is 10–17×. A token whose entry there was zeroed gets a huge spurious logit through the tied LM head.

## (b) What I changed and why

Only the exactly-zero entries of the 2-D matrices were rewritten; every other value is bit-identical to `damaged.safetensors` (asserted in `step12_final.py`). All methods are weight-only statistics; method and hyper-parameters were chosen by hiding a further 1 % of known entries and measuring reconstruction R² on them, not by text loss.

| Matrices | Method | Held-out variance recovered |
|---|---|---|
| `wte` | row+column mean, plus weighted mean of the 16 most similar token embeddings, plus Gaussian-conditional (ridge) prediction of the remainder from the token's other 760 dims | 62 % vs zero-fill (93–99 % in columns 64/87/138/266, 57–87 % in the LayerNorm-heavy columns 36/496/430/314) |
| `wpe` | Gaussian-conditional imputation across positions | 99 % |
| `c_attn` | same, best of row-wise / column-wise / average per matrix | 8–36 % (mean 21 %) |
| `attn.c_proj` | same | 3–22 % (mean 14 %) |
| `mlp.c_fc` | same | 6–18 % (mean 15 %) |
| `mlp.c_proj` | same | 3–15 % (mean 10 %) |

Code: `impute_lib.py`, `step07_impute_all.py` (blocks and `wpe`, log in `step07_log.json`), `step12_final.py` (`wte` and final assembly).

## (c) Dev perplexity (first 8,192 tokens of `dev.txt`, 512-token windows)

| Model | Perplexity | Top-1 accuracy |
|---|---|---|
| `damaged.safetensors` | 1.12e29 | 3.9 % |
| intermediate `cand_v1` (ridge only for `wte`) | 24.80 | 41.15 % |
| **`repaired.safetensors`** | **24.77** | **41.17 %** |

The repaired model produces sane text ("The capital of France is the capital of the French Republic…") and prefers "The keys … are" over "The keys … is".

## (d) What I could not fix, and confidence

- **Transformer-block weights are mostly not recovered.** These matrices have little low-rank redundancy, so 80–90 % of the squared error on their zeroed entries remains (filled with small, shrunk estimates instead of 0). The functional cost appears small: leaving all attention or all MLP zeros unrepaired changes dev perplexity by about 1 % each. As a rough size-of-effect probe, a fresh unrepaired 1 % zeroing of all block/`wpe` matrices moves the repaired model from 24.77 to 27.03.
- **`wte` is only partly recovered** (about 38 % of zeroed-entry variance remains). Rare tokens with a zeroed entry in a high-gain dimension may still be mis-scored by up to a few nats; this would show up more on factual-recall or rare-word benchmarks than on dev perplexity.
- Any zeroed entry whose true value was an isolated large weight is not recoverable by these methods.
- I do not know the original model's perplexity on this text; my guess is the repair is within a few percent of it, but that is not measured.
- Confidence in the diagnosis (i.i.d. zeroing, ~1 %): high. Confidence that there is no additional non-zero corruption: moderate — the zero count already matches the stated ~1 %, and the repaired model behaves normally.

## (e) Rule compliance

I followed the rules: no network access or downloads, nothing read outside this directory, no gradients/optimisers/loss-driven weight fitting (dev perplexity was only used to compare two candidates and for diagnosis), `torch.set_num_threads(1)` in every script, all scripts kept here. One deviation in spirit from the resource guidance: on a few occasions two single-threaded scripts ran concurrently, and the final evaluation command ran slightly over two minutes in the foreground before being backgrounded.
