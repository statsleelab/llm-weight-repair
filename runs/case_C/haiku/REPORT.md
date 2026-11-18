# REPORT: repair of corrupted GPT-2 (124M) checkpoint

## (a) Diagnosis

- **Corruption type: block erasure (zeroing), one rectangle per 2-D weight.** Every 2-D tensor (`wte`, `wpe`, all 36 `c_attn`, `c_proj`, `c_fc` weights in the 12 blocks) contains exactly one contiguous rectangle of entries that are exactly 0.0. The rectangle covers about 10% of rows times about 10% of columns, so about 1% of each matrix. Checked with `step03_zero_structure.py`: the zero count equals rows x cols of the rectangle exactly, and there are no zeros outside it. Trained GPT-2 weights contain no exact zeros, so these are the corrupted entries.
- Example block positions (row range, col range): `wte` rows 25013-30038, cols 240-316; `wpe` rows 734-835, cols 123-199; `h.0.attn.c_attn` rows 622-698, cols 1539-1768. The full list with each block's choices is in `repair_info.json`.
- Biases and LayerNorm parameters are untouched, as the task says.
- Detection: `step01_diagnose.py` showed exact-zero counts of about 1.00% in every 2-D tensor, and `step02`/`step03` showed the overdispersed, rectangular structure (per-row zero counts of 0 or exactly 77/230/307 for the damaged rows).

## (b) What I changed and why

- Only the zeroed rectangle of each 2-D tensor was rewritten. Everything else is byte-identical in value.
- Method: ridge regression that predicts the damaged columns of the rectangle from the intact columns, fitted on the intact rows, then applied to the damaged rows. The transposed (row-direction) version was also computed, and the final choice was col, row, or the average of the two.
- Hyperparameters per tensor (ridge strength from {1e-3, 1e-2, 0.1, 0.3, 1, 3, 10} and direction) were chosen by hold-out error on three disjoint same-shape blocks in the *intact* part of the same matrix. The dev text was not used for selection.
- Hold-out results (RMSE on the held-out blocks, zero-fill RMS in brackets; see `repair_info.json`):
  - `wpe`: about 0.014 (0.124). Position embeddings are smooth, so this is almost fully recovered.
  - `wte`: about 0.107 (0.144). Good recovery for the rectangle's 77 dimensions.
  - attention and MLP matrices: about 0.08-0.19 against zero-fill RMS of about 0.08-0.2, i.e. only about 10-40% of the block's energy is recovered. These weight matrices are close to unstructured (the top 128 singular values of `h.0.attn.c_attn` carry only 57% of energy), so linear prediction from the rest of the matrix is weak.
- Output: `repaired.safetensors`, 148 keys, same names and shapes, float32.

## (c) Dev perplexity (first 8000 dev tokens, 512-token non-overlapping windows)

| model | NLL (nats) | perplexity | top-1 |
|---|---|---|---|
| damaged | 5.563 | 260.6 | 0.288 |
| repaired | 3.245 | 25.7 | 0.411 |

Dev perplexity was computed once, for the final comparison only. It was not used to choose any repair setting.

## (d) What I could not fix, and confidence

- The 10%x10% blocks in attention and MLP matrices are only partly recovered. Each repaired block is a regularised conditional-mean estimate, not the original values. Expect factual recall and fine-grained grammar to remain degraded relative to the original model, even though perplexity is now much closer to normal.
- The `c_proj` blocks (attention and MLP) gain the least over zero-fill in every layer (validation RMSE about 90-100% of zero-fill RMS), so they are effectively not repaired.
- Activation-based constraints (forward passes of the damaged model) were not used to fit weights. There are no target outputs of the original model, so they cannot identify the missing block without training.
- Confidence: high that the corruption is the block-zeroing described above and that `wpe` is almost fully restored. Moderate that the overall perplexity improvement carries over to the held-out benchmarks. Low that the attention and MLP blocks are close to the originals.

## (e) Rules

I followed the rules: no downloads or network access, nothing read outside this directory, no gradients or training, `torch.set_num_threads(1)` in every script, every script kept as a `step*.py` file in this directory, and no sub-agents. Wall-clock time was about 40 minutes.
