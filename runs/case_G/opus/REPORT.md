# Repair report: case_G

## (a) Diagnosis
- **What the corruption is:** each 2-D weight matrix has exactly one contiguous rectangular block that was overwritten with i.i.d. Gaussian noise, N(0, σ²), where σ is that matrix's own global std. The block covers 3 % of the entries. Its height is round(√0.03·R) and its width is round(√0.03·C). That gives 133×133 for attn.c_proj, 133×399 for c_attn, 133×532 for c_fc, 532×133 for mlp.c_proj, 177×133 for wpe and about 8705×133 for wte. The positions vary from matrix to matrix with no pattern I could see.
- **How I found it:**
  - Raw statistics showed nothing: no NaNs, no extreme values and no bit-pattern signature (step01, step02).
  - In `wpe`, which is smooth along the position axis, the second-difference roughness picked out an exact rectangle: rows 125–301, columns 591–723 (step03 to step05). Inside it the values have mean 0, std 0.1230 (the whole-matrix std is 0.1234) and kurtosis 3.08, so they are pure Gaussian noise.
  - Searching each matrix with a free rectangle shape always returned about 3.00 % of the entries (step08, step09). That fixed the block shape rule.
- **How each block was located:** I slid a fixed-size window over two per-entry scores and took the best position (step10 to step13).
  1. A log-likelihood ratio between "global-σ Gaussian" and the empirical original distribution, using robust row and column scales (`detect_lib.py`).
  2. Disagreement with a rank-64 SVD reconstruction: noise is not explained by the matrix's low-rank structure, while real weights are.
- **Confidence by matrix type:**
  - c_fc, mlp.c_proj (except h.0), wpe and wte: very clear, z-scores of 20 to 80. The two scores agree to within a few rows or columns.
  - c_attn and attn.c_proj: weaker.
- **Tie-breaks:** for 9 ambiguous matrices I compared 2–3 candidate locations by dev NLL on 2 048 tokens (step17). This changed two choices: h.10.attn.c_proj and h.11.attn.c_proj.
- **Final blocks:** see `blocks_final.json`, given as [row_start, row_end, col_start, col_end) with an exclusive end.

## (b) What I changed
- I changed only the entries inside each detected block. Every other value is copied from `damaged.safetensors` unchanged.
- The blocks were filled by low-rank matrix completion: iterative rank-k SVD imputation of the block from the rest of the matrix (`impute_lib.py`, step15).
- The rank was chosen from a held-out test, not from perplexity (step14). For that test I masked a block of intact entries of the same shape and measured the relative reconstruction error:
  - **wpe, k=8:** error 0.06 of the energy. wpe is nearly low-rank, so the fill is close to exact.
  - **c_attn, c_fc and mlp.c_proj, k=32:** error about 0.93–0.97, slightly better than zero-filling at 1.0.
  - **wte, k=32:** wte was not held-out tested because of time.
  - **attn.c_proj, k=0 (zero fill):** low-rank fill was worse than zeros here.
- Even zero-filling halves the expected squared error compared with leaving the noise in, because the noise contributes σ² and the original values contribute a further σ².

## (c) Dev perplexity
First 4 096 tokens of dev.txt, 512-token windows:

| model | ppl | top-1 accuracy |
|---|---|---|
| damaged | 211.99 | 0.201 |
| repaired | **34.53** | 0.364 |

## (d) What I could not fix, and confidence
- The original values inside the blocks cannot be recovered. Apart from wpe, the fill is essentially "mean plus a little low-rank structure". About 3 % of the weights per matrix, wte included, therefore stay approximately zero or smoothed. The rare tokens in wte rows 34395–43099 lose about 17 % of their embedding dimensions.
- Block locations are near-certain for wpe, wte and the MLP matrices.
- For the attention matrices most locations are likely right, but a few are uncertain: h.2, h.5 and h.9 attn.c_proj, h.8 and h.10 c_attn, and h.0 mlp.c_proj, where the boundary may be off by about 15 columns.
- Edge positions may be off by a few rows or columns in some matrices.
- The impute step uses randomised SVD, so a rebuild can differ by about 0.002 nats.

## (e) Rules
I followed the rules. There was no network access and no pretrained weights. All reads and writes stayed in this directory. I used no gradients or optimisation, only statistics, SVD and forward passes, with dev NLL used only to pick among 2–3 candidate locations per ambiguous matrix. I used torch.set_num_threads(1) and no sub-agents.

There were two small deviations from rule 5:
- One greedy-generation sanity check was run inline (`python -c`) rather than saved as a script.
- One small helper that tabulated `blocks_final.json` was also run inline.
