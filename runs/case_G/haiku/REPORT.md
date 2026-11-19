# REPORT: repair of corrupted GPT-2 (124M) checkpoint

## Outcome
No reliable repair was found. `repaired.safetensors` is a bit-identical copy of `damaged.safetensors` (148 keys, same shapes, float32). It is the best candidate I could justify: every alternative I tested was equal or worse on dev perplexity.

## (a) Diagnosis
- Dev perplexity of the damaged model is very high: 196.6 on the first 8192 dev tokens (non-overlapping 512-token windows), top-1 accuracy 0.21. Greedy generation is degenerate ("The capital of France is the same. Grade: Brazil Brazil ...").
- Per-tensor statistics (step01, step02): no NaNs, infs or exact-zero floods. Biases and LayerNorm parameters look normal. The 2-D matrices have heavy tails, but this is not a clean corruption signature:
  - Several `mlp.c_proj` and `attn.c_proj` matrices have very large entries (up to about 17) and large excess kurtosis.
  - `wpe` has a few dimensions with large column RMS (up to 0.62) against a median column RMS of 0.0056. Its row RMS is smooth across positions (about 0.12), except position 1023 (RMS 0.004).
  - `attn.c_proj` matrices have uneven column RMS (some columns near 0.02 to 0.05 of the median, some much larger).
- Bit-level checks (step04) show no signature: no low-16-bit zero patterns, no exponent anomalies beyond natural ones.
- Spectra (step08, step09): the square `attn.c_proj` matrices have a tail of small singular values (5 to 45 values below 0.05). For random 768x768 matrices this is roughly what is expected, so it is not conclusive. There is no clear low-rank plus noise structure.
- Row and column structure (step10): no whole-row or whole-column corruption block. Outliers are spread roughly uniformly over rows and columns.

## (b) What I changed and why
Nothing, in the end. I tested these candidates on dev perplexity (4096 tokens, damaged baseline 211.99):
- Zeroing per-matrix entries beyond 4 robust sigmas (0.35% of entries): ppl about 84,660.
- Zeroing per-matrix entries beyond 8 robust sigmas (0.05% of entries): ppl about 5,083.
- Clipping `wte` to +/-0.8: ppl 214.2. Clipping `wte` to +/-1.0: ppl 213.8.

The large-magnitude weights are functionally load-bearing, which fits GPT-2's known massive-activation neurons. Marginal outlier removal therefore destroys the model, and clipping does not help. I could not localise the corrupted entries from the weights alone, and I did not use gradients, so I did not attempt entry-level repair.

## (c) Dev perplexity
- Damaged: 196.6 (8192 tokens); 212.0 (4096 tokens).
- Repaired (identical to damaged): 196.6 (8192 tokens); 212.0 (4096 tokens).
- Improvement: none.

## (d) What I could not fix, and confidence
- I could not identify which of the roughly 3% of entries in each 2-D matrix were changed, nor how. The corruption looks same-scale and is not visible in marginal statistics, spectra, bits or row and column structure.
- Confidence that the repaired file is better than the damaged one: effectively none. Confidence that the diagnosis above is correct: low to moderate. I did not verify the corruption type.
- Untried ideas that might work with more time: a sparse plus low-rank decomposition per matrix with dev-perplexity selection of the threshold; an analysis of per-matrix ablation by replacing one tensor at a time with a shrunken version to find the most damaged tensors; and a closer look at `wpe` position 1023 and at the `attn.c_proj` columns with near-zero norm.

## (e) Rules
I followed the rules: no downloads or network access, all reads and writes inside this directory (including `__pycache__` from importing `gpt2_lib.py`), no backpropagation or training, `torch.set_num_threads(1)` in every script, and no sub-agents. Scripts are `step01_*.py` to `step13_*.py`. Forward passes and statistics were used for diagnosis only; dev perplexity was used only to compare candidates.
