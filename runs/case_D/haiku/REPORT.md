# Report: repair of damaged GPT-2 checkpoint (haiku)

## Outcome in one line
No repair candidate I tried improved dev perplexity, so `repaired.safetensors` is the damaged checkpoint re-saved unchanged (float32, same 148 keys and shapes). This is the best candidate I have, not a repair.

## (a) Diagnosis
- Dev perplexity of the damaged model: 35.07 (full dev.txt, 13,954 tokens, 512-token windows). 36.08 on the first 8,000 tokens.
- Per-tensor statistics (step01, step02) show no obvious global signature: no NaNs, no zeroed blocks, no bf16-style low-mantissa patterns (step03), and no value-quantisation artefacts.
- `wpe.weight` is the only tensor with strong structure. Rows (positions) vary smoothly: mean cosine between adjacent rows is 0.989. Row norms are ~3.4 everywhere except row 0 (norm 9.9) and row 1023 (norm 0.118, uncorrelated with row 1022, cosine 0.03).
- Positions ~128-230 show a dense band of "jagged" entries: about 65-70 of 768 dims per row deviate strongly from neighbouring positions, while most dims in those rows are tiny (about 0.003). Overall about 1.2% of wpe entries have second differences above 0.024. This is consistent with the stated ~1% corruption, but I could not confirm it was corruption rather than genuine high-frequency content.
- Weight matrices show heavy tails (kurtosis up to about 790, e.g. `h.3.mlp.c_proj.weight` with max |w| about 17, and `h.1.mlp.c_proj.weight` with max about 13.7). These look like possible corruption, but see (b).

## (b) What I changed and why (and what I rejected)
No change is in the final file. Candidates tested by dev perplexity (8,000-token subset, baseline 36.08):
1. Smoothness-based wpe repair, block-adaptive threshold K=5 and K=6 (flagged about 1.2-1.6k entries, interpolated from neighbours): ppl 39.46 and 39.38. Worse.
2. Absolute second-difference threshold 0.0237 (flagged 9,001 entries, replaced by neighbour average): ppl 37.38. Worse.
3. Stricter thresholds |d| > 0.10 (4,011 entries) and |d| > 0.05 (6,145 entries): ppl 37.24 and 37.34. Worse.
4. Zeroing weight entries with |w| > 8 or 12 robust sigma in all c_attn/c_proj/c_fc matrices (48k and 18k entries): ppl 4110 and 12281. Catastrophic. The extreme-magnitude weights are functionally essential, probably genuine outlier channels, so magnitude alone cannot locate corruption.

Reading of these results: the entries my detectors flag are either not the corrupted ones, or interpolating them loses information the model needs. The corruption is most likely in the bulk of the distribution, where it is invisible to per-tensor statistics.

## (c) Dev perplexity
- Damaged (full dev, window 512): 35.069
- Repaired (identical to damaged): 35.069
- Repair gives no improvement; I did not measure on held-out benchmarks.

## (d) What I could not fix, and confidence
- I could not locate the corrupted entries in the weight matrices (c_attn, c_proj, c_fc) or in wte. Without a reference, I have no reliable detector for bulk-distribution corruption such as sign flips or small additive noise.
- Confidence that the output is better than the damaged model: none. Confidence that it is worse: low, since it is the same tensors. The wpe smoothness detector looked promising on paper, and the perplexity result contradicts it, so I suspect the jagged wpe band is natural.
- Unexamined leads: row 1023 of wpe (norm 0.118, anomalous against its neighbours), and the tied wte embedding, which I did not analyse beyond per-tensor statistics.

## (e) Rules
I followed the rules: no downloads or network access; all reads and writes stayed in this directory; no training or gradient-based or loss-driven weight optimisation (dev perplexity was used only to compare a handful of candidates); `torch.set_num_threads(1)` in every script; all scripts saved as `step*.py` here; no sub-agents; no `ps`/`pgrep`. Intermediate files here are `wpe_repaired_K4.0.npy`, `wpe_repaired_K5.0.npy` and `wpe_repaired_K6.0.npy` (wpe candidates, not used in the output), plus the `__pycache__` directory created by importing `gpt2_lib`.
