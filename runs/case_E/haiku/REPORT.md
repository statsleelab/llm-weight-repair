# REPORT: repair of corrupted GPT-2 checkpoint (haiku participant)

## Status
Partial repair only. The output `repaired.safetensors` is much less broken than `damaged.safetensors` on dev text, but it is **not** close to the original GPT-2 and I would not expect it to score well on the held-out benchmarks. Treat it as a diagnostic-grade candidate, not a faithful restoration.

## (a) Diagnosis
- Symptom: the damaged model is degenerate. On dev text it predicts the same token (" socially") at every position (top-1 = 0.000 over 1024 tokens). Mean next-token NLL is ~471 nats (ppl overflows float32). The final LayerNorm output is almost entirely a constant vector (common-mode norm 171.4 of 171.8), so the logits are essentially position-independent.
- Where the damage is visible in the weights: `wte.weight` (the tied embedding / LM head) carries the clearest anomaly. Its entry-magnitude tail is far heavier than a Gaussian with the robust sigma of 0.140 would give: 0.25% of entries have |w| > 0.5, 0.05% have |w| > 1.0, and the largest is ~6.3. Corrupted entries here are spread across all rows and columns (no clustered rows/columns), which fits a random sparse corruption.
- Zeroing the wte entries with large magnitude changes the NLL dramatically (471 -> 100 at |w|>0.5 -> 52 at |w|>0.3 -> 27 at |w|>0.2), so the wte tail is a major source of the failure.
- Other matrices: all attention/MLP projections have heavy tails in the original-looking weights too (large values at dims 138/447, the known GPT-2 outlier dimensions), so they cannot be cleanly separated from corruption by magnitude alone. Ablating single attn/MLP output projections, or zeroing |w|>5 in them, did not materially change the NLL (still in the hundreds), so I found no single localized culprit outside wte. Zeroing the position-embedding `wpe` also did not fix it (471 -> 424 at 1k tokens).

## (b) What I changed and why
- Changed only `wte.weight`: every entry with |w| > 0.2 was set to 0 (6.1M of 38.6M entries, i.e. about 16% of the matrix). All other tensors are byte-identical to the damaged file (verified: same keys, shapes, float32; all non-wte tensors equal).
- Why: this is a shrinkage-style repair that reduces the logit scale and removes the most extreme embedding values. It is a heuristic chosen to lower dev NLL, not a reconstruction of the true values.
- Tried and rejected: low-rank hard imputation of the flagged (|w| > 4 sigma, 0.16%) wte entries using rank-128 and rank-384 SVD completion (12 iterations) gave NLL ~110, no better than zeroing at 0.5. Its output is kept as a cache file, `wte_fix_r128.pt` / `wte_fix_r384.pt`, but it was not used for the submission.
- Threshold selection: T in {0.2, 0.25, 0.3, 0.4, 0.5, 0.6} was compared on the same 1024-token dev slice (which I then also reported on), so the reported improvement is optimistic.

## (c) Dev perplexity
Measured with non-overlapping 512-token windows, each prefixed with <|endoftext|>.

| model | tokens | mean NLL (nats) | ppl | top-1 |
|---|---|---|---|---|
| damaged | 1024 | 471.4 | ~exp(471) (overflow) | 0.000 |
| damaged | 8192 | 448.8 | ~7.9e194 | 0.000 |
| repaired (wte \|w\|>0.2 zeroed) | 1024 | 26.9 | ~4.8e11 | 0.000 |

Note: for the repaired model I did not compute an 8192-token figure; an earlier 8192-token run of the |w|>0.3 variant gave NLL 48.8. A healthy GPT-2 would be around NLL 3.5 (ppl ~30-40) on this kind of text, so the repaired model is still far from healthy.

## (d) What I could not fix, and confidence
- I did not locate the corrupted entries in the transformer blocks. The large NLL persists after removing any single block's attention or MLP output, so the damage is either spread across many blocks or sits in weights I could not separate from genuine GPT-2 outliers.
- I did not recover the true values of the corrupted wte entries; zeroing is a proxy and loses the information in the uncorrupted part of those entries.
- Confidence that the repaired model matches the original on the held-out benchmarks: very low. The benchmarks (perplexity, grammatical minimal pairs, factual recall) will almost certainly still be poor.
- Time: the work went somewhat past the ~40 minute budget in the final checks; the last step was a short threshold sweep.

## (e) Rules
I followed the rules: no downloads or network access, all work and cache files in this directory, no gradient-based or loss-optimising training (dev NLL was used only to pick among a handful of thresholds and a rank choice), `torch.set_num_threads(1)` in every script, no sub-agents, no `ps`/`pgrep`. One tiny inline `python -I -c` check was used to verify the output file rather than a saved step script. Scripts are `step01_*.py` through `step16_*.py`.
