# Repair report (haiku)

## (a) Diagnosis
- Every 2-D matrix has kurtosis ~75-110 and a population of entries 5-100 robust-sd from the bulk (about 0.5-1.5% per matrix, both signs roughly symmetric, magnitudes ~1-15). This is consistent with sparse large-magnitude corruption, but I could not confirm it: clean GPT-2 weights also have heavy tails, and I have no reference.
- Damage is catastrophic for the forward pass: damaged dev NLL = 370.3 nats/token (ppl overflows, ~7e160). Residual norms explode in block 0 (embedding ~7, after block 0 ~500, after block 11 ~2000; logits up to ~700). The tied embedding/LM head (`wte`) is the most sensitive piece: masking its outliers alone moves NLL from inf to ~10-12.
- `wte` shows some columns (e.g. 36, 430, 496) that are nearly constant across tokens (MAD ~0) with ~1% large outliers; I read the constant offset as natural (rogue-dimension-like) and only treated the outliers as corrupt.
- `wpe` has very small clean scale (~0.005 per entry) with large corrupted-looking values; its per-column robust scale varies from 0.002 to 0.05.

## (b) What I changed
- Only `wte.weight` was modified. Detection: |w - column median| / (matrix-wide MAD * 1.4826) > T; flagged entries are set to the column median. Selected T=0.7 from {1.0, 0.7} by NLL on the first 4096 dev tokens (T=0.7 flags 48% of entries; T=1.0 flags 32%).
- Rejected alternatives (all measured on dev):
  - Masking outliers in all c_attn/c_proj/c_fc matrices made dev NLL worse (lin T=4: 44.8 vs 21.8 on 4096 tokens with wte T=2.5), so I did not apply it.
  - Low-rank hard-impute of wte (rank 32/64/128) was no better than column-median fill.
  - Masking |wpe| > 0.1 hurt (14.95 vs 11.84).
- Everything else (wpe, all linear weights, biases, LayerNorms) is copied unchanged. Keys, shapes and float32 dtype match the input.

## (c) Dev perplexity (full dev.txt, 13,954 tokens, 512-token windows)
- Damaged: NLL 370.34 (ppl ~6.9e160, effectively infinite)
- Repaired: NLL 10.27 (ppl ~2.9e4, top-1 0.0002)
- Note: the T=0.7 setting is effectively a strong shrinkage of the embedding toward column medians. It lowers NLL but is not a faithful reconstruction of sparse corruption, so the improvement may not transfer to the held-out benchmarks.

## (d) What I could not fix and confidence
- The model is still far from a working GPT-2 (ppl ~3e4 vs. an expected ~30). I did not identify the corrupted entries in the transformer blocks, and I did not find a repair that helps there.
- The true corruption pattern (which entries, how they were altered) remains unknown to me. Confidence that this repair is close to the original weights: very low. Confidence that it beats the damaged checkpoint on dev: high.

## (e) Rules
I followed the rules: no downloads or network access, no reads outside this directory, no gradient-based or loss-driven weight optimisation (only outlier statistics, column medians, SVD-based imputation tests, and dev-NLL to choose among a few thresholds), torch.set_num_threads(1) in every script, no sub-agents. One deviation: one final evaluation run exceeded the foreground timeout and was moved to the background; it was not restarted.
