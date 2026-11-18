# Report

## (a) Diagnosis
Corruption is **zeroing of ~1.0 % of entries, uniformly at random** in every 2-D weight (wte, wpe, all c_attn/c_proj/c_fc). Detected in step01/02: each matrix has 0.986-1.031 % exact zeros (natural float32 weights virtually never equal 0.0), no row/column concentration, no NaN/Inf, no outlier magnitudes. Biases/LN contain no zeros. Zero entries are therefore identified exactly (mask = W == 0).

## (b) Repair
Only masked (zero) entries were rewritten; known entries untouched. Missing values were filled by hard-impute low-rank matrix completion (iterated truncated eigen/SVD projection, 12 iters) per matrix. Rank per matrix chosen by hold-out CV on 3 % of the *known weights* (weight reconstruction RMSE, no text loss); rank 0 (keep zero) where nothing beat zero. Chosen ranks: wpe 16 (RMSE 0.016 vs 0.126 for zero: very low-rank), wte 64 (0.117 vs 0.144), c_fc ~32, c_attn 8-128, c_proj 0-16 (gains small, 1-5 %). Scripts: step01-05.

## (c) Dev perplexity (first 8000 tokens of dev.txt, 512 windows)
Damaged: ~1.2e29 (catastrophic). Repaired: 29.67.

## (d) Not fixed
Most weight matrices are close to full-rank noise, so imputed values for c_attn/c_fc/c_proj recover only a few percent of the error of the zeroed entries; ~1 % residual error in those matrices remains. wpe is recovered nearly exactly, wte partially. Likely further gains possible from function-aware methods (not tried). Confidence: high on diagnosis, moderate on repair quality; the original perplexity is unknown to me.

## (e) Rules
Followed all rules: no network/pretrained weights, no training or gradient use, stayed in the directory, one thread, no sub-agents. Note: one `ps` call incidentally displayed other participants' process command lines; I did not use that information.
