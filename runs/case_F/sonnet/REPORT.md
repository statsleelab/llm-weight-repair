# Repair report (case F, sonnet)

## (a) Diagnosis
- About 3% of entries of every 2-D weight (wte, wpe, c_attn, c_proj, c_fc) were **replaced by i.i.d. Gaussian noise, std about 1.24**, independent of the original value (measured on wpe, where smoothness along position gives a clean reference: corr(corrupt, clean estimate)=0.06, kurtosis about 3, 2.94% of entries).
- Detected through |x| distribution (a plateau of values 0.5-3 on top of an N(0,~0.14) bulk), wpe smoothness, and forward-pass diagnostics.
- Damaged dev ppl is about 4e175 (4096 tokens, window 512).
- Further findings: GPT-2 has legitimately huge weights in "massive" residual dims 138/378/447 (early mlp.c_proj, paired entries in one neuron row) and heavy tails in layers 10-11 c_proj, so naive outlier clipping destroys the model. Dims with large ln_f gain (6, 36, 107, 314, 374, 430, 442, 479, 496, 627, 703) are extremely sensitive: leftover corruption in residual writes to them is amplified.

## (b) What I changed (scripts step01..step38, repairlib.py)
- wpe: median-of-neighbours (positions +-1,+-2) along position, entries deviating >0.1 replaced (special thresholds for the first 4 and last 3 rows).
- Other matrices: a robust rank-1 scale model S_ij = a_i*b_j (iterated medians); entries with |w| > 3.5*S zeroed, except "legit" large entries (|w|>8, or >3 in massive dims when the neuron row has >=2 large entries in the massive dims).
- Because ln_f gain amplifies them, all c_proj columns and wte columns for residual dims with ln_f gain > 2 (57 dims) were zeroed entirely. This is lossy but cut ppl by many orders of magnitude.

## (c) Perplexity (dev.txt, first 4096 tokens, window 512)
- damaged: about 4.3e175
- repaired: about 1.7e4 (still very bad).

## (d) Not fixed / confidence
- The model is NOT properly repaired. Roughly 1% of entries (corrupt values that landed near zero) are indistinguishable from clean ones by value, and residual errors in sensitive dims still dominate. The first real token (position 1) still develops a spurious massive activation, which in the original only occurs at position 0.
- Zeroing 57 residual dims in c_proj/wte throws away real signal.
- Low confidence that the benchmarks improve beyond near-chance. Better would need structured (low-rank / cross-layer) estimation of the small hidden corruptions, which I did not get to.

## (e) Rules
Followed: no network or downloads, stayed in the working dir, no gradients or training, one thread, no sub-agents. I went slightly over the suggested tool-call budget (about 40 calls).
