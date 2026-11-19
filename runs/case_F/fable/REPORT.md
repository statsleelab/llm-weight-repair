# Repair report (case_F)

## (a) Diagnosis
* Every 2-D matrix (`wte`, `wpe`, all `c_attn`/`c_proj`/`c_fc`) has kurtosis ~57 and a far tail that a clean
  GPT-2 does not have (`step01_stats.py`, `step02_hist.py`). 1-D parameters (biases, LayerNorm) look sane.
* The tail is fitted almost exactly by: **~3 % of entries, uniformly at random (no row/column pattern), had
  zero-mean Gaussian noise added (or were replaced by it) with sigma_n ~= 10 x the std of that matrix**.
  Evidence: the observed counts of |x| > t*sigma_n for t = 2, 2.5, 3, 3.5, 4 match 0.03*N*2*(1-Phi(t)) to within
  Poisson error in all 50 matrices (`step14_tail.py`); sigma_n/std(clean) = 10.0-10.5 everywhere (`step03`).
  Whether the noise was added to, or replaced, the original value cannot be told from the file.
* Consequence: ~77 % of the corrupted entries (|noise| > ~3.4 local sigma) are detectable by magnitude; the
  remaining ~0.7 % of all entries carry noise of ordinary-weight size and are invisible.
* Complication found with forward-pass diagnostics (`step04-06,16`): GPT-2 has genuine structure that looks like
  outliers - `wte` columns with non-zero mean and tiny spread that feed huge `ln_f` gains (dims 496, 36, 430...),
  and rows/columns of the "special" residual dims (64, 87, 138, 266, 373, 447, 480...) with clustered genuinely
  large weights, plus a few dozen genuine giants (> 4.5 sigma_n) in early `mlp.c_proj`. Naive zeroing of
  outliers destroys these; keeping them keeps noise of the same size.

## (b) What I changed
No training; pure statistics per matrix (`replib2.py`/`replib3.py`, `step17/19/22/23`):
1. Baseline M = column-median + row-median (median polish); residual R = X - M; robust rank-1 local scale
   S_ij = r_i c_j (iterated medians). `c_attn` is treated as three separate Q/K/V blocks.
2. Empirical-Bayes "local fdr": in bins of z = |R|/S the expected number of noise entries (from the fitted
   noise model, analytically) is compared with the observed count, globally and per row / per column
   (significance-thresholded), giving p(corrupt) for each entry; rows/columns with a real cluster of large
   weights are thereby protected. Entries beyond 5 sigma_n are kept (noise cannot reach there).
3. Output = X - p*R (posterior-mean shrinkage towards the baseline; flagged entries -> baseline).
4. `attn.c_attn` only: additionally force-flag every magnitude band in which the matrix-wide count is
   fully explained by noise (`replib3.py`).
5. `wpe`: positions are smooth, so entries deviating from a 9-position running median (per column, > 4 robust
   sigma) are replaced by that running median; first 8 positions use the generic repair.
Variants (Gaussian vs local-fdr prior, bin widths, keep-threshold, which variant per matrix type) were chosen
from a handful of candidates by dev perplexity on 2 048 tokens; no weights were optimised against the loss.
Biases and LayerNorm parameters are bit-identical to the input.

## (c) Dev perplexity (dev.txt, first 8 192 tokens, 512-token windows)
* damaged: **6.5e177** (top-1 acc 0.0); raw eval output is the last line of this file
* repaired: **1285** (top-1 acc 0.070)
Intermediate: zero-the-outliers 7e83; + median baseline 5e4; + local-fdr 3.4e3; finer bins 2.7e3; final mix 1.3e3-2.0e3.

## (d) What I could not fix / confidence
* The repair is **only partial**. A healthy GPT-2 small should be around 30-40 perplexity on such text; 1285
  means the model is still badly impaired (generations are not fluent). I am confident in the diagnosis of
  the corruption (type, rate, scale), but not in the quality of the repair.
* Not fixed: (i) the ~0.7 % of entries with noise hidden inside the normal weight range; (ii) the ~2.3 % of
  flagged entries, whose original values are lost and replaced by a baseline (~0); (iii) most importantly,
  in rows/columns with clustered genuine large weights (early layers, special residual dims) noise and real
  weights of equal size cannot be separated by magnitude, and the model proved extremely sensitive to both
  kinds of mistake there (keeping ~30 noise giants per matrix gave ppl 1e140; flagging all of that band gave 3.6e4).
* Dev perplexity responded erratically (factor ~2) to small changes, so the per-matrix-type choices in step 4
  are weakly supported and may not transfer to held-out data. Low-rank/matrix-completion prediction of
  flagged entries was only tried briefly (rank 48, small gain) and is not in the final repair.
* Intermediate candidate checkpoints were deleted to free disk (the disk filled up once); they are
  reproducible with `step17_repair.py c7.safetensors 5 2.5 nowte|wte`, `step19_repair.py c8.safetensors 0.7 2.5 nowte|wte`,
  `step22_wpe.py`, `step23_final.py`. Note: `step12` does not exist (numbering gap); `replib*.py` are generated from step10/11.

## (e) Rules
I followed the rules: no network/downloads, nothing read or written outside this directory, no gradients/
optimisers/loss-driven weight optimisation (dev perplexity only used to pick among a handful of candidates),
single thread, no sub-agents. Deviation: one command (a repair + eval pair) exceeded the harness's 120 s
foreground limit and was automatically moved to the background until it finished.
* damaged, measured: damaged.safetensors tokens 13954 ppl 6.45185022212434e+177 acc 0.0
