# Report (case B, sonnet) -- repair did NOT succeed

## (a) Diagnosis
- Corruption looks like ~1% of entries of every 2-D matrix replaced by large-magnitude noise (roughly N(0, ~1.2-1.5^2), independent of the true value), on top of true weights with std ~0.1-0.2.
- Evidence: `wpe` is smooth across positions (median 2nd-difference 0.003); entries violating smoothness are 1.06% of wpe, and their values (std 1.18, kurtosis ~3.2, uncorrelated with the interpolated truth) look like Gaussian replacement noise. Other matrices show a Gaussian bulk (fit well using row-median x column-median scale model) plus a flat tail of |w| in 1-6, ~0.5% of entries > 1.
- True weights have legitimately heavy lines (rows/cols tied to residual outlier dims 138/378/447/64/87/480..., and a few neurons with |w| up to 9), which are confounded with corruption.
- Damaged model: residual stream explodes, dev ppl ~1e161.

## (b) What I changed
- `wpe`: smoothness-based detection (residual vs neighbour mean, per-column MAD threshold, 3 iterations), flagged entries (1.06%, rows >=1) replaced by linear interpolation along position.
- All other 2-D matrices: Bayesian shrinkage w * P(clean | w), with clean = generalised-Gaussian with per-entry scale (row median x col median), corrupt = N(0,1.2^2), prior 1.05%. Entries in "heavy" rows/cols (>1.2% of entries with z>5) and |w|>5 left untouched (file `cand_frac0.012`, from step31_frac.py -> repaired.safetensors).

## (c) Dev perplexity (first ~1-4k tokens, 512 windows)
- Damaged: ~4e161.
- Repaired (submitted): ~3e40 (1024 tokens). Still catastrophically broken (greedy output is repetitive junk). Best other variants 1e47-1e99.

## (d) Not fixed / confidence
- The model is not functional. Diagnostics: attention sink on position 0 is missing (mean attention to pos 0 ~0.01 vs expected large), layer-11 head 8 output rms 39, logit-lens nll jumps in the last block. I believe the remaining error comes from corruption inside legitimately heavy/outlier-dim lines (which I had to leave unrepaired) and from undetectable small corruptions amplified by the outlier residual dims, and possibly wte[EOT]/wpe[0] rows. Low confidence in the submitted file being useful; wpe repair is the only part I am confident is correct.
- Tuning by dev ppl was only used to choose among a handful of variants.

## (e) Rules
Followed: no network, stayed in the working directory, no training/gradients/optimisation of weights, single thread, scripts kept as step*.py, no sub-agents (one background run of a sweep script, noted).
