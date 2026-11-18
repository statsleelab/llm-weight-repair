# Repair report (case_D / sonnet)

## (a) Diagnosis
- About 1% of entries in the 2-D matrices were overwritten or perturbed by roughly Gaussian values of absolute size ~0.15 (similar to the matrices' overall std). No NaN/Inf, no zeroing, no exponent or bit-flip artefacts, no duplicate-value pattern (step01-07).
- Clear evidence exists only in `wpe`, which is smooth along the position axis. Residuals against a neighbour-median prediction showed ~0.75% of entries (at z>10) off by N(0, ~0.15), independent of the column scale (step04, step08).
- In the other matrices the noise is the same size as the genuine weights, so it is invisible in marginal statistics. Large entries there are mostly genuine: zeroing them (z-score clipping) made perplexity much worse (step12). A rank-1 scale-model Bayesian shrink also hurt for c_attn, c_fc and mlp c_proj (step11).

## (b) Changes
- `wpe`: entries flagged as outliers against a local neighbour estimate (|z|>10, one pass, step09) were replaced by that estimate (about 6k entries).
- `wte`: Bayesian shrinkage with a 1% corruption prior (step10). Only about 1.2k entries get a posterior above 0.5, and they are pulled towards zero.
- All other matrices are unchanged. Script: `step14_final.py`. Output: `repaired.safetensors`.

## (c) Dev perplexity (first 8192 tokens of dev.txt, 512-token windows)
- Damaged: 35.72
- Repaired: 35.60

## (d) Not fixed
- The bulk of the corruption in c_attn, c_proj, c_fc and most of wte is not individually identifiable by statistics. I could not find a structural prior strong enough to locate it. I estimate about 90% of corrupted entries remain.
- The gain is small and the true original perplexity is unknown. My confidence that the repaired model is better than the damaged one is moderate. The improvement could partly be noise.
- Several wpe variants (more iterations, a lower threshold) were not clearly better or were slightly worse, so I kept the conservative one.

## (e) Rules
Followed all rules: no network and no pretrained weights, no files read outside this directory, no training or gradients, one thread, no sub-agents. Dev perplexity was used only to choose between a handful of candidate repairs.
Deviation: one background run (step13) wrote its output to the harness's output file. I did not read it and re-ran it with local output (`eval13.log`).
