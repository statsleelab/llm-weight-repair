# Repair report – case_D

## (a) Diagnosis
* No NaN/Inf, no zeros and no exponent-scale outliers, so this is not bit-flip, zeroing or scaling damage.
* `wpe` is a smooth function of position, so I used it as a probe (step03–08). Rows (positions) **151–251** and columns **59–135** form a clean **rectangle** that has lost all smoothness (101×77 = 0.99 % of entries). The values inside it are **i.i.d. Gaussian noise**: mean ≈ 0, std ≈ the std of the whole matrix (0.122), kurtosis 2.92, lag-1 autocorrelation ≈ 0, and correlation ≈ 0 with an interpolation of the block's surroundings. So the original values are gone and have been replaced by random noise.
* Hypothesis: every 2-D matrix has a single contiguous rectangle (~1 % of entries) replaced by N(0, σ_matrix²) noise.
* Detection for the other matrices (step11–13): I split each matrix into 16×16 windows and scored each window with a "pure N(0,σ²) noise" test (variance ≈ σ², mean ≈ 0, no row or column scale heterogeneity). Then I ran a maximum-sum sub-rectangle search over the windows that look most like noise. I accepted a candidate (step15) only if it covered 0.4–2.2 % of the matrix, did not span (almost) a whole dimension, and had block std/σ within 1 ± 0.08 and kurtosis within 3 ± 0.4. **25 matrices** have accepted or confirmed rectangles: `wte`, `wpe`, `h.0.attn.c_proj` (confirmed visually from the window map), most `mlp.c_fc` and `mlp.c_proj` matrices, and `h.4`/`h.8`/`h.9 attn.c_attn`. The rectangles are listed in `accepted_rects.pt` and `step13_out.txt`. Except for `wpe`, the boundaries are only accurate to about 16 entries.

## (b) Repair
* `wpe` block: linear interpolation along position between rows 150 and 252.
* Every other accepted block: ridge-regression imputation. On the rows outside the block, I regress the block's columns on all the other columns, then predict the block rows from their uncorrupted columns. Primal or dual form depends on shape. λ = α·trace(G)/p, with α = 0.1 picked from {0.1, 1.0} by dev perplexity.
* Other fills I compared: zero-fill was catastrophic (ppl ~1e15, because some columns have large non-zero means). Column-mean fill gave 34.3 at 4k tokens, against 33.6 for ridge.
* All other weights are unchanged.

## (c) Dev perplexity (first 8192 tokens of dev.txt, 512-token windows)
* damaged: **35.72**
* repaired: **31.77**

## (d) Limitations and confidence
* I am highly confident about the type of corruption (rectangular blocks of Gaussian noise), because it was verified exactly on `wpe`.
* About 25 matrices are probably still corrupted but were not repaired, because the detector was ambiguous for them. This is especially true for most `attn.c_attn` and `attn.c_proj` matrices, where natural V/head structure looks like noise to my test. Several of those candidates were rejected.
* The coarse 16-entry boundaries mean some clean entries were overwritten by predictions, and some noisy edge entries were missed.
* The imputation recovers only part of the lost information: the held-out relative MSE is about 0.9–1.0, so most of the gain comes from removing large random values rather than from restoring the original weights exactly.

## (e) Rules
I followed all rules: no network or downloads, only this directory was used, no training, autograd or loss-driven weight optimisation (dev ppl was used only to choose between a few fill methods and one ridge hyper-parameter), single thread, and every script is kept as `step*.py`.
