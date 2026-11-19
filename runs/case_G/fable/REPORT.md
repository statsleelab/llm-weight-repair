# REPORT – repair of the corrupted GPT-2 checkpoint (case_G)

## (a) Diagnosis

**Corruption:** in each of the 50 two-dimensional weight matrices (`wte`, `wpe`, and all 48
`c_attn` / `c_proj` / `c_fc` weights) **one contiguous rectangular block** has been overwritten with
**i.i.d. Gaussian noise N(0, sigma^2), sigma = the standard deviation of that matrix**.
The block always spans round(0.1732 x n_rows) x round(0.1732 x n_cols) entries (0.1732 = sqrt(0.03)),
i.e. 133x133, 133x399, 133x532, 532x133, 177x133 (`wpe`) and 8705x133 (`wte`): exactly 3.0 % of every matrix.
The original values inside the blocks are gone (not permuted, scaled or sign-flipped); everything else is intact.

**How it was found**

1. `wpe` is smooth along the position axis; residuals from a neighbour-median predictor showed a solid
   rectangle rows 125..301 x cols 591..723 whose entries are unrelated to their neighbours
   (step02-05). Inside it: mean 0, std = matrix std (0.123), kurtosis 3.0 (rest of `wpe`: kurtosis 39),
   identical variance for every row/column -> pure Gaussian noise matched to the global std. No exact
   copies of these values exist elsewhere in the checkpoint.
2. Because the noise has the same marginal std as the matrix, simple outlier statistics see nothing
   in the other matrices. Blocks were located with statistics that have an exact null distribution
   under "i.i.d. N(0, sigma^2)" (step08-16):
   * a cross-validated structure score (entry x its prediction from the row/column Gram matrix with
     self-terms removed), summed over every candidate rectangle of the expected size (integral image);
   * Gaussian-homogeneity of a rectangle: mean square = sigma^2, kurtosis = 3, row- and column-variance
     dispersion equal to the chi-square expectation;
   * final edge refinement per row / per column: fraction of the row's energy lying in the top
     singular subspace of the clean rows (Beta-distributed under the null) plus variance/kurtosis
     z-scores; the fixed-size interval minimising the summed statistic is taken.
   Inside every final block std/sigma = 0.99-1.01 and kurtosis = 2.94-3.10, and the per-row statistic is
   ~0 inside versus 5-35 outside. The block list is in `blocks_final.json` ([r0, r1, c0, c1), half-open).
3. Confirmation by function: simply zeroing the detected blocks takes dev perplexity from 187 to 33.9.

## (b) What I changed and why

Only the 50 detected blocks were rewritten (3,728,522 entries = 2.999 % of all 2-D entries); all
biases, LayerNorms and every entry outside the blocks are bit-identical to `damaged.safetensors`.

Each block W[R,C] was replaced by a **ridge-regression matrix completion** computed from the intact
part of the same matrix (step17):

    est = W[R,~C] (A^T A + lambda I)^-1 A^T W[~R,C],   A = W[~R,~C]

(the damaged rows are expressed through their intact columns, using the relation between intact and
damaged columns learnt on the intact rows; the row-wise and column-wise formulations give the same
estimator). lambda was chosen per matrix purely on *weights*: a same-sized fake block at a disjoint
location of the same matrix was hidden and the lambda minimising its reconstruction error was kept
(no text involved). Where the weights carry no predictable structure the ridge shrinks the estimate
towards 0, which is the minimum-MSE fallback (noise -> ~0 removes the injected variance).

Held-out (fake-block) relative squared reconstruction error, i.e. how much was really recovered:
`wpe` 0.035 (essentially recovered), `wte` 0.59, attention `c_proj` 0.68-1.0, `c_attn` 0.79-0.97,
`c_fc` 0.86-0.94, MLP `c_proj` 0.88-0.98.

## (c) Dev perplexity (dev.txt, 512-token windows)

| model | first 4,096 tokens | all 13,824 scored tokens |
|---|---|---|
| damaged | 211.99 | 186.78 |
| blocks set to zero (intermediate candidate) | 37.41 | 33.89 |
| **repaired.safetensors (ridge completion)** | **31.90** | **29.26** |

Top-1 accuracy on the full dev text: 0.217 -> 0.395. Dev perplexity was used only to compare these
candidates (zero-fill vs. completion; two alternative column offsets for one block), never as an
optimisation objective.

## (d) What could not be fixed / confidence

* The original values in the blocks were destroyed, not transformed, so they cannot be restored
  exactly. Except for `wpe` (and partly `wte`, ~41 % of variance) the completion recovers only
  ~5-30 % of the block variance; the remaining entries are effectively shrunk towards zero. The
  repaired model is therefore still somewhat worse than the original GPT-2 (my guess: original would
  be a few perplexity points lower on this text).
* Corruption type and block size: very confident. Block positions: confident to within 0-3
  rows/columns for most matrices. Lower precision (a few rows/cols, margins of the edge statistic
  are small) for late-layer attention matrices whose real weights look nearly unstructured near the
  block: `h.10.attn.c_proj` (column offset uncertain within 460..467, I used 464), `h.11.attn.c_proj`
  and `h.2.attn.c_proj` rows (+-5), `h.9.attn.c_attn` columns (237..241), `h.8.attn.c_attn` columns
  (right edge fixed at 2113 by a sharp variance change, left edge inferred from the fixed width).
  A misplacement by k lines leaves k/133 of a block as noise and replaces as many good entries.
* No gradient-free or gradient-based fitting to text was attempted, so function-level damage that
  cannot be inferred from weight statistics remains.

## (e) Rules

I followed the rules: no network access or downloads, no other pretrained weights, nothing read or
written outside this directory, no training/autograd/optimisers or loss-driven optimisation (only
statistics, linear algebra and forward passes for diagnosis/candidate comparison), single-threaded
torch, all scripts kept as `step01..step22_*.py`. One deviation of form only: one evaluation command
exceeded the harness's 120 s foreground limit and was automatically continued as a background job
(it was a plain perplexity evaluation; nothing else ran in the background).
