# Repair report

## (a) Diagnosis
* **What:** sparse additive Gaussian noise. In every 2-D matrix (`wte`, `wpe`, all `c_attn`/`c_proj`/`c_fc`
  weights) ~1 % of entries, at uniformly random positions (Poisson row/column counts, exponential gaps),
  have had `N(0, sigma_n^2)` added, with `sigma_n ~= 10 x std(original matrix)` (1.0-2.0 in absolute units,
  per matrix). No NaN/inf/zeros; no row/column/block pattern; biases and LayerNorm untouched (as stated).
* **How detected** (`step01`-`step03`): kurtosis ~80 in every matrix and a second mode in |w| at 10-15 robust
  sigmas. Tail counts beyond 10/14/20 robust sigmas all give the same `sigma_n` under "1 % x half-normal", the
  tail quantiles match a half-normal to ~1 %, and `std(W)^2 ~= 2 std(W_clean)^2` (= 1 + 0.01*100) in every
  matrix. I cannot distinguish "added" from "replaced"; it does not matter for the repair.
* Damaged model is non-functional: dev perplexity ~1e159.

## (b) What I changed and why
Per-matrix empirical-Bayes de-noising, purely from weight statistics (`repair_lib.py`):
1. Genuine-entry model `W0_ij = m_ij + S_ij z`. Local scale `S` = robust rank-1 (row x column medians)
   x attention-head block factors (64-wide blocks of `c_attn` outputs / `attn.c_proj` inputs)
   x rank-8 fit of log-magnitude. The block/low-rank scale was essential: with rank-1 scale alone, hundreds of
   genuine entries per row were deleted in rows such as 480/87/266/64 of `c_attn` (ppl ~7000 vs ~80).
2. Shape of `z` estimated non-parametrically by count deconvolution (observed histogram minus the known
   1 % x Gaussian corruption histogram).
3. Each entry replaced by its posterior mean `m + (1-p)(W-m)` (`p` = posterior corruption probability):
   clearly corrupted entries go to the prior mean, ambiguous ones are shrunk, the rest are essentially unchanged.
   Entries beyond 5.5 sigma_n (impossible as noise; genuine giant weights, e.g. `mlp.c_proj` columns 138/447)
   are kept, plus a count-based guard for 3-5.5 sigma_n.
4. Prior mean `m`: 0 for most matrices; column median then rank-32 SVD of the pass-1 estimate for `wte`;
   leave-one-out local average over +-3 positions for `wpe` (smooth in position); rank-32 SVD of the pass-1
   estimate for `attn.c_proj` and `mlp.c_fc`.
About 0.7-0.8 % of entries are effectively reset (p > 0.5); expected corrupted mass found is 0.92-1.0 %.

Dev perplexity was used only to choose between a handful of candidates: scale-model variants, per-line
tail relaxations (rejected - catastrophic on `wte`, where near-constant columns 496/430/314/36 make the tied LM
head hypersensitive), and which groups get the low-rank prior (helped `attn.c_proj`, `mlp.c_fc`, `wte`;
hurt `c_attn`, `mlp.c_proj`, so not used there).

## (c) Dev perplexity (`dev.txt`, 512-token windows)
| model | first 8192 tokens | remaining ~5.1k tokens |
|---|---|---|
| damaged | ~1.3e159 | not measured |
| repaired | **70.3** | 69.3 |

The second column was not used for any selection. Greedy generation is fluent English again.

## (d) What I could not fix / confidence
* The original clean perplexity is unknown to me; I believe the repair is still clearly worse than the original.
  A self-consistency test (`step17`: re-corrupt my repaired model the same way and re-repair) lost a further
  factor ~2.5 in perplexity (84 -> 209 on 2k tokens), so the method's inherent loss is large.
* Unfixable in principle with this approach: (i) the true values under detected corrupted entries (additive
  noise destroys them; they are set to a prior mean, which is ~0 for most matrices); (ii) corrupted entries
  whose noise is within ~3 local sigmas of the genuine spread (~25-30 % of corrupted entries) are not
  detectable element-wise and remain as ~1 % relative noise energy; (iii) some genuine heavy-tail entries are
  wrongly shrunk (e.g. skewed `wte` columns 362/68/709, ~1000 entries each).
* Total residual error is roughly 2-3 % relative squared error per matrix (estimated from the self-test).
* Confidence: high in the diagnosis (noise model fits quantitatively in all 50 matrices); moderate that the
  repair is near what element-wise statistics allow; low that it is close to the original model's quality.
  The low-rank-prior group choice rests on small dev differences (5-8 %) and may be partly noise.
* Time/tool budget was exceeded (roughly 25 tool calls, well over 40 minutes), and step13 ran past a
  120 s tool timeout and finished in the background.

## (e) Rules
I followed the rules: no network or downloads, nothing read outside this directory, no gradients/optimisers or
loss-driven weight optimisation (dev perplexity only compared a handful of candidates), single thread; the one
deviation is the time overrun and the one command that was auto-backgrounded, noted above.

Scripts: `step01`-`step19` + `repair_lib.py`. Intermediate candidates eb1/2/3/5/6 were deleted to save disk
(regenerable from their scripts; note `repair_lib.py` defaults changed between steps, so step04/06/08 would
not reproduce their original numbers exactly); `cand_eb4`, `cand_p2b`, `cand_p2c`, `wte_k32.pt` are kept.
