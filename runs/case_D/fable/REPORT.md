# Repair report – case_D

## (a) Diagnosis
* **Kind of corruption:** in every 2-D weight matrix one contiguous **rectangular block** covering
  10 % of the rows x 10 % of the columns (= 1 % of the entries; e.g. 102x77 in `wpe`, 77x230 in
  `c_attn`, 77x307 in `c_fc`, 307x77 in `mlp.c_proj`, 5026x77 in `wte`) has been **overwritten with
  i.i.d. zero-mean Gaussian noise whose std equals the std of that matrix**. The original values in
  the block are gone (no sign flip / permutation / scaling relation to recover).
* **How found:** global statistics showed no NaN/Inf/zeros/outliers. `wpe` is smooth along the
  position axis, so a median-of-neighbouring-rows residual exposed the damaged entries directly:
  they form a sharp rectangle rows 151-252 x cols 59-135, with Gaussian quantiles, kurtosis 2.92,
  std 0.122 (matrix std 0.123), no correlation with neighbours or any other part of the matrix
  (`step02`-`step04`).
* **Locating the block in the other 49 matrices** (not smooth, so harder):
  1. leave-one-out ridge residual of each entry from the rest of its row/column (computed in closed
     form from the regularised Gram inverse); noise entries are unpredictable, so the normalised
     squared residual, box-filtered with the known block size, peaks on the block (`step05`,
     `step07`; z = 15-70 for 43 matrices, and free-size edge refinement in `step06` returned exactly
     the expected block size for most of them);
  2. a Gaussianity scan (box mean 0, variance = matrix variance, skew 0, kurtosis 3) as a cross-check,
     decisive where the natural weights are heavy-tailed (`step10`, `step13`, `step14`).
  For 7 matrices where the two disagreed or the evidence was weak, I compared 2-4 candidate
  positions by dev NLL (`step16`), as allowed for a handful of candidates. Final list:
  `blocks_final.json`.

## (b) What I changed and why
Only the entries inside the 50 located blocks were changed (1,245,860 entries, 1.0 %); everything
else, including all biases/LayerNorms, is bit-identical (`step17_verify.py` asserts this).
Each block was replaced by a **ridge-regression (least-squares) imputation**: the missing part of the
affected rows/columns is predicted from their intact part, using the linear relation learned on the
fully intact rows (or columns) of the same matrix (`step08_repair.py`). The ridge strength was picked
per matrix by holding out an intact pseudo-block of the same size elsewhere in the matrix and
measuring weight-reconstruction error (no text involved); when the prediction was no better than
zero, the block was set to zero. Held-out reconstruction error relative to zero-filling: `wpe`
0.016 (essentially recovered), `wte` 0.62, attention/MLP matrices 0.65-0.98 (mostly shrunk toward
zero; `h.10/h.11.attn.c_proj` zero-filled). So the gain is mainly from removing the noise, plus
partial recovery of the signal.

## (c) Dev perplexity (512-token windows, `dev.txt`)
| model | tokens 0-6144 | tokens 6144-12288 |
|---|---|---|
| damaged | 39.09 (acc 0.365) | 34.68 (acc 0.388) |
| blocks zeroed (earlier block list) | 27.71 | – |
| **repaired (imputed)** | **27.18** (acc 0.401) | **24.33** (acc 0.429) |

## (d) Not fixed / confidence
* The original block contents cannot be recovered exactly; outside `wpe` only ~5-35 % of their
  variance is restored, so about 1 % of each matrix is still effectively "shrunk to the mean". The
  true clean perplexity is unknown to me, so I cannot say how much of the gap remains.
* Block position confidence: high (edges within a few rows/cols) for ~43 matrices. **Lower** for
  `h.10.attn.c_proj`, `h.9.attn.c_attn`, `h.8.attn.c_attn`, `h.10.attn.c_attn`, `h.11.attn.c_proj`,
  `h.2.attn.c_attn`, `h.4.attn.c_proj`: there the position may be off (possibly by tens of
  columns, and for `h.10.attn.c_proj` / `h.9.attn.c_attn` possibly wrong), which would leave a strip
  of noise and remove a strip of good weights. Dev-NLL differences between candidates there were
  tiny (~0.001-0.005 nats), i.e. these blocks matter little either way; for `h.10.attn.c_attn`
  zero-filling the chosen block was slightly *worse* on dev than leaving it (-0.001 nats), so that
  one is uncertain.
* Even for confident blocks, edges may be off by 1-3 rows/columns.
* The dev-NLL candidate comparison used tokens 6144-10240, so the second column of the table is
  slightly optimistic for those 7 choices; the first column was not used for any choice.

## (e) Rules
I followed the rules: no network/downloads, nothing read or written outside this directory, no
gradients/optimisers/loss-driven weight optimisation (dev NLL used only to choose among a handful of
candidate block positions), `torch.set_num_threads(1)` in every script, all scripts kept here.
Deviation to note: two scans exceeded the tool's 120 s foreground limit and were automatically moved
to the background by the harness (they finished on their own; I did not start parallel copies), and
`step16` took about 6 minutes.
