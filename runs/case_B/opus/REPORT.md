# Checkpoint repair report (case_B / opus)

## (a) Diagnosis
* **Corruption type:** sparse, random-position replacement or additive noise. About 1% of the entries in every 2-D matrix were overwritten with values from a broad zero-centred distribution (|value| is roughly 0.5 to 5, with heavy tails). The noise does not depend on the original entry's row or column scale. Evidence:
  * Every matrix has the same excess of outliers: about 0.45% of entries above 8 robust SD, with the sign split 50/50 (step01, step02).
  * In near-constant `wte` columns (496, 430: true value ≈ -0.206 / -0.257 with MAD ≈ 1e-5), about 0.9-1.0% of rows hold arbitrary values from -4.6 to +3.2 (step17). Their ratio to the true value is spread continuously, so the change is not multiplicative.
  * Flagged-value magnitudes do not follow the local entry scale (step05).
* **Where:** all 2-D matrices (`wte`, `wpe`, every `c_attn`/`c_proj`/`c_fc`), spread roughly uniformly.
* **Detection model (common.py):** each matrix is modelled as an additive location `mu_ij = a_i + b_j` (median polish; needed because many `wte` columns have large constant offsets) plus a rank-1 multiplicative scale `s_ij = r_i * c_j` (robust median fit). The model is refitted iteratively with outliers masked out. Each entry gets z = (w - mu) / s.

## (b) What I changed
Each entry was replaced by a posterior-mean estimate `mu + (1 - p) * (w - mu)`. Here p is the posterior probability that the entry is corrupted, under a two-part mixture:
* clean entries: `mu + s * Student-t(nu=3)`
* corrupted entries: Cauchy(0, 1)
* prior corruption rate: pi = 0.1

The Cauchy corruption model was essential. With a Gaussian corruption model, very large corrupted values got p ≈ 0 and were kept, which was catastrophic (step31).

One explicit protection: the genuine massive-activation weights in `mlp.c_proj` of layers 0-3, output columns 138/447/378 with |w| > 4, are never changed. These are real GPT-2 structure (the attention-sink / massive-activation circuit), and zeroing them was a major failure mode. Protecting them took ppl from 82k to 9k (step32).

Biases and LayerNorm parameters were not touched. The hyper-parameters (nu, pi, protection on/off) were chosen from about 10 candidates by dev perplexity on 4096 tokens. No weights were fitted to text loss.

## (c) Dev perplexity (first 8192 tokens of dev.txt, 512-token windows)
| model | ppl | top-1 acc |
|---|---|---|
| damaged | 1.3e159 | 0.000 |
| repaired | **1304** | 0.114 |

## (d) What I could not fix / confidence
**The repair is only partial.** A healthy GPT-2 should be around 30-40 ppl on this text, so the repaired model is still very badly damaged. Greedy generation repeats itself ("the capital of the capital of ...").

The main unsolved problem: roughly a third to half of the corrupted entries have values inside the clean weight distribution and cannot be found from magnitude alone. On top of that, GPT-2 has genuinely heavy-tailed, structured weights (massive-activation neurons, special residual dims 64/87/138/266/373/447/480...). Being more aggressive therefore also destroys real weights, and being less aggressive leaves damage in place.

A simulation (step28-29) re-corrupted my own repaired model with 1% noise and ran the same pipeline on it. Perplexity still went up about ×11, which confirms that leftover undetectable corruption does large harm. The most likely improvements are better modelling of the structured heavy-tailed weights (protecting all massive-activation circuitry, e.g. the `c_fc` neurons that feed it) and smoother position-wise modelling of `wpe`. Confidence that the changed entries are mostly true corruptions is high. Confidence that the model is close to the original is low.

## (e) Rules
I followed the rules: no network access or downloads, only files in this directory were read, no gradients or optimisers, torch.set_num_threads(1) in every script, and every script was kept as step*.py plus common.py. Dev perplexity was used only to choose among a handful of candidate settings. One deviation: a few large intermediate caches were written to my session scratchpad directory, not to this directory.
