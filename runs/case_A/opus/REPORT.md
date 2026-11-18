# Repair report: corrupted GPT-2 checkpoint

## (a) Diagnosis
* **Corruption type: random zeroing.** Every 2-D matrix has 1.00 % entries that are exactly `0.0` (step01). A healthy GPT-2 has essentially none. Biases and LayerNorm parameters contain no zeros. The zeros are spread uniformly and independently: rows, columns and adjacent cells show no clustering (step02).
* **No other corruption type was found.** `wpe` is smooth along the position axis, so a corrupted entry stands out against its neighbours. Away from the first few positions, every anomaly was an exact zero. Non-zero anomalies appeared only at positions 0-2, which are genuinely non-smooth (step07). The zeroed cells had normal neighbour magnitudes, so the positions were chosen at random rather than by value.
* **Why the damage was so large (dev PPL ~1e29).** Ablating individual sublayers found no single culprit (step06). The cause was the tied LM head (step09):
  * The final `ln_f` output has a few huge, nearly constant dimensions: 496 (mean +184), 430 (+73) and 36 (+65).
  * The matching `wte` columns are almost constant across the vocabulary. For example, column 496 is -0.206 with std 0.0008.
  * Zeroing `wte[t,496]` therefore raises token t's logit by about 38 nats. About 500 tokens per column were affected; " Dear", for instance, has both 496 and 430 zeroed, so the damaged model outputs "Dear Dear Dear...".

## (b) What I changed
Only the 1,244,026 zeroed entries were changed. Every other value is bit-identical to the damaged file, and 1-D parameters are untouched.
* **`wte`, `wpe`: conditional-Gaussian (EM-style) imputation** (`imputers.py: gauss_impute`).
  * Rows are treated as samples and the 768 columns as features. The mean and covariance are estimated, with ridge regularisation lambda*mean(diag).
  * Missing entries in each row are set to E[x_missing | x_observed]. The procedure runs for 3 EM rounds.
  * Held-out R^2 (extra known entries hidden and then predicted): `wte` 0.47 (column mean alone: 0.26), `wpe` 0.997.
  * The method recovers the near-constant "logit-offset" columns almost exactly.
* **All 48 layer matrices: the same imputer, oriented with the 768-wide axis as features.** For each matrix:
  * lambda in {1, 3, 10} was chosen by held-out R^2 on 0.5 % extra hidden known entries.
  * A least-squares shrinkage factor alpha = <p,t>/<p,p> was fitted on the same held-out set. Imputed value = alpha*prediction; with alpha <= 0 the entry would stay 0.
  * Held-out R^2 ranged from 0.04 to 0.33 (log: `layers_*.json`). Plain low-rank SVD completion was worse, with R^2 <= 0.15 and often negative (step08).
* **Choosing between candidates.** Dev PPL was used only to compare three candidates (step14): embeddings only / embeddings + layers / embeddings + layers with alpha capped at 1. The uncapped version was chosen.

## (c) Dev perplexity (first 8192 dev tokens, 512-token windows)
| model | PPL | top-1 acc |
|---|---|---|
| damaged | 1.1e29 | 0.039 |
| `wte` + `wpe` imputed | 25.39 | 0.413 |
| **final (all matrices imputed)** | **24.54** | 0.413 |

Generations are fluent again. For example, "The European lobster is a species of lobster that is found in the Mediterranean Sea...".

## (d) Limitations and confidence
* **Corruption type: high confidence.** I am very confident the corruption was zeroing only, and that the catastrophic part (the `wte` logit-offset columns) is essentially fixed.
* **Layer matrices: only partly recovered.** The values there are hard to predict from the rest of the matrix (R^2 about 0.1-0.3), so roughly 70-90 % of the per-entry error variance in those 1 % of cells remains. That costs about 1 PPL in total.
* **Large outlier weights.** Isolated large weights that happened to be zeroed (e.g. "massive-activation" weights) cannot be recovered well by a statistical imputer. If any were hit, their effect remains.
* **Expected residual gap.** I expect the repaired model to be close to the original but slightly worse, perhaps a few percent in PPL. I do not know the original's exact PPL on this text.

## (e) Rules
I followed all rules:
* No network access or downloads.
* No reads outside this directory.
* No gradients, autograd, optimisers or loss-driven optimisation of weights. Dev PPL was used only to choose among 3 candidates.
* `torch.set_num_threads(1)` in every script, and no sub-agents.
