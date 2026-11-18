# Repair report – case C

## (a) Diagnosis
* Every 2-D matrix had an unusually large number of entries that were exactly 0.0 (17 710 in each c_attn, 5 929 in each attn.c_proj, 23 639 in each c_fc and mlp.c_proj, 7 854 in wpe, 387 002 in wte). That is exactly 1 % of each matrix. A trained GPT-2 has almost no exact zeros.
* The zero positions (step02/step03) always form **one contiguous rectangular block per matrix, ~10 % of the rows × ~10 % of the columns** (e.g. c_attn 77×230, c_fc 77×307, mlp.c_proj 307×77, wpe rows 734–835 × cols 123–199, wte token rows 25013–30038 × cols 240–316). The blocks are 100 % zero, and there are no zeros outside them. So the corruption is **block zeroing**. I found no other anomalies: there are no NaN/Inf values, and the large values present match GPT-2's usual outlier dimensions.
* Damage attribution (step06): putting the damage back one group at a time into an otherwise repaired model shows that the **wte block causes almost all of the loss**. It is tied to the LM head, and the final hidden state has a large constant component on those dims (norm ≈ 33), so zeroing shifts the logits of about 5 000 tokens. With wte unrepaired, ppl is about 280. Each layer's blocks cost only ~0.0–0.3 ppl, and wpe has a negligible effect on dev.

## (b) What I changed
Only the zeroed blocks were overwritten. Biases and LayerNorm parameters were left as they were.
* **All matrices (step05):** I imputed each block by ridge regression from the intact parts of the same matrix. "col" mode predicts the block columns from the other columns, with the intact rows as samples; "row" mode is the transpose; I also tried averaging the two. The orientation and λ were chosen per matrix by **cross-validation on simulated blocks of the same shape inside the intact part of the matrix**. This is a weight-space criterion and uses no text.
* **wpe:** highly predictable (CV relative error 0.11), using the average of the row and column ridge predictions.
* **Transformer blocks:** only weakly predictable (CV relative error 0.85–0.99), so the predictions are small, shrunk values close to the row/column means. For the 5 blocks whose CV error was ≥ 1 (h0/h9/h10/h11 attn.c_proj, h1 mlp.c_proj), I kept zeros (step08).
* **wte (step07/08):** CV used held-out token rows adjacent to the damaged range (20000–25012 and 30039–35064) in the same 77 columns. The error was weighted by E[hhᵀ] of the ln_f output on those dims, measured with a diagnostic forward pass, so that it tracks logit error. Logit-weighted error relative to zeros: column mean 0.012, ridge (λ≈1) 0.0033, ridge + 0.5 × softmax-weighted kNN (k=30, cosine on the 691 known dims) of the ridge residuals 0.0028. I used the last of these.
* Dev perplexity was used only to pick among 3 final candidates: ridge-all 25.75, ridge wte 25.71, ridge + kNN wte 25.62.

## (c) Dev perplexity (first 8 192 dev tokens, 512-token windows)
| model | ppl | top-1 acc |
|---|---|---|
| damaged | 255.86 | 0.290 |
| **repaired** | **25.62** | 0.413 |

## (d) Limitations and confidence
* The values in the transformer-layer blocks cannot really be recovered: the weights are close to incompressible. Those blocks hold shrunk estimates or zeros, so some function is lost in those ~1 % of weights. The attribution suggests this costs about 1–2 ppl in total.
* The wte block is about 25–30 % wrong in plain weight-space terms (relative error ≈ 0.7), but the part that matters for logits, the mean direction, is well recovered. Rare-token behaviour for ids 25013–30038 may still be slightly degraded, which could affect factual recall involving those tokens.
* I am highly confident in the diagnosis (exact rectangles of zeros, nothing else). I am fairly confident that this is close to the best achievable without training.

## (e) Rules
I followed all rules: no network access or downloads, no files read or written outside this directory, no gradients or optimisation against text loss (only closed-form least squares/kNN on the weights; dev ppl was used to choose between 3 candidates), torch.set_num_threads(1) in every script, and all scripts are kept as step01–step09.
