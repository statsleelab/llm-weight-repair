# Report

## (a) Diagnosis
The corruption is one contiguous rectangular block per 2-D weight matrix. Each side is about 17.3% of the matrix dimension (sqrt(0.03)), so the block covers about 3.0% of the entries. The block entries are replaced by Gaussian-like noise with roughly the matrix-wide std. Biases and LayerNorm parameters were not touched.

How I found the blocks:
- **wpe:** position rows are very smooth, so residuals against neighbouring rows show a clean block at rows 124-302, columns 591-723.
- **All other matrices:** I fitted a multiplicative row-scale x column-scale model of log|w| (step14/step22). Natural weights correlate with that model and the noise block does not. I averaged the correlation over tiles and then box-filtered with the known block size (step25), which gave a clear minimum for the MLP matrices (c_fc, mlp c_proj) and wte. Exact offsets were refined by a fixed-size sliding search (step26/28).
- **wte:** rows ~34394-43098, columns 620-752.
- **MLP c_fc / mlp c_proj:** boxes are located in every layer, with box correlation about 0 versus a matrix median of 0.2-0.3.
- **attn c_attn / c_proj:** boxes were located the same way, but the evidence is weaker (box correlation about 0 versus a median of 0.05-0.15; the best box is separated from the runner-up by a smaller margin, and very weakly for some layers). Their boundaries and even locations are less certain.

## (b) Repair
- **wpe:** linear interpolation across the block rows, per column, between rows 123 and 303.
- **c_fc, c_attn, attn c_proj:** the block is set to 0 (the mean of the noise-free weights, so least harmful).
- **mlp c_proj and wte:** the block is filled with a ridge regression (lambda = 0.3 x mean Gram diagonal), trained on the clean rows, from the surviving columns to the corrupted columns. This is only a tiny improvement over zero (NLL 3.659 -> 3.644).

## (c) Dev perplexity (first 4096 tokens of dev.txt, 512-token windows)
| model | NLL | perplexity | top-1 acc |
|---|---|---|---|
| damaged | 5.357 | ~212 | 0.201 |
| zero-fill + wpe interpolation | 3.659 | ~38.8 | 0.353 |
| repaired (final) | 3.644 | ~38.2 | 0.352 |

## (d) Not fixed / confidence
- Zeroed or ridge-estimated block values are not the true originals. The lost information (about 3% of each matrix) is only approximated, so the model is likely still noticeably worse than the original (clean GPT-2 would have perplexity of roughly 25-30 on such text).
- Block boundaries are accurate to a few rows or columns.
- Confidence is high for the wpe, wte and MLP block locations, and moderate for the attention matrices (weaker evidence; a wrong box would leave noise in place and zero real weights).
- I did not try better fills for c_fc, c_attn and attn c_proj (for example low-rank completion).

## (e) Rules
No network, no outside-directory access, no training or gradients, one thread, no sub-agents. Dev perplexity was used only to compare the final candidates (damaged, zero-fill and ridge fill), not as an optimisation objective.
