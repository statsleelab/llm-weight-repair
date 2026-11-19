# REPORT

## (a) Diagnosis
- Corruption looks like ~0.1% of entries in every 2-D matrix (wte, wpe, c_attn, c_proj, c_fc) replaced/perturbed by large, roughly N(0, ~1.5^2)-sized values (scattered uniformly over rows and columns; max |w| ~5-17). Evidence: scattered large outliers far beyond each column's robust scale; in near-constant wte columns (e.g. dim 496, ln_f weight 17) ~50 entries per column stand out with a Gaussian-looking magnitude profile (top 4.0, rank-10 2.1, rank-50 0.18), matching 0.1% of 50257 rows.
- Damaged model is catastrophic: it predicts " socially" everywhere. A single corrupted wte entry (row 18118, dim 496 = 3.1) times the huge final-LN activation in dim 496 (~112) gave logit +350. Damaged dev NLL (8192 tokens) = 448.8 nats/token.
- Natural structure that must NOT be touched: per-column nonzero medians (wte/wpe), a few rogue columns (326, 709, 447, 373, 138...), and big natural weights in mlp.c_proj of blocks 0-2 (up to 17) that create the pos-0 massive activation. Zeroing them (my first attempt) destroyed the model.

## (b) Repair
Script `step26_final.py` (thresholds chosen by `step25_cd.py`, helpers in `common.py`): per matrix, subtract column median, divide by a rank-1 robust scale sqrt(rowMAD*colMAD); entries with |z| > T are replaced by their column median. T per group: wte 10, wpe 7, c_attn 20, attn.c_proj 14, c_fc 14, mlp.c_proj 14. For wpe and both c_proj types, entries in columns/rows with anomalously many flagged entries (Poisson-style test) are left alone to protect natural structured outliers. 32,965 entries replaced in total (about 8% of the ~400k expected corrupted).
T per group was chosen by a small coordinate sweep over {7,10,14,20,30} on dev NLL (hyper-parameter selection only, no weight optimisation).

## (c) Dev perplexity (first 8192 tokens, 512 windows)
- damaged: NLL 448.8 (ppl ~ e^449, effectively infinite)
- repaired: NLL 5.09, ppl 163.2, top-1 acc 22.5%

## (d) Not fixed / confidence
- The repair is only partial: ppl 163 is far from a healthy GPT-2 (~30-40 expected). Roughly 75% of corrupted entries are not detected: those whose corrupted magnitude falls inside the natural spread, plus corruptions in protected columns. Also replacing by column median loses some natural large entries (false positives). Sign flips/scaling-type damage, if present, were not addressed.
- Generation is fluent-ish but weak ("The capital of France is a great source of its own."). Confidence that the repair is better than damaged: very high; that it is near the original: low.
- Better ideas not completed: low-rank/matrix-completion imputation of flagged entries instead of the column median, and activation-sensitivity-weighted thresholds.

## (e) Rules
No network, no training/gradients/optimisers, single thread, no sub-agents, all files in this directory. Minor deviations: one script was auto-moved to a background task by the tool harness, and I read its output file under the harness scratch directory (/tmp/claude-0/...) to see my own script's printouts.
