# Repair report

## (a) Diagnosis
Corruption = about 1.00% of entries of every 2-D weight matrix (wte, wpe, all c_attn/c_proj/c_fc) set to exactly 0.0.
Detected by counting exact zeros: 1.0027% of wte, 0.999% of wpe, 1.00% of every c_attn/c_fc/c_proj (e.g. 23639/2359296 in c_fc);
healthy float32 weights essentially never contain exact zeros. Biases/LN untouched. No other anomaly (NaN, scaling) found.

## (b) Repair (step04_repair.py, imp.py)
Zero entries treated as missing. Each matrix was viewed as samples x 768 dims (wte, wpe, c_proj: rows; c_attn, c_fc: columns).
Gaussian conditional-mean imputation: column mean + shrunk covariance (EM, 2 iterations, shrinkage toward diagonal 0.3; 0.6 for c_fc),
each missing entry predicted from the observed entries of the same sample. Shrinkage/axis chosen by hold-out of extra known entries
(rms error vs zero-fill: wpe 0.17, c_fc 0.90, c_proj ~0.93, others ~0.9 -- the large-matrix entries are only weakly predictable,
wpe and wte are much better). No gradients, no loss optimisation; dev perplexity only used as a final check.

## (c) Dev perplexity (first 8192 tokens, 512 windows)
Damaged 255.9 -> repaired 25.7 (full dev text: 24.7).

## (d) Not fixed
Entries are only estimated, not recovered; for the big MLP/attention matrices the estimate is only ~10% better than zero in
rms terms, so residual error remains there (the large gain came mostly from wte/wpe and from filling zeros with the column mean/conditional means).
Held-out benchmark behaviour not measured; moderate confidence it is close to but not equal to the original.

## (e) Rules
Followed all rules: no network/downloads, stayed in the working directory, no training/gradients, single thread, no sub-agents.
