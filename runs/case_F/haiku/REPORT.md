# REPORT: repair of damaged GPT-2 checkpoint (Haiku participant)

**Outcome: not a successful repair.** The delivered `repaired.safetensors` is my best of the few candidates I tried, but its dev perplexity is still astronomically large (about 1.8e67 on 8192 dev tokens), so it is not a functional language model. Treat it as a partial repair only.

## (a) Diagnosis

- **Corruption type: sparse, roughly uniform, large-magnitude entry replacement.** Each 2-D matrix has a narrow bulk plus isolated outliers far outside it.
- **wpe.weight** (1024 x 768): the clean bulk is smooth along position, with per-column robust std of about 0.003 for most of the 768 dims. About 3 % of entries show up as isolated spikes of roughly +/-0.5 to +/-3 against neighbouring positions. A subset of about 80 columns carries a genuine smooth position profile of large amplitude, so I did not treat those columns as corrupted in themselves.
- **2-D matrices** (c_attn, c_proj, c_fc, and wte): flagging entries beyond 4.5 robust sigma of each matrix's own core flags about 2 % of entries in almost every matrix. That fits a ~3 % corruption rate, with some corrupted values too small to detect.
- **h.0.attn.c_proj.weight is the exception.** Its core is very narrow (IQR sigma about 0.037), yet about 16 % of entries lie beyond 4.5 sigma. This is the block that blows up the residual stream: after block 0 the median residual norm jumps from about 8 to about 1200 (healthy GPT-2 is roughly 50 to 100 at that depth). Zeroing this single matrix's outliers reduces the attention-output norm from about 286 to about 43.
- Diagnostic commands: `step01` to `step07`, `step11` to `step13`.

## (b) What I changed

1. **wpe.weight** (`step08_wpe_repair.py`): for each column, predicted each position from the median of its +/-2 neighbours (excluding flagged entries), flagged entries whose residual exceeded 6x the column's robust residual scale, iterated three times, and replaced flagged entries with the neighbour median. About 3.1 % of wpe entries were replaced.
2. **2-D matrices, 4.5-sigma detector** (`step09_repair2d.py`): flagged entries with |w - median| > 4.5 x core sigma, then imputed them with a rank-96 hard-impute estimate fitted to the unflagged entries (randomised SVD, 4 iterations). Zero-imputation is the alternative.
3. **h.0.attn.c_proj.weight**: re-detected with a 3-sigma threshold (21 % flagged) and zero-imputed.

Final candidate (`wpe+zero+fix0`): wpe from step 1, every other 2-D matrix zero-imputed at 4.5 sigma, h.0.attn.c_proj zero-imputed at 3 sigma. The output is float32 with the same 148 keys and shapes as the damaged file.

## (c) Dev perplexity (first 8192 tokens of dev.txt, 512-token windows)

| Candidate | Dev PPL (printed value, order of magnitude) | Top-1 |
|---|---|---|
| damaged (as delivered) | > 1e50 (exact digit count not recorded) | 0.000 |
| wpe repaired only | > 1e50 | 0.000 |
| wpe + 2-D zero-impute (4.5 sigma) | about 3e75 | 0.000 |
| wpe + 2-D rank-96 LR-impute (4.5 sigma) | about 8e80 | 0.000 |
| wpe + 2-D LR-impute + h.0.attn.c_proj LR-impute (3 sigma) | about 4e88 | 0.000 |
| **wpe + 2-D zero-impute + h.0.attn.c_proj zero-impute (3 sigma), delivered** | **about 1.8e67** | **0.000** |

Orders of magnitude were read off the printed values; I did not compute exact log-perplexities, and the damaged-model figure was not counted precisely. The delivered candidate was chosen as the lowest of the candidates tried.

## (d) What I could not fix, and confidence

- **Residual logit scale.** After block 0 the residual norms look plausible (about 84 at layer 1 rising to about 580 before ln_f), but logits still have std about 11.6 and absolute maximum about 200, and no top-1 predictions are correct. The most likely cause is undetected corruption in wte (which also serves as the LM head and feeds logits directly) and in the late-layer matrices, where roughly 1 % of corrupted entries remain below the detection threshold.
- **Entries below threshold.** Corrupted values with magnitude below about 4.5 x core sigma (roughly 0.6 in c_attn) are not detected in the 2-D matrices. Imputation is only a shrinkage-type estimate: the rank-96 fit explains only about 40 % of the energy, so imputed values carry error of roughly 0.8 x the bulk sigma.
- **Detection thresholds were not tuned** beyond the 3 and 4.5 sigma values above. A more systematic search over per-matrix thresholds, a sparse-plus-low-rank decomposition, or a check of ln_f.weight (max about 17, which looks unusual) could help, but I ran out of the time budget.
- **Confidence:** high that wpe spikes and the h.0.attn.c_proj corruption are real and were largely removed. Low that the delivered model is close to the original in any benchmarked sense.

## (e) Rules

I followed the rules: no downloads or network access, no reads or writes outside this directory, no gradient-based training, `torch.set_num_threads(1)` in every script, and every script saved as a file. Caches live in `cache/` inside this directory. The only use of dev perplexity was to choose among the handful of candidates listed above.
