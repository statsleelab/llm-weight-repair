# Repair report: case_E

## (a) Diagnosis
- About 0.1 % of the entries in each 2-D matrix were **replaced with random values whose scale follows the whole tensor's std, not the local scale** (roughly N(0, (~10·σ_tensor)²), up to about 45σ). In `wpe`, the smoothness across positions made this clear: corrupted entries do not match their neighbours, and the ratio corrupted/predicted is random, so the corruption is replacement, not scaling (step09).
- How I found it: robust z-scores per entry, centred on the column median and divided by the column and row MAD (steps 02–05, `repair_lib*.py`). The global histogram has a long excess tail from about 6σ to 45σ that a clean tensor would not have (step05).
- Why the damaged model is so bad (perplexity ~1e205): `ln_f` output is huge in residual dimensions 496, 430 and 36 (up to ~190). The `wte` columns for those dimensions are nearly constant (e.g. col 496 ≈ -0.206, MAD ~3e-5). A single corrupted `wte` entry in one of those columns adds hundreds of nats to that token's logit, so the model outputs " socially"/" XXX" every time (steps 10, 12).
- GPT-2 also has real massive weights (columns 447/138/378/373/481 in `c_proj`, up to ~180σ) that produce the ~3000-scale massive activation. These must not be "repaired". Corruption never goes above ~45σ, so anything above 50σ_tensor is protected.

## (b) What I changed (`step21_save.py`; 73,531 entries total)
- `wte`: column-centred robust z > 12 → replaced with the column median.
- `wpe`: positions ≥ 3: residual from the median of the 4 neighbouring positions, in column-MAD units, > 12 → replaced with that neighbour median. Positions 0–2 are not smooth, so I left them alone.
- All other matrices (`c_attn`, `c_proj`, `c_fc`): row/column robust z > 16 and |v| < 50σ_tensor → replaced with the column median (≈0).
- Thresholds were picked from a handful of candidates on 1024 dev tokens (steps 16–20). No weights were fitted to a loss.
- Two attempts were dropped: "heavy row/column" protection made results worse, and a least-squares prediction of the critical `wte` columns gave no gain (step19).

## (c) Dev perplexity (first 2048 dev tokens, 512-token windows)
- damaged: 1.25e205
- repaired: **370.6** (on 1024 tokens: 283)
- Greedy output is now fluent but repetitive: "The capital of France is the French capital of France, the French capital ...".

## (d) What is not fixed / confidence
- The repaired perplexity (~370) is still far above a healthy GPT-2 (~30–40 expected on this text). A large amount of damage remains:
  - corrupted entries that look like normal values cannot be told apart by magnitude;
  - the thresholds have a recall / false-positive trade-off, and removing real large weights clearly hurts (lower thresholds were worse);
  - there is probably remaining corruption in heavy-tailed early layers (`h.0`/`h.1`) and in the low-variance `wte` columns, where even small errors are amplified by the large `ln_f` dimensions.
- I am confident about the kind of corruption. I am not confident the repair is close to the original model. I stopped at the ~40-minute budget.

## (e) Rules
I followed the rules: no network, all work inside this directory, no gradients or optimisation (only statistics, a least-squares fit and forward passes), set_num_threads(1), no sub-agents. Note: one script (step19) ran past the tool timeout and finished as a background job.
