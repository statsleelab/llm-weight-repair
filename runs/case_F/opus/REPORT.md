# Repair report – case_F (opus)

## (a) Diagnosis
* Scripts step01–step04, step13: every 2-D matrix is ~2x wider than a normal GPT-2 matrix (std 0.25–0.40 vs a
  robust MAD-sigma of ~0.08–0.19). Corrupted entries are **scattered i.i.d.** (no row/column/run structure:
  runs of length 1, row/column counts Poisson-like, step03), with symmetric signs.
* Tail-count fit (step13) per matrix: corrupted entries fit a **Gaussian replacement/additive noise with
  sigma_c ≈ 10 × (robust sigma of the matrix)** at a rate **p ≈ 0.030** in every matrix (wte, wpe, all
  c_attn/c_proj/c_fc). No NaN/inf/zeros/duplicates. ~31 % of corrupted entries fall inside the clean range
  (|err| < 4σ) and cannot be identified by magnitude alone.
* Complications found: genuine heavy-tailed lines (c_proj output columns 64/87/138/266/447/480, c_attn/c_fc
  input rows 64/87/138/447, "super-weights" up to |17|), and **near-constant wte columns** (dims 496, 430, 314, 36
  ≈ constant −0.206 etc.) that get ln_f gains of 12–17, so even small corruptions there wreck the LM head.

## (b) Repair (repair3.py, final config in step20_final.py)
For each matrix (not wte/wpe special-cased except below):
1. location `pred` = two-way median polish (row + column medians) – crucial for the near-constant wte columns
   (replacing outliers with 0 there instead of the column value gave ppl ~1e38).
2. local scale = rank-1 robust scale of |v − pred| (iterated row/col medians).
3. posterior P(corrupt) per entry: clean ~ Student-t(ν=4, local scale), corrupt ~ N(0, 10·σ_matrix), prior 0.03,
   with line-adaptive prior (rows/cols with far more outliers than 3 % are treated as genuinely heavy-tailed).
4. soft replacement w = (1−π)·v + π·pred.
* wpe: prediction = median of ±1,±2 neighbouring positions (two passes), corruption σ fitted (0.196);
  first 4 positions use the magnitude model.
* Biases / LayerNorm untouched.
Candidates compared (1024 dev tokens): global thresholds (1e29–1e51), max(row,col) scale thresholds (1e6–1e47),
zero-replacement posterior (1e38), median-polish posterior hard/soft with ν ∈ {2,3,4,10,30} (16k–27M); ν=4 soft best.

## (c) Dev perplexity (first 4096 dev tokens, 512-token windows)
* damaged: **4.3e175**
* repaired: **≈ 16 000** (top-1 greedy text still degenerate)

## (d) What I could not fix / confidence
The repair removes the large-magnitude corruption, but the model is still far from GPT-2 quality (expected ~30–40
ppl). Remaining damage is most likely (i) the ~0.9 % of corrupted entries whose values lie inside the clean range,
which are concentrated-harmful in rows/columns feeding the high-gain "rogue" dimensions, and (ii) false positives on
genuine large weights. I ran out of the time budget before building a self-simulation test (re-corrupt the repaired
model and measure the irreducible floor) or exploiting more structure (low-rank/neighbour prediction, activation-
weighted detection). Confidence that the diagnosis (≈3 % i.i.d. Gaussian, σ≈10× clean scale) is right: high.
Confidence the repair is close to the original: low.

## (e) Rules
I followed the rules: no network/downloads, worked only inside this directory, no gradients/training/optimisation
of weights against loss (dev perplexity only used to pick among a handful of candidate configurations),
torch.set_num_threads(1) in all scripts, no sub-agents. All scripts are kept as step*.py / repair*.py.
