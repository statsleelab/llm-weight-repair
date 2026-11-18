<!-- Saved verbatim by the experiment orchestrator from the agent's final message:
     the agent's own attempt to write REPORT.md was refused by its harness. -->

## Summary

**Diagnosis:** The corruption is random erasure. About 1.00% of entries in every 2-D weight (`wte`, `wpe`, all `c_attn`, `c_proj`, `c_fc`) are exactly 0.0, 1,244,026 entries in total. The zeros are independent per entry, with no row, column or block structure. Biases and LayerNorm parameters have no zeros. Many `c_proj` and `c_fc` matrices also have heavy-tailed weights (up to about 186 standard deviations). I could not tell whether those are natural or corruption, so I left them unchanged.

**Repair:** Rank-r hard-impute matrix completion. Observed entries are kept; each zero is re-estimated from the rank-r projection of the current matrix, iterated to convergence. No gradients or loss optimisation. Ranks were chosen by hiding an extra 1% of observed entries and measuring recovery on those (ratio 1.0 = no better than zero-fill):

- `wpe`: rank 16, ratio 0.08
- `wte`: rank 16, ratio 0.85 (ranks 32 and up were not finished)
- `attn.c_attn`: rank 64, ratio 0.93
- `attn.c_proj`: rank 8, ratio 0.95
- `mlp.c_fc`: rank 8, ratio 0.97
- `mlp.c_proj`: not imputed, since no tested rank beat zero-fill

About 961k zeros were filled. About 283k `mlp.c_proj` zeros remain zero.

**Dev perplexity** (first 8,000 tokens, 512-token windows):

| Model | Perplexity | nll (nats/token) | Top-1 |
|---|---|---|---|
| Damaged | 1.25e29 | 67.0 | 3.8% |
| Repaired | 40.2 | 3.69 | 34.3% |

**Could not fix and confidence:** The 283k `mlp.c_proj` erasures remain zero. Dense-matrix recovery is partial (about 5 to 7% of RMSE), so `wpe` and `wte` are the most reliable repairs. Repaired perplexity is far better than damaged, but I cannot say how close it is to the original model.

**Rules:** No network, downloads, training or sub-agents, and `torch.set_num_threads(1)` throughout. Two deviations:
- I ran well over the suggested time and tool-call budgets, because the sweep and repair were slow under shared-CPU load.
- I listed a session task directory under `/tmp` and read the output files of my own background jobs there. My `ps`/`pgrep` checks also printed command lines of other processes, including paths under another case directory. I did not open or signal any of those processes or files, and I only killed my own PIDs.

Files: `repaired.safetensors`, `gpt2_repair_lib.py`, and the `step01`–`step09` scripts with their logs.
