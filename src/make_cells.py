"""Stage damaged checkpoints for the follow-up LLM runs at doses other than 1%.
    python make_cells.py <clean.safetensors> <stage_dir> <secret_dir>
Same seeds as dose_response.py, so the checkpoints equal the grid cells."""
import sys, torch
from pathlib import Path
import gpt2_lib as G
from damage import CONDITIONS, damage

CELLS = {"E": ("scatter_noise", 0.001), "F": ("scatter_noise", 0.03), "G": ("block_noise", 0.03)}
SEED = 20261118
if __name__ == "__main__":
    clean = G.load_weights(sys.argv[1]); stage, secret = Path(sys.argv[2]), Path(sys.argv[3])
    for letter, (cond, frac) in CELLS.items():
        i = list(CONDITIONS).index(cond)
        pattern, value, _, k = CONDITIONS[cond]
        dmg, masks = damage(clean, pattern, value, frac, SEED + i, k)
        G.save_weights(dmg, stage / f"case_{letter}.safetensors")
        torch.save(masks, secret / f"masks_case_{letter}.pt")
        print(letter, cond, frac, sum(int(m.sum()) for m in masks.values()), flush=True)
