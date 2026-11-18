"""A fixed, non-LLM repair baseline, written BEFORE any agent run.

It encodes the two most obvious generic fixes and nothing else:
  * an entry that is exactly 0.0 is treated as erased -> column mean
  * an entry more than 8 robust-sigmas from its column median -> column median
(column = along axis 0, robust sigma = 1.4826 * MAD).

    python heuristic_baseline.py damaged.safetensors repaired.safetensors
"""
import sys
import torch
import gpt2_lib as G


def repair(sd):
    out = {}
    for k, w in sd.items():
        if w.ndim != 2:
            out[k] = w
            continue
        w = w.clone()
        zero = w == 0
        med = w.median(0, keepdim=True).values
        mad = (w - med).abs().median(0, keepdim=True).values * 1.4826 + 1e-12
        outl = ((w - med).abs() > 8 * mad) & ~zero
        w = torch.where(outl, med.expand_as(w), w)
        keep = (~zero).float()
        mean = (w * keep).sum(0, keepdim=True) / keep.sum(0, keepdim=True).clamp(min=1)
        out[k] = torch.where(zero, mean.expand_as(w), w)
    return out


if __name__ == "__main__":
    torch.set_num_threads(2)
    G.save_weights(repair(G.load_weights(sys.argv[1])), sys.argv[2])
