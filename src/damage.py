"""Damage operators.  Each takes the clean state dict and returns
(damaged_state_dict, masks) where masks[name] is a bool tensor marking every
entry that was touched.  Only 2-D weight matrices are damaged (they hold
99.9% of GPT-2's parameters); biases and LayerNorm gains are left alone.

Two corruption *patterns*, mirroring real storage failures:
  scatter -- i.i.d. random entries        ("bit rot")
  block   -- one contiguous rectangle per ("bad sector": a run of rows x a
             matrix                         run of columns is lost)
and two corruption *values*:
  zero    -- entries are erased (set to exactly 0)
  noise   -- entries are overwritten / perturbed with Gaussian noise
"""
from __future__ import annotations
import math
import torch


def targets(sd):
    return [k for k, v in sd.items() if v.ndim == 2]


def _scatter_mask(shape, frac, g):
    return torch.rand(shape, generator=g) < frac


def _block_mask(shape, frac, g):
    r, c = shape
    s = math.sqrt(frac)                       # same aspect ratio as the matrix
    nr, nc = max(1, round(r * s)), max(1, round(c * s))
    r0 = int(torch.randint(0, r - nr + 1, (1,), generator=g))
    c0 = int(torch.randint(0, c - nc + 1, (1,), generator=g))
    m = torch.zeros(shape, dtype=torch.bool)
    m[r0:r0 + nr, c0:c0 + nc] = True
    return m


def damage(sd, pattern: str, value: str, frac: float, seed: int, noise_k: float = 1.0):
    """pattern in {scatter, block}; value in {zero, noise}.

    noise/scatter: w += N(0, (noise_k * std(W))^2)   (additive perturbation)
    noise/block  : w  = N(0, (noise_k * std(W))^2)   (block overwritten)
    """
    g = torch.Generator().manual_seed(seed)
    out, masks = {k: v.clone() for k, v in sd.items()}, {}
    for name in targets(sd):
        w = out[name]
        m = (_scatter_mask if pattern == "scatter" else _block_mask)(w.shape, frac, g)
        if value == "zero":
            w[m] = 0.0
        else:
            eps = torch.randn(w.shape, generator=g) * (noise_k * sd[name].std())
            w[m] = eps[m] + (w[m] if pattern == "scatter" else 0.0)
        masks[name] = m
    return out, masks


CONDITIONS = {
    # name: (pattern, value, frac, noise_k)
    "scatter_zero":  ("scatter", "zero", 0.01, 0.0),
    "scatter_noise": ("scatter", "noise", 0.01, 10.0),
    "block_zero":    ("block", "zero", 0.01, 0.0),
    "block_noise":   ("block", "noise", 0.01, 1.0),
}
