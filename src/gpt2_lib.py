"""Minimal, dependency-light GPT-2 (124M) for the weight-repair study.

Only needs torch, numpy, safetensors and `regex`.  The model is a plain
functional forward pass over a flat ``dict[str, torch.Tensor]`` so that every
weight is trivially readable and writable -- which is the whole point of the
study.  Parameter names follow the original OpenAI / Hugging Face checkpoint
(``wte.weight``, ``h.3.attn.c_attn.weight``, ...).  The LM head is tied to
``wte.weight``.

Conv1D convention (as in the original checkpoint): ``y = x @ W + b`` with
``W`` of shape ``(n_in, n_out)``.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import regex as re
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

N_LAYER, N_HEAD, N_EMBD, N_CTX, VOCAB = 12, 12, 768, 1024, 50257
EOT = 50256


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------
def load_weights(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a ``.safetensors`` checkpoint into a flat dict of float32 tensors."""
    return {k: v.float() for k, v in load_file(str(path)).items()}


def save_weights(sd: dict[str, torch.Tensor], path: str | Path) -> None:
    save_file({k: v.detach().float().contiguous() for k, v in sd.items()}, str(path))


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


@torch.no_grad()
def forward(sd: dict[str, torch.Tensor], ids: torch.Tensor, return_hidden: bool = False):
    """ids: LongTensor (B, T) with T <= 1024.  Returns logits (B, T, VOCAB).

    With ``return_hidden=True`` also returns the list of residual-stream states
    (embedding output, then after each of the 12 blocks, before ``ln_f``).
    """
    B, T = ids.shape
    x = sd["wte.weight"][ids] + sd["wpe.weight"][:T]
    hidden = [x]
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    for i in range(N_LAYER):
        p = f"h.{i}."
        h = F.layer_norm(x, (N_EMBD,), sd[p + "ln_1.weight"], sd[p + "ln_1.bias"], 1e-5)
        qkv = h @ sd[p + "attn.c_attn.weight"] + sd[p + "attn.c_attn.bias"]
        q, k, v = qkv.split(N_EMBD, dim=-1)
        q = q.view(B, T, N_HEAD, -1).transpose(1, 2)
        k = k.view(B, T, N_HEAD, -1).transpose(1, 2)
        v = v.view(B, T, N_HEAD, -1).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        a = (att @ v).transpose(1, 2).reshape(B, T, N_EMBD)
        x = x + a @ sd[p + "attn.c_proj.weight"] + sd[p + "attn.c_proj.bias"]
        h = F.layer_norm(x, (N_EMBD,), sd[p + "ln_2.weight"], sd[p + "ln_2.bias"], 1e-5)
        m = gelu_new(h @ sd[p + "mlp.c_fc.weight"] + sd[p + "mlp.c_fc.bias"])
        x = x + m @ sd[p + "mlp.c_proj.weight"] + sd[p + "mlp.c_proj.bias"]
        hidden.append(x)
    x = F.layer_norm(x, (N_EMBD,), sd["ln_f.weight"], sd["ln_f.bias"], 1e-5)
    logits = x @ sd["wte.weight"].T
    return (logits, hidden) if return_hidden else logits


# --------------------------------------------------------------------------
# tokenizer (byte-level BPE, identical to the original GPT-2 encoder)
# --------------------------------------------------------------------------
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class Tokenizer:
    def __init__(self, directory: str | Path):
        d = Path(directory)
        self.encoder = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        self.decoder = {v: k for k, v in self.encoder.items()}
        merges = (d / "merges.txt").read_text(encoding="utf-8").split("\n")[1:-1]
        self.bpe_ranks = {tuple(m.split()): i for i, m in enumerate(merges)}
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self._bpe = lru_cache(maxsize=200_000)(self._bpe_uncached)

    def _bpe_uncached(self, token: str) -> tuple[str, ...]:
        word = tuple(token)
        while len(word) > 1:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            a, b = bigram
            new, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new.append(a + b)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = tuple(new)
        return word

    def encode(self, text: str) -> list[int]:
        out = []
        for tok in self.pat.findall(text):
            tok = "".join(self.byte_encoder[b] for b in tok.encode("utf-8"))
            out.extend(self.encoder[t] for t in self._bpe(tok))
        return out

    def decode(self, ids) -> str:
        text = "".join(self.decoder[int(i)] for i in ids)
        return bytearray(self.byte_decoder[c] for c in text).decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# convenience metrics
# --------------------------------------------------------------------------
@torch.no_grad()
def nll_windows(sd, ids: list[int], window: int = 512, max_tokens: int | None = None, batch: int = 4):
    """Mean next-token NLL (nats) and top-1 accuracy over non-overlapping
    windows, each prefixed with <|endoftext|>."""
    if max_tokens:
        ids = ids[:max_tokens]
    chunks = [ids[i : i + window] for i in range(0, len(ids) - 1, window)]
    chunks = [c for c in chunks if len(c) == window]
    tot, n, hit = 0.0, 0, 0
    for b in range(0, len(chunks), batch):
        x = torch.tensor([[EOT] + c for c in chunks[b : b + batch]])
        logits = forward(sd, x[:, :-1])
        tgt = x[:, 1:]
        tot += F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="sum").item()
        hit += (logits.argmax(-1) == tgt).sum().item()
        n += tgt.numel()
    return tot / n, hit / n


def perplexity(sd, ids, **kw) -> float:
    return math.exp(nll_windows(sd, ids, **kw)[0])


@torch.no_grad()
def generate(sd, tok: Tokenizer, prompt: str, n: int = 30) -> str:
    """Greedy continuation -- handy for eyeballing whether the model is sane."""
    ids = tok.encode(prompt)
    for _ in range(n):
        logits = forward(sd, torch.tensor([ids[-N_CTX:]]))
        ids.append(int(logits[0, -1].argmax()))
    return tok.decode(ids)


@torch.no_grad()
def sentence_logprob(sd, tok: Tokenizer, sentences: list[str]) -> list[float]:
    """Total log-probability of each sentence (conditioned on <|endoftext|>)."""
    enc = [[EOT] + tok.encode(s) for s in sentences]
    L = max(map(len, enc))
    x = torch.full((len(enc), L), EOT)
    for i, e in enumerate(enc):
        x[i, : len(e)] = torch.tensor(e)
    lp = forward(sd, x[:, :-1]).log_softmax(-1)
    tok_lp = lp.gather(-1, x[:, 1:, None])[..., 0]
    return [tok_lp[i, : len(e) - 1].sum().item() for i, e in enumerate(enc)]
