#!/usr/bin/env bash
# Download GPT-2 (124M), its tokenizer, WikiText-2 and BLiMP into ./assets
set -euo pipefail
mkdir -p assets/gpt2 assets/data
B=https://s3.amazonaws.com/models.huggingface.co/bert   # same files as huggingface.co/gpt2
curl -fsSL -o assets/gpt2/pytorch_model.bin "$B/gpt2-pytorch_model.bin"
curl -fsSL -o assets/gpt2/vocab.json        "$B/gpt2-vocab.json"
curl -fsSL -o assets/gpt2/merges.txt        "$B/gpt2-merges.txt"
echo "7c5d3f4b8b76583b422fcb9189ad6c89d5d97a094541ce8932dce3ecabde1421  assets/gpt2/pytorch_model.bin" | sha256sum -c -
R=https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2
for s in test valid; do curl -fsSL -o "assets/data/wikitext2_$s.txt" "$R/$s.txt"; done
[ -d assets/data/blimp_repo ] || git clone -q --depth 1 https://github.com/alexwarstadt/blimp assets/data/blimp_repo
