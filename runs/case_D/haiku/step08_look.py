import torch, sys
torch.set_num_threads(1)
from safetensors.torch import load_file
import numpy as np
np.set_printoptions(linewidth=200, precision=3, suppress=True)
raw = load_file("damaged.safetensors")
W = raw["wpe.weight"].numpy().astype(np.float64)
P = np.zeros_like(W); P[1:-1] = (W[:-2] + W[2:]) / 2
R = W - P
bad = np.abs(R) > 0.0237
colc = bad[160:230].sum(0)
j_hot = int(np.argmax(colc)); j_cold = int(np.argmin(colc))
print("hot col", j_hot, "count in 160-230:", colc[j_hot], " cold col", j_cold)
print("hot col values rows 150..175:", W[150:176, j_hot])
print("cold col values rows 150..175:", W[150:176, j_cold])
print("row 170 first 12 dims:", W[170,:12])
print("row 169 first 12 dims:", W[169,:12])
print("row 171 first 12 dims:", W[171,:12])
print("row 0 first 12:", W[0,:12])
print("row 1 first 12:", W[1,:12])
print("row 2 first 12:", W[2,:12])
