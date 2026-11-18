import torch, numpy as np, sys
torch.set_num_threads(1)
from gpt2_lib import *
from step11_null import nullmap
sd = load_weights("damaged.safetensors")
for k in sys.argv[1:]:
    s=nullmap(sd[k]).numpy()
    r=np.argsort(np.argsort(s.flatten())).reshape(s.shape)/s.size
    # print as ascii, maybe transposed if wide
    print(k, s.shape)
    for row in r:
        print(''.join('#' if a<0.01 else ('+' if a<0.03 else ('.' if a<0.2 else ' ')) for a in row))
