import torch, sys
torch.set_num_threads(1)
from gpt2_lib import *
from repair_lib import *
ids=torch.load("dev_ids.pt").tolist()
sd=load_weights("damaged.safetensors")
for T in [float(x) for x in sys.argv[1:]]:
    r=repair(sd,T=T)
    print("hard T",T,"ppl",perplexity(r,ids,max_tokens=4096),flush=True)
