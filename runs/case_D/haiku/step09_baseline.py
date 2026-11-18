import torch, sys, time
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/runs/case_D/haiku")
from gpt2_lib import load_weights, Tokenizer, perplexity
tok = Tokenizer("tokenizer")
ids = tok.encode(open("dev.txt", encoding="utf-8").read())
print("dev tokens:", len(ids))
sd = load_weights("damaged.safetensors")
t = time.time()
print("damaged dev ppl (8000 tok, window 512):", perplexity(sd, ids, window=512, max_tokens=8000), "time", round(time.time()-t,1))
