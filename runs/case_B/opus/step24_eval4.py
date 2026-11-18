import torch; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
D = load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read())
def ppl(sd): return perplexity(sd,ids,max_tokens=2049)
cache=torch.load("/tmp/claude-0/-home-user/19958a03-350d-552e-9865-2b580b11f60f/scratchpad/cache4.pt")
x=torch.tensor([[EOT]+ids[:16]])
def build(T,pk,wpe_mode="generic"):
    sd=dict(D); n=0
    for k,(z,mu,sc) in cache.items():
        v=D[k]; r=v-mu; t0=r[z.abs()>15].std().item()
        m=(z.abs()>T)&(r.abs()<pk*t0)
        o=v.clone(); o[m]=mu[m]; sd[k]=o; n+=m.sum().item()
    return sd,n
for T,pk in [(5,5),(4,5),(5,3),(4,3)]:
    sd,n=build(T,pk); _,hid=forward(sd,x,return_hidden=True)
    print(f"T={T} pk={pk} n={n} ppl={ppl(sd):.2f} pos0 L4 norm {hid[4][0,0].norm():.0f} pos5 L4 {hid[4][0,5].norm():.0f}",flush=True)
