import torch; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
sd = load_weights("damaged.safetensors")
for k in ["h.0.attn.c_proj.weight","h.5.mlp.c_fc.weight","h.2.mlp.c_proj.weight","wpe.weight","h.10.attn.c_attn.weight"]:
    w=sd[k]; z,s=zscore(w)
    g=(w.abs()/(s/1.4826))  # unused
    m=z.abs()>15
    glob=s.median().item()
    vals=w[m]; sig=s[m]
    q=sig.quantile(torch.tensor([.25,.75]))
    lo=vals[sig<q[0]].abs(); hi=vals[sig>q[1]].abs()
    print(k,"global sd(robust median sigma)",round(glob,4),"std of matrix",round(w.std().item(),4))
    print("   flagged |val| median, low-sigma entries %.3f, high-sigma entries %.3f"%(lo.median(),hi.median()))
    print("   flagged val std %.3f ; ratio to median sigma %.1f"%(vals.std(),vals.std()/glob))
