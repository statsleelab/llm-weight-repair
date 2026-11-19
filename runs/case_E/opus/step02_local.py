import torch; torch.set_num_threads(1)
from gpt2_lib import *
sd = load_weights("damaged.safetensors")
def lz(v):
    # robust row/col scale via median abs
    r = v.abs().median(1,keepdim=True).values
    c = v.abs().median(0,keepdim=True).values
    s = torch.sqrt(r*c)/0.6745 * (v.abs().median()/ (v.abs().median(1).values.median()*v.abs().median(0).values.median()).sqrt())
    return v/ (r*c/ v.abs().median()) * 0.6745
for k in ["h.3.mlp.c_proj.weight","h.5.attn.c_proj.weight","wpe.weight","h.0.attn.c_attn.weight"]:
    v=sd[k]; z=lz(v); n=v.numel()
    print(k, "counts |z|>t:", [(t,(z.abs()>t).sum().item()) for t in [4,5,6,8,10,15,20,30]], "expected 0.1%:",n//1000)
    # top entries
    idx=z.abs().flatten().topk(10).indices
    print("  top z:", [round(z.flatten()[i].item(),1) for i in idx], [round(v.flatten()[i].item(),3) for i in idx])
    # bit patterns
    b = v.flatten().view(torch.int32)
    low = (b & 0xFFFF)
    print("  frac low16==0:", (low==0).float().mean().item(), " frac low8==0", ((b&0xFF)==0).float().mean().item())
