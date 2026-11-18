import torch; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
D = load_weights("damaged.safetensors")
for k,v in D.items():
    if v.dim()!=2: continue
    if k=="wpe.weight":
        _,m=repair_wpe(v,6)
    else:
        _,m,tau=repair2(v,5)
    rf=m.float().mean(1); cf=m.float().mean(0)
    br=(rf>0.05).nonzero().flatten().tolist(); bc=(cf>0.05).nonzero().flatten().tolist()
    print(f"{k:26s} frac {m.float().mean()*100:.3f}% rows>5%: {br[:10]}{'...' if len(br)>10 else ''} ({len(br)}) cols>5%: {bc[:10]} ({len(bc)})")
    if k=="wpe.weight": print("   wpe flagged in first rows:", m[:8].sum(1).tolist(), "last rows", m[-4:].sum(1).tolist())
