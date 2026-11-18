import torch; torch.set_num_threads(1)
from gpt2_lib import *
from common import *
D = load_weights("damaged.safetensors")
for k,v in D.items():
    if v.dim()!=2 or k=="wpe.weight": continue
    z,s=zscore(v); m=z.abs()>6
    vals=v[m].abs()
    big=(m&(v.abs()>5)).nonzero().tolist()
    print(f"{k:26s} flagged {m.sum().item():6d} |val|>4: {(vals>4).sum().item():3d} >5: {(vals>5).sum().item():3d} >7: {(vals>7).sum().item()}", [(i,j,round(v[i,j].item(),2),round(z[i,j].item())) for i,j in big[:6]])
