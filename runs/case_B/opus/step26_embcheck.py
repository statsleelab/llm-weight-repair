import torch; torch.set_num_threads(1)
from gpt2_lib import *
R=torch.load("/tmp/claude-0/-home-user/19958a03-350d-552e-9865-2b580b11f60f/scratchpad/r5g.pt")
D=load_weights("damaged.safetensors")
W=R["wte.weight"]; P=R["wpe.weight"]; PD=D["wpe.weight"]
for j in [496,430,36,314,374]:
    c=W[:,j]; md=c.median(); mad=(c-md).abs().median()
    print("wte col",j,"median %.4f mad %.5f"%(md,mad), "max dev", (c-md).abs().max().item(), "n dev>10mad", ((c-md).abs()>10*mad+1e-4).sum().item())
    print("   wpe col",j,"damaged first 6", [round(t,3) for t in PD[:6,j].tolist()], "repaired", [round(t,3) for t in P[:6,j].tolist()], "rows 100-104", [round(t,3) for t in P[100:104,j].tolist()])
print("wpe row norms first 8 damaged", PD[:8].norm(dim=1).tolist())
print("wpe row norms first 8 repaired", P[:8].norm(dim=1).tolist())
print("wpe row norm typical", P[100:110].norm(dim=1).tolist())
