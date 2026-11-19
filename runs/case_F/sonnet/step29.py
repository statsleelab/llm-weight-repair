from repairlib import *
for k in ['wte.weight','h.0.attn.c_attn.weight','h.3.mlp.c_fc.weight','h.10.mlp.c_proj.weight','h.11.attn.c_proj.weight']:
    w=sd[k]; P=posterior(k,w); 
    print(k,"P>.5 frac %.4f"%(P>0.5).float().mean().item(),"sum P frac %.4f"%P.mean().item(),"zeroed>3:",((P>0.5)&(w.abs()>3)).sum().item(),"of >3:",(w.abs()>3).sum().item())
