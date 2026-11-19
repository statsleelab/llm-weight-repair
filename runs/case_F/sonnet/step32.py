from repairlib import *
for n in ['ln_f.weight','h.11.ln_2.weight','h.5.ln_1.weight','h.0.ln_1.weight']:
    w=sd[n]; print(n,"median %.2f"%w.median().item(),"top",[(i.item(),round(w[i].item(),2)) for i in w.abs().topk(5).indices],"at 496,447,373,138,378,314:",[round(w[j].item(),2) for j in (496,447,373,138,378,314)])
w=sd['wte.weight']
for c in [496,314,373,447]:
    col=w[:,c]; print(c,"absmedian %.5f"%col.abs().median().item(),"frac>0.3 %.4f"%(col.abs()>0.3).float().mean().item(),"std %.3f"%col.std().item())
