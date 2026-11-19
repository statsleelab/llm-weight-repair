from repairlib import *
for k in ['wte.weight','h.3.mlp.c_fc.weight','h.3.mlp.c_proj.weight','h.5.attn.c_attn.weight','h.10.mlp.c_proj.weight']:
    w=sd[k]; m=flag(k,w,4,0.0)
    cf=m.float().mean(0); rf=m.float().mean(1)
    print(k,"overall %.4f"%m.float().mean().item(),"col frac top",[(i.item(),round(cf[i].item(),3)) for i in cf.topk(5).indices],"row frac top",[(i.item(),round(rf[i].item(),3)) for i in rf.topk(4).indices])
    cm=(w-w.median(0).values).abs().median(0).values*1.4826
    print("   col mad min/med/max",cm.min().item(),cm.median().item(),cm.max().item())
