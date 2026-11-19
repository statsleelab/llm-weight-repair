from repairlib import *
for L in [1,2,3,4]:
  for c in [138,447,378]:
    w=sd[f'h.{L}.mlp.c_proj.weight'][:,c]
    v,i=w.abs().sort(descending=True)
    print(L,c,"top",[(i[j].item(),round(w[i[j]].item(),2)) for j in range(10)],"median %.3f mad %.3f"%(w.median(),(w-w.median()).abs().median()*1.48))
