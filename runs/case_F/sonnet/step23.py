from repairlib import *
w=sd['wpe.weight']; f=fix_wpe(w)
m=(w!=f).sum(1)
print("flag counts rows 0..12",m[:12].tolist(),"median",m.median().item(), "last rows",m[-5:].tolist())
for r in range(6):
    print(r,"norm %.1f"%w[r].norm().item(),"absmedian %.3f"%w[r].abs().median().item(), "absmax %.2f"%w[r].abs().max().item(), "q90 %.3f"%w[r].abs().quantile(.9).item())
print("norm row 10, 500:",w[10].norm().item(),w[500].norm().item(), f[10].norm().item())
