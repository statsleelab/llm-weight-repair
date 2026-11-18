import torch,numpy as np
from step12_repair import *
fx,flag=fix_wpe(sd['wpe.weight'])
w=sd['wpe.weight'].numpy().astype(np.float64); f=fx.numpy().astype(np.float64)
m=flag
obs=w[m]; est=f[m]; dl=obs-est
print("n",m.sum())
print("|obs| q",np.quantile(np.abs(obs),[.05,.25,.5,.75,.95,.99]))
print("|est| q",np.quantile(np.abs(est),[.05,.25,.5,.75,.95,.99]))
print("obs std",obs.std(),"kurt",((obs-obs.mean())**4).mean()/obs.var()**2)
print("hist |obs|",np.histogram(np.abs(obs),bins=[0,.05,.1,.2,.4,.8,1.2,1.6,2,3,4,8])[0])
print("corr obs est",np.corrcoef(obs,est)[0,1], "sign agree",np.mean(np.sign(obs)==np.sign(est)))
print("ratio obs/est median (|est|>0.1)",np.median(obs[np.abs(est)>.1]/est[np.abs(est)>.1]), "n",(np.abs(est)>.1).sum())
# are corrupted columns scale related? col scale vs |obs|
cs=np.median(np.abs(f),0)
cc=np.nonzero(m)[1]
print("corr of log|obs| with log colscale",np.corrcoef(np.log(np.abs(obs)+1e-6),np.log(cs[cc]+1e-6))[0,1])
# unflagged remaining: count with extremely big
print("flag per row max",m.sum(1).max(),"row0 flag",m[0].sum())
