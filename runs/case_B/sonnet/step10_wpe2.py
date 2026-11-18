import torch,numpy as np
from gpt2_lib import *
sd=load_weights("damaged.safetensors")
w=sd['wpe.weight'].numpy().astype(np.float64)
d=w[1:-1]-(w[:-2]+w[2:])/2
flag=np.abs(d)>0.06
# isolated: entry flagged and neighbors t-1,t+1 not flagged at their own d... approximate by using wider-neighborhood interpolation
T,C=w.shape
idx=np.argwhere(flag)+[1,0]
print("flagged",len(idx),"frac",len(idx)/(T*C))
# estimate orig via median of w[t-2],w[t-1],w[t+1],w[t+2] linear extrapolation
est=[];obs=[]
for t,c in idx:
    if 2<=t<T-2:
        e=(w[t-2,c]+w[t-1,c]+w[t+1,c]+w[t+2,c])/4  # crude
        est.append(e);obs.append(w[t,c])
est=np.array(est);obs=np.array(obs)
dl=obs-est
print("delta quantiles",np.quantile(np.abs(dl),[.1,.25,.5,.75,.9,.99]))
print("obs quantiles",np.quantile(np.abs(obs),[.1,.25,.5,.75,.9,.99]))
print("corr(est,obs)",np.corrcoef(est,obs)[0,1],"ratio med",np.median(obs/np.where(abs(est)<1e-3,np.nan,est)) if True else 0)
print("obs/est sign agree",np.mean(np.sign(obs)==np.sign(est)))
print("delta/ obs std",dl.std(),obs.std(), "kurt delta",((dl-dl.mean())**4).mean()/dl.var()**2)
