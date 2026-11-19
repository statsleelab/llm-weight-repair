from repairlib import *
w=sd['wpe.weight']; f=fix_wpe(w)
m=(w!=f)
print("count",m.sum().item(),"frac",m.float().mean().item())
a=w[m]; c=f[m]
d=a-c
print("corrupt val quantiles",torch.quantile(a.abs(),torch.tensor([.1,.25,.5,.75,.9,.99])))
print("clean est abs quantiles",torch.quantile(c.abs(),torch.tensor([.1,.5,.9,.99])))
print("corr(a,c)",torch.corrcoef(torch.stack([a,c]))[0,1].item())
print("std a",a.std().item(),"std d",d.std().item(),"kurt-ish", (a**4).mean().item()/(a.var().item()**2))
print("sign agreement",(a.sign()==c.sign()).float().mean().item())
# ratio
r=(a/c)[c.abs()>0.05]; print("ratio quantiles",torch.quantile(r,torch.tensor([.1,.25,.5,.75,.9])))
# per column corrupt counts
print("cols count min/max",m.sum(0).min().item(),m.sum(0).max().item(),"rows",m.sum(1).min().item(),m.sum(1).max().item())
print("hist of a",torch.histc(a,bins=16,min=-4,max=4).tolist())
