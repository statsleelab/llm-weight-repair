from common import *
sd=load_weights("damaged.safetensors")
w=sd["wte.weight"]
for c in [326,709,138,447,373]:
    col=w[:,c]
    print(c,"mean",col.mean().item(),"std",col.std().item(),"median",col.median().item(),"mad",(col-col.median()).abs().median().item(),"frac>1",(col.abs()>1).float().mean().item(), "quantiles",col.quantile(torch.tensor([.01,.25,.5,.75,.99])).tolist())
print("ln_f w",sd["ln_f.weight"][[326,709,138,447,373]], "ln_f stats",sd["ln_f.weight"].mean(),sd["ln_f.weight"].std())
print("ln_f.bias",sd["ln_f.bias"][[326,709,138,447,373]])
