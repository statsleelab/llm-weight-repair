from common import *
sd=load_weights("damaged.safetensors")
tok=Tokenizer("tokenizer")
ids=tok.encode(open("dev.txt").read())[:1024]
def build(T,kc,Tproj=None):
    r=dict(sd); n=0
    for k in KEYS(sd):
        if "c_proj" in k or k=="wpe.weight":
            o,m,med=flag(sd[k],Tproj or T,kc)
        else:
            z,_,med=zscore(sd[k]); o=z.abs()>T
        n+=o.sum().item()
        r[k]=torch.where(o,med.expand_as(sd[k]),sd[k])
    return r,n
if __name__=="__main__":
    for T,kc in [(14,5)]:
        r,n=build(T,kc); print(T,kc,n,nll_windows(r,ids,window=256))
