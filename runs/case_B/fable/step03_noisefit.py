import torch, numpy as np, math
torch.set_num_threads(1)
from gpt2_lib import load_weights, Tokenizer, perplexity
from statistics import NormalDist
_nd=NormalDist()
class norm:
    isf=staticmethod(lambda p: np.array([_nd.inv_cdf(1-float(x)) for x in np.atleast_1d(p)]) if np.ndim(p) else _nd.inv_cdf(1-float(p)))
    sf=staticmethod(lambda x: 1-_nd.cdf(float(x)))

sd = load_weights("damaged.safetensors")
# Hypothesis: 1% entries get N(0, sigma_n) noise. Estimate sigma_n from tail counts at two thresholds, and tail mean.
for k in ["wte.weight","h.0.mlp.c_fc.weight","h.3.attn.c_attn.weight","h.6.mlp.c_fc.weight","h.6.mlp.c_proj.weight","h.6.attn.c_proj.weight","h.11.mlp.c_proj.weight","h.9.attn.c_attn.weight"]:
    a=sd[k].flatten().double().numpy(); N=len(a)
    rs=np.median(np.abs(a))*1.4826
    # clean std estimate: std of entries below 6 rs
    cs=a[np.abs(a)<6*rs].std()
    out=[]
    for T in (10,14,20):
        frac=(np.abs(a)>T*rs).mean()
        # frac = pi*2*(1-Phi(T rs/sig)) with pi=.01
        sig=T*rs/norm.isf(frac/0.02)
        out.append(sig)
    tail=np.abs(a[np.abs(a)>10*rs])
    # ratio diagnostics: E[x^2 | x>T]
    print(f"{k:26s} rs={rs:.4f} cleanstd={cs:.4f} totstd={a.std():.4f} sig_n(T=10,14,20)={np.round(out,3)} ratio sig/cleanstd={np.round(np.array(out)/cs,2)} implied sig from var: {math.sqrt(max(a.var()-cs**2,0)/0.01):.3f}")
    # quantiles of tail vs. halfnormal with sig=out[0]
    q=np.array([.1,.3,.5,.7,.9,.99])
    s=out[0]; p0=2*norm.sf(10*rs/s)
    theo=s*norm.isf(p0*(1-q)/2)
    print("    tail quantiles emp",np.quantile(tail,q).round(3)," theo",theo.round(3))
try:
    pass
except Exception as e: print(e)
tok=Tokenizer("tokenizer"); ids=tok.encode(open("dev.txt").read()); print("dev tokens",len(ids))
print("damaged ppl (8k tok):", perplexity(sd, ids, max_tokens=8192))
